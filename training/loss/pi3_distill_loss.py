import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import *
import math
import utils3d


from einops import rearrange
from .loss_utils import check_and_fix_inf_nan, filter_by_quantile

from third_party.pi3.utils.geometry import homogenize_points, se3_inverse, depth_edge
from third_party.pi3.utils.alignment import align_points_scale

from .edge_loss import MultiScaleDeriLoss, MultiScaleCombinedGradientLoss
# from datasets import __HIGH_QUALITY_DATASETS__, __MIDDLE_QUALITY_DATASETS__

# ---------------------------------------------------------------------------
# Some functions from MoGe
# ---------------------------------------------------------------------------

def weighted_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)

def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)

def angle_diff_vec3(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12):
    return torch.atan2(torch.cross(v1, v2, dim=-1).norm(dim=-1) + eps, (v1 * v2).sum(dim=-1))

# ---------------------------------------------------------------------------
# PointLoss: Scale-invariant Local Pointmap
# ---------------------------------------------------------------------------

class PointLoss(nn.Module):
    def __init__(self, local_align_res=4096, train_conf=False, expected_dist_thresh=0.02, use_edge_loss=False):
        super().__init__()
        self.local_align_res = local_align_res
        self.criteria_local = nn.L1Loss(reduction='none')

        if use_edge_loss:
            # self.edge_loss = MultiScaleDeriLoss(operator='Scharr', norm=1, scales=6, trim=False, ssi=False, amp=False)
            self.edge_loss = MultiScaleDeriLoss(operator='Scharr', norm=1, scales=4, trim=False, ssi=True, amp=False)
            # self.edge_loss = MultiScaleCombinedGradientLoss(
            #     norm=1, 
            #     scales=4, 
            #     trim=False,
            #     ssi=True,
            #     amp=False,
            #     scharr_weight=1.0,    # First-order edges
            #     laplace_weight=0.5    # Second-order edges
            # )
        else:
            self.edge_loss = None

        self.train_conf = train_conf
        if self.train_conf:
            # self.prepare_segformer()
            self.conf_loss_fn = torch.nn.BCEWithLogitsLoss()
            self.expected_dist_thresh = expected_dist_thresh

    def prepare_segformer(self):
        from pi3.models.segformer.model import EncoderDecoder
        self.segformer = EncoderDecoder()
        self.segformer.load_state_dict(torch.load('ckpts/segformer.b0.512x512.ade.160k.pth', map_location=torch.device('cpu'), weights_only=False)['state_dict'])
        self.segformer = self.segformer.cuda()

    def predict_sky_mask(self, imgs):
        with torch.no_grad():
            output = self.segformer.inference_(imgs)
            output = output == 2
        return output

    def prepare_ROE(self, pts, mask, target_size=4096):
        B, N, H, W, C = pts.shape
        output = []
        
        for i in range(B):
            valid_pts = pts[i][mask[i]]

            if valid_pts.shape[0] > 0:
                valid_pts = valid_pts.permute(1, 0).unsqueeze(0)  # (1, 3, N1)
                # NOTE: Is is important to use nearest interpolate. Linear interpolate will lead to unstable result!
                valid_pts = F.interpolate(valid_pts, size=target_size, mode='nearest')  # (1, 3, target_size)
                valid_pts = valid_pts.squeeze(0).permute(1, 0)  # (target_size, 3)
            else:
                valid_pts = torch.ones((target_size, C), device=valid_pts.device)

            output.append(valid_pts)

        return torch.stack(output, dim=0)
    
    def normal_loss(self, points, gt_points, mask):
        not_edge = ~depth_edge(gt_points[..., 2], rtol=0.03)
        mask = torch.logical_and(mask, not_edge)

        leftup, rightup, leftdown, rightdown = points[..., :-1, :-1, :], points[..., :-1, 1:, :], points[..., 1:, :-1, :], points[..., 1:, 1:, :]
        upxleft = torch.cross(rightup - rightdown, leftdown - rightdown, dim=-1)
        leftxdown = torch.cross(leftup - rightup, rightdown - rightup, dim=-1)
        downxright = torch.cross(leftdown - leftup, rightup - leftup, dim=-1)
        rightxup = torch.cross(rightdown - leftdown, leftup - leftdown, dim=-1)

        gt_leftup, gt_rightup, gt_leftdown, gt_rightdown = gt_points[..., :-1, :-1, :], gt_points[..., :-1, 1:, :], gt_points[..., 1:, :-1, :], gt_points[..., 1:, 1:, :]
        gt_upxleft = torch.cross(gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1)
        gt_leftxdown = torch.cross(gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1)
        gt_downxright = torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)
        gt_rightxup = torch.cross(gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1)

        mask_leftup, mask_rightup, mask_leftdown, mask_rightdown = mask[..., :-1, :-1], mask[..., :-1, 1:], mask[..., 1:, :-1], mask[..., 1:, 1:]
        mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
        mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
        mask_downxright = mask_leftdown & mask_rightup & mask_leftup
        mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown

        MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(1), math.radians(90), math.radians(3)

        loss = mask_upxleft * _smooth(angle_diff_vec3(upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_leftxdown * _smooth(angle_diff_vec3(leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_downxright * _smooth(angle_diff_vec3(downxright, gt_downxright).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_rightxup * _smooth(angle_diff_vec3(rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)

        loss = loss.mean() / (4 * max(points.shape[-3:-1]))

        return loss

    def forward(self, pred, gt):
        pred_local_pts = pred['local_points']
        gt_local_pts = gt['local_points']
        valid_masks = gt['valid_masks']
        details = dict()
        final_loss = 0.0

        B, N, H, W, _ = pred_local_pts.shape

        weights_ = gt_local_pts[..., 2]
        weights_ = weights_.clamp_min(0.1 * weighted_mean(weights_, valid_masks, dim=(-2, -1), keepdim=True))
        weights_ = 1 / (weights_ + 1e-6)

        # alignment
        with torch.no_grad():
            xyz_pred_local = self.prepare_ROE(pred_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            xyz_gt_local = self.prepare_ROE(gt_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            xyz_weights_local = self.prepare_ROE((weights_[..., None]).reshape(B, N, H, W, 1), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()[:, :, 0]

            S_opt_local = align_points_scale(xyz_pred_local, xyz_gt_local, xyz_weights_local)
            S_opt_local[S_opt_local <= 0] *= -1

        aligned_local_pts = S_opt_local.view(B, 1, 1, 1, 1) * pred_local_pts

        # local point loss
        local_pts_loss = self.criteria_local(aligned_local_pts[valid_masks].float(), gt_local_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]

        # conf loss
        if self.train_conf:
            pred_conf = pred['conf']

            # probability loss
            valid = local_pts_loss.detach().mean(-1, keepdims=True) < self.expected_dist_thresh
            local_conf_loss = self.conf_loss_fn(pred_conf[valid_masks], valid.float())

            if 'sky_masks' not in gt.keys():
                sky_mask = self.predict_sky_mask(gt['imgs'].reshape(B*N, 3, H, W)).reshape(B, N, H, W)
            else:
                sky_mask = gt['sky_masks']

            sky_mask[valid_masks] = False
            if sky_mask.sum() == 0:
                sky_mask_loss = 0.0 * aligned_local_pts.mean()
            else:
                sky_mask_loss = self.conf_loss_fn(pred_conf[sky_mask], torch.zeros_like(pred_conf[sky_mask]))
            
            final_loss += 0.05 * (local_conf_loss + sky_mask_loss)
            details['local_conf_loss'] = (local_conf_loss + sky_mask_loss)

        final_loss += local_pts_loss.mean()
        details['local_pts_loss'] = local_pts_loss.mean()

        # normal loss
        # normal_batch_id = [i for i in range(len(gt['dataset_names'])) if gt['dataset_names'][i] in __HIGH_QUALITY_DATASETS__ + __MIDDLE_QUALITY_DATASETS__]
        normal_batch_id = [i for i in range(len(gt['depth_type'])) if gt['depth_type'][i] == 'synthetic']
        if len(normal_batch_id) == 0:
            normal_loss =  0.0 * aligned_local_pts.mean()
        else:
            normal_loss = self.normal_loss(aligned_local_pts[normal_batch_id], gt_local_pts[normal_batch_id], valid_masks[normal_batch_id])
        
        # Handle both scalar and tensor normal_loss
        normal_loss_mean = normal_loss if normal_loss.dim() == 0 else normal_loss.mean()
        final_loss += normal_loss_mean
        details['normal_loss'] = normal_loss_mean


        # edge loss
        if self.edge_loss is not None:
            edge_batch_id = [i for i in range(len(gt['depth_type'])) if gt['depth_type'][i] == 'synthetic']
            # edge_batch_id = [i for i in range(len(gt['depth_type']))] # calculate for all datasets
            if len(edge_batch_id) == 0:
                edge_loss =  0.0 * aligned_local_pts.mean()
            else:
                px = gt['px'][edge_batch_id]
                aligned_local_depth = aligned_local_pts[edge_batch_id][..., 2] # B T H W
                gt_local_depth = gt_local_pts[edge_batch_id][..., 2]
                aligned_local_cano_inverse_depth = 1 / aligned_local_depth * (px/W)[..., None, None]
                gt_local_cano_inverse_depth = 1 / gt_local_depth * (px/W)[..., None, None]

                aligned_local_cano_inverse_depth = rearrange(aligned_local_cano_inverse_depth, 'b t h w -> (b t) 1 h w')
                gt_local_cano_inverse_depth = rearrange(gt_local_cano_inverse_depth, 'b t h w -> (b t) 1 h w')

                valid_masks_ = rearrange(valid_masks[edge_batch_id], 'b t h w -> (b t) 1 h w')


                edge_loss = self.edge_loss(aligned_local_cano_inverse_depth, gt_local_cano_inverse_depth, valid_masks_)
            
            # Handle both scalar and tensor edge_loss
            edge_loss_mean = edge_loss if edge_loss.dim() == 0 else edge_loss.mean()
            final_loss += edge_loss_mean * 0.1 # NOTE hardcode edge loss weight = 0.1
            details['edge_loss'] = edge_loss_mean

        # [Optional] Global Point Loss
        if 'global_points' in pred and pred['global_points'] is not None:
            gt_pts = gt['global_points']

            pred_global_pts = pred['global_points'] * S_opt_local.view(B, 1, 1, 1, 1)
            global_pts_loss = self.criteria_local(pred_global_pts[valid_masks].float(), gt_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]

            final_loss += global_pts_loss.mean()
            details['global_pts_loss'] = global_pts_loss.mean()

        return final_loss, details, S_opt_local

class PointLossTeacher(PointLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, pred, pred_teacher, gt):
        pred_local_pts = pred['local_points']
        pred_teacher_local_pts = pred_teacher['local_points']
        valid_masks = gt['valid_masks']

        # valid_masks = gt['valid_masks']
        details = dict()
        final_loss = 0.0

        B, N, H, W, _ = pred_local_pts.shape

        weights_ = pred_teacher_local_pts[..., 2]
        weights_ = weights_.clamp_min(0.1 * weighted_mean(weights_, valid_masks, dim=(-2, -1), keepdim=True))
        weights_ = 1 / (weights_ + 1e-6)

        # alignment
        with torch.no_grad():
            xyz_pred_local = self.prepare_ROE(pred_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            xyz_gt_local = self.prepare_ROE(pred_teacher_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            xyz_weights_local = self.prepare_ROE((weights_[..., None]).reshape(B, N, H, W, 1), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()[:, :, 0]

            S_opt_local = align_points_scale(xyz_pred_local, xyz_gt_local, xyz_weights_local)
            # print(f"S_opt_local: {S_opt_local}")
            S_opt_local[S_opt_local <= 0] *= -1

        aligned_local_pts = S_opt_local.view(B, 1, 1, 1, 1) * pred_local_pts

        # local point loss
        local_pts_loss = self.criteria_local(aligned_local_pts[valid_masks].float(), pred_teacher_local_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]

        final_loss += local_pts_loss.mean()
        details['distill_local_pts_loss'] = local_pts_loss.mean()


        return final_loss, details, S_opt_local

# ---------------------------------------------------------------------------
# CameraLoss: Affine-invariant Camera Pose
# ---------------------------------------------------------------------------

class CameraLoss(nn.Module):
    def __init__(self, alpha=100):
        super().__init__()
        self.alpha = alpha

    def rot_ang_loss(self, R, Rgt, eps=1e-6):
        """
        Args:
            R: estimated rotation matrix [B, 3, 3]
            Rgt: ground-truth rotation matrix [B, 3, 3]
        Returns:  
            R_err: rotation angular error 
        """
        residual = torch.matmul(R.transpose(1, 2), Rgt)
        trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
        cosine = (trace - 1) / 2
        R_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))  # handle numerical errors and NaNs
        return R_err.mean()         # [0, 3.14]
    
    def forward(self, pred, gt, scale, prefix_loss_name='', valid_batch=None):
        pred_pose = pred['camera_poses']
        gt_pose = gt['camera_poses']

        B, N, _, _ = pred_pose.shape

        pred_pose_align = pred_pose.clone()
        pred_pose_align[..., :3, 3] *=  scale.view(B, 1, 1)
        
        pred_w2c = se3_inverse(pred_pose_align)
        gt_w2c = se3_inverse(gt_pose)
        
        pred_w2c_exp = pred_w2c.unsqueeze(2)
        pred_pose_exp = pred_pose_align.unsqueeze(1)
        
        gt_w2c_exp = gt_w2c.unsqueeze(2)
        gt_pose_exp = gt_pose.unsqueeze(1)
        
        pred_rel_all = torch.matmul(pred_w2c_exp, pred_pose_exp)
        gt_rel_all = torch.matmul(gt_w2c_exp, gt_pose_exp)

        mask = ~torch.eye(N, dtype=torch.bool, device=pred_pose.device)

        t_pred = pred_rel_all[..., :3, 3][:, mask, ...]
        R_pred = pred_rel_all[..., :3, :3][:, mask, ...]
        
        t_gt = gt_rel_all[..., :3, 3][:, mask, ...]
        R_gt = gt_rel_all[..., :3, :3][:, mask, ...]

        if valid_batch is not None:
            # print(f"valid_batch: {valid_batch.shape}, {t_pred.shape}")
            t_pred = t_pred[valid_batch, ...]
            R_pred = R_pred[valid_batch, ...]
            t_gt = t_gt[valid_batch, ...]
            R_gt = R_gt[valid_batch, ...]
            # print(f"after: {t_pred.shape}")

            # breakpoint()


        trans_loss = F.huber_loss(t_pred, t_gt, reduction='mean', delta=0.1)
        
        rot_loss = self.rot_ang_loss(
            R_pred.reshape(-1, 3, 3), 
            R_gt.reshape(-1, 3, 3)
        )
        
        total_loss = self.alpha * trans_loss + rot_loss

        details = {
            f'{prefix_loss_name}trans_loss': trans_loss,
            f'{prefix_loss_name}rot_loss': rot_loss,
        }

        return total_loss, details

# ---------------------------------------------------------------------------
# Final Loss
# ---------------------------------------------------------------------------

class Pi3DistillLoss(nn.Module):
    def __init__(
        self,
        train_conf=False,
        use_edge_loss=False,
    ):
        super().__init__()
        self.point_loss = PointLoss(train_conf=train_conf, use_edge_loss=use_edge_loss)
        self.point_loss_teacher = PointLossTeacher(train_conf=train_conf)
        self.camera_loss = CameraLoss()

    def prepare_gt(self, gt_raw, device, use_moge=True):
        metric_depth = gt_raw["metric_depth"].squeeze(2)
        intrinsics = gt_raw["intrinsics"]
        poses = gt_raw["extrinsics"]# B,T,4,4
        masks = gt_raw["valid_mask"].bool().squeeze(2) # B T H W
        sky_masks = gt_raw["inf_mask"].bool().squeeze(2) if "inf_mask" in gt_raw else None

        imgs = gt_raw["rgb"]

        depth_type = gt_raw.get("depth_type", None)
        is_metric_scale = gt_raw.get("is_metric_scale", None)
        if depth_type is None or depth_type[0] == "placeholder":
            return dict(
                imgs=imgs,
                global_points=None,
                local_points=None,
                valid_masks=None,
                sky_mask=None,
                camera_poses=None,
                depth_type=depth_type,
                is_metric_scale=None,
                norm_factor=None,
            )

        H, W = metric_depth.shape[-2:]

        if use_moge:
            gt_intrinsics = intrinsics.clone()
        else:
            gt_intrinsics = intrinsics.clone()
            gt_intrinsics[..., 0,:] = gt_intrinsics[..., 0,:] / W
            gt_intrinsics[..., 1,:] = gt_intrinsics[..., 1,:] / H

        
        # Get fx from the intrinsics matrix (position [0, 0])
        px_normalized = gt_intrinsics[..., 0, 0]  # Shape: (B, T)
        px = px_normalized * W  # Shape: (B, T)

        with torch.cuda.amp.autocast(enabled=False):
            gt_pts = utils3d.torch.depth_to_points(metric_depth, intrinsics=gt_intrinsics)
            gt_focal = 1 / (1 / gt_intrinsics[..., 0, 0] ** 2 + 1 / gt_intrinsics[..., 1, 1] ** 2) ** 0.5 

        B, N, H, W, _ = gt_pts.shape

        # transform to first frame camera coordinate
        w2c_target = se3_inverse(poses[:, 0])
        poses = torch.einsum('bik, bnkj -> bnij', w2c_target, poses) # to camera 0
        gt_pts = torch.einsum('bnij, bnhwj -> bnhwi', poses, homogenize_points(gt_pts))[..., :3]

        # normalize points
        valid_batch = masks.sum([-1, -2, -3]) > 0
        if valid_batch.sum() > 0:
            B_ = valid_batch.sum()
            all_pts = gt_pts[valid_batch].clone()
            all_pts[~masks[valid_batch]] = 0
            all_pts = all_pts.reshape(B_, N, -1, 3)
            all_dis = all_pts.norm(dim=-1)
            norm_factor = all_dis.sum(dim=[-1, -2]) / (masks[valid_batch].float().sum(dim=[-1, -2, -3]) + 1e-8)

            gt_pts[valid_batch] = gt_pts[valid_batch] / norm_factor[..., None, None, None, None]
            poses[valid_batch, ..., :3, 3] /= norm_factor[..., None, None]

        extrinsics = se3_inverse(poses)
        gt_local_pts = torch.einsum('bnij, bnhwj -> bnhwi', extrinsics, homogenize_points(gt_pts))[..., :3]
        
        
        return dict(
            imgs=imgs,
            global_points=gt_pts,
            local_points=gt_local_pts,
            valid_masks=masks,
            sky_mask=sky_masks,
            camera_poses=poses,
            depth_type=depth_type,
            is_metric_scale=is_metric_scale,
            px=px,
            norm_factor=norm_factor,
        )

    
    def normalize_pred(
        self, 
        pred_local_points, 
        pred_camera_poses,
        pred_conf=None,
        pred_global_points=None,
        pred_metric_scale=None,
        gt=None,
    ):
        # local_points = pred['local_points']
        # camera_poses = pred['camera_poses']
        B, N, H, W, _ = pred_local_points.shape

        masks = gt['valid_masks']
        # normalize predict points
        all_pts = pred_local_points.clone()
        all_pts[~masks] = 0
        all_pts = all_pts.reshape(B, N, -1, 3)
        all_dis = all_pts.norm(dim=-1)
        norm_factor = all_dis.sum(dim=[-1, -2]) / (masks.float().sum(dim=[-1, -2, -3]) + 1e-8)
        pred_local_points  = pred_local_points / norm_factor[..., None, None, None, None]

        if pred_global_points is not None:
            pred_global_points = pred_global_points / norm_factor[..., None, None, None, None]

        camera_poses_normalized = pred_camera_poses.clone()
        camera_poses_normalized[..., :3, 3] /= norm_factor.view(B, 1, 1)


        # if pred_metric_scale is not None:
        #     pred_metric_scale = pred_metric_scale / norm_factor.view(B, 1)

        # pred['local_points'] = local_points
        # pred['camera_poses'] = camera_poses_normalized
        pred_dict = dict(
            local_points=pred_local_points,
            camera_poses=camera_poses_normalized,
            conf=pred_conf,
            global_points=pred_global_points,
            metric_scale=pred_metric_scale,
            norm_factor=norm_factor,
        )

        return pred_dict

    def forward(
        self, 
        pred, 
        pred_teacher,
        gt_raw,
        device,
        use_moge=True,
        acts_tea=None,
        acts_stu=None,
        mapping_layers_tea=None,
        mapping_layers_stu=None,
        use_real_loss=True,
        camera_loss_weight=0.1,
        kd_alpha=1.0,
        feat_distill_type='mse',
        metric_scale_loss=False,
    ):
        gt = self.prepare_gt(gt_raw, device=device, use_moge=use_moge)

        

        if gt.get('valid_masks', None) is None:
            # Explicitly detach and manage memory for teacher confidence computation
            teacher_conf = torch.sigmoid(pred_teacher['conf'].detach()).squeeze(-1)
            
            # Compute quantile with explicit memory management
            teacher_conf_flat = teacher_conf.flatten(2, 3)
            conf_threshold = torch.quantile(
                teacher_conf_flat, 0.3, dim=-1
            )
            # Immediately delete the flattened tensor to free memory
            del teacher_conf_flat
            
            teacher_masks = (teacher_conf > conf_threshold[..., None, None]).bool()
            # Clean up confidence tensor
            del teacher_conf

            gt['valid_masks'] = teacher_masks
        
        # valid_masks_per_batch = gt['valid_masks'].reshape(batch_size, -1)
        # valid_batch = (valid_masks_per_batch.sum(-1) > (num_frames * img_h * img_w) * 0.01)

        # gt['valid_masks'][~valid_batch] = False

        # if invalid_batch.sum() > 0 and gt.get('local_points', None) is None: # if less than 10% of the points are valid, use all points
            
        #     for k in gt.keys():
        #         if isinstance(gt[k], torch.Tensor):
        #             gt[k] = 
            # final_loss = torch.tensor(0.0, device=device, requires_grad=True)
            # details = dict()
            # return final_loss, details

        pred_teacher_dict = self.normalize_pred(
            pred_teacher['local_points'], 
            pred_teacher['camera_poses'], 
            pred_teacher.get('conf', None), 
            pred_teacher.get('global_points', None), 
            gt=gt,
        )

        pred_dict = self.normalize_pred(
            pred['local_points'], 
            pred['camera_poses'], 
            pred.get('conf', None), 
            pred.get('global_points', None), 
            pred.get('metric_scale', None),
            gt=gt,
        )

        

        final_loss = 0.0
        details = dict()

        if use_real_loss and gt.get('local_points', None) is not None:
            # Local Point Loss
            point_loss, point_loss_details, scale = self.point_loss(pred_dict, gt)
            final_loss += check_and_fix_inf_nan(point_loss)
            details.update(point_loss_details)

            # Camera Loss
            camera_loss, camera_loss_details = self.camera_loss(pred_dict, gt, scale, prefix_loss_name='')
            final_loss += check_and_fix_inf_nan(camera_loss) * camera_loss_weight
            details.update(camera_loss_details)

            if pred_dict.get('metric_scale', None) is not None and metric_scale_loss:
                metric_scale_batch_id = [i for i in range(len(gt['is_metric_scale'])) if gt['is_metric_scale'][i] == 1]
                if len(metric_scale_batch_id) > 0:

                    # Scale loss: Ls = ||log(pred_scale) - stopgrad(log(target_scale))||^2
                    # Note: pred['metric_scale'] is already in log space from model output
                    pred_log_scale = torch.log(pred_dict['metric_scale'][metric_scale_batch_id])  # Shape: (B, T)

                    gt_norm_factor = gt['norm_factor'][metric_scale_batch_id]
                    pred_norm_factor = pred_dict['norm_factor'][metric_scale_batch_id].detach()
                    target_scale = scale[metric_scale_batch_id].detach() * gt_norm_factor / pred_norm_factor # Shape: (B,), stopgrad on target scale
                    target_log_scale = torch.log(target_scale)  # Shape: (B,)
                    
                    # Unsqueeze target to match prediction shape: (B,) -> (B, 1) to broadcast to (B, T)
                    target_log_scale = target_log_scale.unsqueeze(1).expand(-1, pred_log_scale.shape[1])  # Shape: (B, T)
                    
                    # Compute log scale loss
                    scale_loss = F.mse_loss(pred_log_scale, target_log_scale)
                    final_loss += scale_loss * 0.2 # NOTE hardcode metric scale loss weight = 0.2
                    details['metric_scale_loss'] = scale_loss
                else:
                    details['metric_scale_loss'] = torch.tensor(0.0, device=device, requires_grad=True) 


        # Distill Point Loss
        distill_point_loss, distill_point_loss_details, teacher_scale = self.point_loss_teacher(pred_dict, pred_teacher_dict, gt)
        final_loss += check_and_fix_inf_nan(distill_point_loss) * kd_alpha
        details.update(distill_point_loss_details)

        # Distill Camera Loss
        distill_camera_loss, distill_camera_loss_details = self.camera_loss(pred_dict, pred_teacher_dict, teacher_scale, prefix_loss_name='distill_')
        final_loss += check_and_fix_inf_nan(distill_camera_loss) * camera_loss_weight * kd_alpha
        details.update(distill_camera_loss_details)

        # Feature Distill Loss
        if acts_tea is not None and acts_stu is not None:
            feat_distill_loss = 0.0
            for (m_tea, m_stu) in zip(mapping_layers_tea, mapping_layers_stu):
                a_tea = acts_tea[m_tea]
                a_stu = acts_stu[m_stu]
                
                if feat_distill_type == 'mse':
                    feat_distill_loss_ = F.mse_loss(a_stu.float(), a_tea.detach().float(), reduction="mean")
                elif feat_distill_type == 'cosine':
                    feat_distill_loss_ = (1.0 - F.cosine_similarity(a_stu.float(), a_tea.detach().float(), dim=-1)).mean()
                else:
                    raise ValueError(f"Invalid feat_distill_type: {feat_distill_type}")
                feat_distill_loss += feat_distill_loss_
            final_loss += check_and_fix_inf_nan(feat_distill_loss) * 0.5 * kd_alpha
            details['feat_distill_loss'] = feat_distill_loss

        # print(f"details: {details}")
        # for k, v in details.items():
        #     if torch.any(torch.isnan(v)) or torch.any(torch.isinf(v)):
        #         print(f"Bug in {k}: {v}")
        #         breakpoint()

        return final_loss, details


class Pi3DistillLossMixedBatch(nn.Module):
    """
    Modified version of Pi3DistillLoss that handles mixed batches where some samples 
    have valid depth and others are placeholders.
    
    Key differences from Pi3DistillLoss:
    - prepare_gt: Processes only valid samples, creates valid_depth_batch mask
    - forward: Computes real GT losses only on samples with valid depth
    - Distillation losses computed on all samples
    """
    def __init__(self, train_conf=False, use_edge_loss=False):
        super().__init__()
        self.point_loss = PointLoss(train_conf=train_conf, use_edge_loss=use_edge_loss)
        self.point_loss_teacher = PointLossTeacher(train_conf=train_conf)
        self.camera_loss = CameraLoss()

    def prepare_gt(self, gt_raw, device, use_moge=True):
        """
        Modified to handle mixed batches with both valid and placeholder samples.
        
        Returns:
            dict with additional 'valid_depth_batch' mask indicating which samples have valid depth
        """
        metric_depth = gt_raw["metric_depth"].squeeze(2)
        intrinsics = gt_raw["intrinsics"]
        poses = gt_raw["extrinsics"]# B,T,4,4
        masks = gt_raw["valid_mask"].bool().squeeze(2) # B T H W
        sky_masks = gt_raw["inf_mask"].bool().squeeze(2) if "inf_mask" in gt_raw else None

        imgs = gt_raw["rgb"]
        B = imgs.shape[0]

        # Check depth_type per sample in batch
        depth_type = gt_raw.get("depth_type", None)
        if depth_type is None:
            # If no depth_type provided, assume all are placeholders
            valid_depth_batch = torch.zeros(B, dtype=torch.bool, device=device)
        else:
            # Create mask for samples with valid depth (not placeholder)
            valid_depth_batch = torch.tensor(
                [dt != "placeholder" for dt in depth_type], 
                dtype=torch.bool, 
                device=device
            )
        
        # If no valid samples in batch, return early with None values
        if not valid_depth_batch.any():
            return dict(
                imgs=imgs,
                global_points=None,
                local_points=None,
                valid_masks=None,
                sky_mask=None,
                camera_poses=None,
                depth_type=depth_type,
                valid_depth_batch=valid_depth_batch,
            )

        H, W = metric_depth.shape[-2:]

        if use_moge:
            gt_intrinsics = intrinsics.clone()
        else:
            gt_intrinsics = intrinsics.clone()
            gt_intrinsics[..., 0,:] = gt_intrinsics[..., 0,:] / W
            gt_intrinsics[..., 1,:] = gt_intrinsics[..., 1,:] / H

        
        # Get fx from the intrinsics matrix (position [0, 0])
        px_normalized = gt_intrinsics[..., 0, 0]  # Shape: (B, T)
        px = px_normalized * W  # Shape: (B, T)

        # Initialize outputs with zeros for all batch samples
        N = poses.shape[1]
        gt_pts = torch.zeros(B, N, H, W, 3, device=device, dtype=imgs.dtype)
        gt_local_pts = torch.zeros(B, N, H, W, 3, device=device, dtype=imgs.dtype)
        poses_normalized = poses.clone()

        # Only process samples with valid depth
        if valid_depth_batch.any():
            with torch.cuda.amp.autocast(enabled=False):
                gt_pts_valid = utils3d.torch.depth_to_points(
                    metric_depth[valid_depth_batch], 
                    intrinsics=gt_intrinsics[valid_depth_batch]
                )
                gt_focal = 1 / (1 / gt_intrinsics[valid_depth_batch, :, 0, 0] ** 2 + 
                               1 / gt_intrinsics[valid_depth_batch, :, 1, 1] ** 2) ** 0.5 

            # Transform to first frame camera coordinate for valid samples
            poses_valid = poses[valid_depth_batch]
            w2c_target = se3_inverse(poses_valid[:, 0])
            poses_valid = torch.einsum('bik, bnkj -> bnij', w2c_target, poses_valid) # to camera 0
            gt_pts_valid = torch.einsum('bnij, bnhwj -> bnhwi', poses_valid, 
                                       homogenize_points(gt_pts_valid))[..., :3]

            # Normalize points for valid samples
            valid_pixels = masks[valid_depth_batch].sum([-1, -2, -3]) > 0
            if valid_pixels.any():
                B_valid = valid_pixels.sum()
                all_pts = gt_pts_valid[valid_pixels].clone()
                all_pts[~masks[valid_depth_batch][valid_pixels]] = 0
                all_pts = all_pts.reshape(B_valid, N, -1, 3)
                all_dis = all_pts.norm(dim=-1)
                norm_factor = all_dis.sum(dim=[-1, -2]) / (masks[valid_depth_batch][valid_pixels].float().sum(dim=[-1, -2, -3]) + 1e-8)

                gt_pts_valid[valid_pixels] = gt_pts_valid[valid_pixels] / norm_factor[..., None, None, None, None]
                poses_valid[valid_pixels, ..., :3, 3] /= norm_factor[..., None, None]

            # Compute local points for valid samples
            extrinsics_valid = se3_inverse(poses_valid)
            gt_local_pts_valid = torch.einsum('bnij, bnhwj -> bnhwi', extrinsics_valid, 
                                             homogenize_points(gt_pts_valid))[..., :3]
            
            # Copy valid results back to full batch tensors
            gt_pts[valid_depth_batch] = gt_pts_valid
            gt_local_pts[valid_depth_batch] = gt_local_pts_valid
            poses_normalized[valid_depth_batch] = poses_valid
        
        # Mask out invalid samples in the masks tensor
        masks = masks.clone()
        masks[~valid_depth_batch] = False
        if sky_masks is not None:
            sky_masks = sky_masks.clone()
            sky_masks[~valid_depth_batch] = False
        
        return dict(
            imgs=imgs,
            global_points=gt_pts,
            local_points=gt_local_pts,
            valid_masks=masks,
            sky_mask=sky_masks,
            camera_poses=poses_normalized,
            depth_type=depth_type,
            px=px,
            valid_depth_batch=valid_depth_batch,  # New: mask indicating which samples have valid depth
        )

    def normalize_pred(
        self, 
        pred_local_points, 
        pred_camera_poses,
        pred_conf=None,
        pred_global_points=None,
        gt=None,
    ):
        """
        Normalize predictions using masks from gt.
        Note: gt['valid_masks'] already contains hybrid masks (GT for valid, teacher for invalid)
        as set up in forward() method.
        """
        B, N, H, W, _ = pred_local_points.shape

        masks = gt['valid_masks']

        # normalize predict points
        all_pts = pred_local_points.clone()
        all_pts[~masks] = 0
        all_pts = all_pts.reshape(B, N, -1, 3)
        all_dis = all_pts.norm(dim=-1)
        norm_factor = all_dis.sum(dim=[-1, -2]) / (masks.float().sum(dim=[-1, -2, -3]) + 1e-8)

        pred_local_points_normalized = pred_local_points / norm_factor[..., None, None, None, None]
        pred_camera_poses_normalized = pred_camera_poses.clone()
        pred_camera_poses_normalized[..., :3, 3] /= norm_factor[..., None, None]

        # unproject local points using camera poses
        pred_global_points_normalized = torch.einsum('bnij, bnhwj -> bnhwi', pred_camera_poses_normalized, homogenize_points(pred_local_points_normalized))[..., :3]

        return dict(
            global_points=pred_global_points_normalized,
            local_points=pred_local_points_normalized,
            camera_poses=pred_camera_poses_normalized,
        )

    def forward(
        self, 
        pred, 
        pred_teacher, 
        gt_raw, 
        device,
        use_moge=True,
        acts_tea=None,
        acts_stu=None,
        mapping_layers_tea=None,
        mapping_layers_stu=None,
        use_real_loss=True,
        camera_loss_weight=0.1,
        kd_alpha=1.0,
        feat_distill_type='mse',
    ):
        """
        Modified forward to handle mixed batches.
        
        Key changes:
        - Only computes real GT losses (point + camera) on samples with valid depth
        - Distillation losses computed on all samples
        """
        gt = self.prepare_gt(gt_raw, device=device, use_moge=use_moge)

        # Get valid_depth_batch mask
        valid_depth_batch = gt.get('valid_depth_batch', None)

        # print(f"valid_depth_batch: {valid_depth_batch}")
        
        # Compute teacher masks only for samples without valid depth
        if valid_depth_batch is not None:
            invalid_depth_batch = ~valid_depth_batch
            
            if invalid_depth_batch.any():
                # Only compute teacher confidence for invalid samples
                teacher_conf_invalid = torch.sigmoid(pred_teacher['conf'][invalid_depth_batch].detach()).squeeze(-1)
                
                # Compute quantile for invalid samples (flatten spatial dimensions H, W)
                teacher_conf_flat = teacher_conf_invalid.flatten(2, 3)  # (num_invalid, N, H*W)
                conf_threshold = torch.quantile(teacher_conf_flat, 0.3, dim=-1)  # (num_invalid, N)
                del teacher_conf_flat
                
                # Create teacher masks for invalid samples only
                teacher_masks_invalid = (teacher_conf_invalid > conf_threshold[:, :, None, None]).bool()
                del teacher_conf_invalid
                
                # Initialize valid_masks if it's None (happens when all samples are placeholder)
                if gt['valid_masks'] is None:
                    gt['valid_masks'] = torch.zeros_like(pred_teacher['conf'].squeeze(-1), dtype=torch.bool)
                
                # Create hybrid masks in-place: GT masks for valid samples, teacher masks for invalid samples
                gt['valid_masks'][invalid_depth_batch] = teacher_masks_invalid
        else:
            # Fallback: compute teacher masks for all samples (original behavior)
            teacher_conf = torch.sigmoid(pred_teacher['conf'].detach()).squeeze(-1)
            teacher_conf_flat = teacher_conf.flatten(2, 3)
            conf_threshold = torch.quantile(teacher_conf_flat, 0.3, dim=-1)
            del teacher_conf_flat
            teacher_masks = (teacher_conf > conf_threshold[..., None, None]).bool()
            del teacher_conf
            
            if gt.get('valid_masks', None) is None:
                gt['valid_masks'] = teacher_masks

        pred_teacher_dict = self.normalize_pred(
            pred_teacher['local_points'], 
            pred_teacher['camera_poses'], 
            pred_teacher.get('conf', None), 
            pred_teacher.get('global_points', None), 
            gt=gt,
        )

        pred_dict = self.normalize_pred(
            pred['local_points'], 
            pred['camera_poses'], 
            pred.get('conf', None), 
            pred.get('global_points', None), 
            gt=gt,
        )

        final_loss = 0.0
        details = dict()

        # Get valid depth batch mask
        valid_depth_batch = gt.get('valid_depth_batch', None)
        has_valid_depth_samples = (valid_depth_batch is not None and valid_depth_batch.any())

        if use_real_loss and gt.get('local_points', None) is not None and has_valid_depth_samples:
            # Only compute real losses for samples with valid depth
            # Create masked versions of pred and gt for samples with valid depth
            pred_dict_masked = {
                k: v[valid_depth_batch] if isinstance(v, torch.Tensor) and v.shape[0] == valid_depth_batch.shape[0] else v
                for k, v in pred_dict.items()
            }
            gt_masked = {
                k: v[valid_depth_batch] if isinstance(v, torch.Tensor) and v.shape[0] == valid_depth_batch.shape[0] else v
                for k, v in gt.items()
            }
            
            # Local Point Loss
            point_loss, point_loss_details, scale = self.point_loss(pred_dict_masked, gt_masked)
            final_loss += check_and_fix_inf_nan(point_loss)
            details.update(point_loss_details)

            # Camera Loss
            camera_loss, camera_loss_details = self.camera_loss(pred_dict_masked, gt_masked, scale, prefix_loss_name='')
            final_loss += check_and_fix_inf_nan(camera_loss) * camera_loss_weight
            details.update(camera_loss_details)


        # Distill Point Loss - always computed (uses teacher masks for all samples)
        distill_point_loss, distill_point_loss_details, teacher_scale = self.point_loss_teacher(pred_dict, pred_teacher_dict, gt)
        final_loss += check_and_fix_inf_nan(distill_point_loss) * kd_alpha
        details.update(distill_point_loss_details)

        # Distill Camera Loss
        distill_camera_loss, distill_camera_loss_details = self.camera_loss(pred_dict, pred_teacher_dict, teacher_scale, prefix_loss_name='distill_')
        final_loss += check_and_fix_inf_nan(distill_camera_loss) * camera_loss_weight * kd_alpha
        details.update(distill_camera_loss_details)

        # Feature Distill Loss
        if acts_tea is not None and acts_stu is not None:
            feat_distill_loss = 0.0
            for (m_tea, m_stu) in zip(mapping_layers_tea, mapping_layers_stu):
                a_tea = acts_tea[m_tea]
                a_stu = acts_stu[m_stu]
                
                if feat_distill_type == 'mse':
                    feat_distill_loss_ = F.mse_loss(a_stu.float(), a_tea.detach().float(), reduction="mean")
                elif feat_distill_type == 'cosine':
                    feat_distill_loss_ = (1.0 - F.cosine_similarity(a_stu.float(), a_tea.detach().float(), dim=-1)).mean()
                else:
                    raise ValueError(f"Invalid feat_distill_type: {feat_distill_type}")
                feat_distill_loss += feat_distill_loss_
            final_loss += check_and_fix_inf_nan(feat_distill_loss) * 0.5 * kd_alpha
            details['feat_distill_loss'] = feat_distill_loss

        return final_loss, details

