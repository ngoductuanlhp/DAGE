import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import *
import math
import utils3d
import numpy as np

from einops import rearrange
from third_party.pi3.utils.geometry import homogenize_points, se3_inverse, depth_edge
from third_party.pi3.utils.alignment import align_points_scale

from .edge_loss import MultiScaleDeriLoss, MultiScaleCombinedGradientLoss
# from third_party.pi3.models.segformer.model import EncoderDecoder

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


# class ConfidenceLoss(nn.Module):
#     def __init__(self):
#         super().__init__(expected_dist_thresh=0.02)
#         self.loss_fn = nn.BCEWithLogitsLoss()
#         self.expected_dist_thresh = expected_dist_thresh

#     def forward(self, pred, gt):
#         valid = local_pts_loss.detach().mean(-1, keepdims=True) < self.expected_dist_thresh
#         local_conf_loss = self.conf_loss_fn(pred_conf[valid_masks], valid.float())

#         if 'sky_masks' not in gt.keys():
#             sky_mask = self.predict_sky_mask(gt['imgs'].reshape(B*N, 3, H, W)).reshape(B, N, H, W)
#         else:
#             sky_mask = gt['sky_masks']

#         sky_mask[valid_masks] = False
#         if sky_mask.sum() == 0:
#             sky_mask_loss = 0.0 * aligned_local_pts.mean()
#         else:
#             sky_mask_loss = self.conf_loss_fn(pred_conf[sky_mask], torch.zeros_like(pred_conf[sky_mask]))
        
#         final_loss += 0.05 * (local_conf_loss + sky_mask_loss)
#         details['local_conf_loss'] = (local_conf_loss + sky_mask_loss)

class PointLoss(nn.Module):
    def __init__(self, local_align_res=4096, train_conf=False, expected_dist_thresh=0.03, use_edge_loss=False):
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
            self.prepare_segformer()
            self.conf_loss_fn = torch.nn.BCEWithLogitsLoss()
            self.expected_dist_thresh = expected_dist_thresh

    def prepare_segformer(self):

        import albumentations as A
        import segmentation_models_pytorch as smp
        # self.segformer = EncoderDecoder()
        # self.segformer.load_state_dict(torch.load('ckpts/segformer.b0.512x512.ade.160k.pth', map_location=torch.device('cpu'), weights_only=False)['state_dict'])
        # self.segformer = self.segformer.cuda()
        checkpoint = "smp-hub/segformer-b0-512x512-ade-160k"
        self.segformer = smp.from_pretrained(checkpoint).eval().cuda()
        self.segformer_preprocessing = A.Compose.from_pretrained(checkpoint)
    
    def preprocess_batch_sky_mask(self, imgs):
        """
        Process batch of images for segformer with aspect ratio preservation.
        Args:
            imgs: torch.Tensor of shape (B, 3, H, W) in range [0, 1]
        Returns:
            torch.Tensor with max dimension = 512, normalized
        """
        B, C, H, W = imgs.shape
        
        # 1. Resize so max dimension is 512, keeping aspect ratio
        max_dim = max(H, W)
        scale = 512 / max_dim
        new_h = int(H * scale) // 32 * 32
        new_w = int(W * scale) // 32 * 32
        
        imgs_resized = F.interpolate(imgs, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        # 2. Normalize (mean/std from 0-255 scale to 0-1 scale)
        mean = torch.tensor([123.675, 116.28, 103.53], device=imgs.device).view(1, 3, 1, 1) / 255.0
        std = torch.tensor([58.395, 57.12, 57.375], device=imgs.device).view(1, 3, 1, 1) / 255.0
        
        imgs_normalized = (imgs_resized - mean) / std
        
        return imgs_normalized
        
    @torch.no_grad()
    def predict_sky_mask(self, imgs):
        """
        Args:
            imgs: torch.Tensor of shape (B, 3, H, W) in range [0, 1]
        """
        B, C, H, W = imgs.shape
        
        # Preprocess entire batch at once (max_dim=512, aspect ratio preserved)
        input_tensor = self.preprocess_batch_sky_mask(imgs)
        
        # Run segformer on the batch
        output = self.segformer(input_tensor)
        
        sky_mask = output.argmax(dim=1) == 2
        sky_mask = F.interpolate(sky_mask.float()[:, None], size=(H, W), mode='nearest').squeeze(1).bool()
        
        return sky_mask

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

    def get_pred_gt_scale(self, pred, gt):
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

        return S_opt_local

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
              
            # NOTE for conf, only train with conf loss
            return final_loss, details, S_opt_local

        final_loss += local_pts_loss.mean()
        details['local_pts_loss'] = local_pts_loss.mean()

        # normal loss
        # normal_batch_id = [i for i in range(len(gt['dataset_names'])) if gt['dataset_names'][i] in __HIGH_QUALITY_DATASETS__ + __MIDDLE_QUALITY_DATASETS__]
        normal_batch_id = [i for i in range(len(gt['depth_type'])) if gt['depth_type'][i] == 'synthetic']
        if len(normal_batch_id) == 0:
            normal_loss =  0.0 * aligned_local_pts.mean()
        else:
            normal_loss = self.normal_loss(aligned_local_pts[normal_batch_id], gt_local_pts[normal_batch_id], valid_masks[normal_batch_id])
        final_loss += normal_loss.mean()

        details['normal_loss'] = normal_loss.mean()

        # # Handle both scalar and tensor normal_loss, or Python float
        # if isinstance(normal_loss, torch.Tensor):
        #     normal_loss_mean = normal_loss if normal_loss.dim() == 0 else normal_loss.mean()
        # else:
        #     normal_loss_mean = normal_loss  # Python float
        # final_loss += normal_loss_mean
        # details['normal_loss'] = normal_loss_mean

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

            final_loss += edge_loss.mean() * 0.02 # NOTE hardcode edge loss weight = 0.1
            details['edge_loss'] = edge_loss.mean()

            #     final_loss += edge_loss.mean() * 0.1 # NOTE hardcode edge loss weight = 0.1
            # details['edge_loss'] = edge_loss.mean()
            # Handle both scalar and tensor edge_loss, or Python float
            # if isinstance(edge_loss, torch.Tensor):
            #     edge_loss_mean = edge_loss if edge_loss.dim() == 0 else edge_loss.mean()
            # else:
            #     edge_loss_mean = edge_loss  # Python float
            # final_loss += edge_loss_mean * 0.02 # NOTE hardcode edge loss weight = 0.1
            # details['edge_loss'] = edge_loss_mean

        # [Optional] Global Point Loss
        if 'global_points' in pred and pred['global_points'] is not None:
            gt_pts = gt['global_points']

            pred_global_pts = pred['global_points'] * S_opt_local.view(B, 1, 1, 1, 1)
            global_pts_loss = self.criteria_local(pred_global_pts[valid_masks].float(), gt_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]

            final_loss += global_pts_loss.mean()
            details['global_pts_loss'] = global_pts_loss.mean()

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
    
    def forward(self, pred, gt, scale):
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

        trans_loss = F.huber_loss(t_pred, t_gt, reduction='mean', delta=0.1)
        
        rot_loss = self.rot_ang_loss(
            R_pred.reshape(-1, 3, 3), 
            R_gt.reshape(-1, 3, 3)
        )
        
        total_loss = self.alpha * trans_loss + rot_loss

        return total_loss, dict(trans_loss=trans_loss, rot_loss=rot_loss)

# ---------------------------------------------------------------------------
# Final Loss
# ---------------------------------------------------------------------------

class Pi3Loss(nn.Module):
    def __init__(
        self,
        train_conf=False,
        use_edge_loss=False,
    ):
        super().__init__()
        self.point_loss = PointLoss(train_conf=train_conf, use_edge_loss=use_edge_loss)
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
        else:
            norm_factor = None

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
    
    def normalize_pred(self, pred, gt):
        local_points = pred['local_points']
        camera_poses = pred['camera_poses']
        B, N, H, W, _ = local_points.shape
        masks = gt['valid_masks']

        # normalize predict points
        all_pts = local_points.clone()
        all_pts[~masks] = 0
        all_pts = all_pts.reshape(B, N, -1, 3)
        all_dis = all_pts.norm(dim=-1)
        norm_factor = all_dis.sum(dim=[-1, -2]) / (masks.float().sum(dim=[-1, -2, -3]) + 1e-8)
        local_points  = local_points / norm_factor[..., None, None, None, None]

        if 'global_points' in pred and pred['global_points'] is not None:
            pred['global_points'] /= norm_factor[..., None, None, None, None]

        camera_poses_normalized = camera_poses.clone()
        camera_poses_normalized[..., :3, 3] /= norm_factor.view(B, 1, 1)

        pred['local_points'] = local_points
        pred['camera_poses'] = camera_poses_normalized

        pred['norm_factor'] = norm_factor

        # if 'metric_scale' in pred and pred['metric_scale'] is not None:
        #     pred['metric_scale'] = pred['metric_scale'] / norm_factor.view(B, 1)

        return pred

    def forward(
        self, 
        pred, 
        gt_raw,
        device,
        use_moge=True,
        skip_cam_loss=False,
        metric_scale_loss=False,
    ):
        gt = self.prepare_gt(gt_raw, device=device, use_moge=use_moge)
        pred = self.normalize_pred(pred, gt)

        final_loss = 0.0
        details = dict()

        # Local Point Loss
        point_loss, point_loss_details, scale = self.point_loss(pred, gt)
        final_loss += point_loss
        details.update(point_loss_details)

        # Camera Loss
        if not skip_cam_loss:
            camera_loss, camera_loss_details = self.camera_loss(pred, gt, scale)
            final_loss += camera_loss * 0.1
            details.update(camera_loss_details)


        if pred.get('metric_scale', None) is not None and gt['norm_factor'] is not None and metric_scale_loss:
            metric_scale_batch_id = [i for i in range(len(gt['is_metric_scale'])) if gt['is_metric_scale'][i] == 1]
            if len(metric_scale_batch_id) > 0:

                # Scale loss: Ls = ||log(pred_scale) - stopgrad(log(target_scale))||^2
                # Note: pred['metric_scale'] is already in log space from model output
                pred_log_scale = torch.log(pred['metric_scale'][metric_scale_batch_id])  # Shape: (B, T)

                gt_norm_factor = gt['norm_factor'][metric_scale_batch_id]
                pred_norm_factor = pred['norm_factor'][metric_scale_batch_id].detach()
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

        return final_loss, details


class Pi3MetricOnlyLoss(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.point_loss = PointLoss(train_conf=False, use_edge_loss=False)
        # self.camera_loss = CameraLoss()

    def prepare_gt(self, gt_raw, device, use_moge=True):

        metric_depth = gt_raw["metric_depth"].squeeze(2)
        intrinsics = gt_raw["intrinsics"]
        poses = gt_raw["extrinsics"]# B,T,4,4
        masks = gt_raw["valid_mask"].bool().squeeze(2) # B T H W
        sky_masks = gt_raw["inf_mask"].bool().squeeze(2) if "inf_mask" in gt_raw else None

        imgs = gt_raw["rgb"]

        depth_type = gt_raw.get("depth_type", None)
        is_metric_scale = gt_raw.get("is_metric_scale", None)


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
        else:
            norm_factor = None

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
    
    def normalize_pred(self, pred, gt):
        local_points = pred['local_points']
        camera_poses = pred['camera_poses']
        B, N, H, W, _ = local_points.shape
        masks = gt['valid_masks']

        # normalize predict points
        all_pts = local_points.clone()
        all_pts[~masks] = 0
        all_pts = all_pts.reshape(B, N, -1, 3)
        all_dis = all_pts.norm(dim=-1)
        norm_factor = all_dis.sum(dim=[-1, -2]) / (masks.float().sum(dim=[-1, -2, -3]) + 1e-8)
        local_points  = local_points / norm_factor[..., None, None, None, None]

        if 'global_points' in pred and pred['global_points'] is not None:
            pred['global_points'] /= norm_factor[..., None, None, None, None]

        camera_poses_normalized = camera_poses.clone()
        camera_poses_normalized[..., :3, 3] /= norm_factor.view(B, 1, 1)

        pred['local_points'] = local_points
        pred['camera_poses'] = camera_poses_normalized

        pred['norm_factor'] = norm_factor

        # if 'metric_scale' in pred and pred['metric_scale'] is not None:
        #     pred['metric_scale'] = pred['metric_scale'] / norm_factor.view(B, 1)

        return pred

    def forward(
        self, 
        pred, 
        gt_raw,
        device,
        use_moge=True,
    ):
        gt = self.prepare_gt(gt_raw, device=device, use_moge=use_moge)
        pred = self.normalize_pred(pred, gt)

        scale_loss = torch.tensor(0.0, device=device, requires_grad=True, dtype=torch.float32)
        details = dict()

        # Local Point Loss
        # point_loss, point_loss_details, scale = self.point_loss(pred, gt)
        # final_loss += point_loss
        # details.update(point_loss_details)

        with torch.no_grad():
            scale = self.point_loss.get_pred_gt_scale(pred, gt)


        if pred.get('metric_scale', None) is not None and gt['norm_factor'] is not None:
            metric_scale_batch_id = [i for i in range(len(gt['is_metric_scale'])) if gt['is_metric_scale'][i] == 1]
            if len(metric_scale_batch_id) > 0:

                # Scale loss: Ls = ||log(pred_scale) - stopgrad(log(target_scale))||^2
                # Note: pred['metric_scale'] is already in log space from model output
                pred_log_scale = torch.log(pred['metric_scale'][metric_scale_batch_id])  # Shape: (B, T)

                gt_norm_factor = gt['norm_factor'][metric_scale_batch_id]
                pred_norm_factor = pred['norm_factor'][metric_scale_batch_id].detach()
                target_scale = scale[metric_scale_batch_id].detach() * gt_norm_factor / pred_norm_factor # Shape: (B,), stopgrad on target scale
                target_log_scale = torch.log(target_scale)  # Shape: (B,)
                
                # Unsqueeze target to match prediction shape: (B,) -> (B, 1) to broadcast to (B, T)
                target_log_scale = target_log_scale.unsqueeze(1).expand(-1, pred_log_scale.shape[1])  # Shape: (B, T)
                
                # Compute log scale loss
                scale_loss = F.mse_loss(pred_log_scale, target_log_scale)
                details['metric_scale_loss'] = scale_loss
            else:
                details['metric_scale_loss'] = torch.tensor(0.0, device=device, requires_grad=True)
                

        return scale_loss, details

