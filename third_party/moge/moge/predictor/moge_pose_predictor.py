import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from einops import rearrange
import numpy as np
from typing import Union, Dict

from third_party.moge.moge.model.moge_model_v1_pose_v2 import MoGeModelV1PoseV2
from third_party.moge.moge.model.moge_model_v2_pose_v2 import MoGeModelV2PoseV2

from third_party.pi3.models.pi3 import Pi3
from third_party.pi3.models.pi3v3_training import Pi3V3
from third_party.pi3.models.pi3v5_training import Pi3V5
from third_party.moge.moge.model.cut3r import Cut3r
import cv2


from third_party.pi3.utils.geometry import homogenize_points, se3_inverse
from third_party.pi3.utils.alignment import align_points_scale
from training_unip.util.timer import CUDATimer
from collections import defaultdict


def weighted_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)

class MoGePosePredictor(nn.Module):
    def __init__(self, model_name: str, prior_model_name: str, device: torch.device, model_pretrained_path: str, prior_pretrained_path: str = None):
        super(MoGePosePredictor, self).__init__()
        # self.model = eval(model_name)()

        self.model_name = model_name
        self.prior_model_name = prior_model_name

        if self.prior_model_name == "Pi3":
            if prior_pretrained_path is not None:
                self.prior_model = Pi3.from_pretrained(prior_pretrained_path).to(device).eval()
            else:
                self.prior_model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
        elif self.prior_model_name == "Pi3V3":
            self.prior_model = Pi3V3.from_pretrained(prior_pretrained_path).to(device).eval()
        elif self.prior_model_name == "Pi3V5":
            self.prior_model = Pi3V5.from_pretrained(prior_pretrained_path, strict=False).to(device).eval()
        elif self.prior_model_name == "Cut3r":
            self.prior_model = Cut3r.from_pretrained("/home/colligo/dngo/projects/CUT3R/src/cut3r_512_dpt_4_64.pth", strict=False).to(device).eval()
        else:
            raise ValueError(f"Prior model {self.prior_model_name} not supported")

        self.model = eval(model_name).from_pretrained(pretrained_model_name_or_path=model_pretrained_path, strict=True).to(device).eval()

        self.prior_patch_size = 14 if "Pi3" in self.prior_model_name else 16
        self.prior_max_size = 518 if "Pi3" in self.prior_model_name else 512

    def prepare_ROE(self, pts, mask, target_size=4096):
        C = pts.shape[-1]

        valid_pts = pts[mask]
        if valid_pts.shape[0] > 0:
            valid_pts = valid_pts.permute(1, 0).unsqueeze(0)  # (1, 3, N1)
            # NOTE: Is is important to use nearest interpolate. Linear interpolate will lead to unstable result!
            valid_pts = F.interpolate(valid_pts, size=target_size, mode='nearest')  # (1, 3, target_size)
            valid_pts = valid_pts.squeeze(0).permute(1, 0)  # (target_size, 3)
        else:
            valid_pts = torch.ones((target_size, C), device=valid_pts.device)

        return valid_pts


    def get_align_scale(self, pred_local_pts, prior_local_pts, prior_mask, local_align_res=4096):

        weights_ = prior_local_pts[..., 2]
        # breakpoint()
        weights_ = weights_.clamp_min(0.1 * weighted_mean(weights_, prior_mask, dim=(-2, -1), keepdim=True))
        weights_ = 1 / (weights_ + 1e-6)


        xyz_pred_local = self.prepare_ROE(pred_local_pts, prior_mask, target_size=local_align_res).contiguous()
        xyz_prior_local = self.prepare_ROE(prior_local_pts, prior_mask, target_size=local_align_res).contiguous()
        xyz_weights_local = self.prepare_ROE((weights_[..., None]), prior_mask, target_size=local_align_res).contiguous()[..., 0]

        S_opt_local = align_points_scale(xyz_pred_local, xyz_prior_local, xyz_weights_local)

        return S_opt_local

    @torch.inference_mode()
    def forward(self, video, device: torch.device, prior_max_size=None, align=True, num_tokens=None, prior_video=None, valid_mask=None) -> Dict[str, torch.Tensor]:
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video)

        video = video.to(device)
        if len(video.shape) == 4:
            video = video.unsqueeze(0)

        if video.shape[-1] == 3:
            video = video.permute(0,1,4,2,3) # from b t h w 3 to b t 3 h w

        _, num_frames, _, original_height, original_width = video.shape

        if prior_video is None:

            if prior_max_size is None:
                prior_max_size = self.prior_max_size

            original_aspect_ratio = original_width / original_height
            if original_width > original_height:
                prior_width = min(prior_max_size, original_width // self.prior_patch_size * self.prior_patch_size) # NOTE hardcode here 
                prior_height = int((prior_width / original_aspect_ratio) // self.prior_patch_size * self.prior_patch_size)
            else:
                prior_height = min(prior_max_size, original_height // self.prior_patch_size * self.prior_patch_size) # NOTE hardcode here 
                prior_width = int((prior_height * original_aspect_ratio) // self.prior_patch_size * self.prior_patch_size)
            prior_patch_h, prior_patch_w = prior_height // self.prior_patch_size, prior_width // self.prior_patch_size
            prior_video = F.interpolate(
                rearrange(video, 'b t c h w -> (b t) c h w'), (prior_height, prior_width), mode='bilinear', antialias=True
            ).clamp(0, 1)
            prior_video = rearrange(prior_video, '(b t) c h w -> b t c h w', t=num_frames)

        else:
            prior_height, prior_width = prior_video.shape[-2:]
            prior_patch_h, prior_patch_w = prior_height // self.prior_patch_size, prior_width // self.prior_patch_size
        
        # with CUDATimer("Prior forward", enabled=True, num_frames=video_prior.shape[1]):
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            prior_output = self.prior_model(prior_video)

            prior_hidden = prior_output["hidden"]
            prior_global_pts = prior_output['points']
            if "Pi3" in self.prior_model_name:
                prior_conf = prior_output['conf']
                if prior_conf is None:
                    prior_conf = 10.0 * torch.ones_like(prior_global_pts[..., 0])
            else:
                prior_conf = prior_output['local_points_conf']
            prior_local_pts = prior_output['local_points']
            camera_poses = prior_output['camera_poses']
            prior_patch_start_idx = prior_output["patch_start_idx"]


        
        # with CUDATimer("Model forward", enabled=True, num_frames=video_prior.shape[1]):
        # with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):

        model_output = self.model.infer(
            video,
            global_video_tokens=prior_hidden,
            global_video_tokens_shape=(prior_patch_h, prior_patch_w),
            global_patch_start_idx=prior_patch_start_idx,
            num_tokens=num_tokens,
            resolution_level=9,
            force_projection=True,
            apply_mask=False,
        )

        pred_local_pts = model_output['points'][0]
        pred_mask = model_output['mask'][0]
        pred_depth = pred_local_pts[..., 2] # T H W

        if not align:
            return {
                'local_points': pred_local_pts,
                'mask': pred_mask,
            }


        # with CUDATimer("Align forward", enabled=True, num_frames=video_prior.shape[1]):
        prior_global_pts = prior_global_pts[0]
        prior_local_pts = prior_local_pts[0]
        prior_conf = prior_conf[0]
        camera_poses = camera_poses[0]

        prior_depth = prior_local_pts[..., 2]

        if "Pi3" in self.prior_model_name:
            prior_mask = torch.sigmoid(prior_conf.squeeze(-1)) > 0.2
        else:
            prior_mask = prior_conf > 1.5

        
        pred_local_pts_resized = F.interpolate(pred_local_pts.permute(0, 3, 1, 2), size=(prior_height, prior_width), mode='nearest').permute(0, 2, 3, 1)
        # pred_depth_resized = F.interpolate(pred_depth.unsqueeze(1), size=(prior_height, prior_width), mode='nearest').squeeze(1) # T H W
        # pred_mask_resized = F.interpolate(pred_mask.float().unsqueeze(1), size=(prior_height, prior_width), mode='nearest').squeeze(1).bool()

        # scale = get_scale_depth(pred_depth_resized, prior_depth, pred_mask_resized, query_frame="all", align_resolution=32, trunc=1.0)
        # pred_local_pts = pred_local_pts * scale

        if valid_mask is not None:
            prior_mask = F.interpolate(valid_mask.float().unsqueeze(1), size=(prior_height, prior_width), mode='nearest').squeeze(1).bool()


        scale = self.get_align_scale(pred_local_pts_resized, prior_local_pts, prior_mask, local_align_res=4096)
        pred_local_pts = pred_local_pts * scale

        # for t in range(pred_local_pts.shape[0]):
        #     scale_t = self.get_align_scale(pred_local_pts_resized[t], prior_local_pts[t], prior_mask[t], local_align_res=4096)
        #     pred_local_pts[t] = pred_local_pts[t] * scale_t


        
        to_camera0 = se3_inverse(camera_poses[0])
        camera_poses = torch.einsum('ik, tkj -> tij', to_camera0, camera_poses) # to camera 0

        pred_global_pts = torch.einsum('tij, thwj -> thwi', camera_poses, homogenize_points(pred_local_pts))[..., :3]
        # prior_global_pts = torch.einsum('tij, thwj -> thwi', camera_poses, homogenize_points(prior_local_pts))[..., :3]



        output = {
            'points': pred_global_pts,
            'local_points': pred_local_pts,
            'mask': pred_mask,
            "prior_local_points": prior_local_pts,
            "prior_global_points": prior_global_pts,
            "prior_conf": prior_conf,
            "prior_mask": prior_mask,
            'camera_poses': camera_poses,
        }

        return output

    @torch.inference_mode()
    def forward_long(self, video, device: torch.device, prior_max_size=None, align=True, num_tokens=None, prior_video=None) -> Dict[str, torch.Tensor]:
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video)

        video = video.to(device)
        if len(video.shape) == 4:
            video = video.unsqueeze(0)

        if video.shape[-1] == 3:
            video = video.permute(0,1,4,2,3) # from b t h w 3 to b t 3 h w

        _, num_frames, _, original_height, original_width = video.shape

        if prior_video is None:

            if prior_max_size is None:
                prior_max_size = self.prior_max_size

            original_aspect_ratio = original_width / original_height
            if original_width > original_height:
                prior_width = min(prior_max_size, original_width // self.prior_patch_size * self.prior_patch_size) # NOTE hardcode here 
                prior_height = int((prior_width / original_aspect_ratio) // self.prior_patch_size * self.prior_patch_size)
            else:
                prior_height = min(prior_max_size, original_height // self.prior_patch_size * self.prior_patch_size) # NOTE hardcode here 
                prior_width = int((prior_height * original_aspect_ratio) // self.prior_patch_size * self.prior_patch_size)
            prior_patch_h, prior_patch_w = prior_height // self.prior_patch_size, prior_width // self.prior_patch_size
            prior_video = F.interpolate(
                rearrange(video, 'b t c h w -> (b t) c h w'), (prior_height, prior_width), mode='bilinear', antialias=True
            ).clamp(0, 1)
            prior_video = rearrange(prior_video, '(b t) c h w -> b t c h w', t=num_frames)

        else:
            prior_height, prior_width = prior_video.shape[-2:]
            prior_patch_h, prior_patch_w = prior_height // self.prior_patch_size, prior_width // self.prior_patch_size

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            prior_output = self.prior_model(prior_video)

            prior_hidden = prior_output["hidden"]
            prior_global_pts = prior_output['points']
            if "Pi3" in self.prior_model_name:
                prior_conf = prior_output['conf']
                if prior_conf is None:
                    prior_conf = 10.0 * torch.ones_like(prior_global_pts[..., 0])
            else:
                prior_conf = prior_output['local_points_conf']
            prior_local_pts = prior_output['local_points']
            camera_poses = prior_output['camera_poses']
            prior_patch_start_idx = prior_output["patch_start_idx"]

            # torch.cuda.empty_cache()


        
        # with CUDATimer("Model forward", enabled=True, num_frames=video_prior.shape[1]):
        chunk_size = 8
        num_chunks = num_frames // chunk_size + 1 if num_frames % chunk_size != 0 else num_frames // chunk_size
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, num_frames)
            chunk_video = video[:, chunk_start:chunk_end]
            chunk_output = self.model.infer(
                chunk_video,
                global_video_tokens=prior_hidden[:, chunk_start:chunk_end],
                global_video_tokens_shape=(prior_patch_h, prior_patch_w),
                global_patch_start_idx=prior_patch_start_idx,
                num_tokens=num_tokens,
                resolution_level=9,
                force_projection=True,
                apply_mask=False,
            )
            # chunk_output['points'] = chunk_output['points'][:, chunk_start:chunk_end]
            # chunk_output['mask'] = chunk_output['mask'][:, chunk_start:chunk_end]
            if chunk_idx == 0:
                model_output = defaultdict(list)
                for k in chunk_output:
                    model_output[k].append(chunk_output[k])
            else:
                for k in chunk_output:
                    model_output[k].append(chunk_output[k])
            # torch.cuda.empty_cache()

        for k in model_output:
            model_output[k] = torch.cat(model_output[k], dim=1)

        pred_local_pts = model_output['points'][0]
        pred_mask = model_output['mask'][0]
        pred_depth = pred_local_pts[..., 2] # T H W


        if not align:
            return {
                'local_points': pred_local_pts,
                'mask': pred_mask,
            }


        # with CUDATimer("Align forward", enabled=True, num_frames=video_prior.shape[1]):
        prior_global_pts = prior_global_pts[0]
        prior_local_pts = prior_local_pts[0]
        prior_conf = prior_conf[0]
        camera_poses = camera_poses[0]

        prior_depth = prior_local_pts[..., 2]

        if "Pi3" in self.prior_model_name:
            prior_mask = torch.sigmoid(prior_conf.squeeze(-1)) > 0.2
        else:
            prior_mask = prior_conf > 1.5

        
        pred_local_pts_resized = F.interpolate(pred_local_pts.permute(0, 3, 1, 2), size=(prior_height, prior_width), mode='nearest').permute(0, 2, 3, 1)
        # pred_depth_resized = F.interpolate(pred_depth.unsqueeze(1), size=(prior_height, prior_width), mode='nearest').squeeze(1) # T H W
        # pred_mask_resized = F.interpolate(pred_mask.float().unsqueeze(1), size=(prior_height, prior_width), mode='nearest').squeeze(1).bool()

        # scale = get_scale_depth(pred_depth_resized, prior_depth, pred_mask_resized, query_frame="all", align_resolution=32, trunc=1.0)
        # pred_local_pts = pred_local_pts * scale

        if num_frames > 100:
            pred_local_pts_resized_align = pred_local_pts_resized[:100]
            prior_local_pts_align = prior_local_pts[:100]
            prior_mask_align = prior_mask[:100]
        else:
            pred_local_pts_resized_align = pred_local_pts_resized
            prior_local_pts_align = prior_local_pts
            prior_mask_align = prior_mask

        scale = self.get_align_scale(pred_local_pts_resized_align, prior_local_pts_align, prior_mask_align, local_align_res=4096)
        pred_local_pts = pred_local_pts * scale


        
        to_camera0 = se3_inverse(camera_poses[0])
        camera_poses = torch.einsum('ik, tkj -> tij', to_camera0, camera_poses) # to camera 0

        pred_global_pts = torch.einsum('tij, thwj -> thwi', camera_poses, homogenize_points(pred_local_pts))[..., :3]

        prior_global_pts = torch.einsum('tij, thwj -> thwi', camera_poses, homogenize_points(prior_local_pts))[..., :3]



        output = {
            'points': pred_global_pts,
            'local_points': pred_local_pts,
            'mask': pred_mask,
            "prior_local_points": prior_local_pts,
            "prior_global_points": prior_global_pts,
            "prior_conf": prior_conf,
            "prior_mask": prior_mask,
            "prior_video": prior_video[0],
            'camera_poses': camera_poses,
        }

        return output
