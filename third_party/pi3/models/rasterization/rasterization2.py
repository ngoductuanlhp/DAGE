from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum, rearrange, repeat

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy

from .sh_utils import eval_sh, RGB2SH
from .act_gs import reg_dense_offsets, reg_dense_scales, reg_dense_rotation, reg_dense_sh, reg_dense_opacities, reg_dense_weights
from third_party.pi3.utils.geometry import homogenize_points, se3_inverse
from third_party.pi3.utils.da3_geometry import sample_image_grid, get_world_rays
from third_party.pi3.utils.da3_transform import cam_quat_xyzw_to_world_quat_wxyz
# from src.models.utils.frustum import calculate_unprojected_mask
# from src.models.utils.geometry import depth_to_world_coords_points, closed_form_inverse_se3
# from src.models.utils.camera_utils import vector_to_camera_matrices



class DeferredBP(torch.autograd.Function):
    @staticmethod
    def render(xyz, feature, scale, rotation, opacity, test_w2c, test_intr, 
               W, H, raster_kwargs):
        rgbd, alpha, _ = rasterization(
            means=xyz, 
            quats=rotation, 
            scales=scale, 
            opacities=opacity, 
            colors=feature,
            viewmats=test_w2c, 
            Ks=test_intr, 
            width=W, 
            height=H, 
            render_mode="RGB+ED",
            **raster_kwargs,
        ) # (1, H, W, 3) 
        image, depth = rgbd[..., :3], rgbd[..., 3:]
        return image, alpha, depth     # (1, H, W, 3)

    @staticmethod
    def forward(ctx, xyz, feature, scale, rotation, opacity, test_w2cs, test_intr,
                W, H, raster_kwargs):
        ctx.save_for_backward(xyz, feature, scale, rotation, opacity, test_w2cs, test_intr)
        ctx.W = W
        ctx.H = H
        # ctx.near_plane = near_plane
        # ctx.far_plane = far_plane
        ctx.raster_kwargs = raster_kwargs
        with torch.no_grad():
            # B, V = test_intr.shape[:2]
            # images = torch.zeros(B, V, H, W, 3).to(xyz.device)
            # alphas = torch.zeros(B, V, H, W, 1).to(xyz.device)
            # depths = torch.zeros(B, V, H, W, 1).to(xyz.device)
            # for ib in range(B):
            #     for iv in range(V):
            #         image, alpha, depth = DeferredBP.render(
            #             xyz[ib], feature[ib], scale[ib], rotation[ib], opacity[ib], 
            #             test_w2cs[ib,iv:iv+1], test_intr[ib,iv:iv+1], 
            #             W, H, near_plane, far_plane, backgrounds[ib,iv:iv+1],
            #             raster_kwargs
            #         )
            #         images[ib, iv:iv+1] = image
            #         alphas[ib, iv:iv+1] = alpha
            #         depths[ib, iv:iv+1] = depth

            V = test_intr.shape[0]
            images = torch.zeros(V, H, W, 3).to(xyz.device)
            alphas = torch.zeros(V, H, W, 1).to(xyz.device)
            depths = torch.zeros(V, H, W, 1).to(xyz.device)
            for iv in range(V):
                image, alpha, depth = DeferredBP.render(
                    xyz, feature, scale, rotation, opacity, 
                    test_w2cs[iv:iv+1], test_intr[iv:iv+1], 
                    W, H,
                    raster_kwargs
                )
                images[iv:iv+1] = image
                alphas[iv:iv+1] = alpha
                depths[iv:iv+1] = depth

        images = images.requires_grad_()
        alphas = alphas.requires_grad_()
        depths = depths.requires_grad_()
        return images, alphas, depths

    @staticmethod
    def backward(ctx, images_grad, alphas_grad, depths_grad):
        xyz, feature, scale, rotation, opacity, test_w2cs, test_intr = ctx.saved_tensors
        xyz = xyz.detach().requires_grad_()
        feature = feature.detach().requires_grad_()
        scale = scale.detach().requires_grad_()
        rotation = rotation.detach().requires_grad_()
        opacity = opacity.detach().requires_grad_()
        W = ctx.W
        H = ctx.H
        # near_plane = ctx.near_plane
        # far_plane = ctx.far_plane
        raster_kwargs = ctx.raster_kwargs
        with torch.enable_grad():
            # B, V = test_intr.shape[:2]
            # for ib in range(B):
            #     for iv in range(V):
            #         image, alpha, depth = DeferredBP.render(
            #             xyz[ib], feature[ib], scale[ib], rotation[ib], opacity[ib], 
            #             test_w2cs[ib,iv:iv+1], test_intr[ib,iv:iv+1], 
            #             W, H, near_plane, far_plane, backgrounds[ib,iv:iv+1],
            #             raster_kwargs,
            #         )
            #         render_split = torch.cat([image, alpha, depth], dim=-1)
            #         grad_split = torch.cat([images_grad[ib, iv:iv+1], alphas_grad[ib, iv:iv+1], depths_grad[ib, iv:iv+1]], dim=-1) 
            #         render_split.backward(grad_split)
            V = test_intr.shape[0]
            for iv in range(V):
                image, alpha, depth = DeferredBP.render(
                    xyz, feature, scale, rotation, opacity, 
                    test_w2cs[iv:iv+1], test_intr[iv:iv+1], 
                    W, H,
                    raster_kwargs,
                )
                render_split = torch.cat([image, alpha, depth], dim=-1)
                grad_split = torch.cat([images_grad[iv:iv+1], alphas_grad[iv:iv+1], depths_grad[iv:iv+1]], dim=-1) 
                render_split.backward(grad_split)

        return xyz.grad, feature.grad, scale.grad, rotation.grad, opacity.grad, None, None, None, None, None, None, None, None


class Rasterizer:
    def __init__(self, rasterization_mode="classic", packed=True, abs_grad=True, with_eval3d=False,
                 camera_model="pinhole", sparse_grad=False, distributed=False, grad_strategy=DefaultStrategy, deferred_bp=False):
        self.rasterization_mode = rasterization_mode
        self.packed = packed
        self.abs_grad = abs_grad
        self.camera_model = camera_model
        self.sparse_grad = sparse_grad
        self.grad_strategy = grad_strategy
        self.distributed = distributed
        self.with_eval3d = with_eval3d
        self.deferred_bp = deferred_bp

    def rasterize_splats(
        self,
        means,
        quats,
        scales,
        opacities,
        colors,
        world_to_cam: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        
        render_colors, render_alphas, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=world_to_cam,  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.packed,
            absgrad=(
                self.abs_grad
                if isinstance(self.grad_strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.sparse_grad,
            rasterize_mode=self.rasterization_mode,
            distributed=self.distributed,
            camera_model=self.camera_model,
            with_eval3d=self.with_eval3d,
            render_mode="RGB+ED",
            **kwargs,
        )
        return render_colors[..., :3], render_colors[..., 3:], render_alphas

    def rasterize_batches(self, means, quats, scales, opacities, colors, viewmats, Ks, width, height, kwargs):
        rendered_colors, rendered_depths, rendered_alphas = [], [], []
        batch_size = len(means)
        
        for i in range(batch_size):
            means_i = means[i]  # [N, 4]
            quats_i = quats[i]  # [N, 4]
            scales_i = scales[i]  # [N, 3]
            opacities_i = opacities[i]  # [N,]
            colors_i = colors[i]  # [N, 3]
            viewmats_i = viewmats[i]  # [V, 4, 4]
            Ks_i = Ks[i]  # [V, 3, 3]


            world_to_cam_i = se3_inverse(viewmats_i)

            if self.deferred_bp:
                render_colors_i, render_depths_i, render_alphas_i = DeferredBP.apply(
                    means_i, colors_i, scales_i, quats_i, opacities_i, world_to_cam_i, Ks_i, width, height, kwargs
                )
            else:
                render_colors_i, render_depths_i, render_alphas_i = self.rasterize_splats(
                    means_i, quats_i, scales_i, opacities_i, colors_i, world_to_cam_i, Ks_i, width, height, kwargs
                )
            
            
            rendered_colors.append(render_colors_i)  # V H W 3
            rendered_depths.append(render_depths_i)  # V H W 1
            rendered_alphas.append(render_alphas_i)  # V H W 1
            
        rendered_colors = torch.stack(rendered_colors, dim=0)  # B V H W 3
        rendered_depths = torch.stack(rendered_depths, dim=0)  # B V H W 1
        rendered_alphas = torch.stack(rendered_alphas, dim=0)  # B V H W 1
        
        return rendered_colors, rendered_depths, rendered_alphas


    

class GaussianSplatRendererV2(nn.Module):
    def __init__(
        self,
        feature_dim: int = 256,       # Output channels of gs_feat_head (kept for backward compatibility)
        sh_degree: int = 0,
        predict_offset: bool = False,
        predict_residual_sh: bool = True,
        enable_prune: bool = True,
        voxel_size: float = 0.002,    # Default voxel size for prune_gs
        using_gtcamera_splat: bool = False,
        render_novel_views: bool = False,
        enable_conf_filter: bool = False,  # Enable confidence filtering
        conf_threshold_percent: float = 30.0,  # Confidence threshold percentage
        max_gaussians: int = 5000000,  # Maximum number of Gaussians
        debug=False,
        enable_prune_by_opacity: bool = False,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.sh_degree = sh_degree
        self.nums_sh = (sh_degree + 1) ** 2
        self.predict_offset = predict_offset
        self.predict_residual_sh = predict_residual_sh
        self.voxel_size = voxel_size
        self.enable_prune = enable_prune
        self.using_gtcamera_splat = using_gtcamera_splat
        self.render_novel_views = render_novel_views
        self.enable_conf_filter = enable_conf_filter
        self.conf_threshold_percent = conf_threshold_percent
        self.max_gaussians = max_gaussians
        self.debug = debug

        self.enable_prune_by_opacity = enable_prune_by_opacity
        self.opacity_ratio = 0.5
        self.random_ratio = 0.1

        # Rasterizer
        self.rasterizer = Rasterizer(deferred_bp=True)

    # ======== Main entry point: Complete GS rendering and fill results back to predictions ========
    def render(
        self,
        gs_points: torch.Tensor,                   # [B, S(+V), H, W, 3]
        # gs_mask: torch.Tensor,                     # [B, S(+V), H, W]
        gs_params: torch.Tensor,                   # [B*S, H, W, C] - already computed Gaussian parameters
        images: torch.Tensor,                      # [B, S+V, 3, H, W]
        # predictions: Dict[str, torch.Tensor],      # From WorldMirror: pose/depth/pts3d etc
        # views: Dict[str, torch.Tensor],
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        context_nums: int = None,
        context_predictions: torch.Tensor = None,
        skip_render_gs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns predictions with the following fields filled:
        - rendered_colors / rendered_depths / (rendered_alphas during training)
        - gt_colors / gt_depths / valid_masks
        - splats / rendered_extrinsics / rendered_intrinsics
        """
        H, W = images.shape[-2:]
        S = context_nums if context_nums is not None else images.shape[1]
        V = images.shape[1] - S
        
        # 2) Select predicted cameras
        Bx = images.shape[0]
        # pred_all_extrinsic, pred_all_intrinsic = self.prepare_prediction_cameras(predictions, S + V, hw=(H, W))
        # pred_all_extrinsic = pred_all_extrinsic.reshape(Bx, S + V, 4, 4)
        # pred_all_source_extrinsic = pred_all_extrinsic[:, :S]
        pred_all_extrinsic = extrinsics
        pred_all_intrinsic = intrinsics
        pred_all_source_extrinsic = extrinsics[:, :S]

        pred_all_intrinsic_normed = pred_all_intrinsic.clone().detach()
        pred_all_intrinsic_normed[..., 0, :] /= W
        pred_all_intrinsic_normed[..., 1, :] /= H


        scale_factor = 1.0
        if context_predictions is not None:
            pred_source_extrinsic, _ = self.prepare_prediction_cameras(context_predictions, S, hw=(H, W))
            pred_source_extrinsic = pred_source_extrinsic.reshape(Bx, S, 4, 4)
            scale_factor = pred_source_extrinsic[:, :, :3, 3].mean(dim=(1, 2), keepdim=True) / (
                pred_all_source_extrinsic[:, :, :3, 3].mean(dim=(1, 2), keepdim=True) + 1e-6
            )

        pred_all_extrinsic[..., :3, 3] = pred_all_extrinsic[..., :3, 3] * scale_factor

        render_viewmats, render_Ks = pred_all_extrinsic, pred_all_intrinsic
        # render_images = images
        # gt_colors = render_images.permute(0, 1, 3, 4, 2)
        
        # 3) Generate splats from gs_params + predictions, and perform voxel merging
        splats = self.prepare_splats(
            gs_points=gs_points, 
            images=images, 
            gs_params=gs_params, 
            context_nums=S, 
            target_nums=V, 
            # context_predictions=context_predictions, 
            cam2worlds=pred_all_extrinsic, 
            intr_normed=pred_all_intrinsic_normed, 
            debug=False
        )

        ctx_local_gs_points = splats["means_cam"]

        output_dict = {
            "splats": splats,
            "ctx_local_gs_points": ctx_local_gs_points,
        }

        if skip_render_gs:
            return output_dict


        # Apply confidence filtering before pruning
        # if self.enable_conf_filter and "depth_conf" in predictions:
        #     splats = self.apply_confidence_filter(splats, predictions["depth_conf"])
        
        if self.enable_prune:
            splats = self.prune_gs(splats, voxel_size=self.voxel_size)
        
        # print(f"enable_prune_by_opacity: {self.enable_prune_by_opacity}, splats: {splats['opacities'][0].shape}")
        if self.enable_prune_by_opacity:
            splats = self.prune_gs_by_opacity(splats)


        rendered_colors, rendered_depths, rendered_alphas = self.rasterizer.rasterize_batches(
            splats["means"], splats["quats"], splats["scales"], splats["opacities"],
            splats["sh"] if "sh" in splats else splats["colors"],
            render_viewmats.detach(), render_Ks.detach(),
            width=W, height=H,
            kwargs=dict(
                sh_degree=min(self.sh_degree, 0) if "sh" in splats else None,
                # near_plane=0.001,
                # far_plane=100.0,
            )
        )

        output_dict["rendered_colors"] = rendered_colors
        output_dict["rendered_depths"] = rendered_depths
        output_dict["rendered_alphas"] = rendered_alphas


        return output_dict

    # ======== Main entry point: Complete GS rendering and fill results back to predictions ========
    def render_test(
        self,
        gs_points: torch.Tensor,                   # [B, S(+V), H, W, 3]
        # gs_mask: torch.Tensor,                     # [B, S(+V), H, W]
        gs_params: torch.Tensor,                   # [B*S, C, H, W] - already computed Gaussian parameters
        images: torch.Tensor,                      # [B, S+V, 3, H, W]
        # predictions: Dict[str, torch.Tensor],      # From WorldMirror: pose/depth/pts3d etc
        # views: Dict[str, torch.Tensor],
        ctx_intrinsics: torch.Tensor,
        ctx_extrinsics: torch.Tensor,
        tgt_intrinsics: torch.Tensor,
        tgt_extrinsics: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns predictions with the following fields filled:
        - rendered_colors / rendered_depths / (rendered_alphas during training)
        - gt_colors / gt_depths / valid_masks
        - splats / rendered_extrinsics / rendered_intrinsics
        """
        B = gs_points.shape[0]
        H, W = images.shape[-2:]
        # S = context_nums if context_nums is not None else images.shape[1]
        # V = images.shape[1] - S 
        S = gs_points.shape[1]
        V = tgt_intrinsics.shape[1]
        
        # 2) Select predicted cameras
        # pred_all_extrinsic, pred_all_intrinsic = self.prepare_prediction_cameras(predictions, S + V, hw=(H, W))
        # pred_all_extrinsic = pred_all_extrinsic.reshape(Bx, S + V, 4, 4)
        # pred_all_source_extrinsic = pred_all_extrinsic[:, :S]
        # pred_all_extrinsic = extrinsics
        # pred_all_intrinsic = intrinsics
        # pred_all_source_extrinsic = extrinsics[:, :S]

        # render_viewmats, render_Ks = pred_all_extrinsic, pred_all_intrinsic
        render_viewmats = tgt_extrinsics
        render_Ks = tgt_intrinsics
        # render_images = images
        # gt_colors = render_images.permute(0, 1, 3, 4, 2)

        cam2worlds = ctx_extrinsics
        intr_normed = ctx_intrinsics.clone().detach()
        intr_normed[..., 0, :] /= W
        intr_normed[..., 1, :] /= H
        
        # 3) Generate splats from gs_params + predictions, and perform voxel merging
        splats = self.prepare_splats(
            gs_points=gs_points, 
            images=images, 
            gs_params=gs_params, 
            context_nums=S, 
            target_nums=V, 
            cam2worlds=cam2worlds, 
            intr_normed=intr_normed, 
            debug=False
        )

        
        if self.enable_prune:
            splats = self.prune_gs(splats, voxel_size=self.voxel_size)

        
        # print(f"enable_prune_by_opacity: {self.enable_prune_by_opacity}, splats: {splats['opacities'][0].shape}")
        if self.enable_prune_by_opacity:
            splats = self.prune_gs_by_opacity(splats, skip_random=True)

        # print(f"after prune_gs_by_opacity: {splats['opacities'][0].shape}")
        # breakpoint()

        

        # Prevent OOM by using chunked rendering
        rendered_colors_list, rendered_depths_list, rendered_alphas_list = [], [], []
        chunk_size = 4
        num_view_render = render_viewmats.shape[1]

        for i in range(0, num_view_render, chunk_size):
            end_idx = min(i + chunk_size, num_view_render)
            viewmats_i = render_viewmats[:, i:end_idx]
            Ks_i = render_Ks[:, i:end_idx]

            rendered_colors, rendered_depths, rendered_alphas = self.rasterizer.rasterize_batches(
                splats["means"], splats["quats"], splats["scales"], splats["opacities"],
                splats["sh"] if "sh" in splats else splats["colors"],
                viewmats_i.detach(), Ks_i.detach(),
                width=W, height=H,
                kwargs=dict(
                    sh_degree=min(self.sh_degree, 0) if "sh" in splats else None,
                    # near_plane=0.001,
                    # far_plane=100.0,
                )
            )
            rendered_colors_list.append(rendered_colors)
            rendered_depths_list.append(rendered_depths)
            rendered_alphas_list.append(rendered_alphas)

        rendered_colors = torch.cat(rendered_colors_list, dim=1)
        rendered_depths = torch.cat(rendered_depths_list, dim=1)
        rendered_alphas = torch.cat(rendered_alphas_list, dim=1)

        # 5) return predictions
        output_dict = {
            "rendered_colors": rendered_colors,
            "rendered_depths": rendered_depths,
            "rendered_alphas": rendered_alphas,
            # "splats": splats,
            "gs_points": gs_points,
        }


        return output_dict

    def apply_confidence_filter(self, splats, gs_depth_conf):
        """
        Apply confidence filtering to Gaussian splats before pruning.
        Discard bottom p% confidence points, keep top (100-p)%.
        
        Args:
            splats: Dictionary containing Gaussian parameters
            gs_depth_conf: Confidence tensor [B, S, H, W]
        
        Returns:
            Filtered splats dictionary
        """
        if not self.enable_conf_filter or gs_depth_conf is None:
            return splats

        device = splats["means"].device
        B, N = splats["means"].shape[:2]

        # Flatten confidence: [B, S, H, W] -> [B, N]
        conf = gs_depth_conf.flatten(1).to(device)
        # Mask invalid/very small values
        conf = conf.masked_fill(conf <= 1e-5, float("-inf"))

        # Keep top (100-p)% points, discard bottom p%
        if self.conf_threshold_percent > 0:
            keep_from_percent = int(np.ceil(N * (100.0 - self.conf_threshold_percent) / 100.0))
        else:
            keep_from_percent = N
        K = max(1, min(self.max_gaussians, keep_from_percent))

        # Select top-K indices for each batch (deterministic, no randomness)
        topk_idx = torch.topk(conf, K, dim=1, largest=True, sorted=False).indices  # [B, K]
        
        filtered = {}
        mask_keys = ["means", "quats", "scales", "opacities", "sh", "weights"]
        
        for key in splats.keys():
            if key in mask_keys and key in splats:
                x = splats[key]
                if x.ndim == 2:  # [B, N]
                    filtered[key] = torch.gather(x, 1, topk_idx)
                else:
                    # Expand indices to match tensor dimensions
                    expand_idx = topk_idx.clone()
                    for i in range(x.ndim - 2):
                        expand_idx = expand_idx.unsqueeze(-1)
                    expand_idx = expand_idx.expand(-1, -1, *x.shape[2:])
                    filtered[key] = torch.gather(x, 1, expand_idx)
            else:
                filtered[key] = splats[key]

        return filtered

    
        
    
    def prune_gs(self, splats, voxel_size=0.002):
        """
        Prune Gaussian splats by merging those in the same voxel.
        
        Args:
            splats: Dictionary containing Gaussian parameters
            voxel_size: Size of voxels for spatial grouping
            
        Returns:
            Dictionary with pruned splats
        """
        B = splats["means"].shape[0]
        merged_splats_list = []
        device = splats["means"].device

        for i in range(B):
            # Extract splats for current batch
            splats_i = {k: splats[k][i] for k in ["means", "quats", "scales", "opacities", "sh", "weights"]}
            
            # Compute voxel indices
            coords = splats_i["means"]
            voxel_indices = (coords / voxel_size).floor().long()
            min_indices = voxel_indices.min(dim=0)[0]
            voxel_indices = voxel_indices - min_indices
            max_dims = voxel_indices.max(dim=0)[0] + 1
            
            # Flatten 3D voxel indices to 1D
            flat_indices = (voxel_indices[:, 0] * max_dims[1] * max_dims[2] + 
                           voxel_indices[:, 1] * max_dims[2] + 
                           voxel_indices[:, 2])
            
            # Find unique voxels and inverse mapping
            unique_voxels, inverse_indices = torch.unique(flat_indices, return_inverse=True)
            K = len(unique_voxels)

            # Initialize merged splats
            merged = {
                "means": torch.zeros((K, 3), device=device),
                "quats": torch.zeros((K, 4), device=device),
                "scales": torch.zeros((K, 3), device=device),
                "opacities": torch.zeros(K, device=device),
                "sh": torch.zeros((K, self.nums_sh, 3), device=device)
            }
            
            # Get weights and compute weight sums per voxel
            weights = splats_i["weights"]
            weight_sums = torch.zeros(K, device=device)
            weight_sums.scatter_add_(0, inverse_indices, weights)
            weight_sums = torch.clamp(weight_sums, min=1e-8)

            # Merge means (weighted average)
            for d in range(3):
                merged["means"][:, d].scatter_add_(0, inverse_indices, 
                                                 splats_i["means"][:, d] * weights)
            merged["means"] = merged["means"] / weight_sums.unsqueeze(1)

            # Merge spherical harmonics (weighted average)
            for d in range(3):
                merged["sh"][:, 0, d].scatter_add_(0, inverse_indices, 
                                                  splats_i["sh"][:, 0, d] * weights)
            merged["sh"] = merged["sh"] / weight_sums.unsqueeze(-1).unsqueeze(-1)

            # Merge opacities (weighted sum of squares)
            merged["opacities"].scatter_add_(0, inverse_indices, weights * weights)
            merged["opacities"] = merged["opacities"] / weight_sums

            # Merge scales (weighted average)
            for d in range(3):
                merged["scales"][:, d].scatter_add_(0, inverse_indices, 
                                                  splats_i["scales"][:, d] * weights)
            merged["scales"] = merged["scales"] / weight_sums.unsqueeze(1)

            # Merge quaternions (weighted average + normalization)
            for d in range(4):
                merged["quats"][:, d].scatter_add_(0, inverse_indices, 
                                                 splats_i["quats"][:, d] * weights)
            quat_norms = torch.norm(merged["quats"], dim=1, keepdim=True)
            merged["quats"] = merged["quats"] / torch.clamp(quat_norms, min=1e-8)

            merged_splats_list.append(merged)

        # Reorganize output
        output = {}
        for key in ["means", "sh", "opacities", "scales", "quats"]:
            output[key] = [merged[key] for merged in merged_splats_list]
        
        return output

    def prune_gs_by_opacity(self, splats, skip_random=False):
        """
        Prune Gaussian splats by keeping top opacity values.
        
        Args:
            splats: Dictionary containing Gaussian parameters
            prune_ratio: Ratio of Gaussians to prune (0.0 to 1.0)
            random_ratio: Ratio of kept Gaussians to randomly sample from pruned ones
                         (relative to keep_ratio, e.g., 0.1 means 10% of kept ones are random)
            
        Returns:
            Dictionary with pruned splats (same format as prune_gs output)
        """

        prune_ratio = self.opacity_ratio
        random_ratio = self.random_ratio

        B = splats["means"].shape[0]
        num_gaussians = splats["means"].shape[1]
        device = splats["means"].device
        
        if prune_ratio <= 0:
            # No pruning needed, just reformat to output format
            output = {}
            for key in ["means", "sh", "opacities", "scales", "quats"]:
                output[key] = [splats[key][i] for i in range(B)]
            return output
        
        # Calculate keep ratios
        keep_ratio = 1 - prune_ratio
        random_ratio_adjusted = keep_ratio * random_ratio
        keep_ratio_deterministic = keep_ratio - random_ratio_adjusted
        num_keep = int(num_gaussians * keep_ratio_deterministic)
        num_keep_random = int(num_gaussians * random_ratio_adjusted)

        if skip_random:
            total_keep = num_keep
        else:
            total_keep = num_keep + num_keep_random
        
        # Vectorized processing across all batches
        # Get opacities for all batches - shape: (B, N) or (B, N, 1)
        opacities = splats["opacities"]
        if opacities.dim() > 2:
            opacities = opacities.squeeze(-1)
        
        # Sort by opacity in descending order for all batches - shape: (B, N)
        idx_sort = opacities.argsort(dim=1, descending=True)
        
        # Keep top ones - shape: (B, num_keep)
        keep_idx = idx_sort[:, :num_keep]
        
        # Add random ones if needed
        if num_keep_random > 0 and not skip_random:
            rest_idx = idx_sort[:, num_keep:]  # (B, num_rest)
            num_rest = rest_idx.shape[1]
            if num_rest > 0:
                # Generate random permutations for each batch
                random_perm = torch.stack([
                    torch.randperm(num_rest, device=device)[:num_keep_random]
                    for _ in range(B)
                ])  # (B, num_keep_random)
                random_idx = torch.gather(rest_idx, 1, random_perm)  # (B, num_keep_random)
                keep_idx = torch.cat([keep_idx, random_idx], dim=1)  # (B, total_keep)
        
        # Vectorized gather for all keys
        pruned_splats = {}
        for key in ["means", "quats", "scales", "opacities", "sh"]:
            v = splats[key]  # (B, N, ...)
            v_shape = v.shape
            
            # Reshape to (B, N, -1) for gather
            v = v.reshape(B, num_gaussians, -1)
            
            # Expand keep_idx to match last dimension
            keep_idx_expanded = keep_idx.unsqueeze(-1).expand(-1, -1, v.shape[-1])  # (B, total_keep, features)
            
            # Gather selected indices for all batches at once
            v = torch.gather(v, 1, keep_idx_expanded)  # (B, total_keep, features)
            
            # Reshape back to original shape (except second dimension)
            pruned_splats[key] = v.reshape(B, total_keep, *v_shape[2:])
        
        # Reorganize output to match prune_gs format (list of tensors per batch)
        output = {}
        for key in ["means", "sh", "opacities", "scales", "quats"]:
            output[key] = [pruned_splats[key][i] for i in range(B)]
        
        return output
    
    def get_scale_multiplier(
        self,
        intrinsics: torch.Tensor,  # "*#batch 3 3"
        pixel_size: torch.Tensor,  # "*#batch 2"
        multiplier: float = 0.1,
    ) -> torch.Tensor:  # " *batch"
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].float().inverse().to(intrinsics),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    def get_scale(self, scales, gs_points, intr_normed, W, H):
        dtype, device = scales.dtype, scales.device
        scale_min = 1e-5
        scale_max = 30.0
        scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
        pixel_size = 1 / torch.tensor((W, H), dtype=dtype, device=device)
        multiplier = self.get_scale_multiplier(intr_normed, pixel_size)
        gs_scales = scales * gs_points[..., 2:3] * multiplier[..., None, None, None]
        gs_scales = rearrange(gs_scales, "b v h w d -> b (v h w) d")

        return gs_scales

    def prepare_splats(self, gs_points, images, gs_params, context_nums, target_nums, cam2worlds, intr_normed, debug=False):
        """
        Prepare Gaussian splats from model predictions and input data.
        
        Args:
            views: Dictionary containing view data (camera poses, intrinsics, etc.)
            predictions: Model predictions including depth, pose_enc, etc.
            images: Input images [B, S_all, 3, H, W]
            gs_params: Gaussian splatting parameters from model
            context_nums: Number of context views (S)
            target_nums: Number of target views (V)
            context_predictions: Optional context predictions for camera poses
            position_from: Method to compute 3D positions ("pts3d", "preddepth+predcamera", "gsdepth+predcamera", "gsdepth+gtcamera")
            debug: Whether to use debug mode with ground truth data
            
        Returns:
            splats: Dictionary containing prepared Gaussian splat parameters
        """

        


        B, S_all, _, H, W = images.shape
        S, V = context_nums, target_nums
        splats = {}

        dtype, device = gs_params.dtype, gs_params.device
        
        # Only take parameters from source view branch
        # gs_params = rearrange(gs_params, "(b s) c h w -> b s h w c", b=B)[:, :S]
        # gs_params = gs_params[:, :S]
        # gs_points = gs_points[:, :S]

        # splats["gs_feats"] = gs_params.reshape(B, S*H*W, -1)

        quats, scales, opacities, residual_sh, xy_offsets = torch.split(
            gs_params, [4, 3, 1, self.nums_sh * 3, 2], dim=-1
        )
        # offsets = 0.

        xy_ray, _ = sample_image_grid((H, W), device)
        xy_ray = xy_ray[None, None, ...].expand(B, S_all, -1, -1, -1)  # b v h w xy
        # offset xy if needed
        pixel_size = 1 / torch.tensor((W, H), dtype=xy_ray.dtype, device=device)
        xy_ray = xy_ray + xy_offsets * pixel_size
        # 1.4) unproject depth + xy to world ray
        origins, directions = get_world_rays(
            xy_ray,
            repeat(cam2worlds, "b v i j -> b v h w i j", h=H, w=W),
            repeat(intr_normed, "b v i j -> b v h w i j", h=H, w=W),
        )
        gs_means_world = origins + directions * gs_points[..., 2:3]
        
        # NOTE for training only
        gs_means_cam = torch.einsum(
            'bnij, bnhwj -> bnhwi', 
            se3_inverse(cam2worlds), 
            homogenize_points(gs_means_world)
        )[..., :3]


        gs_means_world = rearrange(gs_means_world[:, :S], "b v h w d -> b (v h w) d")

        # # NOTE debug

        # gs_points = gs_points[:, :S]
        # global_gs_points = torch.einsum('bnij, bnhwj -> bnhwi', cam2worlds[:, :S], homogenize_points(gs_points))[..., :3]
        # global_gs_points = rearrange(global_gs_points, 'b t h w c -> b (t h w) c')

        # # breakpoint()


        # Apply activation functions to Gaussian parameters
        quats = reg_dense_rotation(quats[:, :S].reshape(B, S * H * W, 4))
        # splats["scales"] = reg_dense_scales(scales.reshape(B, S * H * W, 3)).clamp_max(0.3)
        opacities = reg_dense_opacities(opacities[:, :S].reshape(B, S * H * W))
        residual_sh = reg_dense_sh(residual_sh[:, :S].reshape(B, S * H * W, self.nums_sh * 3))
        scales = self.get_scale(scales[:, :S], gs_points[:, :S], intr_normed[:, :S], W, H)

        

        # cam_quat_xyzw = rearrange(quats, "b v h w c -> b (v h w) c")
        c2w_mat = repeat(
            cam2worlds[:, :S],
            "b v i j -> b (v h w) i j",
            h=H,
            w=W,
        )
        world_quats = cam_quat_xyzw_to_world_quat_wxyz(quats, c2w_mat)
        # gs_rotations_world = world_quats  # b (v h w) c



        # Handle spherical harmonics (SH) coefficients
        if self.predict_residual_sh:
            new_sh = torch.zeros_like(residual_sh)
            new_sh[..., 0, :] = RGB2SH(
                images[:, :S].permute(0, 1, 3, 4, 2).reshape(B, S * H * W, 3)
            )
            splats['sh'] = new_sh + residual_sh
            splats['residual_sh'] = residual_sh
        else:
            splats['sh'] = residual_sh

        splats["quats"] = world_quats
        splats["scales"] = scales
        splats["opacities"] = opacities
        splats["means"] = gs_means_world
        splats["means_cam"] = gs_means_cam

        # 2.3) 3DGS color / SH coefficient (world space)
        # sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        # if not self.pred_color:
        #     sh = sh * self.sh_mask

        # if self.pred_color or self.sh_degree == 0:
        #     # predict pre-computed color or predict only DC band, no need to transform
        #     gs_sh_world = sh
        # else:
        #     gs_sh_world = rotate_sh(sh, cam2worlds[:, :, None, None, None, :3, :3])
        # gs_sh_world = rearrange(gs_sh_world, "b v h w xyz d_sh -> b (v h w) xyz d_sh")

        # 2.4) 3DGS opacity
        # gs_opacities = rearrange(opacities, "b v h w ... -> b (v h w) ...")

        # splats["weights"] = reg_dense_weights(weights.reshape(B, S * H * W))

        # splats["means"] = global_gs_points + offsets

        # depth = predictions["gs_depth"][:, :S].reshape(B * S, H, W)
        # if context_predictions is not None:
        #     pose3x4, intrinsic = vector_to_camera_matrices(
        #         context_predictions["camera_params"][:, :S].reshape(B * S, -1), (H, W)
        #     )
        # else:
        #     pose3x4, intrinsic = vector_to_camera_matrices(
        #         predictions["camera_params"][:, :S].reshape(B * S, -1), (H, W)
        #     )
        # pose4x4 = torch.eye(4, device=pose3x4.device, dtype=pose3x4.dtype)[None].repeat(B * S, 1, 1)
        # pose4x4[:, :3, :4] = pose3x4
        # extrinsics = closed_form_inverse_se3(pose4x4)
        # pts3d, _, _ = depth_to_world_coords_points(depth, extrinsics.detach(), intrinsic.detach())
        # pts3d = pts3d.reshape(B, S * H * W, 3)

                    # # Compute 3D positions based on specified method
                    # if position_from == "pts3d":
                    #     pts3d = predictions["pts3d"][:, :S].reshape(B, S * H * W, 3)
                    #     splats["means"] = pts3d + offsets
                        
                    # elif position_from == "preddepth+predcamera":
                    #     depth = predictions["depth"][:, :S].reshape(B * S, H, W)
                    #     if context_predictions is not None:
                    #         pose3x4, intrinsic = vector_to_camera_matrices(
                    #             context_predictions["camera_params"][:, :S].reshape(B * S, -1), (H, W)
                    #         )
                    #     else:
                    #         pose3x4, intrinsic = vector_to_camera_matrices(
                    #             predictions["camera_params"][:, :S].reshape(B * S, -1), (H, W)
                    #         )
                    #     pose4x4 = torch.eye(4, device=pose3x4.device, dtype=pose3x4.dtype)[None].repeat(B * S, 1, 1)
                    #     pose4x4[:, :3, :4] = pose3x4
                    #     extrinsics = closed_form_inverse_se3(pose4x4)
                    #     pts3d, _, _ = depth_to_world_coords_points(depth, extrinsics.detach(), intrinsic.detach())
                    #     pts3d = pts3d.reshape(B, S * H * W, 3)
                    #     splats["means"] = pts3d + offsets
                        
                    # elif position_from == "gsdepth+predcamera":
                    #     depth = predictions["gs_depth"][:, :S].reshape(B * S, H, W)
                    #     if context_predictions is not None:
                    #         pose3x4, intrinsic = vector_to_camera_matrices(
                    #             context_predictions["camera_params"][:, :S].reshape(B * S, -1), (H, W)
                    #         )
                    #     else:
                    #         pose3x4, intrinsic = vector_to_camera_matrices(
                    #             predictions["camera_params"][:, :S].reshape(B * S, -1), (H, W)
                    #         )
                    #     pose4x4 = torch.eye(4, device=pose3x4.device, dtype=pose3x4.dtype)[None].repeat(B * S, 1, 1)
                    #     pose4x4[:, :3, :4] = pose3x4
                    #     extrinsics = closed_form_inverse_se3(pose4x4)
                    #     pts3d, _, _ = depth_to_world_coords_points(depth, extrinsics.detach(), intrinsic.detach())
                    #     pts3d = pts3d.reshape(B, S * H * W, 3)
                    #     splats["means"] = pts3d + offsets
                        
                    # elif position_from == "gsdepth+gtcamera":
                    #     depth = predictions["gs_depth"][:, :S].reshape(B * S, H, W)
                    #     pose4x4 = views["camera_pose"][:, :S].reshape(B * S, 4, 4)
                    #     intrinsic = views["camera_intrinsics"][:, :S].reshape(B * S, 3, 3)
                    #     extrinsics = pose4x4
                    #     pts3d, _, _ = depth_to_world_coords_points(depth, extrinsics.detach(), intrinsic.detach())
                    #     pts3d = pts3d.reshape(B, S * H * W, 3)
                    #     splats["means"] = pts3d + offsets
                        
                    # else:
                    #     raise ValueError(f"Invalid position_from={position_from}")

        return splats

    def prepare_cameras(self, views, nums):
        viewmats = views['camera_pose'][:, :nums]
        Ks = views['camera_intrinsics'][:, :nums]
        return viewmats, Ks

    def prepare_prediction_cameras(self, predictions, nums, hw: Tuple[int, int]):
        """
        Prepare camera matrices from predicted pose encodings.
        
        Args:
            predictions: Dictionary containing pose_enc predictions
            nums: Number of views to process
            hw: Tuple of (height, width)
            
        Returns:
            viewmats: Camera view matrices [B, S, 4, 4]
            Ks: Camera intrinsic matrices [B, S, 3, 3]
        """
        B = predictions["camera_params"].shape[0]
        H, W = hw
        
        # Convert pose encoding to extrinsics and intrinsics
        pose3x4, intrinsic = vector_to_camera_matrices(
            predictions["camera_params"][:, :nums].reshape(B * nums, -1), (H, W)
        )
        
        # Convert to homogeneous coordinates and compute view matrices
        pose4x4 = torch.eye(4, device=pose3x4.device, dtype=pose3x4.dtype)[None].repeat(B * nums, 1, 1)
        pose4x4[:, :3, :4] = pose3x4

        viewmats = closed_form_inverse_se3(pose4x4).reshape(B, nums, 4, 4)
        Ks = intrinsic.reshape(B, nums, 3, 3)
        
        return viewmats, Ks
            
        
        
if __name__ == "__main__":
    device = "cuda:0"
    means = torch.randn((100, 3), device=device)
    quats = torch.randn((100, 4), device=device)
    scales = torch.rand((100, 3), device=device) * 0.1  
    opacities = torch.rand((100,), device=device)
    colors = torch.rand((100, 3), device=device)

    viewmats = torch.eye(4, device=device)[None, :, :].repeat(10, 1, 1)
    Ks = torch.tensor([
    [300., 0., 150.], [0., 300., 100.], [0., 0., 1.]], device=device)[None, :, :].repeat(10, 1, 1)
    width, height = 300, 200

    rasterizer = Rasterizer()
    splats = {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
    }
    colors, alphas, _ = rasterizer.rasterize_splats(splats, viewmats, Ks, width, height)
    