import torch
import torch.nn.functional as F

import numpy as np

# from eval.moge.utils.geometry_torch import mask_aware_nearest_resize
# from eval.moge.utils.alignment import align_points_scale_xyz_shift

# from third_party.moge.moge.utils.geometry_torch import mask_aware_nearest_resize
# from third_party.moge.moge.utils.alignment import align_points_scale_xyz_shift




def abs_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target
    if valid_mask is not None:
        abs_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    abs_relative_diff = torch.sum(abs_relative_diff, (-1, -2)) / n
    return abs_relative_diff.mean()


def squared_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    square_relative_diff = (
        torch.pow(torch.abs(actual_output - actual_target), 2) / actual_target
    )
    if valid_mask is not None:
        square_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    square_relative_diff = torch.sum(square_relative_diff, (-1, -2)) / n
    return square_relative_diff.mean()


def rmse_linear(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    diff = actual_output - actual_target
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n
    rmse = torch.sqrt(mse)
    return rmse.mean()


def rmse_log(output, target, valid_mask=None):
    diff = torch.log(output) - torch.log(target)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse)
    return rmse.mean()


def log10(output, target, valid_mask=None):
    if valid_mask is not None:
        diff = torch.abs(
            torch.log10(output[valid_mask]) - torch.log10(target[valid_mask])
        )
    else:
        diff = torch.abs(torch.log10(output) - torch.log10(target))
    return diff.mean()


# adapt from: https://github.com/imran3180/depth-map-prediction/blob/master/main.py
def threshold_percentage(output, target, threshold_val, valid_mask=None):
    d1 = output / target
    d2 = target / output
    max_d1_d2 = torch.max(d1, d2)
    zero = torch.zeros(*output.shape)
    one = torch.ones(*output.shape)
    bit_mat = torch.where(max_d1_d2.cpu() < threshold_val, one, zero)
    if valid_mask is not None:
        bit_mat[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    count_mat = torch.sum(bit_mat, (-1, -2))
    threshold_mat = count_mat / n.cpu()
    return threshold_mat.mean()


def delta1_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25, valid_mask)


def delta2_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**2, valid_mask)


def delta3_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**3, valid_mask)


def i_rmse(output, target, valid_mask=None):
    output_inv = 1.0 / output
    target_inv = 1.0 / target
    diff = output_inv - target_inv
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse)
    return rmse.mean()


def silog_rmse(depth_pred, depth_gt, valid_mask=None):
    diff = torch.log(depth_pred) - torch.log(depth_gt)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = depth_gt.shape[-2] * depth_gt.shape[-1]

    diff2 = torch.pow(diff, 2)

    first_term = torch.sum(diff2, (-1, -2)) / n
    second_term = torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
    loss = torch.sqrt(torch.mean(first_term - second_term)) * 100
    return loss


def point_rel_error(point_map, point_map_gt, mask_gt):
    assert point_map.shape == point_map_gt.shape
    assert point_map_gt.shape[:-1] == mask_gt.shape
    # *, H, W, 3
    # *, H, W, 3
    # *, H, W
    error = torch.norm(point_map - point_map_gt, p=2, dim=-1, keepdim=False)
    rel_error = error / torch.clamp_min(torch.norm(point_map_gt, p=2, dim=-1, keepdim=False), 1e-2)

    rel_error = (rel_error * mask_gt.float()).sum((-1, -2)) / mask_gt.float().sum((-1, -2))
    return rel_error.mean()

def depth_rel_error(depth, depth_gt, mask_gt):
    assert depth.shape == depth_gt.shape
    assert depth_gt.shape == mask_gt.shape

    # *, H, W
    # *, H, W
    # *, H, W
    error = (depth - depth_gt).abs()
    rel_error = error / torch.clamp_min(depth_gt.abs(), 1e-2)
    rel_error = (rel_error * mask_gt.float()).sum((-1, -2)) / mask_gt.float().sum((-1, -2))
    return rel_error.mean()

def point_inlier_percent(point_map, point_map_gt, mask_gt):
    assert point_map.shape == point_map_gt.shape
    assert point_map_gt.shape[:-1] == mask_gt.shape
    # *, H, W, 3
    # *, H, W, 3
    # *, H, W
    error = torch.norm(point_map - point_map_gt, p=2, dim=-1, keepdim=False)
    rel_error = error / torch.clamp_min(torch.norm(point_map_gt, p=2, dim=-1, keepdim=False), 1e-2)
    percentage = ((rel_error < 0.25).float() * mask_gt.float()).sum((-1, -2)) / mask_gt.float().sum((-1, -2))
    return percentage.mean()

def depth_inlier_percent(depth, depth_gt, mask_gt):
    assert depth.shape == depth_gt.shape
    assert depth_gt.shape == mask_gt.shape
    # *, H, W
    # *, H, W
    # *, H, W
    error = torch.max(depth.abs()/torch.clamp_min(depth_gt.abs(), 1e-2), depth_gt.abs()/torch.clamp_min(depth.abs(), 1e-2))
    percentage = ((error < 1.25).float() * mask_gt.float()).sum((-1, -2)) / mask_gt.float().sum((-1, -2))
    return percentage.mean()


def recover_scale(points, points_gt, mask=None, weight=None):
    """
    Recover the scale factor for a point map with a target point map by minimizing the mse loss.
    points * scale ~ points_gt

    ### Parameters:
    - `points: torch.Tensor` of shape (T, H, W, 3/1)
    - `points_gt: torch.Tensor` of shape (T, H, W, 3/1)
    - `mask: torch.Tensor` of shape (T, H, W) Optional.
    - `weight: torch.Tensor` of shape (T, H, W) Optional.
    
    ### Returns:
    - `scale`: the estimated scale factor
    """
    ndim = points.shape[-1]
    points = points.reshape(-1, ndim)
    points_gt = points_gt.reshape(-1, ndim)
    mask = None if mask is None else mask.reshape(-1)
    weight = None if weight is None else weight.reshape(-1)
        
    if mask is not None:
        points = points[mask]
        points_gt = points_gt[mask]
        weight = None if weight is None else weight[mask]
    # min_x ||Ax-b||_2^2
    A = points.reshape(-1, 1)
    b = points_gt.reshape(-1, 1)
    if weight is not None:
        weight = torch.tile(weight.reshape(-1, 1), (1, ndim)).reshape(-1)
        A = A * weight[:, None]
        b = b * weight[:, None]
    x = torch.linalg.lstsq(A, b, rcond=None).solution
    return x[0, 0]


def recover_scale_xyz_shift(points, points_gt, mask=None, weight=None, z_shift_only=False):
    """
    Recover the scale factor and shift vector for a point map with a target point map by minimizing the mse loss.
    points * scale + shift ~ points_gt

    ### Parameters:
    - `points: torch.Tensor` of shape (T, H, W, 3/1)
    - `points_gt: torch.Tensor` of shape (T, H, W, 3/1)
    - `mask: torch.Tensor` of shape (T, H, W) Optional.
    - `weight: torch.Tensor` of shape (T, H, W) Optional.
    - `z_shift_only: bool` If True, only apply shift to z-dimension (last dimension)
    
    ### Returns:
    - `scale`: the estimated scale factor (scalar)
    - `shift`: the estimated shift vector (3D or 1D depending on input)
    """
    ndim = points.shape[-1]
    points = points.reshape(-1, ndim)
    points_gt = points_gt.reshape(-1, ndim)
    mask = None if mask is None else mask.reshape(-1)
    weight = None if weight is None else weight.reshape(-1)
        
    if mask is not None:
        points = points[mask]
        points_gt = points_gt[mask]
        weight = None if weight is None else weight[mask]
    
    num_points = points.shape[0]
    if num_points == 0:
        return torch.tensor(0.0, device=points.device, dtype=points.dtype), torch.zeros(ndim, device=points.device, dtype=points.dtype)
    
    if z_shift_only:
        # Only apply shift to z-dimension (last dimension)
        # Set up equations: points[:, j] * scale = points_gt[:, j] for j < ndim-1
        #                   points[:, -1] * scale + shift = points_gt[:, -1] for j = ndim-1
        
        total_equations = num_points * ndim
        num_params = 2  # 1 scale + 1 z-shift
        
        # Flatten points and points_gt
        points_flat = points.reshape(-1)  # (num_points * ndim,)
        points_gt_flat = points_gt.reshape(-1)  # (num_points * ndim,)
        
        # Create design matrix A
        A = torch.zeros(total_equations, num_params, dtype=points.dtype, device=points.device)
        
        # Fill scale column (column 0) - coefficients are the flattened points
        A[:, 0] = points_flat
        
        # Fill z-shift column (column 1) - only for z-dimension equations
        # Put 1.0 in rows corresponding to z-dimension: (ndim-1), (ndim-1)+ndim, (ndim-1)+2*ndim, ...
        z_indices = torch.arange(ndim-1, total_equations, ndim, device=points.device)
        A[z_indices, 1] = 1.0
        
        # Target vector
        b = points_gt_flat
        
        # Apply weights if provided
        if weight is not None:
            weight_expanded = torch.tile(weight.reshape(-1, 1), (1, ndim)).reshape(-1)
            A = A * weight_expanded[:, None]
            b = b * weight_expanded
        
        # Solve the least squares problem
        try:
            solution = torch.linalg.lstsq(A, b, rcond=None).solution
            scale = solution[0]
            # Create shift vector with zeros for x,y and the solved value for z
            shift = torch.zeros(ndim, dtype=points.dtype, device=points.device)
            shift[-1] = solution[1]  # z-shift
            return scale, shift
        except:
            return torch.tensor(0.0, device=points.device, dtype=points.dtype), torch.zeros(ndim, device=points.device, dtype=points.dtype)
    
    else:
        # Original implementation: shift for all dimensions
        # Set up the least squares problem: points * scale + shift = points_gt
        # We need to solve for [scale, shift_x, shift_y, shift_z] (or just [scale, shift] for 1D)
        
        total_equations = num_points * ndim
        num_params = 1 + ndim  # 1 scale + ndim shift components
        
        # Flatten points and points_gt to (num_points * ndim,)
        points_flat = points.reshape(-1)  # (num_points * ndim,)
        points_gt_flat = points_gt.reshape(-1)  # (num_points * ndim,)
        
        # Create design matrix A
        A = torch.zeros(total_equations, num_params, dtype=points.dtype, device=points.device)
        
        # Fill scale column (column 0) - coefficients are the flattened points
        A[:, 0] = points_flat
        
        # Fill shift columns (columns 1 to ndim) - create identity pattern
        # For dimension j, put 1.0 in rows j, j+ndim, j+2*ndim, ...
        shift_pattern = torch.eye(ndim, dtype=points.dtype, device=points.device)
        shift_pattern = shift_pattern.repeat(num_points, 1)  # (num_points * ndim, ndim)
        A[:, 1:] = shift_pattern
        
        # Target vector
        b = points_gt_flat
        
        # Apply weights if provided
        if weight is not None:
            # Expand weight to match the number of equations
            weight_expanded = torch.tile(weight.reshape(-1, 1), (1, ndim)).reshape(-1)
            A = A * weight_expanded[:, None]
            b = b * weight_expanded
        
        # Solve the least squares problem
        try:
            solution = torch.linalg.lstsq(A, b, rcond=None).solution
            scale = solution[0]
            shift = solution[1:1+ndim]
            return scale, shift
        except:
            # Fallback in case of numerical issues
            return torch.tensor(0.0, device=points.device, dtype=points.dtype), torch.zeros(ndim, device=points.device, dtype=points.dtype)


def recover_scale_z_shift(points, points_gt, mask=None, weight=None):
    """
    Convenience function to recover scale and z-shift only (no x,y shift).
    Equivalent to recover_scale_xyz_shift(..., z_shift_only=True)
    
    ### Parameters:
    - `points: torch.Tensor` of shape (T, H, W, 3/1)
    - `points_gt: torch.Tensor` of shape (T, H, W, 3/1)
    - `mask: torch.Tensor` of shape (T, H, W) Optional.
    - `weight: torch.Tensor` of shape (T, H, W) Optional.
    
    ### Returns:
    - `scale`: the estimated scale factor (scalar)
    - `shift`: the estimated shift vector (zeros for x,y, value for z)
    """
    return recover_scale_xyz_shift(points, points_gt, mask, weight, z_shift_only=True)

def recover_scale_xyz_shift_moge(points, points_gt, mask=None, weight=None, z_shift_only=False):
    anchor_frame = 0
    _, lr_mask, lr_index = mask_aware_nearest_resize(None, mask[anchor_frame], (64, 64), return_index=True)

    pred_points_lr_masked, gt_points_lr_masked = points[anchor_frame][lr_index][lr_mask], points_gt[anchor_frame][lr_index][lr_mask]
    scale, shift = align_points_scale_xyz_shift(pred_points_lr_masked, gt_points_lr_masked, 1 / gt_points_lr_masked.norm(dim=-1))

    return scale, shift


# def compute_metrics(
#     pred_pmap,
#     pred_mask,
#     gt_pmap,
#     gt_mask,
#     use_weight=False,
#     scale=True,
#     shift=False,
# ):
#     assert pred_pmap.shape == gt_pmap.shape # t,h,w,3 float32
#     assert pred_mask.shape == gt_mask.shape # t,h,w bool

#     # NOTE for sintel, some scene has inf or nan pointmap
#     gt_mask[torch.any(torch.isnan(gt_pmap), dim=-1)] = False
#     gt_mask[torch.any(torch.isinf(torch.abs(gt_pmap)), dim=-1)] = False
#     gt_pmap = torch.where(gt_mask[..., None], gt_pmap, 0.0)
    
#     if scale:
#         if shift:
#             scale_factor, shift_factor = recover_scale_xyz_shift(
#                 pred_pmap, 
#                 gt_pmap, 
#                 mask=gt_mask,
#                 weight=1.0 / (gt_pmap[..., 2] + 1e-6) if use_weight else None,
#                 z_shift_only=False)

#             # scale_factor, shift_factor = recover_scale_xyz_shift_moge(
#             #     pred_pmap, 
#             #     gt_pmap, 
#             #     mask=gt_mask,
#             #     weight=1.0 / (gt_pmap[..., 2] + 1e-6) if use_weight else None,
#             #     z_shift_only=False)

                
#             print(f"Point scale: {scale_factor.item()}, Point shift: {shift_factor.tolist()}")
#             aligned_pmap = pred_pmap * scale_factor + shift_factor
#         else:
#             scale_factor = recover_scale(
#                 pred_pmap, 
#                 gt_pmap, 
#                 mask=gt_mask,
#                 weight=1.0 / (gt_pmap[..., 2] + 1e-6) if use_weight else None)
#             aligned_pmap = pred_pmap * scale_factor
#     else:
#         aligned_pmap = pred_pmap

#     p_rel_err = point_rel_error(
#         aligned_pmap,
#         gt_pmap, 
#         gt_mask
#     ).item()
#     p_in_percent = point_inlier_percent(
#         aligned_pmap, 
#         gt_pmap, 
#         gt_mask
#     ).item()
    
#     if scale:
#         if shift:
#             scale_factor, shift_factor = recover_scale_xyz_shift(
#                 pred_pmap[..., 2:3], 
#                 gt_pmap[..., 2:3], 
#                 mask=gt_mask,
#                 weight=1.0 / (gt_pmap[..., 2] + 1e-6) if use_weight else None,
#                 z_shift_only=False)
#             print(f"Depth scale: {scale_factor.item()}, Depth shift: {shift_factor.tolist()}")
#             aligned_dmap = pred_pmap[..., 2] * scale_factor + shift_factor[..., 0]    
#         else:
#             scale_factor = recover_scale(
#                 pred_pmap[..., 2:3], gt_pmap[..., 2:3], 
#                 mask=gt_mask,
#                 weight=1.0 / (gt_pmap[..., 2] + 1e-6) if use_weight else None)
            
#             aligned_dmap = pred_pmap[..., 2] * scale_factor 
#     else:
#         aligned_dmap = pred_pmap[..., 2]

#     gt_dmap = gt_pmap[..., 2]

#     d_rel_err = depth_rel_error(
#         aligned_dmap, 
#         gt_dmap, 
#         gt_mask
#     ).item()
#     d_in_percent = depth_inlier_percent(
#         aligned_dmap, 
#         gt_dmap, 
#         gt_mask
#     ).item()

#     if shift is None:
#         metrics = {
#             "points_scale_invariant": {
#                 "rel": p_rel_err,
#                 "delta1": p_in_percent
#             },
#             "depth_scale_invariant": {
#                 "rel": d_rel_err,
#                 "delta1": d_in_percent
#             }
#         }
#     else:
#         metrics = {
#             "points_affine_invariant": {
#                 "rel": p_rel_err,
#                 "delta1": p_in_percent
#             },
#             "depth_affine_invariant": {
#                 "rel": d_rel_err,
#                 "delta1": d_in_percent
#             }
#         }

#     return metrics


def compute_direct_metrics_no_alignment(
    pred_pmap,
    gt_pmap,
    gt_mask,
):
    """Compute metrics directly without any scale or shift alignment.
    
    Args:
    ----
        pred_pmap: Predicted point map (T, H, W, 3)
        gt_pmap: Ground truth point map (T, H, W, 3)
        gt_mask: Valid mask for ground truth (T, H, W)
    
    Returns:
    -------
        dict: Dictionary containing unaligned metrics for both points and depth
    
    """
    # Extract depth maps (z-coordinate)
    pred_dmap = pred_pmap[..., 2]
    gt_dmap = gt_pmap[..., 2]
    
    # Compute point-wise metrics without alignment
    p_rel_err = point_rel_error(
        pred_pmap,
        gt_pmap, 
        gt_mask
    ).item()
    p_in_percent = point_inlier_percent(
        pred_pmap, 
        gt_pmap, 
        gt_mask
    ).item()
    
    # Compute depth metrics without alignment
    d_rel_err = depth_rel_error(
        pred_dmap, 
        gt_dmap, 
        gt_mask
    ).item()
    d_in_percent = depth_inlier_percent(
        pred_dmap, 
        gt_dmap, 
        gt_mask
    ).item()
    
    metrics = {
        "points_direct": {
            "rel": p_rel_err,
            "delta1": p_in_percent
        },
        "depth_direct": {
            "rel": d_rel_err,
            "delta1": d_in_percent
        }
    }
    
    return metrics

def compute_affine_invariant_metrics(
    pred_pmap,
    # pred_mask,
    gt_pmap,
    gt_mask,
    use_weight=False,
):
    
    if gt_pmap.shape[-2] > 1024:
        downsample_ratio = 4
        new_height, new_width = int(gt_pmap.shape[-3] // downsample_ratio), int(gt_pmap.shape[-2] // downsample_ratio)
        gt_pmap_downsampled = F.interpolate(gt_pmap.permute(0, 3, 1, 2), (new_height, new_width), mode='nearest').permute(0, 2, 3, 1)
        gt_mask_downsampled = F.interpolate(gt_mask[:, None].float(), (new_height, new_width), mode='nearest').squeeze(1).bool()
        pred_pmap_downsampled = F.interpolate(pred_pmap.permute(0, 3, 1, 2), (new_height, new_width), mode='nearest').permute(0, 2, 3, 1)
        # pred_mask_downsampled = F.interpolate(pred_mask[:, None].float(), (new_height, new_width), mode='nearest').squeeze(1).bool()

        scale_factor, shift_factor = recover_scale_xyz_shift(
            pred_pmap_downsampled, 
            gt_pmap_downsampled, 
            mask=gt_mask_downsampled,
            weight=1.0 / (gt_pmap_downsampled[..., 2] + 1e-6) if use_weight else None,
            z_shift_only=False)

    else:
        scale_factor, shift_factor = recover_scale_xyz_shift(
            pred_pmap, 
            gt_pmap, 
            mask=gt_mask,
            weight=1.0 / (gt_pmap[..., 2] + 1e-6) if use_weight else None,
            z_shift_only=False)
    # breakpoint()
    aligned_pmap = pred_pmap * scale_factor + shift_factor

    if gt_pmap.shape[-2] > 1024:
        scale_factor, shift_factor = recover_scale_xyz_shift(
            pred_pmap_downsampled[..., 2:3], 
            gt_pmap_downsampled[..., 2:3], 
            mask=gt_mask_downsampled,
            weight=1.0 / (gt_pmap_downsampled[..., 2] + 1e-6) if use_weight else None,
            z_shift_only=False)
    else:
        scale_factor, shift_factor = recover_scale_xyz_shift(
            pred_pmap[..., 2:3], 
            gt_pmap[..., 2:3], 
            mask=gt_mask,
            weight=1.0 / (gt_pmap[..., 2] + 1e-6) if use_weight else None,
            z_shift_only=False)
    # print(f"Depth scale: {scale_factor.item()}, Depth shift: {shift_factor.tolist()}")
    aligned_dmap = pred_pmap[..., 2] * scale_factor + shift_factor[..., 0]    
    gt_dmap = gt_pmap[..., 2]

    p_rel_err = point_rel_error(
        aligned_pmap,
        gt_pmap, 
        gt_mask
    ).item()
    p_in_percent = point_inlier_percent(
        aligned_pmap, 
        gt_pmap, 
        gt_mask
    ).item()

    d_rel_err = depth_rel_error(
        aligned_dmap, 
        gt_dmap, 
        gt_mask
    ).item()
    d_in_percent = depth_inlier_percent(
        aligned_dmap, 
        gt_dmap, 
        gt_mask
    ).item()

    metrics ={
        "points_affine_invariant": {
            "rel": p_rel_err,
            "delta1": p_in_percent
        },
        "depth_affine_invariant": {
            "rel": d_rel_err,
            "delta1": d_in_percent
        }
        
    }
    
    return metrics, aligned_pmap

def compute_scale_invariant_metrics(
    pred_pmap,
    # pred_mask,
    gt_pmap,
    gt_mask,
    use_weight=False,
):
    

    if gt_pmap.shape[-2] > 1024:
        downsample_ratio = 4
        new_height, new_width = int(gt_pmap.shape[-3] // downsample_ratio), int(gt_pmap.shape[-2] // downsample_ratio)
        gt_pmap_downsampled = F.interpolate(gt_pmap.permute(0, 3, 1, 2), (new_height, new_width), mode='nearest').permute(0, 2, 3, 1)
        gt_mask_downsampled = F.interpolate(gt_mask[:, None].float(), (new_height, new_width), mode='nearest').squeeze(1).bool()
        pred_pmap_downsampled = F.interpolate(pred_pmap.permute(0, 3, 1, 2), (new_height, new_width), mode='nearest').permute(0, 2, 3, 1)
        # pred_mask_downsampled = F.interpolate(pred_mask[:, None].float(), (new_height, new_width), mode='nearest').squeeze(1).bool()

        scale_factor = recover_scale(
            pred_pmap_downsampled,
            gt_pmap_downsampled,
            mask=gt_mask_downsampled,
            weight=1.0 / (gt_pmap_downsampled[..., 2] + 1e-6) if use_weight else None)

    else:
        scale_factor = recover_scale(
            pred_pmap, 
            gt_pmap, 
            mask=gt_mask,
            weight=1.0 / (gt_pmap[..., 2] + 1e-6) if use_weight else None)

    aligned_pmap = pred_pmap * scale_factor
    
    if gt_pmap.shape[-2] > 1024:
        scale_factor = recover_scale(
            pred_pmap_downsampled[..., 2:3],
            gt_pmap_downsampled[..., 2:3],
            mask=gt_mask_downsampled,
            weight=1.0 / (gt_pmap_downsampled[..., 2] + 1e-6) if use_weight else None)
    else:
        scale_factor = recover_scale(
            pred_pmap[..., 2:3], 
            gt_pmap[..., 2:3], 
            mask=gt_mask,
            weight=1.0 / (gt_pmap[..., 2] + 1e-6) if use_weight else None)
    aligned_dmap = pred_pmap[..., 2] * scale_factor
    
    gt_dmap = gt_pmap[..., 2]
    
    p_rel_err = point_rel_error(
        aligned_pmap,
        gt_pmap, 
        gt_mask
    ).item()
    p_in_percent = point_inlier_percent(
        aligned_pmap, 
        gt_pmap, 
        gt_mask
    ).item()
    
    d_rel_err = depth_rel_error(
        aligned_dmap, 
        gt_dmap, 
        gt_mask
    ).item()
    d_in_percent = depth_inlier_percent(
        aligned_dmap, 
        gt_dmap, 
        gt_mask
    ).item()
    
    metrics = {
        "points_scale_invariant": {
            "rel": p_rel_err,
            "delta1": p_in_percent
        },
        "depth_scale_invariant": {
            "rel": d_rel_err,
            "delta1": d_in_percent
        }
    }
    return metrics, aligned_pmap

def compute_metrics(
    pred_pmap,
    # pred_mask,
    gt_pmap,
    gt_mask,
    use_weight=False,
    scale=True,
    shift=False,
    compute_direct_metrics=False,
):
    assert pred_pmap.shape == gt_pmap.shape # t,h,w,3 float32
    # assert pred_mask.shape == gt_mask.shape # t,h,w bool

    # NOTE for sintel, some scene has inf or nan pointmap
    gt_mask[torch.any(torch.isnan(gt_pmap), dim=-1)] = False
    gt_mask[torch.any(torch.isinf(torch.abs(gt_pmap)), dim=-1)] = False
    gt_pmap = torch.where(gt_mask[..., None], gt_pmap, 0.0)

    metrics = dict()
    affine_invariant_metrics, affine_invariant_pmap = compute_affine_invariant_metrics(
        pred_pmap,
        # pred_mask,
        gt_pmap,
        gt_mask,
        use_weight=use_weight
    )
    metrics.update(affine_invariant_metrics)
    scale_invariant_metrics, scale_invariant_pmap = compute_scale_invariant_metrics(
        pred_pmap,
        # pred_mask,
        gt_pmap,
        gt_mask,
        use_weight=use_weight
    )
    metrics.update(scale_invariant_metrics)

    if compute_direct_metrics:
        direct_metrics = compute_direct_metrics_no_alignment(
            pred_pmap,
            gt_pmap,
            gt_mask,
        )
        metrics.update(direct_metrics)

    return metrics, affine_invariant_pmap, scale_invariant_pmap


def compute_metrics_per_frame_scale(
    pred_pmap,
    pred_mask,
    gt_pmap,
    gt_mask,
    use_weight=False
):
    assert pred_pmap.shape == gt_pmap.shape # t,h,w,3 float32
    assert pred_mask.shape == gt_mask.shape # t,h,w bool
    
    T = pred_pmap.shape[0]

    aligned_pmap = []

    for t in range(T):
        scale = recover_scale(
            pred_pmap[t], 
            gt_pmap[t], 
            mask=gt_mask[t],
            weight=1.0 / (gt_pmap[t, ..., 2] + 1e-6) if use_weight else None)
        aligned_pmap_ = pred_pmap[t] * scale
        aligned_pmap.append(aligned_pmap_)

    aligned_pmap = torch.stack(aligned_pmap, dim=0)



    p_rel_err = point_rel_error(
        aligned_pmap,
        gt_pmap, 
        gt_mask
    ).item()
    p_in_percent = point_inlier_percent(
        aligned_pmap, 
        gt_pmap, 
        gt_mask
    ).item()
    
    aligned_dmap = []
    for t in range(T):
        scale = recover_scale(
            pred_pmap[t, ..., 2:3], gt_pmap[t, ..., 2:3], 
            mask=gt_mask[t],
            weight=1.0 / (gt_pmap[t, ..., 2] + 1e-6) if use_weight else None)
        aligned_dmap_ = pred_pmap[t, ..., 2] * scale
        aligned_dmap.append(aligned_dmap_)

    aligned_dmap = torch.stack(aligned_dmap, dim=0)
    
    
    # aligned_dmap = pred_pmap[..., 2] * scale
    gt_dmap = gt_pmap[..., 2]

    d_rel_err = depth_rel_error(
        aligned_dmap, 
        gt_dmap, 
        gt_mask
    ).item()
    d_in_percent = depth_inlier_percent(
        aligned_dmap, 
        gt_dmap, 
        gt_mask
    ).item()

    metrics = {
        "points_scale_invariant": {
            "rel": p_rel_err,
            "delta1": p_in_percent
        },
        "depth_scale_invariant": {
            "rel": d_rel_err,
            "delta1": d_in_percent
        }
    }

    return metrics