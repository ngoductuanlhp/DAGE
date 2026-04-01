# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Code adapted from:
# https://github.com/prs-eth/Marigold/blob/v0.1.4/src/util/metric.py

import pandas as pd
import torch
import numpy as np
from scipy import ndimage
from skimage import feature

from .tools import absolute_value_scaling, absolute_value_scaling2, depth2disparity



def dbe_acc_comp(output, target, valid_mask=None):
    depth = output.cpu().numpy()
    gt = target.cpu().numpy()
    if valid_mask is not None:
        valid_mask = valid_mask.cpu().numpy()


    ## Run for Depth ##
    pred_normalized = gt - np.nanmin(gt)
    pred_normalized = pred_normalized/np.nanmax(pred_normalized)
    gt_depth_edge = feature.canny(pred_normalized,sigma=np.sqrt(2),low_threshold=0.1,high_threshold=0.2)

    pred_normalized = depth - np.nanmin(depth)
    pred_normalized = pred_normalized/np.nanmax(pred_normalized)
    pred_depth_edge = feature.canny(pred_normalized,sigma=np.sqrt(2),low_threshold=0.1,high_threshold=0.2)

    ## Run for Disparity ##
    disp_gt = depth2disparity(gt)
    pred_normalized = disp_gt - np.nanmin(disp_gt)
    pred_normalized = pred_normalized/np.nanmax(pred_normalized)
    gt_disp_edge = feature.canny(pred_normalized,sigma=np.sqrt(2),low_threshold=0.1,high_threshold=0.2)

    disp = depth2disparity(depth)
    pred_normalized = disp - np.nanmin(disp)
    pred_normalized = pred_normalized/np.nanmax(pred_normalized)
    pred_disp_edge = feature.canny(pred_normalized,sigma=np.sqrt(2),low_threshold=0.1,high_threshold=0.2)

    ## MERGE ##
    gt_edge = (gt_depth_edge | gt_disp_edge) & valid_mask
    pred_edge = (pred_depth_edge | pred_disp_edge) & valid_mask

    # compute distance transform for chamfer metric
    D_gt = ndimage.distance_transform_edt(1-gt_edge)
    D_est = ndimage.distance_transform_edt(1-pred_edge)

    max_dist_thr = 10.; # Threshold for local neighborhood

    mask_D_gt = D_gt<max_dist_thr; # truncate distance transform map

    E_fin_est_filt = pred_edge*mask_D_gt; # compute shortest distance for all predicted edges

    if np.sum(E_fin_est_filt) == 0: # assign MAX value if no edges could be found in prediction
        dbe_acc = max_dist_thr
        dbe_comp = max_dist_thr
    else:
        dbe_acc = np.nansum(D_gt*E_fin_est_filt)/np.nansum(E_fin_est_filt) # accuracy: directed chamfer distance
        dbe_comp = np.nansum(D_est*gt_edge)/np.nansum(gt_edge) # completeness: directed chamfer distance (reversed)

    return dbe_acc, dbe_comp

def dbe_acc_comp_video(output, target, valid_mask=None):
    num_frames = output.shape[0]

    dbe_acc_avg = 0
    dbe_comp_avg = 0
    for i in range(num_frames):
        dbe_acc_, dbe_comp_ = dbe_acc_comp(output[i], target[i], valid_mask[i] if valid_mask is not None else None)
        dbe_acc_avg += dbe_acc_
        dbe_comp_avg += dbe_comp_
    return dbe_acc_avg / num_frames, dbe_comp_avg / num_frames


def dbe_compl(output, target, valid_mask=None):
    depth = output.cpu().numpy()
    gt = target.cpu().numpy()
    if valid_mask is not None:
        valid_mask = valid_mask.cpu().numpy()

    ## Run for Depth ##
    pred_normalized = gt - np.nanmin(gt)
    pred_normalized = pred_normalized/np.nanmax(pred_normalized)
    gt_depth_edge = feature.canny(pred_normalized,
                            sigma=np.sqrt(2),low_threshold=0.1,high_threshold=0.2)

    pred_normalized = depth - np.nanmin(depth)
    pred_normalized = pred_normalized/np.nanmax(pred_normalized)
    pred_depth_edge = feature.canny(pred_normalized,
                            sigma=np.sqrt(2),low_threshold=0.1,high_threshold=0.2)

    ## Run for Disparity ##
    disp_gt = depth2disparity(gt)
    pred_normalized = disp_gt - np.nanmin(disp_gt)
    pred_normalized = pred_normalized/np.nanmax(pred_normalized)
    gt_disp_edge = feature.canny(pred_normalized,
                            sigma=np.sqrt(2),low_threshold=0.1,high_threshold=0.2)

    disp = depth2disparity(depth)
    pred_normalized = disp - np.nanmin(disp)
    pred_normalized = pred_normalized/np.nanmax(pred_normalized)
    pred_disp_edge = feature.canny(pred_normalized,
                            sigma=np.sqrt(2),low_threshold=0.1,high_threshold=0.2)

    ## MERGE ##
    gt_edge = (gt_depth_edge | gt_disp_edge) & valid_mask
    pred_edge = (pred_depth_edge | pred_disp_edge) & valid_mask
    
    # compute distance transform for chamfer metric
    D_gt = ndimage.distance_transform_edt(1-gt_edge)
    D_est = ndimage.distance_transform_edt(1-pred_edge)

    max_dist_thr = 10.; # Threshold for local neighborhood

    mask_D_gt = D_gt<max_dist_thr; # truncate distance transform map

    E_fin_est_filt = pred_edge*mask_D_gt; # compute shortest distance for all predicted edges

    if np.sum(E_fin_est_filt) == 0: # assign MAX value if no edges could be found in prediction
        dbe_com = max_dist_thr
    else:
        dbe_com = np.nansum(D_est*gt_edge)/np.nansum(gt_edge) # completeness: directed chamfer distance (reversed)

    return dbe_com

def dbe_compl_video(output, target, valid_mask=None):
    num_frames = output.shape[0]

    dbe_compl_avg = 0.0
    for i in range(num_frames):
        dbe_compl_avg += dbe_compl(output[i], target[i], valid_mask[i] if valid_mask is not None else None)
    return dbe_compl_avg / num_frames



def sharpdepth_evaluation(
    predicted_depth_original,
    ground_truth_depth_original,
    max_depth=80,
    custom_mask=None,
    post_clip_min=None,
    post_clip_max=None,
    pre_clip_min=None,
    pre_clip_max=None,
    align_with_lstsq=False,
    align_with_lad=False,
    align_with_lad2=False,
    metric_scale=False,
    lr=1e-4,
    max_iters=1000,
    use_gpu=False,
    align_with_scale=False,
    disp_input=False,
):
    """
    Evaluate the depth map using various metrics and return a depth error parity map, with an option for least squares alignment.

    Args:
        predicted_depth (numpy.ndarray or torch.Tensor): The predicted depth map.
        ground_truth_depth (numpy.ndarray or torch.Tensor): The ground truth depth map.
        max_depth (float): The maximum depth value to consider. Default is 80 meters.
        align_with_lstsq (bool): If True, perform least squares alignment of the predicted depth with ground truth.

    Returns:
        dict: A dictionary containing the evaluation metrics.
        torch.Tensor: The depth error parity map.
    """
    if isinstance(predicted_depth_original, np.ndarray):
        predicted_depth_original = torch.from_numpy(predicted_depth_original)
    if isinstance(ground_truth_depth_original, np.ndarray):
        ground_truth_depth_original = torch.from_numpy(ground_truth_depth_original)
    if custom_mask is not None and isinstance(custom_mask, np.ndarray):
        custom_mask = torch.from_numpy(custom_mask)

    # if the dimension is 3, flatten to 2d along the batch dimension
    ori_predicted_depth_original = predicted_depth_original.clone()
    ori_ground_truth_depth_original = ground_truth_depth_original.clone()
    num_frames = predicted_depth_original.shape[0]
    if predicted_depth_original.dim() == 3:
        _, h, w = predicted_depth_original.shape
        predicted_depth_original = predicted_depth_original.view(-1, w)
        ground_truth_depth_original = ground_truth_depth_original.view(-1, w)
        if custom_mask is not None:
            custom_mask = custom_mask.view(-1, w)

    # put to device
    if use_gpu:
        predicted_depth_original = predicted_depth_original.cuda()
        ground_truth_depth_original = ground_truth_depth_original.cuda()
        ori_predicted_depth_original = ori_predicted_depth_original.cuda()
        ori_ground_truth_depth_original = ori_ground_truth_depth_original.cuda()

    # Filter out depths greater than max_depth
    if max_depth is not None:
        mask = (ground_truth_depth_original > 0) & (
            ground_truth_depth_original < max_depth
        )
    else:
        mask = ground_truth_depth_original > 0
    predicted_depth = predicted_depth_original[mask]
    ground_truth_depth = ground_truth_depth_original[mask]

    # Clip the depth values
    if pre_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=pre_clip_min)
        ori_predicted_depth_original = torch.clamp(ori_predicted_depth_original, min=pre_clip_min)
    if pre_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=pre_clip_max)
        ori_predicted_depth_original = torch.clamp(ori_predicted_depth_original, max=pre_clip_max)

    if disp_input:  # align the pred to gt in the disparity space
        real_gt = ground_truth_depth.clone()
        ground_truth_depth = 1 / (ground_truth_depth + 1e-8)
        ori_ground_truth_depth_original = 1 / (ori_ground_truth_depth_original + 1e-8)

    # various alignment methods
    if metric_scale:
        predicted_depth = predicted_depth
    elif align_with_lstsq:
        # Convert to numpy for lstsq
        predicted_depth_np = predicted_depth.cpu().numpy().reshape(-1, 1)
        ground_truth_depth_np = ground_truth_depth.cpu().numpy().reshape(-1, 1)

        # Add a column of ones for the shift term
        A = np.hstack([predicted_depth_np, np.ones_like(predicted_depth_np)])

        # Solve for scale (s) and shift (t) using least squares
        result = np.linalg.lstsq(A, ground_truth_depth_np, rcond=None)
        s, t = result[0][0], result[0][1]

        # convert to torch tensor
        s = torch.tensor(s, device=predicted_depth_original.device)
        t = torch.tensor(t, device=predicted_depth_original.device)

        # Apply scale and shift
        predicted_depth = s * predicted_depth + t
        ori_predicted_depth_original = s * ori_predicted_depth_original + t
    elif align_with_lad:
        s, t = absolute_value_scaling(
            predicted_depth,
            ground_truth_depth,
            s=torch.median(ground_truth_depth) / torch.median(predicted_depth),
        )
        predicted_depth = s * predicted_depth + t
        ori_predicted_depth_original = s * ori_predicted_depth_original + t
    elif align_with_lad2:
        s_init = (
            torch.median(ground_truth_depth) / torch.median(predicted_depth)
        ).item()
        s, t = absolute_value_scaling2(
            predicted_depth,
            ground_truth_depth,
            s_init=s_init,
            lr=lr,
            max_iters=max_iters,
        )
        predicted_depth = s * predicted_depth + t
        ori_predicted_depth_original = s * ori_predicted_depth_original + t
    elif align_with_scale:
        # Compute initial scale factor 's' using the closed-form solution (L2 norm)
        dot_pred_gt = torch.nanmean(ground_truth_depth)
        dot_pred_pred = torch.nanmean(predicted_depth)
        s = dot_pred_gt / dot_pred_pred

        # Iterative reweighted least squares using the Weiszfeld method
        for _ in range(10):
            # Compute residuals between scaled predictions and ground truth
            residuals = s * predicted_depth - ground_truth_depth
            abs_residuals = (
                residuals.abs() + 1e-8
            )  # Add small constant to avoid division by zero

            # Compute weights inversely proportional to the residuals
            weights = 1.0 / abs_residuals

            # Update 's' using weighted sums
            weighted_dot_pred_gt = torch.sum(
                weights * predicted_depth * ground_truth_depth
            )
            weighted_dot_pred_pred = torch.sum(weights * predicted_depth**2)
            s = weighted_dot_pred_gt / weighted_dot_pred_pred

        # Optionally clip 's' to prevent extreme scaling
        s = s.clamp(min=1e-3)

        # Detach 's' if you want to stop gradients from flowing through it
        s = s.detach()

        # Apply the scale factor to the predicted depth
        predicted_depth = s * predicted_depth
        ori_predicted_depth_original = s * ori_predicted_depth_original

    else:
        # Align the predicted depth with the ground truth using median scaling
        scale_factor = torch.median(ground_truth_depth) / torch.median(predicted_depth)
        predicted_depth *= scale_factor
        ori_predicted_depth_original *= scale_factor

    if disp_input:
        # convert back to depth
        ground_truth_depth = real_gt
        predicted_depth = depth2disparity(predicted_depth)
        ori_predicted_depth_original = depth2disparity(ori_predicted_depth_original)

    # Clip the predicted depth values
    if post_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=post_clip_min)
        ori_predicted_depth_original = torch.clamp(ori_predicted_depth_original, min=post_clip_min)
    if post_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=post_clip_max)
        ori_predicted_depth_original = torch.clamp(ori_predicted_depth_original, max=post_clip_max)

    if custom_mask is not None:
        assert custom_mask.shape == ground_truth_depth_original.shape
        mask_within_mask = custom_mask.cpu()[mask]
        predicted_depth = predicted_depth[mask_within_mask]
        ground_truth_depth = ground_truth_depth[mask_within_mask]

    dbe_accuracy, dbe_completeness = dbe_acc_comp_video(ori_predicted_depth_original, ori_ground_truth_depth_original, mask.reshape(num_frames, h, w))
    # dbe_completeness = dbe_compl_video(ori_predicted_depth_original, ori_ground_truth_depth_original, mask.reshape(num_frames, h, w))
    

    num_valid_pixels = (
        torch.sum(mask).item()
        if custom_mask is None
        else torch.sum(mask_within_mask).item()
    )
    if num_valid_pixels == 0:
        (
            dbe_accuracy,
            dbe_completeness,
        ) = (0, 0)
    
    results = {}
    results["dbe_accuracy"] = dbe_accuracy
    results["dbe_completeness"] = dbe_completeness
    results["valid_pixels"] = num_valid_pixels

    return results
