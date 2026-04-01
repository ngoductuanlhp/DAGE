# Last modified: 2024-03-30
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------

# @GonzaloMartinGarcia
# The following code is built upon Marigold's infer.py, and was adapted to include some new settings.
# All additions made are marked with # add.

import os
import sys
# Add project root to path (go up 4 levels from this file)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "third_party"))


import argparse
import logging
import os
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import h5py
import mediapy as media
from dage.utils.timer import CUDATimer

# add
import sys

from third_party.depth_pro import depth_pro

from training.util.seed_all import seed_all

from decord import VideoReader, cpu
from evaluation.geocrafter.metrics import compute_metrics, compute_metrics_per_frame_scale
from evaluation.video_depth_geocrafter.boundary_metrics import compute_boundary_metrics
from evaluation.video_depth_geocrafter.shaprness_metric import dbe_acc_comp_video

from evaluation.moge.utils.tools import key_average, flatten_nested_dict, timeit, import_file_as_module

from evaluation.geocrafter.config import BENCHMARK_CONFIGS


def depth_to_pointmap(depth, intrinsics):
    """
    Convert depth map to point map using camera intrinsics.
    
    Args:
        depth: (H, W) depth map
        intrinsics: camera intrinsics dict with 'fx', 'fy', 'cx', 'cy'
    
    Returns:
        point_map: (H, W, 3) point map
    """
    H, W = depth.shape
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    # Create pixel grid
    y, x = torch.meshgrid(
        torch.arange(H, device=depth.device, dtype=depth.dtype),
        torch.arange(W, device=depth.device, dtype=depth.dtype),
        indexing='ij'
    )
    
    # Unproject to 3D
    z = depth
    x_3d = (x - cx) * z / fx
    y_3d = (y - cy) * z / fy
    
    point_map = torch.stack([x_3d, y_3d, z], dim=-1)  # H W 3
    return point_map


def run_benchmark_inference_and_eval(
        benchmark_name, 
        benchmark_config, 
        predictor,
        transform,
        device,     
        output_dir,
        prior_max_size=None,
):
    """Run inference and evaluation on a single benchmark configuration in memory."""
    logging.info(f"Running inference and evaluation on benchmark: {benchmark_name}")
    
    # Create benchmark-specific output directory
    benchmark_output_dir = os.path.join(output_dir, benchmark_name)
    os.makedirs(benchmark_output_dir, exist_ok=True)

    meta_file_path = os.path.join(benchmark_config["path"], 'filename_list.txt')
    samples = []
    with open(meta_file_path, "r") as f:
        for line in f.readlines():
            video_path, data_path = line.split()
            samples.append(dict(
                video_path=os.path.join(benchmark_config["path"], video_path),
                data_path=os.path.join(benchmark_config["path"], data_path)
            ))
    
    metrics_list = []
    runtime_list = []
    fps_list = []

    with torch.no_grad(), tqdm(total=len(samples), desc=f"Processing {benchmark_name}", leave=False) as pbar:

        for i, sample in enumerate(samples):

            video_path = sample["video_path"]
            data_path = sample["data_path"]
            height, width = benchmark_config["height"], benchmark_config["width"]
            downsample_ratio = benchmark_config.get("downsample_ratio", 1.0)
            use_weight = benchmark_config["use_weight"]

            # Load ground truth first
            with h5py.File(data_path, "r") as file:
                gt_mask = file['valid_mask'][:].astype(np.bool_) # T H W
                gt_pmap = file['point_map'][:].astype(np.float32) # T H W C
            gt_pmap = torch.from_numpy(gt_pmap).to(device).float()
            gt_mask = torch.from_numpy(gt_mask).to(device).bool()

            # Load video data
            vid = VideoReader(video_path, ctx=cpu(0))
            original_height, original_width = vid.get_batch([0]).shape[1:3]
            frames_idx = list(range(0, len(vid), 1))
            frames = vid.get_batch(frames_idx).asnumpy().astype(np.float32)

            frames_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2).to(device) / 255.0 # T 3 H W
            original_height, original_width = frames_tensor.shape[-2], frames_tensor.shape[-1]

            if downsample_ratio > 1.0:
                frames_tensor = F.interpolate(frames_tensor, (round(frames_tensor.shape[-2]/downsample_ratio), round(frames_tensor.shape[-1]/downsample_ratio)), mode='bicubic', antialias=True).clamp(0, 1)

            # DepthPro inference - process frame by frame
            num_frames = frames_tensor.shape[0]
            pred_pmaps = []
            pred_masks = []
            
            # We'll get the focal length from the first frame's prediction
            predicted_focal_length = None
            
            with CUDATimer("Model forward", enabled=True, num_frames=num_frames) as runtime:
                for frame_idx in range(num_frames):
                    frame = frames_tensor[frame_idx]  # 3 H W
                    
                    # Convert to PIL Image for DepthPro
                    frame_pil = Image.fromarray(
                        (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    )
                    
                    # Apply transform
                    frame_transformed = transform(frame_pil)
                    
                    # Run inference
                    # For the first frame, let DepthPro predict the focal length (f_px=None)
                    # For subsequent frames, use the predicted focal length from first frame
                    # with torch.amp.autocast(enabled=True, dtype=torch.bfloat16, device_type='cuda'):
                    prediction = predictor.infer(
                        frame_transformed, 
                        f_px=predicted_focal_length
                    )
                    
                    depth = prediction["depth"]  # H W, depth in meters
                    focallength_px = prediction["focallength_px"]  # Predicted focal length
                    
                    # Store focal length from first frame for consistency across video
                    if predicted_focal_length is None:
                        predicted_focal_length = focallength_px
                        # Create intrinsics using predicted focal length
                        # Assume principal point at image center and square pixels (fx=fy)
                        cx, cy = original_width / 2, original_height / 2
                        fx = fy = predicted_focal_length
                        intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
                    
                    # Convert depth to point map
                    point_map = depth_to_pointmap(depth, intrinsics)  # H W 3
                    
                    # Create mask (DepthPro provides valid depth everywhere)
                    mask = torch.ones_like(depth, dtype=torch.bool)
                    
                    pred_pmaps.append(point_map)
                    pred_masks.append(mask)
            
            # Stack predictions
            pred = torch.stack(pred_pmaps, dim=0)  # T H W 3
            pred_mask = torch.stack(pred_masks, dim=0)  # T H W

            # Apply upsampling if downsampled
            if downsample_ratio > 1.0:
                pred = F.interpolate(pred.permute(0, 3, 1, 2), (original_height, original_width), mode='bilinear').permute(0, 2, 3, 1)
                pred_mask = (F.interpolate(pred_mask.unsqueeze(1).float(), (original_height, original_width), mode='bilinear') > 0.5).squeeze(1)

            # Compute metrics immediately without saving to disk
            metrics, affine_invariant_pmap, scale_invariant_pmap  = compute_metrics(
                pred, 
                # pred_mask, 
                gt_pmap, 
                gt_mask,
                use_weight=use_weight,
                compute_direct_metrics=True,
            )
            
                    # if benchmark_name in ["sintel", "urbansyn", "unreal4k_quad", "monkaa", "unreal4k_2k"]:
                    #     boundary_metrics = compute_boundary_metrics(pred[..., 2], gt_pmap[..., 2])
                    #     metrics.update(boundary_metrics)
                        
                    #     dbe_accuracy, dbe_completeness = dbe_acc_comp_video(
                    #         affine_invariant_pmap[..., 2], 
                    #         gt_pmap[..., 2], 
                    #         gt_mask
                    #     )
                    #     dbe_chamfer = (dbe_accuracy + dbe_completeness) / 2
                    #     metrics["dbe_accuracy"] = dbe_accuracy
                    #     metrics["dbe_completeness"] = dbe_completeness
                    #     metrics["dbe_chamfer"] = dbe_chamfer
            # print(f"metrics: {metrics}")
            metrics_list.append(metrics)

            if i > 0:
                fps_list.append(1000 / runtime.avg_time_ms)
                runtime_list.append(runtime.avg_time_ms)
            
            # logging.debug(f"Metrics for sample {i}: {metrics}")
            
            pbar.update(1)
    

    mean_fps = np.mean(fps_list)
    mean_runtime = np.mean(runtime_list)
    print(f"Mean FPS: {mean_fps:.2f}, Mean runtime: {mean_runtime:.2f} ms")

    # -------------------- Compute and save metrics --------------------
    mean_metrics = key_average(metrics_list)
    
    # Prepare comprehensive results
    results = {
        "summary": {
            "benchmark_name": benchmark_name,
            "num_samples": len(metrics_list),
            "mean_metrics": mean_metrics,
            "benchmark_config": benchmark_config
        },
        "per_sample_metrics": []
    }
    
    # Add per-sample metrics with sample info
    for i, sample_metrics in enumerate(metrics_list):
        sample_info = {
            "sample_id": i,
            "video_path": samples[i]["video_path"],
            "data_path": samples[i]["data_path"],
            "metrics": sample_metrics
        }
        results["per_sample_metrics"].append(sample_info)
    
    # Save comprehensive results as JSON
    metrics_json_filename = os.path.join(benchmark_output_dir, "eval_metrics.json")
    with open(metrics_json_filename, "w+") as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Evaluation metrics for {benchmark_name} saved to {metrics_json_filename}")
    logging.info(f"Mean metrics for {benchmark_name}: {mean_metrics}")
    
    return mean_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run DepthPro inference and evaluation on a single benchmark in memory."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (not used for DepthPro, uses default pretrained).",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for evaluation results."
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=list(BENCHMARK_CONFIGS.keys()),
        help="Single benchmark to process.",
    )
    parser.add_argument(
        "--prior_max_size",
        type=int,
        required=False,
        default=518,
        help="Max prior size (not used for DepthPro).",
    )

    # inference setting
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--no_cuda", action="store_true", help="Run without cuda")

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    output_dir = args.output_dir
    benchmark_name = args.benchmark
    seed = args.seed
    prior_max_size = args.prior_max_size

    print(f"run with prior_max_size: {prior_max_size}")

    # Save arguments
    print(f"arguments: {args}")
    os.makedirs(output_dir, exist_ok=True)
    args_dict = vars(args)
    args_str = '\n'.join(f"{key}: {value}" for key, value in args_dict.items())
    args_path = os.path.join(output_dir, "arguments.txt")
    with open(args_path, 'w') as file:
        file.write(args_str)
    print(f"Arguments saved in {args_path}")

    # -------------------- Preparation --------------------
    logging.info(
        f"Processing settings: benchmark = {benchmark_name}, seed = {seed}"
    )

    # Random seed
    if seed is None:
        import time
        seed = int(time.time())
    seed_all(seed)

    # -------------------- Device --------------------
    device = torch.device("cuda" if not args.no_cuda else "cpu")
    logging.info(f"device = {device}")

    # -------------------- Load Model --------------------
    logging.info("Loading DepthPro model...")
    model, transform = depth_pro.create_model_and_transforms(device=device)
    model = model.to(device)
    model.eval()
    logging.info("DepthPro model loaded successfully")

    # -------------------- Process single benchmark --------------------
    if benchmark_name not in BENCHMARK_CONFIGS:
        logging.error(f"Unknown benchmark: {benchmark_name}")
        sys.exit(1)
        
    benchmark_config = BENCHMARK_CONFIGS[benchmark_name]
    
    mean_metrics = run_benchmark_inference_and_eval(
        benchmark_name, 
        benchmark_config, 
        model,
        transform,
        device, 
        output_dir,
        prior_max_size=args.prior_max_size,
    )

    if mean_metrics is not None:
        print(f"\nProcessing completed for {benchmark_name}!")
        print(f"Results saved in: {output_dir}")
        print(f"Mean metrics: {mean_metrics}")
    else:
        print(f"Processing failed for {benchmark_name}")
        sys.exit(1)

