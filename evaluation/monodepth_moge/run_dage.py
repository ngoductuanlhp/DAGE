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
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

import argparse
import logging
import json
from tabulate import tabulate

import numpy as np
import torch
from tqdm.auto import tqdm
from dage.utils.timer import CUDATimer

from dage.models.dage import DAGE

from training.util.seed_all import seed_all

from evaluation.moge.dataloader import EvalDataLoaderPipeline
from evaluation.moge.metrics import compute_metrics
from evaluation.moge.utils.tools import key_average

# Define benchmark configurations
BENCHMARK_DATA_ROOT = "<PATH TO MOGE EVAL DATA>"

BENCHMARK_CONFIGS = {
    "NYUv2": {
        "path": os.path.join(BENCHMARK_DATA_ROOT, "NYUv2"),
        "width": 640,
        "height": 480,
        "split": ".index.txt",
        "depth_unit": 1.0
    },
    "ETH3D": {
        "path": os.path.join(BENCHMARK_DATA_ROOT, "ETH3D"),
        "width": 2048,
        "height": 1365,
        "split": ".index.txt",
        "include_segmentation": True,
        "depth_unit": 1
    },
    "Sintel": {
        "path": os.path.join(BENCHMARK_DATA_ROOT, "Sintel"),
        "width": 872,
        "height": 432,
        "split": ".index.txt",
        "has_sharp_boundary": True,
        "include_segmentation": True
    },
    "GSO": {
        "path": os.path.join(BENCHMARK_DATA_ROOT, "GSO"),
        "width": 512,
        "height": 512,
        "split": ".index.txt"
    },
    "KITTI": {
        "path": os.path.join(BENCHMARK_DATA_ROOT, "KITTI"),
        "width": 750,
        "height": 375,
        "split": ".index.txt",
        "depth_unit": 1
    },
    "hypersim": {
        "path": os.path.join(BENCHMARK_DATA_ROOT, "hypersim"),
        "width": 640,
        "height": 480,
        "split": ".index.txt",
        "depth_unit": 1
    },
    "iBims-1": {
        "path": os.path.join(BENCHMARK_DATA_ROOT, "iBims-1"),
        "width": 640,
        "height": 480,
        "split": ".index.txt",
        "has_sharp_boundary": True,
        "include_segmentation": True,
        "depth_unit": 1.0
    },
    "DDAD": {
        "path": os.path.join(BENCHMARK_DATA_ROOT, "DDAD"),
        "width": 1400,
        "height": 700,
        "include_segmentation": True,
        "split": ".index.txt",
        "depth_unit": 1.0
    },
    "DIODE": {
        "path": os.path.join(BENCHMARK_DATA_ROOT, "DIODE"),
        "width": 1024,
        "height": 768,
        "split": ".index.txt",
        "include_segmentation": True,
        "depth_unit": 1.0
    },
    "Spring": {
        "path": os.path.join(BENCHMARK_DATA_ROOT, "Spring"),
        "width": 1920,
        "height": 1080,
        "split": ".index.txt",
        "has_sharp_boundary": True
    },
    "HAMMER": {
        "path": os.path.join(BENCHMARK_DATA_ROOT, "HAMMER"),
        "width": 1664,
        "height": 832,
        "split": ".index.txt",
        "depth_unit": 1,
        "has_sharp_boundary": True
    }
}


def evaluate_benchmark(
        benchmark_name,
        benchmark_config,
        predictor,
        device,
        output_dir,
        lr_max_size=252,
):
    """Evaluate a single benchmark configuration."""
    logging.info(f"Evaluating benchmark: {benchmark_name}")

    # Create benchmark-specific output directory
    benchmark_output_dir = os.path.join(output_dir, benchmark_name)
    os.makedirs(benchmark_output_dir, exist_ok=True)

    # Initialize evaluation pipeline
    eval_data_pipe = EvalDataLoaderPipeline(**benchmark_config)

    # -------------------- Inference and evaluation (PER-FRAME) --------------------
    metrics_list = []
    runtime_list = []
    fps_list = []

    logging.info("Processing frames independently...")

    with torch.no_grad():
        with eval_data_pipe:
            pbar = tqdm(total=len(eval_data_pipe), desc=f"Processing {benchmark_name}", leave=False)

            for i in range(len(eval_data_pipe)):
                batch = eval_data_pipe.get()
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # Prepare single frame for inference
                # batch["image"] shape: (C, H, W) in range [0, 255] from dataloader
                frame = batch["image"].float() / 255.0  # (C, H, W) in range [0, 1]

                # DAGE expects (B, N, C, H, W) format
                video_tensor = frame.unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)

                # Run inference on single frame
                with CUDATimer("Model forward", enabled=True, num_frames=1) as runtime:
                    output = predictor.infer(video_tensor, lr_max_size=lr_max_size)

                # Extract prediction (remove temporal dimension)
                pred = output['local_points'][0]  # (H, W, 3) - first (and only) frame

                # Evaluate frame independently (monocular depth evaluation)
                pred_dict = {
                    "points_affine_invariant": pred,
                    "depth_affine_invariant": pred[..., 2]  # Extract z-coordinate (depth)
                }
                metrics, _ = compute_metrics(pred_dict, batch, vis=False)
                metrics_list.append(metrics)

                if i > 0:
                    fps_list.append(1000 / runtime.avg_time_ms)
                    runtime_list.append(runtime.avg_time_ms)

                pbar.update(1)

            pbar.close()

    # -------------------- Compute and save metrics --------------------
    mean_fps = np.mean(fps_list) if len(fps_list) > 0 else 0.0
    mean_runtime = np.mean(runtime_list) if len(runtime_list) > 0 else 0.0
    print(f"Mean FPS: {mean_fps:.2f}, Mean runtime: {mean_runtime:.2f} ms")

    mean_metrics = key_average(metrics_list)

    logging.info(f"Per-frame evaluation completed for {benchmark_name}")
    logging.info(f"Mean metrics: {mean_metrics}")

    # Save metrics
    eval_text = f"Evaluation metrics for {benchmark_name} (PER-FRAME, MONOCULAR EVALUATION):\n"
    eval_text += f"Number of samples: {len(metrics_list)}\n"
    eval_text += f"Mean FPS: {mean_fps:.2f}\n"
    eval_text += f"Mean runtime: {mean_runtime:.2f} ms\n"
    eval_text += f"Benchmark config: {json.dumps(benchmark_config, indent=2)}\n\n"

    metric_names = list(mean_metrics.keys())
    metric_values = [f"{v:.4f}" if isinstance(v, (int, float)) else str(v) for v in mean_metrics.values()]
    eval_text += tabulate([metric_names, metric_values], headers=["Metric", "Value"])

    metrics_filename = os.path.join(benchmark_output_dir, "eval_metrics.txt")
    with open(metrics_filename, "w+") as f:
        f.write(eval_text)

    logging.info(f"Metrics saved to {metrics_filename}")

    # Prepare comprehensive results
    results = {
        "summary": {
            "benchmark_name": benchmark_name,
            "num_samples": len(metrics_list),
            "mean_fps": mean_fps,
            "mean_runtime_ms": mean_runtime,
            "mean_metrics": mean_metrics,
            "benchmark_config": benchmark_config
        },
        "per_sample_metrics": []
    }

    # Add per-sample metrics with sample info
    for i, sample_metrics in enumerate(metrics_list):
        sample_info = {
            "sample_id": i,
            "metrics": sample_metrics
        }
        results["per_sample_metrics"].append(sample_info)

    # Save comprehensive results as JSON
    metrics_json_filename = os.path.join(benchmark_output_dir, "eval_metrics.json")
    with open(metrics_json_filename, "w+") as f:
        json.dump(results, f, indent=2)

    logging.info(f"Evaluation metrics for {benchmark_name} saved to {metrics_json_filename}")

    return mean_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run DAGE evaluation on monocular depth benchmarks (MoGe) with per-frame inference."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/model.pt",
        help="Checkpoint path or hub name.",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for evaluation results."
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=list(BENCHMARK_CONFIGS.keys()),
        help="Single benchmark to evaluate.",
    )

    parser.add_argument(
        "--lr_max_size",
        type=int,
        required=False,
        default=252,
        help="Max LR resolution.",
    )

    # inference setting
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--no_cuda", action="store_true", help="Run without cuda")

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    output_dir = args.output_dir
    benchmark_name = args.benchmark
    seed = args.seed
    lr_max_size = args.lr_max_size

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
        f"Evaluation settings: checkpoint = `{checkpoint_path}`, "
        f"benchmark = {benchmark_name}, seed = {seed}"
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
    predictor = DAGE.from_pretrained(pretrained_model_name_or_path=checkpoint_path, strict=False).to(device).eval()

    # -------------------- Process single benchmark --------------------
    if benchmark_name not in BENCHMARK_CONFIGS:
        logging.error(f"Unknown benchmark: {benchmark_name}")
        sys.exit(1)

    benchmark_config = BENCHMARK_CONFIGS[benchmark_name]

    mean_metrics = evaluate_benchmark(
        benchmark_name,
        benchmark_config,
        predictor,
        device,
        output_dir,
        lr_max_size=lr_max_size,
    )

    if mean_metrics is not None:
        print(f"\nProcessing completed for {benchmark_name}!")
        print(f"Results saved in: {output_dir}")
        print(f"Mean metrics: {mean_metrics}")
    else:
        print(f"Processing failed for {benchmark_name}")
        sys.exit(1)
