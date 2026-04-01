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

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "third_party"))

import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# VGGT-specific imports
from third_party.vggt.models.vggt import VGGT
from third_party.vggt.utils.pose_enc import pose_encoding_to_extri_intri

from training.util.seed_all import seed_all
from dage.utils.data_utils import read_video

from evaluation.relpose.metadata import dataset_metadata
from evaluation.relpose.utils import get_tum_poses, save_tum_poses, process_directory, calculate_averages, eval_metrics, plot_trajectory, load_traj
from third_party.pi3.utils.geometry import homogenize_points, se3_inverse, depth_edge


def extrinsic_to_c2w(extrinsic):
    """
    Convert extrinsic matrix (world-to-camera) to camera-to-world matrix.
    
    Args:
        extrinsic: (T, 4, 4) tensor of extrinsic matrices
        
    Returns:
        c2w: (T, 4, 4) tensor of camera-to-world matrices
    """
    # Extrinsic is world-to-camera, we need camera-to-world
    # c2w = inv(extrinsic)
    c2w = se3_inverse(extrinsic)
    return c2w


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run VGGT camera pose evaluation on multiple benchmarks."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="facebook/VGGT-1B",
        help="Checkpoint path or hub name.",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    parser.add_argument(
        "--eval_dataset", type=str, required=True, help="Evaluation dataset."
    )

    parser.add_argument(
        "--max_size", type=int, default=518, 
        help="Maximum size for inference. Will resize maintaining aspect ratio."
    )

    # inference setting
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--no_cuda", action="store_true", help="Run without cuda")

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)  
    seed = args.seed
    no_cuda = args.no_cuda
    max_size = args.max_size

    # Random seed
    if seed is None:
        import time
        seed = int(time.time())
    seed_all(seed)

    # -------------------- Device --------------------
    cuda_avail = torch.cuda.is_available() and not no_cuda
    if cuda_avail:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Load Model --------------------
    logging.info(f"Loading VGGT model from {checkpoint_path}")
    predictor = VGGT.from_pretrained(checkpoint_path).to(device).to('cuda', dtype=torch.float32)
    predictor = predictor.eval()
    logging.info("VGGT model loaded successfully")

    eval_dataset = args.eval_dataset

    metadata = dataset_metadata.get(eval_dataset)
    img_path = metadata["img_path"]
    mask_path = metadata["mask_path"]
    anno_path = metadata.get("anno_path", None)

    if metadata.get("full_seq", False):
        full_seq = True
    else:
        full_seq = False
        seq_list = metadata.get("seq_list", [])
    if full_seq:
        seq_list = os.listdir(img_path)
        seq_list = [
            seq for seq in seq_list if os.path.isdir(os.path.join(img_path, seq))
        ]
    seq_list = sorted(seq_list)

    ate_list = []
    rpe_trans_list = []
    rpe_rot_list = []

    with torch.no_grad():
        for seq in tqdm(seq_list):
            try:
                dir_path = metadata["dir_path_func"](img_path, seq)

                # Handle skip_condition
                skip_condition = metadata.get("skip_condition", None)
                if skip_condition is not None and skip_condition(output_dir, seq):
                    continue

                mask_path_seq_func = metadata.get(
                    "mask_path_seq_func", lambda mask_path, seq: None
                )
                mask_path_seq = mask_path_seq_func(mask_path, seq)

                # Read video
                video, original_height, original_width, _ = read_video(dir_path)

                # Convert to tensor
                video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
                video = video.to('cuda') / 255.0  # 1 T 3 H W

                # Store original dimensions for interpolation
                inference_height, inference_width = video.shape[-2:]
                
                # Calculate new dimensions based on max_size
                original_aspect_ratio = inference_width / inference_height
                patch_size = 14
                
                if max_size is None:
                    max_size = max(inference_height // patch_size * patch_size, 
                                 inference_width // patch_size * patch_size)
                
                if inference_width > inference_height:
                    new_width = min(max_size, inference_width // patch_size * patch_size)
                    new_height = int((new_width / original_aspect_ratio) // patch_size * patch_size)
                else:
                    new_height = min(max_size, inference_height // patch_size * patch_size)
                    new_width = int((new_height * original_aspect_ratio) // patch_size * patch_size)
                
                # Resize if needed
                if inference_height != new_height or inference_width != new_width:
                    logging.debug(f"Resizing from ({inference_height}, {inference_width}) to ({new_height}, {new_width}) for inference")
                    video = F.interpolate(
                        video.squeeze(0), 
                        (new_height, new_width), 
                        mode='bilinear', 
                        antialias=True
                    ).clamp(0, 1).unsqueeze(0)

                # Inference
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    # Aggregate tokens
                    aggregated_tokens_list, ps_idx = predictor.aggregator(video)
                    
                    # Predict Cameras
                    pose_enc = predictor.camera_head(aggregated_tokens_list)[-1]
                
                # Convert to float32 and get extrinsic matrices
                pose_enc = pose_enc.float()
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, video.shape[-2:])
                
                # Remove batch dimension
                extrinsic = extrinsic.squeeze(0).cpu()  # (T, 4, 4)
                
                # Convert extrinsic (world-to-camera) to camera-to-world
                c2w_matrices = extrinsic_to_c2w(extrinsic)
                
                # Convert to numpy for get_tum_poses
                pred_pose = c2w_matrices.numpy()

                pred_traj = get_tum_poses(pred_pose)

                gt_traj_file = metadata["gt_traj_func"](img_path, anno_path, seq)
                traj_format = metadata.get("traj_format", None)

                if eval_dataset == "sintel":
                    gt_traj = load_traj(
                        gt_traj_file=gt_traj_file, traj_format="sintel", stride=1
                    )
                elif eval_dataset in ["mp3d", "mp3d_short"]:
                    gt_traj = load_traj(
                        gt_traj_file=gt_traj_file, traj_format="mp3d", stride=1
                    )
                elif traj_format is not None:
                    gt_traj = load_traj(
                        gt_traj_file=gt_traj_file,
                        traj_format=traj_format,
                        stride=1,
                    )
                else:
                    gt_traj = None

                if gt_traj is not None:

                    ate, rpe_trans, rpe_rot = eval_metrics(
                        pred_traj,
                        gt_traj,
                        seq=seq,
                        filename=f"{output_dir}/{seq}_eval_metric.txt",
                    )
                    plot_trajectory(
                        pred_traj, gt_traj, title=seq, filename=f"{output_dir}/{seq}.png"
                    )
                else:
                    ate, rpe_trans, rpe_rot = 0, 0, 0
                    bug = True

                ate_list.append(ate)
                rpe_trans_list.append(rpe_trans)
                rpe_rot_list.append(rpe_rot)

            except Exception as e:
                print(f"Error processing {seq}: {e}")
                import traceback
                traceback.print_exc()
                continue

        results = process_directory(output_dir)
        avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results)

        print(f"Ate: {avg_ate:.4f}, RPE Trans: {avg_rpe_trans:.4f}, RPE Rot: {avg_rpe_rot:.4f}")
        with open(f"{output_dir}/summary.txt", "a") as f:
            f.write(
                f"Summary of {eval_dataset} on {len(seq_list)} sequences: \nAverage ATE: {avg_ate:.5f}, Average RPE trans: {avg_rpe_trans:.5f}, Average RPE rot: {avg_rpe_rot:.5f}\n"
            )

