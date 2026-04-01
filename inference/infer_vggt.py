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
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)


import argparse
import logging
from tqdm import tqdm
import glob

import albumentations as A
import segmentation_models_pytorch as smp

import numpy as np
import torch
import torch.nn.functional as F
import mediapy as media
import cv2
from PIL import Image

from third_party.vggt.models.vggt import VGGT
from third_party.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from third_party.vggt.utils.geometry import unproject_depth_map_to_point_map

from dage.utils.timer import CUDATimer
from dage.utils.data_utils import read_video
from dage.utils.vis_disp_utils import pmap_to_disp, pmap_to_depth


if "__main__" == __name__:

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run VGGT inference on videos."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="facebook/VGGT-1B",
        help="Checkpoint path or hub name.",
    )

    parser.add_argument(
        "--output_dir", type=str, default="quali_results/vggt", help="Output directory."
    )
    parser.add_argument(
        "--max_size", type=int, default=518, 
        help="Maximum size for inference. Will resize maintaining aspect ratio."
    )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    output_dir = args.output_dir
    max_size = args.max_size
    os.makedirs(output_dir, exist_ok=True)  

    # -------------------- Device --------------------
    device = torch.device("cuda")
    predictor = VGGT.from_pretrained(checkpoint_path).to(device).to('cuda', dtype=torch.float32).eval()


    folder_path = "./assets/demo_data"
    video_paths = sorted([os.path.join(folder_path, l) for l in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, l)) or l.endswith(".mp4") or l.endswith(".MOV")])


    print(f"Found {len(video_paths)} videos at {folder_path}")
    
    with torch.no_grad():
        for video_path in tqdm(video_paths):
            video_name = os.path.basename(video_path.replace("/images", ""))

            # NOTE read the whole video from path
            video, original_height, original_width, fps = read_video(video_path)

            
            num_frames = len(video)
            video = torch.from_numpy(video).permute(0, 3, 1, 2).float()
            video = video.to('cuda') / 255.0
            
            ori_video = video.clone()
            
            # Calculate new dimensions based on max_size (similar to Pi3)
            original_aspect_ratio = original_width / original_height
            patch_size = 14
            
            if max_size is None:
                max_size = max(original_height // patch_size * patch_size, original_width // patch_size * patch_size)
            
            if original_width > original_height:
                new_width = min(max_size, original_width // patch_size * patch_size)
                new_height = int((new_width / original_aspect_ratio) // patch_size * patch_size)
            else:
                new_height = min(max_size, original_height // patch_size * patch_size)
                new_width = int((new_height * original_aspect_ratio) // patch_size * patch_size)
            
            # Resize if needed
            if original_height != new_height or original_width != new_width:
                logging.info(f"Resizing from ({original_height}, {original_width}) to ({new_height}, {new_width}) for inference")
                video = F.interpolate(video, (new_height, new_width), mode='bilinear', antialias=True).clamp(0, 1)
            
            # Add batch dimension
            video = video.unsqueeze(0)  # (1, T, 3, H, W)

            logging.info(f"Processing video: {num_frames} frames, shape {video.shape}, max_size={max_size}")
            
            # VGGT Inference
            with CUDATimer(name="VGGT", measure_memory=True):
                with torch.amp.autocast(enabled=True, dtype=torch.bfloat16, device_type='cuda'):

                    # Aggregate tokens
                    aggregated_tokens_list, ps_idx = predictor.aggregator(video)
                    
                    # Predict Cameras
                    pose_enc = predictor.camera_head(aggregated_tokens_list)[-1]
                    # Predict Depth Maps
                    depth_map, depth_conf = predictor.depth_head(aggregated_tokens_list, video, ps_idx)
                
                pose_enc = pose_enc.float()
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, video.shape[-2:])
                # Remove batch dimension
                depth_map = depth_map.squeeze(0).float()  # (T, H, W, 1)
                depth_conf = depth_conf.squeeze(0).float()  # (T, H, W)
                extrinsic = extrinsic.squeeze(0).float()  # (T, 4, 4)
                intrinsic = intrinsic.squeeze(0).float()  # (T, 3, 3)
                
                # Unproject depth to point maps
                point_map_global, local_points = unproject_depth_map_to_point_map(
                    depth_map, 
                    extrinsic, 
                    intrinsic
                )
                
                # Convert to torch tensors if they are numpy arrays
                if isinstance(local_points, np.ndarray):
                    local_points = torch.from_numpy(local_points).to(device)
                if isinstance(point_map_global, np.ndarray):
                    point_map_global = torch.from_numpy(point_map_global).to(device)
                
                # Create mask from confidence
                # breakpoint()
                mask = depth_conf >= 1.0
                
                # Interpolate back to original resolution (similar to Pi3)
                if original_height != new_height or original_width != new_width:
                    interp_height = original_height // 3
                    interp_width = original_width // 3
                    logging.info(f"Interpolating outputs back to original resolution ({interp_height}, {interp_width})")
                    # local_points: (T, H, W, 3) -> (T, 3, H, W) -> interpolate -> (T, H', W', 3)
                    local_points = F.interpolate(
                        local_points.permute(0, 3, 1, 2),  # (T, 3, H, W)
                        (interp_height, interp_width), 
                        mode='bilinear', 
                        antialias=True
                    )
                    local_points = local_points.permute(0, 2, 3, 1)  # (T, H', W', 3)
                    
                    # mask: (T, H, W) -> (T, 1, H, W) -> interpolate -> (T, H', W')
                    mask = F.interpolate(
                        mask.unsqueeze(1).float(),  # (T, 1, H, W)
                        (interp_height, interp_width),
                        mode='bilinear',
                        antialias=True
                    )
                    mask = mask.squeeze(1) > 0.5  # (T, H', W'), threshold after interpolation

            # Convert to correct format for saving
            local_points = local_points.cpu()
            point_map_global = point_map_global.cpu()
            mask = mask.cpu()
            camera_poses = extrinsic.cpu()

            # Visualize disparity
            disp_colored = pmap_to_disp(local_points, mask, gamma=0.5)
            disp_colored = (disp_colored.cpu().numpy() * 255.0).astype(np.uint8)
            media.write_video(os.path.join(output_dir, f"{video_name}_disp_colored.mp4"), disp_colored, fps=10)

            depth_colored = pmap_to_depth(local_points, mask)
            depth_colored = (depth_colored.cpu().numpy()*255.0).astype(np.uint8)

            media.write_video(os.path.join(output_dir, f"{video_name}_depth_colored.mp4"), depth_colored, fps=10)

            stride = 1
            local_points = local_points[:, ::stride, ::stride, :].cpu().numpy().astype(np.float16) # T H W 3
            point_map_global = point_map_global[:, ::stride, ::stride, :].cpu().numpy().astype(np.float16) # T H W 3
            global_mask = mask[:, ::stride, ::stride].cpu().numpy().astype(bool)
            video = video[..., ::stride, ::stride, :].cpu().numpy()
            extrinsics = camera_poses.cpu().numpy()

            save_dict = {
                "pointmap": local_points,
                "pointmap_global": point_map_global,
                "pointmap_mask": global_mask,
                "rgb": video,
                "extrinsics": extrinsics,
            }

            np.save(os.path.join(output_dir, f"{video_name}.npy"), save_dict, allow_pickle=True)
                

