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
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

import argparse
import logging
import os
import json
from einops import einsum, rearrange
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
import mediapy as media

from third_party.moge.moge.model.cut3r import Cut3r
from dage.utils.timer import CUDATimer

from dage.utils.data_utils import read_video
from dage.utils.vis_disp_utils import pmap_to_disp, pmap_to_depth, color_video_disp

if "__main__" == __name__:
    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run pointmap VAE evaluation on multiple benchmarks."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/cut3r_512_dpt_4_64.pth",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="quali_results/cut3r", help="Output directory."
    )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)  


    # -------------------- Device --------------------
    device = torch.device("cuda")


    # predictor = MoGePosePredictor(
    #     model_name="MoGeModelV2PoseV2", 
    #     prior_model_name="Pi3", 
    #     device=device, 
    #     model_pretrained_path=checkpoint_path
    # )

    predictor = Cut3r.from_pretrained(checkpoint_path, strict=False).to(device).eval()

    folder_path = "./assets/demo_data"
    video_paths = sorted([os.path.join(folder_path, l) for l in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, l)) or l.endswith(".mp4") or l.endswith(".MOV")])

    with torch.no_grad():
        for video_path in tqdm(video_paths):
            video_name = os.path.basename(video_path)
            video, original_height, original_width, fps = read_video(video_path)
                

            num_frames = len(video)
            video = torch.from_numpy(video).permute(0, 3, 1, 2).float()
            video = video.to('cuda') / 255.0
            video = video.unsqueeze(0)

            num_tokens = (original_height // 14) * (original_width // 14)
            print(f"Process video of shape {video.shape} with num_tokens {num_tokens}")

            prior_max_size = 512
            prior_patch_size = 16

            original_aspect_ratio = original_width / original_height
            if original_width > original_height:
                prior_width = min(prior_max_size, original_width // prior_patch_size * prior_patch_size) # NOTE hardcode here 
                prior_height = int((prior_width / original_aspect_ratio) // prior_patch_size * prior_patch_size)
            else:
                prior_height = min(prior_max_size, original_height // prior_patch_size * prior_patch_size) # NOTE hardcode here 
                prior_width = int((prior_height * original_aspect_ratio) // prior_patch_size * prior_patch_size)
            prior_patch_h, prior_patch_w = prior_height // prior_patch_size, prior_width // prior_patch_size
            video_prior = F.interpolate(
                rearrange(video, 'b t c h w -> (b t) c h w'), (prior_height, prior_width), mode='bilinear', antialias=True
            ).clamp(0, 1)
            video_prior = rearrange(video_prior, '(b t) c h w -> b t c h w', t=num_frames)
        

            output = predictor(video_prior)

            local_points = output['local_points'][0]
            global_points = output['points'][0]
            mask = output['local_points_conf'][0] > 1.0
            camera_poses = output['camera_poses'][0]

            depth = local_points[..., 2]
            depth_min, depth_max = depth[mask].min().item(), depth[mask].max().item()

            depth_colored = pmap_to_depth(local_points, torch.ones_like(mask))
            depth_colored = (depth_colored.cpu().numpy()*255.0).astype(np.uint8)

            media.write_video(os.path.join(output_dir, f"{video_name}_depth_colored.mp4"), depth_colored, fps=10)

            
            stride = 1
            local_points = local_points[:, ::stride, ::stride, :].cpu().numpy().astype(np.float16) # T H W 3
            global_points = global_points[:, ::stride, ::stride, :].cpu().numpy().astype(np.float16) # T H W 3
            global_mask = mask[:, ::stride, ::stride].cpu().numpy().astype(bool)
            video = video[..., ::stride, ::stride, :].cpu().numpy()
            extrinsics = camera_poses.cpu().numpy()

            save_dict = {
                "pointmap": local_points,
                "pointmap_global": global_points,
                "pointmap_mask": global_mask,
                "rgb": video,
                "extrinsics": extrinsics,
            }

            np.save(os.path.join(output_dir, f"{video_name}.npy"), save_dict, allow_pickle=True)
