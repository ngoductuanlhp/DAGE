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
from einops import einsum, rearrange
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
import mediapy as media
import cv2

from third_party.pi3.models.pi3 import Pi3
from third_party.pi3.models.pi3_teacher import Pi3Teacher
from dage.utils.timer import CUDATimer


from dage.utils.data_utils import read_video_from_path, read_image_from_path, read_long_video_from_path, read_video_from_folder, read_video

from dage.utils.vis_disp_utils import pmap_to_disp, color_video_disp, pmap_to_depth


if "__main__" == __name__:

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run pointmap VAE evaluation on multiple benchmarks."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="yyfz233/Pi3",
        help="Checkpoint path or hub name.",
    )

    parser.add_argument(
        "--output_dir", type=str, default="quali_results/pi3", help="Output directory."
    )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)  


    # -------------------- Device --------------------
    device = torch.device("cuda")

    predictor = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
    

    device = "cuda"

    folder_path = "./assets/demo_data"
    video_paths = sorted([os.path.join(folder_path, l) for l in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, l)) or l.endswith(".mp4") or l.endswith(".MOV")])


    print(f"Found {len(video_paths)} videos at {folder_path}")
    with torch.no_grad():
        for video_path in tqdm(video_paths):
            video_name = os.path.basename(video_path)
            video, original_height, original_width, fps = read_video(video_path, stride=1, max_frames=100) # 4k_video


            # Compute new dimensions for resizing
            max_size = 518
            original_aspect_ratio = original_width / original_height
            if original_width > original_height:
                new_width = min(max_size, original_width // 14 * 14)
                new_height = int((new_width / original_aspect_ratio) // 14 * 14)
            else:
                new_height = min(max_size, original_height // 14 * 14)
                new_width = int((new_height * original_aspect_ratio) // 14 * 14)
            
            # Resize video using cv2
            resized_video = []
            for frame in video:
                resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                resized_video.append(resized_frame)
            video = np.stack(resized_video, axis=0)
            original_height, original_width = new_height, new_width


            num_frames = len(video)
            video = torch.from_numpy(video).permute(0, 3, 1, 2).float()
            video = video.to('cuda') / 255.0
            video = video.unsqueeze(0)

            num_tokens = (original_height // 14) * (original_width // 14)
            print(f"Process video of shape {video.shape} with num_tokens {num_tokens}")

            with CUDATimer(name="Pi3", measure_memory=True):
                output = predictor.infer(video, max_size=max_size, conf_threshold=0.00)


            local_points = output['local_points']
            mask = output['mask']
            
            mask = torch.ones_like(mask).bool()
            disp_colored = pmap_to_disp(local_points, mask, gamma=0.5, method='gamma')
            disp_colored = (disp_colored.cpu().numpy()*255.0).astype(np.uint8)
            media.write_video(os.path.join(output_dir, f"{video_name}_disp_colored.mp4"), disp_colored, fps=10)


            stride = 1

            local_points = local_points[:, ::stride, ::stride, :].cpu().numpy().astype(np.float16) # T H W 3
            global_points = output['points'][:, ::stride, ::stride, :].cpu().numpy().astype(np.float16) # T H W 3
            global_mask = output['mask'][:, ::stride, ::stride].cpu().numpy().astype(bool)
            video = video[0].permute(0, 2, 3, 1)[..., ::stride, ::stride, :].cpu().numpy()
            # prior_video = output['prior_video'].permute(0, 2, 3, 1)[..., ::stride, ::stride, :].cpu().numpy()

            save_dict = {
                "pointmap": local_points,
                "pointmap_global": global_points,
                "pointmap_mask": global_mask,
                "rgb": video,
                "extrinsics": output['camera_poses'].cpu().numpy(),
            }

            np.save(os.path.join(output_dir, f"{video_name}.npy"), save_dict, allow_pickle=True)
