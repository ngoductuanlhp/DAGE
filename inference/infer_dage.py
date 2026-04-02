
import os
import sys
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

import argparse
import os
from einops import einsum, rearrange
from tqdm import tqdm

import cv2

import numpy as np
import torch
import torch.nn.functional as F
import mediapy as media

from dage.models.dage import DAGE

from dage.utils.timer import CUDATimer
from dage.utils.data_utils import read_video, resize_to_max_side
from dage.utils.vis_disp_utils import pmap_to_disp, pmap_to_depth, color_video_disp

if "__main__" == __name__:

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run pointmap VAE evaluation on multiple benchmarks."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/model.pt",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="quali_results/dage", help="Output directory."
    )
    parser.add_argument(
        "--lr_max_size", type=int, default=252, help="Max low resolution size."
    )
    parser.add_argument(
        "--hr_max_size", type=int, default=None, help="Max high resolution size."
    )
    parser.add_argument(
        "--chunk_size", type=int, default=None, help="Chunk size for high resolution stream. Default: None, disable chunking."
    )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)  


    # -------------------- Device --------------------
    device = torch.device("cuda")
    predictor = DAGE.from_pretrained(pretrained_model_name_or_path=checkpoint_path, strict=True).to(device).eval()

    folder_path = "./assets/demo_data"
    video_paths = sorted([os.path.join(folder_path, l) for l in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, l)) or l.endswith(".mp4") or l.endswith(".MOV")])

    print(f"Found {len(video_paths)} videos at {folder_path}")
    with torch.no_grad():
        for video_path in tqdm(video_paths):
            video_name = os.path.basename(video_path)
            try:
                # NOTE read the whole video from path
                video, original_height, original_width, fps = read_video(video_path)

                # NOTE read first 100 framesand sample frames at stride of 10
                # video, original_height, original_width, fps = read_video(video_path, stride=10, max_frames=150)

                # NOTE read auto 100 frames across the video
                # video, original_height, original_width, fps = read_video(video_path, force_num_frames=100)
            except Exception as e:
                print(f"Error reading video {video_path}: {e}")
                continue

            num_frames = len(video)
            original_aspect_ratio = original_width / original_height

            lr_max_size = args.lr_max_size
            lr_video, lr_height, lr_width = resize_to_max_side(video, lr_max_size)

            hr_max_size = args.hr_max_size
            
            # NOTE 
            # By default, we run HR stream at 3600 tokens (each token is 14x14 patch), so for a squared image, the resolution is 840x840, for a 9:16 (height:width) image, the resolution is 630x1120
            # If you want to run HR stream at a specific resolution, you can set the --hr_max_size argument to the maximum size you want to use.
            if hr_max_size is None:
                total_tokens = 3600
                hr_height, hr_width = int((total_tokens / original_aspect_ratio) ** 0.5) * 14, int((total_tokens * original_aspect_ratio) ** 0.5) * 14
                hr_video = np.stack(
                    [
                        cv2.resize(frame, (hr_width, hr_height), interpolation=cv2.INTER_LINEAR)
                        for frame in video
                    ],
                    axis=0,
                )
            else:
                hr_video, hr_height, hr_width = resize_to_max_side(video, hr_max_size)

            lr_video = rearrange(torch.from_numpy(lr_video), 't h w c-> 1 t c h w').float().to(device) / 255.0
            hr_video = rearrange(torch.from_numpy(hr_video), 't h w c -> 1 t c h w').float().to(device) / 255.0


            hr_num_tokens = (hr_height // 14) * (hr_width // 14)
            print(f"Process video {video_name} of length {num_frames}: HR stream at {hr_height}x{hr_width} with num_tokens {hr_num_tokens}, LR stream at {lr_height}x{lr_width}")

            chunk_size = args.chunk_size
            output = predictor.infer(hr_video=hr_video, lr_video=lr_video, lr_max_size=lr_max_size, hr_num_tokens=hr_num_tokens, chunk_size=chunk_size, refine=False, enable_autocast=True)

            print(f"Done inference, saving results...")
            
            local_points = output['local_points']
            global_points = output['global_points']
            mask = output['mask']
            camera_poses = output['camera_poses']
            hr_video = hr_video[0].permute(0, 2, 3, 1)

            disp_colored = pmap_to_disp(local_points, torch.ones_like(mask), gamma=0.5, method='gamma')
            disp_colored = (disp_colored.cpu().numpy()*255.0).astype(np.uint8)
            media.write_video(os.path.join(output_dir, f"{video_name}_disp_colored.mp4"), disp_colored, fps=10)

            depth_colored = pmap_to_depth(local_points, torch.ones_like(mask))
            depth_colored = (depth_colored.cpu().numpy()*255.0).astype(np.uint8)

            media.write_video(os.path.join(output_dir, f"{video_name}_depth_colored.mp4"), depth_colored, fps=10)
            
            stride = 1
            local_points = local_points[:, ::stride, ::stride, :].cpu().numpy().astype(np.float16) # T H W 3
            global_points = global_points[:, ::stride, ::stride, :].cpu().numpy().astype(np.float16) # T H W 3
            global_mask = mask[:, ::stride, ::stride].cpu().numpy().astype(bool)
            hr_video = hr_video[..., ::stride, ::stride, :].cpu().numpy()
            extrinsics = camera_poses.cpu().numpy()

            save_dict = {
                "pointmap": local_points,
                "pointmap_global": global_points,
                "pointmap_mask": global_mask,
                "rgb": hr_video,
                "extrinsics": extrinsics,
            }

            np.save(os.path.join(output_dir, f"{video_name}.npy"), save_dict, allow_pickle=True)
                