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
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party"))


import argparse
import logging
import os
from einops import rearrange

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm



sys.path.append(os.getcwd())
# from GeoWizard.geowizard.models.geowizard_pipeline import DepthNormalEstimationPipeline

from training.util.seed_all import seed_all


from dage.models.dage import DAGE
from dage.utils.data_utils import read_video


from evaluation.relpose.metadata import dataset_metadata
from evaluation.relpose.utils import get_tum_poses, save_tum_poses, process_directory, calculate_averages, eval_metrics, plot_trajectory, load_traj


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

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
        "--output_dir", type=str, required=True, help="Output directory."
    )

    parser.add_argument(
        "--eval_dataset", type=str, required=True, help="Evaluation dataset."
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

    # predictor = Pi3V3.from_pretrained(pretrained_model_name_or_path=checkpoint_path, strict=True).to(device).eval()
    predictor = DAGE.from_pretrained(pretrained_model_name_or_path=checkpoint_path, strict=True).to(device).eval()


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

    # breakpoint()

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

                video, original_height, original_width, _ = read_video(dir_path)


                video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
                video = video.to('cuda') / 255.0

                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    pose_output = predictor.infer(video, lr_max_size=252)

                pred_pose = pose_output["camera_poses"]


                pred_traj = get_tum_poses(pred_pose)
                # os.makedirs(f"{output_dir}/{seq}", exist_ok=True)
                # save_tum_poses(pred_pose, f"{output_dir}/{seq}/pred_traj.txt")

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
                continue

        results = process_directory(output_dir)
        avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results)

        print(f"Ate: {avg_ate:.4f}, RPE Trans: {avg_rpe_trans:.4f}, RPE Rot: {avg_rpe_rot:.4f}")
        with open(f"{output_dir}/summary.txt", "a") as f:
            f.write(
                f"Summary of {eval_dataset} on {len(seq_list)} sequences: \nAverage ATE: {avg_ate:.5f}, Average RPE trans: {avg_rpe_trans:.5f}, Average RPE rot: {avg_rpe_rot:.5f}\n"
            )
