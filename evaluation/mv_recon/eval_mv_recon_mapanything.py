import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "third_party"))
sys.path.append(os.path.join(project_root, "Unip"))
import time
import torch
import torch.nn.functional as F
import sys
from einops import rearrange
import argparse
import numpy as np
import open3d as o3d
import os.path as osp
import cv2
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torch.utils.data._utils.collate import default_collate
import tempfile
from tqdm import tqdm
import mediapy as media

from third_party.mapanything.models import MapAnything
from third_party.mapanything.utils.image import load_images
import torchvision.transforms as tvf
from PIL import Image

from dage.utils.timer import CUDATimer


from evaluation.mv_recon.data_pi3 import SevenScenes, NRGBD, DTU, ETH3D
from evaluation.mv_recon.utils import accuracy, completion, umeyama, rigid_transform



def prepare_images_for_mapanything(batch, norm_type="dinov2"):
    """
    Convert batch images to MapAnything format.
    MapAnything's load_images returns list of dicts with:
    - img: normalized tensor (1, 3, H, W)
    - true_shape: original shape
    - idx, instance, data_norm_type
    
    The dataset images are already converted to tensors (range [0, 1]) by ImgNorm (tvf.ToTensor).
    We need to apply MapAnything's normalization (DINOv2 by default).
    """
    from third_party.mapanything.utils.image import IMAGE_NORMALIZATION_DICT
    
    # Get normalization parameters
    if norm_type not in IMAGE_NORMALIZATION_DICT.keys():
        raise ValueError(
            f"Unknown norm_type: {norm_type}. Available: {list(IMAGE_NORMALIZATION_DICT.keys())}"
        )
    
    img_norm = IMAGE_NORMALIZATION_DICT[norm_type]
    # Apply only Normalize since images are already tensors in [0, 1] range
    normalize = tvf.Normalize(mean=img_norm.mean, std=img_norm.std)
    
    views = []
    for idx, view_data in enumerate(batch):
        img_tensor = view_data["img"][0]  # (3, H, W), already in [0, 1] range
        
        # Apply MapAnything's normalization (DINOv2)
        img_normalized = normalize(img_tensor)  # (3, H, W)
        img_normalized = img_normalized.unsqueeze(0)  # (1, 3, H, W)
        
        H, W = img_tensor.shape[1:]
        
        views.append({
            "img": img_normalized,
            "true_shape": np.int32([H, W]),
            "idx": idx,
            "instance": str(idx),
            "data_norm_type": [norm_type],
        })
    
    return views


def get_args_parser():
    parser = argparse.ArgumentParser("3D Reconstruction evaluation", add_help=False)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="facebook/map-anything",
        help="Checkpoint path or hub name.",
    )

    parser.add_argument(
        "--conf_thresh", type=float, default=0.0, help="confidence threshold"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--revisit", type=int, default=1, help="revisit times")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--memory_efficient", action="store_true", help="Use memory efficient inference")
    parser.add_argument("--apply_confidence_mask", action="store_true", help="Apply confidence masking")
    parser.add_argument("--confidence_percentile", type=int, default=10, help="Confidence percentile for masking")
    return parser


def main(args):
    datasets_all = {
        "7scenes": SevenScenes(
            split="test",
            ROOT="/mnt/localssd/mv_recon_eval/7scenes",
            resolution=(518, 518),
            num_seq=1,
            full_video=True,
            kf_every=200,
            load_img_size=518,
        ),
        "NRGBD": NRGBD(
            split="test",
            ROOT="/mnt/localssd/mv_recon_eval/neural_rgbd",
            resolution=(518, 518),
            num_seq=1,
            full_video=True,
            kf_every=500,
            load_img_size=518,
        ),
        # "DTU": DTU(
        #     split="test",
        #     ROOT="/mnt/localssd/mv_recon_eval/dtu",
        #     resolution=(518, 518),
        #     num_seq=1,
        #     full_video=True,
        #     kf_every=5,
        #     load_img_size=518,
        # ),
        # "ETH3D": ETH3D(
        #     ROOT="/mnt/localssd/mv_recon_eval/eth3d/",
        #     load_img_size=518,
        #     kf_every=5,
        #     resolution=(518, 518),
        # ),
    }

    checkpoint_path = args.checkpoint

    accelerator = Accelerator()
    device = torch.device("cuda")

    # Load MapAnything model
    print(f"Loading MapAnything model from {checkpoint_path}...")
    predictor = MapAnything.from_pretrained(checkpoint_path).to(device)
    predictor.eval()
    print("Model loaded successfully!")

    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for name_data, dataset in datasets_all.items():
            save_path = osp.join(args.output_dir, name_data)
            os.makedirs(save_path, exist_ok=True)
            log_file = osp.join(save_path, f"logs_{accelerator.process_index}.txt")

            acc_all = 0
            acc_all_med = 0
            comp_all = 0
            comp_all_med = 0
            nc1_all = 0
            nc1_all_med = 0
            nc2_all = 0
            nc2_all_med = 0

            fps_all = []
            time_all = []

            for data_idx in tqdm(range(len(dataset))):
                batch = default_collate([dataset[data_idx]])
                ignore_keys = set(
                    [
                        "depthmap",
                        "dataset",
                        "label",
                        "instance",
                        "idx",
                        "true_shape",
                        "rng",
                    ]
                )
                for view in batch:
                    for name in view.keys():
                        if name in ignore_keys:
                            continue
                        if isinstance(view[name], tuple) or isinstance(
                            view[name], list
                        ):
                            view[name] = [
                                x.to(device, non_blocking=True) for x in view[name]
                            ]
                        else:
                            view[name] = view[name].to(device, non_blocking=True)

                scene_id = view["label"][0].rsplit("/", 1)[0]

                # Prepare images for MapAnything using proper preprocessing
                # MapAnything expects list of dicts with specific format (output of load_images)
                views = prepare_images_for_mapanything(batch, norm_type="dinov2")

                with CUDATimer("Model forward", enabled=True, num_frames=len(views)):
                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                        predictions = predictor.infer(
                            views,
                            memory_efficient_inference=args.memory_efficient,
                            use_amp=True,
                            amp_dtype="bf16",
                            apply_mask=False,
                            mask_edges=False,
                            apply_confidence_mask=args.apply_confidence_mask,
                            confidence_percentile=args.confidence_percentile,
                        )

                # Extract predicted 3D points from MapAnything output
                # pts3d is already in world coordinates (B, H, W, 3)
                pred_pts_list = []
                for pred in predictions:
                    pts3d = pred["pts3d"]  # (1, H, W, 3)
                    pred_pts_list.append(pts3d[0])  # (H, W, 3)
                
                pred_pts = torch.stack(pred_pts_list, dim=0)  # (N, H, W, 3)
                pred_pts = pred_pts.cpu().numpy()

                # Get ground truth points and valid mask
                gt_pts = torch.stack([view["pts3d"][0] for view in batch], dim=0).cpu().numpy()
                valid_mask = torch.stack([view["valid_mask"][0] for view in batch], dim=0).cpu().numpy()

                # Get colors for visualization
                images = torch.stack([view["img"][0] for view in batch], dim=0)
                colors = images.permute(0, 2, 3, 1)[valid_mask].cpu().numpy().reshape(-1, 3)

                # Apply rigid transformation alignment (no scale)
                R, t = rigid_transform(pred_pts[valid_mask].T, gt_pts[valid_mask].T)
                pred_pts = np.einsum('nhwj, ij -> nhwi', pred_pts, R) + t.T
                # c, R, t = umeyama(pred_pts[valid_mask].T, gt_pts[valid_mask].T)
                # pred_pts = c * np.einsum('nhwj, ij -> nhwi', pred_pts, R) + t.T


                # Filter invalid points
                pred_pts = pred_pts[valid_mask].reshape(-1, 3)
                gt_pts = gt_pts[valid_mask].reshape(-1, 3)

                # Create point clouds for evaluation
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pred_pts)
                pcd.colors = o3d.utility.Vector3dVector(colors)

                pcd_gt = o3d.geometry.PointCloud()
                pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)
                pcd_gt.colors = o3d.utility.Vector3dVector(colors)

                # ICP alignment refinement
                if "DTU" in name_data:
                    threshold = 100
                else:
                    threshold = 0.1

                trans_init = np.eye(4)
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    pcd,
                    pcd_gt,
                    threshold,
                    trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                )

                transformation = reg_p2p.transformation
                pcd = pcd.transform(transformation)

                # Estimate normals
                pcd.estimate_normals()
                pcd_gt.estimate_normals()
                pred_normal = np.asarray(pcd.normals)
                gt_normal = np.asarray(pcd_gt.normals)

                # Compute metrics
                acc, acc_med, nc1, nc1_med = accuracy(
                    pcd_gt.points, pcd.points, gt_normal, pred_normal
                )
                comp, comp_med, nc2, nc2_med = completion(
                    pcd_gt.points, pcd.points, gt_normal, pred_normal
                )

                print(
                    f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
                )
                print(
                    f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}",
                    file=open(log_file, "a"),
                )

                acc_all += acc
                comp_all += comp
                nc1_all += nc1
                nc2_all += nc2

                acc_all_med += acc_med
                comp_all_med += comp_med
                nc1_all_med += nc1_med
                nc2_all_med += nc2_med

                # Release cuda memory
                torch.cuda.empty_cache()

            acc_all = acc_all / len(dataset)
            comp_all = comp_all / len(dataset)
            nc1_all = nc1_all / len(dataset)
            nc2_all = nc2_all / len(dataset)
            acc_all_med = acc_all_med / len(dataset)
            comp_all_med = comp_all_med / len(dataset)
            nc1_all_med = nc1_all_med / len(dataset)
            nc2_all_med = nc2_all_med / len(dataset)

            print(f"Mean metrics: Acc: {acc_all:.3f}, Comp: {comp_all:.3f}, NC1: {nc1_all:.3f}, NC2: {nc2_all:.3f} - Acc_med: {acc_all_med:.3f}, Compc_med: {comp_all_med:.3f}, NC1c_med: {nc1_all_med:.3f}, NC2c_med: {nc2_all_med:.3f}")

            if accelerator.is_main_process:
                to_write = ""
                # Copy the error log from each process to the main error log
                for i in range(8):
                    if not os.path.exists(osp.join(save_path, f"logs_{i}.txt")):
                        break
                    with open(osp.join(save_path, f"logs_{i}.txt"), "r") as f_sub:
                        to_write += f_sub.read()

                with open(osp.join(save_path, f"logs_all.txt"), "w") as f:
                    log_data = to_write
                    metrics = defaultdict(list)
                    for line in log_data.strip().split("\n"):
                        match = regex.match(line)
                        if match:
                            data = match.groupdict()
                            # Exclude 'scene_id' from metrics as it's an identifier
                            for key, value in data.items():
                                if key != "scene_id":
                                    metrics[key].append(float(value))
                            metrics["nc"].append(
                                (float(data["nc1"]) + float(data["nc2"])) / 2
                            )
                            metrics["nc_med"].append(
                                (float(data["nc1_med"]) + float(data["nc2_med"])) / 2
                            )
                    mean_metrics = {
                        metric: sum(values) / len(values)
                        for metric, values in metrics.items()
                    }

                    c_name = "mean"
                    print_str = f"{c_name.ljust(20)}: "
                    for m_name in mean_metrics:
                        print_num = np.mean(mean_metrics[m_name])
                        print_str = print_str + f"{m_name}: {print_num:.3f} | "
                    print_str = print_str + "\n"
                    f.write(to_write + print_str)


from collections import defaultdict
import re

pattern = r"""
    Idx:\s*(?P<scene_id>[^,]+),\s*
    Acc:\s*(?P<acc>[^,]+),\s*
    Comp:\s*(?P<comp>[^,]+),\s*
    NC1:\s*(?P<nc1>[^,]+),\s*
    NC2:\s*(?P<nc2>[^,]+)\s*-\s*
    Acc_med:\s*(?P<acc_med>[^,]+),\s*
    Compc_med:\s*(?P<comp_med>[^,]+),\s*
    NC1c_med:\s*(?P<nc1_med>[^,]+),\s*
    NC2c_med:\s*(?P<nc2_med>[^,]+)
"""

regex = re.compile(pattern, re.VERBOSE)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)

