import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "Unip"))

import time
import torch
import argparse
import numpy as np
import open3d as o3d
import os.path as osp
import cv2
from torch.utils.data import DataLoader
# from add_ckpt_path import add_path_to_dust3r
from accelerate import Accelerator
from torch.utils.data._utils.collate import default_collate
import tempfile
from tqdm import tqdm
import mediapy as media
import torch.nn.functional as F
from einops import rearrange


from third_party.moge.moge.predictor.moge_pose_predictor import MoGePosePredictor
from third_party import MoGe
from third_party.pi3.models.pi3 import Pi3
from third_party.pi3.models.pi3_teacher import Pi3Teacher
from third_party.moge.moge.model.dust3r.geometry import geotrf

from evaluation.moge.utils.geometry_torch import (
    weighted_mean, 
    mask_aware_nearest_resize,
    intrinsics_to_fov
)

from evaluation.moge.utils.alignment import (
    align_points_scale_z_shift, 
    align_points_scale_xyz_shift, 
    align_points_xyz_shift,
    align_affine_lstsq, 
    align_depth_scale, 
    align_depth_affine, 
    align_points_scale,
)


from evaluation.mv_recon.data_pi3 import SevenScenes, NRGBD
from evaluation.mv_recon.utils import accuracy, completion, umeyama

import h5py
import open3d as o3d



@torch.no_grad()
def get_scale_shift(pred_points, gt_points, mask, query_frame="all", align_resolution=32, trunc=1.0):

    if isinstance(query_frame, int):
        first_pred = pred_points[query_frame]
        first_gt = gt_points[query_frame]
        first_mask = mask[query_frame]

        # Align using the first frame (same procedure as image variant)
        (pred_lr, gt_lr), lr_mask = mask_aware_nearest_resize((first_pred, first_gt), mask=first_mask, size=(align_resolution, align_resolution))
        # (pred_lr, gt_lr), lr_mask = mask_aware_nearest_resize((first_pred[None], first_gt[None]), mask=first_mask[None], size=(align_resolution, align_resolution))

        scale, shift = align_points_scale_z_shift(
            pred_lr.flatten(-3, -2),
            gt_lr.flatten(-3, -2),
            lr_mask.flatten(-2, -1) / gt_lr[..., 2].flatten(-2, -1).clamp_min(1e-2),
            trunc=trunc,
        )
    elif query_frame == "all":
        (pred_lr, gt_lr), lr_mask = mask_aware_nearest_resize((pred_points, gt_points), mask=mask, size=(align_resolution, align_resolution))
        scale, shift = align_points_scale_z_shift(
            pred_lr.flatten(-4, -2),
            gt_lr.flatten(-4, -2),
            lr_mask.flatten(-3, -1) / gt_lr[..., 2].flatten(-3, -1).clamp_min(1e-2),
            trunc=trunc,
        )
    else:
        raise ValueError(f"Invalid query_frame: {query_frame}")

    valid = scale > 0
    # Zero-out invalid results so they do not contribute later.
    scale = torch.where(valid, scale, torch.zeros_like(scale))
    shift = torch.where(valid[..., None], shift, torch.zeros_like(shift))

    return scale, shift

@torch.no_grad()
def get_scale_depth(pred_points, gt_points, mask, query_frame="all", align_resolution=32, trunc=1.0):

    (pred_lr, gt_lr), lr_mask = mask_aware_nearest_resize((pred_points, gt_points), mask=mask, size=(align_resolution, align_resolution))

    scale = align_depth_scale(pred_lr.flatten(), gt_lr.flatten(), 1 / gt_lr.flatten())

    return scale

def get_args_parser():
    parser = argparse.ArgumentParser("3D Reconstruction evaluation", add_help=False)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="GonzaloMG/marigold-e2e-ft-depth",
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
    return parser


def main(args):
    # add_path_to_dust3r(args.weights)
    # from eval.mv_recon.data import SevenScenes, NRGBD

    # if args.size == 512:
    #     resolution = (512, 384)
    # elif args.size == 224:
    #     resolution = 224
    # else:
    #     raise NotImplementedError

    datasets_all = {
        "7scenes": SevenScenes(
            split="test",
            ROOT="/mnt/localssd/mv_recon_eval/7scenes",
            resolution=(518, 518),
            num_seq=1,
            full_video=True,
            kf_every=200, # 200,
            load_img_size=518,
        ),  # 20),
        "NRGBD": NRGBD(
            split="test",
            ROOT="/mnt/localssd/mv_recon_eval/neural_rgbd",
            resolution=(518, 518),
            num_seq=1,
            full_video=True,
            kf_every=500,
            load_img_size=518,
            # kf_every=500,
        ),
    }


    checkpoint_path = args.checkpoint

    accelerator = Accelerator()
    device = torch.device("cuda")




    prior_model = Pi3Teacher.from_pretrained("checkpoints/pi3_converted.pt").to(device).eval()
    moge_model = MoGe(
        model_name='MoGeModelV2',
        pretrained_path="/home/colligo/dngo/cache/moge_v2_vitl/model.pt",
    ).eval().to('cuda')


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

            # with accelerator.split_between_processes(list(range(len(dataset)))) as idxs:
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
                    for name in view.keys():  # pseudo_focal
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

                video = []
                for view in batch:
                    img_tensor = view["img"]
                    video.append(img_tensor)
                video = torch.stack(video, dim=1)

                ori_video = video.clone()
                original_height, original_width = video.shape[-2:]

                original_aspect_ratio = original_width / original_height
                vggt_width = 518 # NOTE hardcode here 
                vggt_height = int((vggt_width / original_aspect_ratio) // 14 * 14)
                video_vggt = F.interpolate(
                    rearrange(video, 'b t c h w -> (b t) c h w'), (vggt_height, vggt_width), mode='bilinear', antialias=True
                ).clamp(0, 1)
                video_vggt = rearrange(video_vggt, '(b t) c h w -> b t c h w', b=video.shape[0])

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    prior_output = prior_model(video_vggt)

                
                vggt_global_pointmaps = prior_output['points']
                vggt_local_pointmaps = prior_output['local_points']

                moge_pred, moge_pred_mask  = moge_model.forward_video(video)
                # moge_pred, moge_pred_mask = moge_output_dict['points'], moge_output_dict['mask']

                moge_pred = moge_pred[0]
                moge_pred_mask = moge_pred_mask[0]
                moge_depth = moge_pred[..., 2]

                vggt_depths = vggt_local_pointmaps[0, ..., 2]

                moge_depth_resized = F.interpolate(moge_depth.unsqueeze(1), size=(vggt_height, vggt_width), mode='nearest').squeeze(1) # T H W 1
                moge_pred_mask_resized = F.interpolate(moge_pred_mask.float().unsqueeze(1), size=(vggt_height, vggt_width), mode='nearest').squeeze(1).bool()

                # scale, shift = get_scale_shift(moge_depth_resized, vggt_depths, moge_pred_mask_resized, query_frame="all", align_resolution=32, trunc=1.0)
                moge_pred_aligned = []
                for t in range(moge_depth_resized.shape[0]):
                    scale_t = get_scale_depth(moge_depth_resized[t], vggt_depths[t], moge_pred_mask_resized[t], query_frame="all", align_resolution=32, trunc=1.0)
                    moge_pred_aligned.append(moge_pred[t] * scale_t)
                moge_pred_aligned = torch.stack(moge_pred_aligned, dim=0)
                moge_pred = moge_pred_aligned

                vggt_cam2world = prior_output['camera_poses'][0]
                moge_pred = geotrf(vggt_cam2world, moge_pred)


                # output = predictor(video, device, prior_max_size=140)

                pred_pts = moge_pred.cpu().numpy()
                # pred_mask = moge_pred_mask.cpu().numpy()
                pred_pose = vggt_cam2world.cpu().numpy()

                gt_pts = torch.stack([view["pts3d"][0] for view in batch], dim=0).cpu().numpy()
                valid_mask = torch.stack([view["valid_mask"][0] for view in batch], dim=0).cpu().numpy()
                gt_camera_poses = torch.stack([view["camera_pose"][0] for view in batch], dim=0).cpu().numpy()

                images = torch.stack([view["img"][0] for view in batch], dim=0)
                colors = images.permute(0, 2, 3, 1).cpu().numpy()
                colors_masked = colors[valid_mask].reshape(-1, 3)
                
                # 5. coarse align
                c, R, t = umeyama(pred_pts[valid_mask].T, gt_pts[valid_mask].T)
                pred_pts = c * np.einsum('nhwj, ij -> nhwi', pred_pts, R) + t.T

                # 6. filter invalid points
                pred_pts_masked = pred_pts[valid_mask].reshape(-1, 3)
                gt_pts_masked = gt_pts[valid_mask].reshape(-1, 3)

                # 7. save predicted & ground truth point clouds
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pred_pts_masked)
                pcd.colors = o3d.utility.Vector3dVector(colors_masked)
                # o3d.io.write_point_cloud(osp.join(save_path, f"{scene_id.replace('/', '_')}-mask.ply"), pcd)

                pcd_gt = o3d.geometry.PointCloud()
                pcd_gt.points = o3d.utility.Vector3dVector(gt_pts_masked)
                pcd_gt.colors = o3d.utility.Vector3dVector(colors_masked)
                # o3d.io.write_point_cloud(osp.join(save_path, f"{scene_id.replace('/', '_')}-gt.ply"), pcd_gt)

                pred_save_dict = {
                    "pointmap_global": pred_pts,
                    "pointmap_mask": valid_mask,
                    "rgb": colors,
                    "extrinsics": pred_pose,
                }

                np.save(os.path.join(save_path, f"{scene_id.replace('/', '_')}.npy"), pred_save_dict, allow_pickle=True)

                # 8. ICP align refinement
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
                
                # 9. estimate normals
                pcd.estimate_normals()
                pcd_gt.estimate_normals()
                pred_normal = np.asarray(pcd.normals)
                gt_normal = np.asarray(pcd_gt.normals)

                # 10. compute metrics
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

                # release cuda memory
                torch.cuda.empty_cache()

            acc_all = acc_all / len(dataset)
            comp_all = comp_all / len(dataset)
            nc1_all = nc1_all / len(dataset)
            nc2_all = nc2_all / len(dataset)
            acc_all_med = acc_all_med / len(dataset)
            comp_all_med = comp_all_med / len(dataset)
            nc1_all_med = nc1_all_med / len(dataset)
            nc2_all_med = nc2_all_med / len(dataset)

            # print("Mean metrics:")
            print(f"Mean metrics: Acc: {acc_all:.3f}, Comp: {comp_all:.3f}, NC1: {nc1_all:.3f}, NC2: {nc2_all:.3f} - Acc_med: {acc_all_med:.3f}, Compc_med: {comp_all_med:.3f}, NC1c_med: {nc1_all_med:.3f}, NC2c_med: {nc2_all_med:.3f}")

            # accelerator.wait_for_everyone()
            # Get depth from pcd and run TSDFusion
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
