"""
Load a pointmap .npy (DA3 / DAGE format), merge all frames into one point cloud, and
view it in viser — no timestep slider or playback.

Optional voxel deduplication: quantize merged points to a 3D grid and keep the
highest-confidence sample per voxel (reduces duplicate surfaces seen in many views).
Default voxel size = (bbox diagonal) / 200.

Example:
  python visualization/vis_pointmaps_all.py --data_path tmp/da3_re10k/scene.npy
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
import viser
import viser.transforms as tf


project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)


# from src.utils.cuda_timer import CUDATimer



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concatenate all frames' points into one cloud and visualize (no timestep UI)."
    )
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument(
        "--indices",
        type=int,
        default=None,
        nargs="+",
        help="Only use these frame indices (default: all).",
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=None,
        help="Uniformly sample this many frames (default: all).",
    )
    parser.add_argument("--downsample_ratio", type=int, default=1, help="Spatial H/W downsample.")
    parser.add_argument("--point_size", type=float, default=0.01)
    parser.add_argument("--scale_factor", type=float, default=1.0)
    parser.add_argument(
        "--max_points",
        type=int,
        default=1000000,
        help="If more points than this, randomly subsample (0 = no limit).",
    )
    parser.add_argument(
        "--show_cameras",
        action="store_true",
        help="Draw camera frustums for each frame (off by default).",
    )
    parser.add_argument("--port", type=int, default=7891)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for voxel dedup / FPS (cuda or cpu).",
    )
    args = parser.parse_args()

    print(f"Loading {args.data_path}")
    data = np.load(args.data_path, allow_pickle=True).item()

    if "pointmap" in data:
        point_map = data["pointmap"].astype(np.float32)
        mask = data["pointmap_mask"].astype(bool)
    if "pointmap_global" in data:
        point_map = data["pointmap_global"].astype(np.float32)
        mask = data["pointmap_mask"].astype(bool)
        print("Using global point map")
    if "pointmap" not in data and "pointmap_global" not in data:
        raise KeyError("Expected 'pointmap' or 'pointmap_global' in npy.")

    extrinsics = data["extrinsics"].astype(np.float32) if "extrinsics" in data else None

    if args.indices is not None:
        indices = np.array(args.indices, dtype=np.int32)
    elif args.sample_num is not None:
        indices = np.linspace(0, len(point_map) - 1, args.sample_num)
        indices = np.round(indices).astype(np.int32)
    else:
        indices = np.array(list(range(len(point_map))))

    point_map = torch.tensor(point_map[indices]).float()
    mask = torch.tensor(mask[indices]).bool()
    if extrinsics is not None:
        extrinsics = extrinsics[indices]

    frames = torch.tensor(data["rgb"][indices]).float()
    if frames.max() <= 1.0:
        frames = frames * 255.0

    if extrinsics is not None:
        if extrinsics.shape[-2] == 3:
            extrinsics = np.concatenate(
                [
                    extrinsics,
                    np.array([0, 0, 0, 1], dtype=np.float32)[None, None, :].repeat(
                        extrinsics.shape[0], axis=0
                    ),
                ],
                axis=-2,
            )
        extrinsics = torch.tensor(extrinsics).float()
        in_camera0 = torch.linalg.inv(extrinsics[0])
        point_map = torch.einsum("ij, nhwj -> nhwi", in_camera0[:3, :3], point_map) + in_camera0[:3, 3]
        extrinsics = torch.einsum("ij, njk -> nik", in_camera0, extrinsics)

    H, W = point_map.shape[1:3]
    if frames.shape[1:3] != (H, W):
        frames = (
            F.interpolate(frames.permute(0, 3, 1, 2), (H, W), mode="bicubic")
            .clamp(0, 255)
            .permute(0, 2, 3, 1)
        )

    if args.downsample_ratio > 1:
        H, W = H // args.downsample_ratio, W // args.downsample_ratio
        point_map = (
            F.interpolate(point_map.permute(0, 3, 1, 2), (H, W)).permute(0, 2, 3, 1)
        )
        frames = F.interpolate(frames.permute(0, 3, 1, 2), (H, W)).permute(0, 2, 3, 1)
        mask = F.interpolate(mask.float()[:, None], (H, W))[:, 0] > 0.5

    conf_t: torch.Tensor | None = None
    if "conf" in data:
        conf_t = torch.tensor(data["conf"][indices].astype(np.float32)).float()
        if conf_t.shape[1:3] != (H, W):
            conf_t = F.interpolate(conf_t[:, None], (H, W), mode="nearest")[:, 0]

    num_frames = len(frames)
    chunks_p = []
    chunks_c = []
    chunks_conf: list[torch.Tensor] = []
    chunks_view: list[np.ndarray] = []
    chunks_xy: list[np.ndarray] = []
    for i in range(num_frames):
        valid = mask[i]
        n_valid = int(valid.sum().item())
        chunks_p.append(point_map[i][valid].detach().cpu().numpy())
        chunks_c.append(frames[i][valid].detach().cpu().numpy().astype(np.uint8))
        chunks_view.append(np.full(n_valid, indices[i], dtype=np.int32))
        # torch.nonzero gives (row, col) = (y, x) for each valid pixel
        yx = torch.nonzero(valid).cpu().numpy().astype(np.int32)  # (n_valid, 2)
        chunks_xy.append(yx * args.downsample_ratio)  # scale back to original image coords
        if conf_t is not None:
            chunks_conf.append(conf_t[i][valid].detach().cpu())

    positions = np.concatenate(chunks_p, axis=0)
    colors = np.concatenate(chunks_c, axis=0)

    print(f"Merged {num_frames} frames -> {positions.shape[0]} points (H={H}, W={W})")

    if args.max_points > 0:
        print(f"Subsampling from {positions.shape[0]} points to {args.max_points} points for visualization")
        permutation = np.random.permutation(positions.shape[0])[:args.max_points]
        positions = positions[permutation]
        colors = colors[permutation]

    positions = positions * args.scale_factor

    server = viser.ViserServer(port=args.port)
    server.request_share_url()

    server.scene.add_frame(
        "/world",
        wxyz=tf.SO3.exp(np.array([0.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )

    server.scene.add_point_cloud(
        name="/world/merged",
        points=positions,
        colors=colors,
        point_size=args.point_size,
        point_shape="rounded",
    )

    if args.show_cameras and extrinsics is not None:
        for i in range(num_frames):
            norm_i = i / (num_frames - 1) if num_frames > 1 else 0.0
            color_rgb = cm.viridis(norm_i)[:3]
            fi = frames[i]
            pseudo_focal = float(fi.shape[1])
            fov = 2 * np.arctan2(fi.shape[0] / 2, pseudo_focal)
            aspect = fi.shape[1] / fi.shape[0]
            camera_to_world = extrinsics[i]
            frustum_scale = 0.02
            axes_scale = 0.05
            img = frames[i].numpy().astype(np.uint8)
            server.scene.add_camera_frustum(
                f"/world/frustum_{i:04d}",
                fov=fov,
                aspect=aspect,
                scale=frustum_scale,
                image=img,
                wxyz=tf.SO3.from_matrix(camera_to_world[:3, :3].numpy()).wxyz,
                position=camera_to_world[:3, 3].numpy(),
                color=color_rgb,
            )
            server.scene.add_frame(
                f"/world/frustum_{i:04d}/axes",
                axes_length=frustum_scale * axes_scale * 10,
                axes_radius=frustum_scale * axes_scale,
            )

    print("Viser running (merged cloud only). Ctrl+C to exit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
