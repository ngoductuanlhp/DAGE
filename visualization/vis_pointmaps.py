# usage: python visualize/vis_pointmaps_button.py --data_path tmp/da3_re10k/39b58270c2e99310.npy
# Same as vis_pointmaps.py but uses a Play button instead of checkbox to start/stop playback.
import os
import sys
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time
import matplotlib.cm as cm

from kornia.filters import canny
from kornia.morphology import dilation
import viser
import viser.extras
import viser.transforms as tf


def compute_edge(depth: torch.Tensor):
    magnitude, edges = canny(depth[None, None, :, :], low_threshold=0.4, high_threshold=0.5)
    magnitude = magnitude[0, 0]
    edges = edges[0, 0]
    return edges > 0


def dilation_mask(mask: torch.Tensor, kernel_size: int = 3):
    mask = mask.float()
    mask = dilation(mask[None, None, :, :], torch.ones((kernel_size,kernel_size), device=mask.device))
    return mask[0, 0] > 0.5


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument("--indices", type=int, default=None, nargs='+', help='only load these frames for visualization')
    parser.add_argument("--sample_num", type=int, default=None, help='only sample several frames for visualization')
    parser.add_argument("--downsample_ratio", type=int, default=2, help='downsample ratio')
    parser.add_argument("--point_size", type=float, default=0.005, help='point size')
    parser.add_argument("--camera_frustum_scale", type=float, default=0.06, help='camera frustum scale')
    parser.add_argument("--scale_factor", type=float, default=1.0, help='point cloud scale factor for visualization')
    parser.add_argument("--edge_dilation_radius", type=int, default=3, help='remove floater points for visualization')
    parser.add_argument("--keyframe_interval", type=int, default=5, help='accumulate pointmaps at frames 0, N, 2N, ...; other frames show single frame')
    parser.add_argument("--port", type=int, default=7891, help='port')
    args = parser.parse_args()

    print(f"Loading from {args.data_path}")
    data = np.load(args.data_path, allow_pickle=True).item()

    depth_map = None
    if 'pointmap' in data:
        point_map = data['pointmap'].astype(np.float32)
        mask = data['pointmap_mask'].astype(bool)
        depth_map = point_map[..., 2:3]

    if 'pointmap_global' in data:
        point_map = data['pointmap_global'].astype(np.float32)
        mask = data['pointmap_mask'].astype(bool)
        print(f"Using global point map")

    if "extrinsics" in data:
        extrinsics = data['extrinsics'].astype(np.float32)
    else:
        extrinsics = None

    if args.indices:
        indices = np.array(args.indices, dtype=np.int32)
    elif args.sample_num:
        indices = np.linspace(0, len(point_map)-1, args.sample_num)
        indices = np.round(indices).astype(np.int32)
    else:
        indices = np.array(list(range(len(point_map))))

    point_map = torch.tensor(point_map[indices]).float()
    mask = torch.tensor(mask[indices]).bool()

    if depth_map is not None:
        depth_map = torch.tensor(depth_map[indices]).float()

    if extrinsics is not None:
        extrinsics = extrinsics[indices]

    frames = torch.tensor(data['rgb'][indices]).float()

    if frames.max() <= 1.0:
        frames = frames * 255.0

    if extrinsics is not None:
        if extrinsics.shape[-2] == 3:
            extrinsics = np.concatenate([extrinsics, np.array([0,0,0,1], dtype=np.float32)[None, None, :].repeat(extrinsics.shape[0], axis=0)], axis=-2)
        extrinsics = torch.tensor(extrinsics).float()
        in_camera0 = torch.linalg.inv(extrinsics[0])
        point_map = torch.einsum('ij, nhwj -> nhwi', in_camera0[:3, :3], point_map) + in_camera0[:3, 3]
        extrinsics = torch.einsum('ij, njk -> nik', in_camera0, extrinsics)

    H, W = point_map.shape[1:3]

    if frames.shape[1:3] != (H, W):
        frames = F.interpolate(frames.permute(0,3,1,2), (H, W), mode='bicubic').clamp(0, 255).permute(0,2,3,1)
    


    if args.downsample_ratio > 1:
        print(f"Downsampling to {H // args.downsample_ratio}x{W // args.downsample_ratio} for visualization")
        H, W = H // args.downsample_ratio, W // args.downsample_ratio
        point_map = F.interpolate(point_map.permute(0,3,1,2), (H, W)).permute(0,2,3,1)
        frames = F.interpolate(frames.permute(0,3,1,2), (H, W)).permute(0,2,3,1)
        mask = F.interpolate(mask.float()[:, None], (H, W))[:, 0] > 0.5
        if depth_map is not None:
            depth_map = F.interpolate(depth_map.permute(0,3,1,2), (H, W)).permute(0,2,3,1)

    max_hw = max(H, W)
    if max_hw > 600:
        print(f"Trying to visualize {frames.shape[0]}x{H}x{W} point map, which is huge. Consider setting downsample_ratio > 1 for visualization.")



    print(f"Visualizing {len(frames)} frames, H: {H}, W: {W}, keyframe interval: {args.keyframe_interval}")
    server = viser.ViserServer(port=args.port)
    server.request_share_url()
    num_frames = len(frames)

    frame_nodes: list[viser.FrameHandle] = []

    # Play state: use list to allow mutation from button callback
    playing = [False]
    keyframe_interval = args.keyframe_interval

    def get_visible_frames(current: int) -> set[int]:
        """Keyframes (0, N, 2N, ...) always stay visible (accumulated). Non-keyframes are temporary."""
        # Always show accumulated keyframes up to and including the latest keyframe <= current
        last_keyframe = (current // keyframe_interval) * keyframe_interval
        keyframes_visible = set(range(0, last_keyframe + 1, keyframe_interval))
        # Add current frame (keyframes stay; non-keyframes are temporary overlay)
        keyframes_visible.add(current)
        return keyframes_visible

    def get_all_keyframes() -> set[int]:
        """All keyframes in the sequence (for final accumulated display)."""
        last_keyframe = ((num_frames - 1) // keyframe_interval) * keyframe_interval
        return set(range(0, last_keyframe + 1, keyframe_interval))

    def update_frame_visibility(visible_set: set[int]) -> None:
        for i, node in enumerate(frame_nodes):
            node.visible = i in visible_set
        server.flush()

    def update_play_controls():
        """Update disabled state of frame controls based on playing and show_all."""
        disable = playing[0] or gui_show_all.value
        gui_timestep.disabled = disable

    def reset_to_original():
        """Reset to initial state: frame 0, stop playing, show frame 0."""
        playing[0] = False
        gui_timestep.value = 0
        update_play_controls()
        update_frame_visibility(get_visible_frames(0))

    # Add playback UI - Play button instead of checkbox
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=False,
        )
        gui_play_button = server.gui.add_button("Play", disabled=False)
        gui_reset_button = server.gui.add_button("Reset", disabled=False)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=1, initial_value=10
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("10", "20", "30", "60")
        )
        gui_show_all = server.gui.add_checkbox("Show All", False)

    @gui_reset_button.on_click
    def _(_) -> None:
        reset_to_original()

    @gui_play_button.on_click
    def _(_) -> None:
        playing[0] = not playing[0]
        update_play_controls()

    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    @gui_show_all.on_update
    def _(_) -> None:
        if gui_show_all.value:
            for node in frame_nodes:
                node.visible = True
            gui_timestep.disabled = True
        else:
            gui_timestep.disabled = playing[0]
            update_frame_visibility(get_visible_frames(gui_timestep.value))

    @gui_timestep.on_update
    def _(_) -> None:
        if gui_show_all.value:
            return
        update_frame_visibility(get_visible_frames(gui_timestep.value))

    server.scene.add_frame(
        "/frames",
        wxyz=tf.SO3.exp(np.array([np.pi / 2.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )

    for i in tqdm(range(num_frames)):
        valid_mask = mask[i]
        position = point_map[i][valid_mask].reshape(-1, 3).cpu().numpy()
        color = frames[i][valid_mask].reshape(-1, 3).cpu().numpy().astype(np.uint8)

        # print(f"frame {i} has {position.shape[0]} points")

        frame_nodes.append(server.scene.add_frame(
            f"/frames/t{i}",
            show_axes=False,
            wxyz=tf.SO3.exp(np.array([0.0, 0.0, np.pi])).wxyz
        ))

        point_size = args.point_size
        server.scene.add_point_cloud(
            name=f"/frames/t{i}/point_cloud",
            points=position * args.scale_factor,
            colors=color,
            point_size=point_size,
            point_shape="rounded",
        )

        if extrinsics is not None and i < len(extrinsics):
            norm_i = i / (num_frames - 1) if num_frames > 1 else 0
            color_rgba = cm.viridis(norm_i)
            color_rgb = color_rgba[:3]
            pseudo_focal = frames[i].shape[1]
            fov = 2 * np.arctan2(frames[i].shape[0] / 2, pseudo_focal)
            aspect = frames[i].shape[1] / frames[i].shape[0]
            camera_to_world = extrinsics[i]
            camera_frustum_scale = args.camera_frustum_scale
            downsample_factor = 1
            axes_scale = 0.05

            server.scene.add_camera_frustum(
                f"/frames/t{i}/frustum",
                fov=fov,
                aspect=aspect,
                scale=camera_frustum_scale,
                image=frames[i][::downsample_factor, ::downsample_factor].numpy().astype(np.uint8),
                wxyz=tf.SO3.from_matrix(camera_to_world[:3, :3].numpy()).wxyz,
                position=camera_to_world[:3, 3].numpy(),
                color=color_rgb,
            )

            server.scene.add_frame(
                f"/frames/t{i}/frustum/axes",
                axes_length=camera_frustum_scale * axes_scale * 10,
                axes_radius=camera_frustum_scale * axes_scale,
            )

    update_frame_visibility(get_visible_frames(gui_timestep.value))
    while True:
        if playing[0]:
            if gui_timestep.value >= num_frames - 1:
                # Reached end: stop playing and show final accumulated keyframes
                playing[0] = False
                update_play_controls()
                update_frame_visibility(get_all_keyframes())
            else:
                gui_timestep.value = gui_timestep.value + 1

        time.sleep(1.0 / gui_framerate.value)
