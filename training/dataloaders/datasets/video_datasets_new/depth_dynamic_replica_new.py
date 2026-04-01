import os
import io
import glob
import pandas as pd
from torch.utils.data import Dataset
from training.dataloaders.data_io import download_file, upload_file
from training.dataloaders.config import DATASET_CONFIG

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import random
from collections import defaultdict
import h5py
import cv2
from training.dataloaders.datasets.video_datasets_new.synchronized_video_transform import SynchronizedTransformVideoCut3r
from training.dataloaders.datasets.video_datasets_new.synchronized_video_transform_moge import SynchronizedTransform_MoGe

from training.dataloaders.datasets.video_datasets_new.frame_sampling_utils import get_seq_from_start_id

# -----------------------------------------------------------------------------
# NOTE:
# This file is a FIRST PASS skeleton for supporting the Waymo depth video
# dataset.  It mirrors the structure of the existing TartanAir / Point Odyssey
# loaders in this repository but **does NOT yet contain the final parsing logic**.
# Wherever you see a `TODO` comment, please replace the placeholder with the
# proper implementation that matches your processed Waymo data.
# -----------------------------------------------------------------------------

class SynchronizedTransform_DynamicReplica:
    """Apply the same random transformation to all frames in a sequence."""

    def __init__(self, H: int, W: int):
        self.resize = transforms.Resize((H, W))
        self.resize_depth = transforms.Resize((H, W), interpolation=Image.NEAREST)
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.to_tensor = transforms.ToTensor()
        self.W = W
        self.H = H

    def __call__(self, rgb_images, depth_images, intrinsics_list):
        # Decide on a single horizontal flip for the entire sequence
        flip = random.random() > 0.5

        rgb_tensors = []
        depth_tensors = []
        intrinsics_tensors = []

        for rgb_image, depth_image, intrinsics in zip(
            rgb_images, depth_images, intrinsics_list
        ):
            # Horizontal flip (if selected once per sequence)
            if flip:
                rgb_image = self.horizontal_flip(rgb_image)
                depth_image = self.horizontal_flip(depth_image)

            # Resize keeping aspect ratio as handled by torchvision
            og_width, og_height = rgb_image.size
            scale_w = self.W / og_width
            scale_h = self.H / og_height
            rgb_image = self.resize(rgb_image)
            depth_image = self.resize_depth(depth_image)

            # Adjust intrinsics to resized resolution
            intrinsics[0, 0] *= scale_w  # fx
            intrinsics[1, 1] *= scale_h  # fy
            intrinsics[0, 2] *= scale_w  # cx
            intrinsics[1, 2] *= scale_h  # cy

            # To tensor
            rgb_tensors.append(self.to_tensor(rgb_image))
            depth_tensors.append(self.to_tensor(depth_image))
            intrinsics_tensors.append(torch.tensor(intrinsics))

        return rgb_tensors, depth_tensors, intrinsics_tensors


class VideoDepthDynamicReplicaNew(Dataset):
    """Skeleton video dataset loader for DynamicReplica depth data.

    NOTE: The per-frame loading logic is left as placeholders – you should
    replace those with real parsing of the `*.jpg`, `*.exr`, and `*.npz`
    files found inside each DynamicReplica segment directory.
    """

    depth_type = "synthetic"
    is_metric_scale = True
    # NOTE original resolution is 1280x720

    def __init__(
        self,
        T: int = 5,
        stride_range: tuple = (1, 3),
        transform: bool = True,
        resize: tuple = (384, 512),
        near_plane: float = 1e-5,
        far_plane: float = 80.0,
        resolutions=None,
        use_moge=False,
        moge_augmentation=None,
        use_cut3r_frame_sampling=False, video_prob=0.6, fix_interval_prob=0.6, block_shuffle=16
    ):
        # Lazily load the parquet only if DynamicReplica key exists in config
        parquet_data = download_file(
            
            DATASET_CONFIG["dynamic_replica"]["parquet_path"],
        )
        self.parquet = io.BytesIO(parquet_data)

        self.T = T
        self.stride_range = stride_range
        self.near_plane = near_plane
        self.far_plane = far_plane

        self.frame_size = resize
        self.resolutions = resolutions
        self.use_moge = use_moge
        self.moge_augmentation = moge_augmentation

        # New sophisticated sampling parameters
        self.use_cut3r_frame_sampling = use_cut3r_frame_sampling
        self.video_prob = video_prob
        self.fix_interval_prob = fix_interval_prob
        self.block_shuffle = block_shuffle

        # Organize parquet rows by video name
        self.video_frames = self._organize_by_video()
        self.frame_list = self._create_frame_list()


        if self.use_moge:
            self.transform = SynchronizedTransform_MoGe(moge_augmentation=moge_augmentation)
        elif self.resolutions is not None:
            self.transform = SynchronizedTransformVideoCut3r(seed=777)
        else:
            self.transform = SynchronizedTransform_DynamicReplica(H=resize[0], W=resize[1]) if transform else None

    # ------------------------------------------------------------------
    # Helper functions – identical structure to other dataset loaders
    # ------------------------------------------------------------------

    def _organize_by_video(self):
        """Organize frames by video name.
        Assumes parquet is pre-sorted by (video_name, frame_number).
        """
        df = pd.read_parquet(self.parquet)
        
        # Sort by video_name and frame_number to ensure temporal order
        df = df.sort_values(['video_name', 'frame_number'])
        
        # Group by video_name (much faster than iterrows)
        # sort=False preserves the original row order within each group
        video_frames = {}
        for video_name, video_df in df.groupby('video_name', sort=False):
            # Convert each row to dict efficiently
            # Row order is preserved, so frames are in temporal order
            frames = video_df[['frame_number', 'rgb_path', 'depth_path', 
                               'intrinsics', 'extrinsics']].to_dict('records')
            video_frames[video_name] = frames
        
        return video_frames

    def _create_frame_list(self):
        """Create a flat list of all frames with video context."""
        # Use list comprehension for faster construction
        frame_list = [
            {
                "video_name": video_name,
                "frame_idx_in_video": i,
            }
            for video_name, frames in self.video_frames.items()
            for i, _ in enumerate(frames)
        ]
        return frame_list

    def __len__(self):
        return len(self.frame_list)

    # ------------------------------------------------------------------
    # PLACEHOLDER single-frame loader – FILL THIS IN!
    # ------------------------------------------------------------------
    def _load_single_frame(self, frame_data):
        """Load RGB image, depth (metric), intrinsics and extrinsics for a single Dynamic-Replica frame.

        The parquet row already stores the paths for the RGB/Depth files **as well as** the per-frame
        intrinsics & extrinsics flattened as 1-D lists.  We therefore only need to download the RGB
        and depth assets – the camera parameters can be reconstructed from the row itself.
        """
        try:
            # ------------------------------------------------------------------
            # RGB image (stored as PNG)
            # ------------------------------------------------------------------
            rgb_bytes = download_file(
                
                frame_data["rgb_path"],
            )
            rgb_image = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
            del rgb_bytes  # Free memory immediately

            # ------------------------------------------------------------------
            # Depth – stored as a `.npy` (float32, metric depth in meters)
            # ------------------------------------------------------------------
            depth_bytes = download_file(
                
                frame_data["depth_path"],
            )
            depth_array = np.load(io.BytesIO(depth_bytes))  # type: ignore
            del depth_bytes  # Free memory immediately
            
            depth_array[~np.isfinite(depth_array)] = 0  # invalid
            depth_image = Image.fromarray(depth_array.astype(np.float32))
            del depth_array  # Free memory immediately

            # ------------------------------------------------------------------
            # Camera intrinsics / extrinsics – saved in parquet as flattened lists
            # ------------------------------------------------------------------
            assert frame_data.get("intrinsics") is not None, f"Intrinsics is None for {frame_data['rgb_path']}"
            assert frame_data.get("extrinsics") is not None, f"Extrinsics is None for {frame_data['rgb_path']}"

            intrinsics = np.array(frame_data["intrinsics"], dtype=np.float32).reshape(3, 3).copy()
            extrinsics = np.array(frame_data["extrinsics"], dtype=np.float32).reshape(4, 4).copy()

            # NOTE fixing issue of pose https://github.com/CUT3R/CUT3R/issues/67
            rot_inv = extrinsics[:3, :3].T
            trans_inv = -rot_inv @ extrinsics[:3, 3]

            trans_inv[..., :2] *= -1
            rot_inv[..., :, :2] *= -1

            new_extrinsics = np.eye(4, dtype=np.float32)
            new_extrinsics[:3, :3] = rot_inv.T
            new_extrinsics[:3, 3] = -rot_inv.T @ trans_inv
            # print(f"new_extrinsics: {new_extrinsics}, extrinsics: {extrinsics}")
            extrinsics = new_extrinsics



            # if frame_data.get("intrinsics") is not None:
            #     intrinsics = np.array(frame_data["intrinsics"], dtype=np.float32).reshape(3, 3)
            # else:
                # intrinsics = np.eye(3, dtype=np.float32)

            # if frame_data.get("extrinsics") is not None:
            #     extrinsics = np.array(frame_data["extrinsics"], dtype=np.float32).reshape(4, 4)
            # else:
                # extrinsics = np.eye(4, dtype=np.float32)

            return rgb_image, depth_image, intrinsics, extrinsics, True
        except Exception as e:
            # In case anything goes wrong we return dummy tensors so that the rest of the
            # data-pipeline keeps flowing.
            print(f"[dynamic_replica] Error loading frame: {str(e)}")
            rgb_dummy = Image.new("RGB", (self.frame_size[1], self.frame_size[0]))
            depth_dummy = Image.fromarray(np.zeros((self.frame_size[0], self.frame_size[1]), dtype=np.float32))
            return rgb_dummy, depth_dummy, np.eye(3, dtype=np.float32), np.eye(4, dtype=np.float32), False

    # ------------------------------------------------------------------
    # The rest of __getitem__ mirrors other dataset loaders (unchanged).
    # ------------------------------------------------------------------
    def _can_sample_sequence(self, video_name, start_idx, stride, num_frames=None):
        if num_frames is None:
            num_frames = self.T
        frames = self.video_frames[video_name]
        if start_idx + (num_frames - 1) * stride >= len(frames):
            return False
        start_frame_num = frames[start_idx]["frame_number"]
        for i in range(num_frames):
            frame_idx = start_idx + i * stride
            expected = start_frame_num + i * stride
            actual = frames[frame_idx]["frame_number"]
            if actual != expected:
                return False
        return True

    def __getitem__(self, idx):
        try:
            if isinstance(idx, tuple):
                len_tuple_idx = len(idx)
                if len_tuple_idx == 2:
                    idx, num_frames = idx
                    resolution_idx = None   
                elif len_tuple_idx == 3:
                    idx, num_frames, resolution_idx = idx

            else:
                len_tuple_idx = 1
                num_frames = self.T
                resolution_idx = None
            
            rng = np.random.default_rng(random.randint(0, 2 ** 32 - 1))

            # Starting frame information
            start_frame_info = self.frame_list[idx]
            video_name = start_frame_info["video_name"]
            start_idx = start_frame_info["frame_idx_in_video"]
            
            frames = self.video_frames[video_name]
            
            if self.use_cut3r_frame_sampling:
                # print("using cut3r frame sampling")
                ids_all = list(range(len(frames)))
                
                try:
                    is_success, frame_positions, is_video = get_seq_from_start_id(
                        num_frames, start_idx, ids_all, rng, min_interval=self.stride_range[0], max_interval=self.stride_range[1], 
                        video_prob=self.video_prob, fix_interval_prob=self.fix_interval_prob, block_shuffle=self.block_shuffle
                    )
                except Exception as e:
                    is_success = False

            else:
                is_video = True
                # Randomly sample stride from range
                initial_stride = random.randint(self.stride_range[0], self.stride_range[1])
                stride = initial_stride

                # Try to find a valid stride by gradually reducing it
                if not self._can_sample_sequence(video_name, start_idx, stride, num_frames=num_frames):
                    # Try reducing stride step by step
                    is_success = False
                    while stride > max(self.stride_range[0], 1):
                        stride -= 1
                        if self._can_sample_sequence(video_name, start_idx, stride, num_frames=num_frames):
                            is_success = True
                            frame_positions = [min(start_idx + i * stride, len(frames) - 1) for i in range(num_frames)]
                            break
                else:
                    is_success = True
                    frame_positions = [min(start_idx + i * stride, len(frames) - 1) for i in range(num_frames)]

            if not is_success:
                # Try to get the new sample
                next_idx = (idx + (num_frames - 1) * random.randint(self.stride_range[0], self.stride_range[1])) % self.__len__()
                if len_tuple_idx == 3:
                    return self.__getitem__((next_idx, num_frames, resolution_idx))
                elif len_tuple_idx == 2:
                    return self.__getitem__((next_idx, num_frames))
                elif len_tuple_idx == 1:
                    return self.__getitem__(next_idx)
                else:
                    raise ValueError(f"Invalid tuple length: {len_tuple_idx}")

            # ------------------------------------------------------------------
            # Collect data for each frame in the sequence
            # ------------------------------------------------------------------
            rgb_images = []
            depth_images = []
            intrinsics_list = []
            extrinsics_list = []
            valid_frames = []

            for frame_idx in frame_positions:
                frame_data = frames[frame_idx]  # frames[frame_idx] is already the frame_data dict
                rgb_img, depth_img, intrinsics, extrinsics, valid = self._load_single_frame(frame_data)

                if valid:
                    rgb_images.append(rgb_img)
                    depth_images.append(depth_img)
                    intrinsics_list.append(intrinsics.copy())
                    extrinsics_list.append(extrinsics)
                    valid_frames.append(True)
                else:
                    # Fallback to previous frame (or dummy if none)
                    if rgb_images:
                        rgb_images.append(rgb_images[-1])
                        depth_images.append(depth_images[-1])
                        intrinsics_list.append(intrinsics_list[-1].copy())
                        extrinsics_list.append(extrinsics_list[-1])
                    else:
                        rgb_images.append(Image.new("RGB", (self.frame_size[1], self.frame_size[0])))
                        depth_images.append(Image.fromarray(np.zeros((self.frame_size[0], self.frame_size[1]), dtype=np.float32)))
                        intrinsics_list.append(np.eye(3, dtype=np.float32))
                        extrinsics_list.append(np.eye(4, dtype=np.float32))
                    valid_frames.append(False)

            # ------------------------------------------------------------------
            # Apply the synchronized data-augmentation (resize/h-flip)
            # ------------------------------------------------------------------
            if self.transform is not None:
                if isinstance(self.transform, SynchronizedTransformVideoCut3r):
                    rgb_tensors, depth_tensors, intrinsics_tensors = self.transform(
                        rgb_images, depth_images, intrinsics_list, resolution=self.resolutions[resolution_idx] if (resolution_idx is not None and self.resolutions is not None) else (self.frame_size[1], self.frame_size[0])
                    )
                elif isinstance(self.transform, SynchronizedTransform_MoGe):
                    processed_data = self.transform(
                        rgb_images, depth_images, intrinsics_list, extrinsics_list, valid_frames, self.near_plane, self.far_plane, self.depth_type, resolution_idx, stride=None, rng=rng
                    )
                    processed_data['is_metric_scale'] = self.is_metric_scale
                    return processed_data # NOTE for moge style, we return the processed data directly
                else:
                    rgb_tensors, depth_tensors, intrinsics_tensors = self.transform(
                        rgb_images, depth_images, intrinsics_list, 
                    )
            else:
                rgb_tensors = [transforms.ToTensor()(img) for img in rgb_images]
                depth_tensors = [transforms.ToTensor()(img) for img in depth_images]
                intrinsics_tensors = [torch.tensor(k) for k in intrinsics_list]

            # ------------------------------------------------------------------
            # Post-processing & tensor stacking – identical to other loaders
            # ------------------------------------------------------------------
            processed_rgb = []
            processed_metric_depth = []
            processed_valid_mask = []
            processed_inf_mask = []
            processed_intrinsics = []
            processed_extrinsics = []

            for t in range(num_frames):
                rgb_tensor = rgb_tensors[t]
                depth_tensor = depth_tensors[t]
                intrinsics_tensor = intrinsics_tensors[t]
                extrinsics_tensor = torch.tensor(extrinsics_list[t], dtype=torch.float32)

                # print(f"depth_tensor.shape: {depth_tensor.shape}, {depth_tensor.min()}, {depth_tensor.max()}")

                # Valid depth mask
                valid_depth_mask = (depth_tensor > self.near_plane) & (depth_tensor < self.far_plane)
                inf_depth_mask = depth_tensor >= self.far_plane
                # Normalise RGB to [-1,1]
                rgb_tensor = rgb_tensor * 2.0 - 1.0

                # Depth clamping similar to other datasets
                if valid_depth_mask.any() and valid_frames[t]:
                    flat_depth = depth_tensor[valid_depth_mask].flatten().float()
                    min_depth = torch.quantile(flat_depth, 0.00)
                    max_depth = torch.quantile(flat_depth, 0.98)
                    valid_depth_mask = (depth_tensor > min_depth) & (depth_tensor < max_depth)
                    if min_depth == max_depth:
                        metric_tensor = torch.zeros_like(depth_tensor)
                        valid_depth_mask = torch.zeros_like(depth_tensor).bool()
                    else:
                        depth_tensor = torch.clamp(depth_tensor, min_depth, max_depth)
                        depth_tensor[~valid_depth_mask] = max_depth
                        metric_tensor = depth_tensor.clone()
                else:
                    metric_tensor = torch.zeros_like(depth_tensor)
                    valid_depth_mask = torch.zeros_like(depth_tensor).bool()

                processed_rgb.append(rgb_tensor)
                processed_metric_depth.append(metric_tensor)
                processed_valid_mask.append(valid_depth_mask)
                processed_inf_mask.append(inf_depth_mask)
                processed_intrinsics.append(intrinsics_tensor)
                processed_extrinsics.append(extrinsics_tensor[:3, :3])  # rotation only

            rgb_sequence = torch.stack(processed_rgb, dim=0)
            metric_depth_sequence = torch.stack(processed_metric_depth, dim=0)
            valid_mask_sequence = torch.stack(processed_valid_mask, dim=0)
            inf_mask_sequence = torch.stack(processed_inf_mask, dim=0)
            intrinsics_sequence = torch.stack(processed_intrinsics, dim=0)
            extrinsics_sequence = torch.stack(processed_extrinsics, dim=0)

            return {
                "rgb": rgb_sequence,
                "metric_depth": metric_depth_sequence,
                "valid_mask": valid_mask_sequence,
                "inf_mask": inf_mask_sequence,
                "intrinsics": intrinsics_sequence,
                "extrinsics": extrinsics_sequence,
                # "stride": stride,
                "valid": all(valid_frames),
                "depth_type": self.depth_type,
            }

        except Exception as e:
            print(f"[dynamic_replica] Error processing item: {str(e)}")
            num_frames = self.T
            dummy_rgb = torch.zeros(num_frames, 3, self.frame_size[0], self.frame_size[1])
            dummy_depth = torch.zeros(num_frames, 1, self.frame_size[0], self.frame_size[1])
            dummy_mask = torch.zeros(num_frames, 1, self.frame_size[0], self.frame_size[1], dtype=torch.bool)
            dummy_intrinsics = torch.eye(3).unsqueeze(0).repeat(num_frames, 1, 1)
            dummy_extrinsics = torch.eye(3).unsqueeze(0).repeat(num_frames, 1, 1)
            return {
                "rgb": dummy_rgb,
                "metric_depth": dummy_depth,
                "valid_mask": dummy_mask,
                "intrinsics": dummy_intrinsics,
                "extrinsics": dummy_extrinsics,
                # "stride": 1,
                "valid": False,
                "depth_type": self.depth_type,
            }


# -----------------------------------------------------------------------------
# Parquet generation – scans the *processed* Dynamic-Replica directory and
# builds a dataframe which is saved locally.
# -----------------------------------------------------------------------------

def generate_parquet_file(
    raw_root: str = "/mnt/localssd/dynamic_replica",
    processed_root: str = "/mnt/localssd/dynamic_replica_processed",
    splits: tuple = ("train", "valid"),
):
    """Pre-process the *raw* Dynamic-Replica dataset and create a parquet index.

    Args:
        raw_root:   Directory that contains the original Dynamic-Replica release.
        processed_root: Where the converted RGB / depth / camera files will be written.
        splits: List of splits to process (default: ("train", "valid", "test")).

    The routine closely mirrors the reference preprocessing script you provided but
    limits itself to the assets we actually need for depth-based tasks (RGB, metric
    depth and per-frame intrinsics / pose).  Optical-flow files are *ignored*.
    """

    import gzip
    import re
    import shutil
    from collections import defaultdict
    from dataclasses import dataclass
    from typing import Optional, List
    import tqdm
    import os.path as osp
    import cv2
    import torch
    import numpy as np
    from pytorch3d.implicitron.dataset.types import (
        FrameAnnotation as ImplicitronFrameAnnotation,
        load_dataclass,
    )
    @dataclass
    class DynamicReplicaFrameAnnotation(ImplicitronFrameAnnotation):
        """A dataclass used to load annotations from .json for Dynamic Replica."""

        camera_name: Optional[str] = None
        instance_id_map_path: Optional[str] = None
        flow_forward: Optional[str] = None
        flow_forward_mask: Optional[str] = None
        flow_backward: Optional[str] = None
        flow_backward_mask: Optional[str] = None
        trajectories: Optional[str] = None


    # ------------------------------------------------------------------
    # Helper utilities (lifted from the reference script)
    # ------------------------------------------------------------------

    def _load_16bit_png_depth(png_path: str) -> np.ndarray:
        """Load the 16-bit depth PNG used in Dynamic-Replica and return float32 depth (m)."""
        with Image.open(png_path) as _pil:
            depth = (
                np.frombuffer(np.array(_pil, dtype=np.uint16), dtype=np.float16)
                .astype(np.float32)
                .reshape((_pil.size[1], _pil.size[0]))
            )
        return depth

    def _get_pytorch3d_camera(entry_viewpoint, image_size):
        """Convert the intrinsics representation shipped with Dynamic-Replica ⟶ pixel units."""
        principal_point = torch.tensor(entry_viewpoint.principal_point, dtype=torch.float)
        focal_length = torch.tensor(entry_viewpoint.focal_length, dtype=torch.float)
        half_image_size_wh_orig = (
            torch.tensor(list(reversed(image_size)), dtype=torch.float) / 2.0
        )
        fmt = entry_viewpoint.intrinsics_format.lower()
        if fmt == "ndc_norm_image_bounds":
            rescale = half_image_size_wh_orig
        elif fmt == "ndc_isotropic":
            rescale = half_image_size_wh_orig.min()
        else:
            raise ValueError(f"Unknown intrinsics format: {fmt}")
        principal_point_px = half_image_size_wh_orig - principal_point * rescale
        focal_length_px = focal_length * rescale
        R = torch.tensor(entry_viewpoint.R, dtype=torch.float)
        T = torch.tensor(entry_viewpoint.T, dtype=torch.float)
        # Convert to pyTorch3D convention (y/ z sign flip)
        R_p3d = R.clone()
        T_p3d = T.clone()
        T_p3d[..., :2] *= -1
        R_p3d[..., :, :2] *= -1


        intr = np.eye(3, dtype=np.float32)
        intr[0, 0] = focal_length_px[0].item()
        intr[1, 1] = focal_length_px[1].item()
        intr[0, 2] = principal_point_px[0].item()
        intr[1, 2] = principal_point_px[1].item()
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.numpy().T
        pose[:3, 3] = -R.numpy().T @ T.numpy()
        return intr, pose

    # ------------------------------------------------------------------
    # Prepare paths & dataframe skeleton
    # ------------------------------------------------------------------
    os.makedirs(processed_root, exist_ok=True)
    data_prefix = DATASET_CONFIG["dynamic_replica"]["prefix"]

    df_rows = []  # list of dicts – will convert to DataFrame later

    for split in splits:
        split_dir = osp.join(raw_root, split)
        annotations_file = osp.join(split_dir, f"frame_annotations_{split}.jgz")
        if not osp.isfile(annotations_file):
            print(f"[dynamic_replica] Warning: annotations file missing for split '{split}'. Skipping.")
            continue

        # ------------------------------------------------------------------
        # Load compressed-json annotations
        # ------------------------------------------------------------------
        with gzip.open(annotations_file, "rt", encoding="utf8") as gz_f:
            frame_annots_list: List[DynamicReplicaFrameAnnotation] = load_dataclass(
                gz_f, List[DynamicReplicaFrameAnnotation]
            )

        # Organise per sequence & camera
        seq_dict = defaultdict(lambda: defaultdict(list))
        for fa in frame_annots_list:
            # camera_name = "left" if "left" in fa.image.path else "right"
            seq_dict[fa.sequence_name][fa.camera_name].append(fa)

        # ------------------------------------------------------------------
        # Iterate sequences / camera
        # ------------------------------------------------------------------
        for seq_name, cam_dict in tqdm.tqdm(seq_dict.items(), desc=f"{split} sequences"):
            for cam_name, frames in cam_dict.items():
                # ------------------------------------------------------------------
                # Skip the right camera as requested – only use "left" views.
                # ------------------------------------------------------------------
                if cam_name == "right":
                    continue
                # Ensure chronological order by timestamp
                frames.sort(key=lambda f: f.frame_timestamp)

                # Output dir structure mirrors reference script
                rgb_out_dir = osp.join(processed_root, split, seq_name, cam_name, "rgb")
                depth_out_dir = osp.join(processed_root, split, seq_name, cam_name, "depth")
                cam_out_dir = osp.join(processed_root, split, seq_name, cam_name, "cam")
                fflow_out_dir = osp.join(processed_root, split, seq_name, cam_name, "flow_forward")
                bflow_out_dir = osp.join(processed_root, split, seq_name, cam_name, "flow_backward")
                traj_out_dir = osp.join(processed_root, split, seq_name, cam_name, "traj")
                os.makedirs(rgb_out_dir, exist_ok=True)
                os.makedirs(depth_out_dir, exist_ok=True)
                os.makedirs(cam_out_dir, exist_ok=True)
                os.makedirs(fflow_out_dir, exist_ok=True)
                os.makedirs(bflow_out_dir, exist_ok=True)
                os.makedirs(traj_out_dir, exist_ok=True)

                for frame_idx, fa in enumerate(tqdm.tqdm(frames, leave=False, desc=f"{seq_name}/{cam_name}")):
                    ts = fa.frame_timestamp
                    idx_str = f"{frame_idx:06d}"
                    src_rgb = osp.join(split_dir, fa.image.path)
                    src_depth_png = osp.join(split_dir, fa.depth.path)

                    # Validate source files exist
                    if not osp.isfile(src_rgb) or not osp.isfile(src_depth_png):
                        print(f"[dynamic_replica] Missing files for {seq_name}/{cam_name} ts={ts}. Skipping frame.")
                        continue

                    # ------------------------------------------------------------------
                    # Copy / convert assets to processed directory
                    # ------------------------------------------------------------------
                    dst_rgb = osp.join(rgb_out_dir, f"{idx_str}.png")
                    if not osp.isfile(dst_rgb):
                        shutil.copy(src_rgb, dst_rgb)

                    depth_arr = _load_16bit_png_depth(src_depth_png)
                    dst_depth = osp.join(depth_out_dir, f"{idx_str}.npy")
                    if not osp.isfile(dst_depth):
                        np.save(dst_depth, depth_arr)

                    # Camera params
                    intr, pose = _get_pytorch3d_camera(fa.viewpoint, fa.image.size)
                    dst_cam = osp.join(cam_out_dir, f"{idx_str}.npz")
                    if not osp.isfile(dst_cam):
                        np.savez(dst_cam, intrinsics=intr, pose=pose)

                    # ------------------------------------------------------------------
                    # Forward / backward optical flow (if available)
                    # ------------------------------------------------------------------
                    if fa.flow_forward and fa.flow_forward["path"]:
                        fflow_src = osp.join(split_dir, fa.flow_forward["path"])
                        fflow_mask_src = osp.join(split_dir, fa.flow_forward_mask["path"])
                        if osp.isfile(fflow_src) and osp.isfile(fflow_mask_src):
                            flow_fwd = cv2.imread(fflow_src, cv2.IMREAD_UNCHANGED)
                            flow_fwd_mask = cv2.imread(fflow_mask_src, cv2.IMREAD_UNCHANGED)
                            np.savez(osp.join(fflow_out_dir, f"{idx_str}.npz"), flow=flow_fwd, mask=flow_fwd_mask)

                    if fa.flow_backward and fa.flow_backward["path"]:
                        bflow_src = osp.join(split_dir, fa.flow_backward["path"])
                        bflow_mask_src = osp.join(split_dir, fa.flow_backward_mask["path"])
                        if osp.isfile(bflow_src) and osp.isfile(bflow_mask_src):
                            flow_bwd = cv2.imread(bflow_src, cv2.IMREAD_UNCHANGED)
                            flow_bwd_mask = cv2.imread(bflow_mask_src, cv2.IMREAD_UNCHANGED)
                            np.savez(osp.join(bflow_out_dir, f"{idx_str}.npz"), flow=flow_bwd, mask=flow_bwd_mask)

                    traj_path = osp.join(split_dir, fa.trajectories["path"])
                    traj_dict = torch.load(traj_path)
                    new_traj_dict = {}

                    for k, v in traj_dict.items():
                        if k in ['traj_3d_world', 'traj_2d', 'verts_inds_vis', 'instances']:
                            new_traj_dict[k] = v
                    dst_traj = osp.join(traj_out_dir, f"{idx_str}.pth")
                    if not osp.isfile(dst_traj):
                        torch.save(new_traj_dict, dst_traj)

                    # ------------------------------------------------------------------
                    # Append dataframe row (convert paths)
                    # ------------------------------------------------------------------
                    df_rows.append(
                        {
                            "rgb_path": dst_rgb.replace(processed_root, data_prefix),
                            "depth_path": dst_depth.replace(processed_root, data_prefix),
                            "intrinsics": intr.flatten().tolist(),
                            "extrinsics": pose.flatten().tolist(),
                            "video_name": f"{seq_name}_{cam_name}",
                            "frame_number": frame_idx,
                            "frame_timestamp": ts,
                        }
                    )

    # ------------------------------------------------------------------
    # Create parquet & upload
    # ------------------------------------------------------------------
    if not df_rows:
        print("[dynamic_replica] No frames processed – parquet not generated.")
        return

    df = pd.DataFrame(df_rows)
    parquet_path = "dynamic_replica_train.parquet"
    df.to_parquet(parquet_path)
    print(f"[dynamic_replica] Parquet generated with {len(df)} rows → {parquet_path}")

    upload_file(
        parquet_path,
        DATASET_CONFIG["dynamic_replica"]["parquet_path"],
    )

    os.remove(parquet_path)


# -----------------------------------------------------------------------------
# CLI helper – run `python depth_dynamic_replica.py` to preprocess & upload parquet
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # traj = torch.load("/mnt/localssd/dynamic_replica/train/11a245-1_obj_source_left/trajectories/000000.pth")
    # breakpoint()
    # print(traj)
    # exit()
    # import argparse
    # parser = argparse.ArgumentParser(description="Generate parquet for Dynamic Replica (raw → processed)")
    # parser.add_argument("--raw_root", type=str, default="/mnt/localssd/dynamic_replica", help="Path to raw Dynamic Replica root directory")
    # parser.add_argument("--processed_root", type=str, default="/mnt/localssd/dynamic_replica_processed", help="Output directory for processed assets")
    # parser.add_argument("--splits", type=str, nargs="*", default=["train", "valid"], help="Splits to process")
    # args = parser.parse_args()

    # generate_parquet_file(raw_root=args.raw_root, processed_root=args.processed_root, splits=tuple(args.splits))

    import tqdm
    import cv2
    from Unip.unip.util.pointmap_util import depth_to_pointmap, save_pointmap_as_ply
    import moviepy.video.io.ffmpeg_writer as video_writer
    from training.dataloaders.batched_sampler import make_sampler
    from torch.utils.data import DataLoader
    
    os.makedirs("data_vis", exist_ok=True)
    print("\nTesting video dataset...")

    resolutions = [(512, 384), (512, 336), (512, 288), (512, 256), (384, 512), (336, 512), (288, 512), (256, 512)]
    video_dataset = VideoDepthDynamicReplicaNew(T=8, stride_range=(5, 10), transform=True, resolutions=resolutions)

    dataloader_sampler = make_sampler(
        video_dataset, 
        batch_size=1, 
        number_of_resolutions=len(resolutions),
        min_num_frames=8, 
        max_num_frames=16, 
        shuffle=True, 
        drop_last=True
    )

    dataloader = DataLoader(
        video_dataset,
        batch_sampler=dataloader_sampler,
        num_workers=1,
        pin_memory=False,
    )

    if hasattr(dataloader, "dataset") and hasattr(dataloader.dataset, "set_epoch"):
        dataloader.dataset.set_epoch(0)
    if (
        hasattr(dataloader, "batch_sampler")
        and hasattr(dataloader.batch_sampler, "sampler")
        and hasattr(dataloader.batch_sampler.sampler, "set_epoch")
    ):
        dataloader.batch_sampler.sampler.set_epoch(0)
    

    print(f"Video dataset length: {len(video_dataset)}")
    
    if len(video_dataset) > 0:

        # Test with a few samples from the dataloader
        sample_count = 0
        max_samples = 10  # Test with 5 samples instead of 20
        
        for batch_idx, video_sample in enumerate(dataloader):

            # if batch_idx % 20 != 0:
            #     continue
            
            if sample_count >= max_samples:
                break
                
            # Extract the first item from the batch since batch_size_per_gpu=1
            if isinstance(video_sample['rgb'], list):
                # Handle case where video_sample might be a list of batches
                video_sample = {k: v[0] if isinstance(v, list) else v for k, v in video_sample.items()}
            else:
                # Extract first item from batch dimension
                video_sample = {k: v[0] if v.dim() > 3 else v for k, v in video_sample.items()}

            print(f"Video sample {sample_count} shapes:")
            print(f"  RGB: {video_sample['rgb'].shape}")  # Should be [T, 3, H, W]
            print(f"  Metric depth: {video_sample['metric_depth'].shape}")  # Should be [T, 1, H, W]
            print(f"  Valid mask: {video_sample['valid_mask'].shape}")  # Should be [T, 1, H, W]
            print(f"  Intrinsics: {video_sample['intrinsics'].shape}")  # Should be [T, 3, 3]
            print(f"  Extrinsics: {video_sample['extrinsics'].shape}")  # Should be [T, 3, 3]
            print(f"  Valid: {video_sample['valid']}")
            print(f"  Stride: {video_sample['stride']}")
            
            # Prepare video writers
            H, W = video_sample['rgb'].shape[2], video_sample['rgb'].shape[3]
            
            rgb_writer = video_writer.FFMPEG_VideoWriter(filename=f"data_vis/dynamic_replica_video_rgb_{sample_count}.mp4", size=(W, H), fps=2)
            depth_writer = video_writer.FFMPEG_VideoWriter(filename=f"data_vis/dynamic_replica_video_depth_{sample_count}.mp4", size=(W, H), fps=2)
            mask_writer = video_writer.FFMPEG_VideoWriter(filename=f"data_vis/dynamic_replica_video_mask_{sample_count}.mp4", size=(W, H), fps=2)
            
            print("Saving video frames...")
            for t in tqdm.tqdm(range(video_sample['rgb'].shape[0])):
                # Process RGB frame
                rgb_frame = ((video_sample['rgb'][t].numpy().transpose(1, 2, 0) + 1) / 2 * 255).astype(np.uint8)
                
                # Process depth frame
                depth = video_sample['metric_depth'][t].numpy().transpose(1, 2, 0).squeeze()
                # Normalize depth for visualization (0-255)
                if depth.max() > depth.min():
                    depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
                else:
                    depth_normalized = np.zeros_like(depth, dtype=np.uint8)
                depth_bgr = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)
                
                # Process mask frame
                valid_mask = (video_sample['valid_mask'][t].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                mask_bgr = cv2.cvtColor(valid_mask.squeeze(), cv2.COLOR_GRAY2BGR)
                
                # Write frames to videos
                rgb_writer.write_frame(rgb_frame)
                depth_writer.write_frame(depth_bgr)
                mask_writer.write_frame(mask_bgr)
                
                # Save pointcloud for first frame only
                if t == 0:
                    intrinsics = video_sample['intrinsics'][t].numpy()
                    pointmap = depth_to_pointmap(depth, intrinsics)
                    save_pointmap_as_ply(pointmap, rgb_frame, f"data_vis/dynamic_replica_video_frame0_pointmap_{sample_count}.ply", far_threshold=1000)

            rgb_writer.close()
            depth_writer.close()
            mask_writer.close()
            
            print("Video sequence test completed successfully!")
            print("Saved videos:")
            print(f"  - data_vis/dynamic_replica_video_rgb_{sample_count}.mp4")
            print(f"  - data_vis/dynamic_replica_video_depth_{sample_count}.mp4") 
            print(f"  - data_vis/dynamic_replica_video_mask_{sample_count}.mp4")
            print(f"  - data_vis/dynamic_replica_video_frame0_pointmap_{sample_count}.ply")
            
            sample_count += 1
    else:
        print("No valid video sequences found in dataset.")
