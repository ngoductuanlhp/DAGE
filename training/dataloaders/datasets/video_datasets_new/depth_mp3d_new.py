#!/usr/bin/env python3
"""
Dataloader for preprocessed Matterport3D (MP3D) dataset.

This dataloader reads preprocessed MP3D data from local storage.
Data is organized into scenes with RGB images, depth maps, and camera parameters.
"""

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import io
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import random
import cv2
from collections import defaultdict

from training.dataloaders.data_io import download_file, upload_file
from training.dataloaders.config import DATASET_CONFIG
from training.dataloaders.datasets.video_datasets_new.synchronized_video_transform import (
    SynchronizedTransformVideoCut3r,
)
from training.dataloaders.datasets.video_datasets_new.synchronized_video_transform_moge import (
    SynchronizedTransform_MoGe,
)
from training.dataloaders.datasets.video_datasets_new.frame_sampling_utils import get_seq_from_start_id


class SynchronizedTransform_MP3DVideo:
    def __init__(self, H, W):
        self.resize = transforms.Resize((H, W))
        self.resize_depth = transforms.Resize((H, W), interpolation=Image.NEAREST)
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.to_tensor = transforms.ToTensor()
        self.W = W
        self.H = H

    def __call__(self, rgb_images, depth_images, intrinsics_list, extrinsics_list=None):
        if extrinsics_list is not None:
            flip = False
        else:
            flip = random.random() > 0.5

        rgb_tensors = []
        depth_tensors = []
        intrinsics_tensors = []
        extrinsics_tensors = []

        for idx, (rgb_image, depth_image, intrinsics) in enumerate(
            zip(rgb_images, depth_images, intrinsics_list)
        ):
            if flip:
                rgb_image = self.horizontal_flip(rgb_image)
                depth_image = self.horizontal_flip(depth_image)

            og_width, og_height = rgb_image.size
            scale_w = self.W / og_width
            scale_h = self.H / og_height
            rgb_image = self.resize(rgb_image)
            depth_image = self.resize_depth(depth_image)
            intrinsics[0, 0] *= scale_w
            intrinsics[1, 1] *= scale_h
            intrinsics[0, 2] *= scale_w
            intrinsics[1, 2] *= scale_h

            rgb_tensor = self.to_tensor(rgb_image)
            depth_tensor = self.to_tensor(depth_image)
            intrinsics_tensor = torch.tensor(intrinsics)

            rgb_tensors.append(rgb_tensor)
            depth_tensors.append(depth_tensor)
            intrinsics_tensors.append(intrinsics_tensor)

            if extrinsics_list is not None:
                extrinsics_tensor = torch.tensor(extrinsics_list[idx])
                extrinsics_tensors.append(extrinsics_tensor)

        if extrinsics_list is not None:
            return rgb_tensors, depth_tensors, intrinsics_tensors, extrinsics_tensors
        else:
            return rgb_tensors, depth_tensors, intrinsics_tensors


# pylint: disable=too-many-locals, too-many-branches, too-many-statements


class VideoDepthMP3DNew(Dataset):
    """
    Dataloader for preprocessed Matterport3D dataset.
    
    Loads data from local storage using parquet index.
    """
    depth_type = "lidar"
    is_metric_scale = True
    # NOTE original resolution is 1280x1024

    def __init__(
        self,
        T=5,
        stride_range=(1, 3),
        transform=True,
        resize=(480, 640),
        near_plane=1e-3,
        far_plane=200.0,
        resolutions=None,
        use_moge=False,
        moge_augmentation=None,
        use_cut3r_frame_sampling=False,
        video_prob=0.6,
        fix_interval_prob=0.6,
        block_shuffle=16,
    ):
        self.T = T
        self.stride_range = stride_range
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.frame_size = resize

        self.resolutions = resolutions
        self.use_moge = use_moge
        self.moge_augmentation = moge_augmentation

        self.use_cut3r_frame_sampling = use_cut3r_frame_sampling
        self.video_prob = video_prob
        self.fix_interval_prob = fix_interval_prob
        self.block_shuffle = block_shuffle

        # Load data from local storage
        parquet_data = download_file(
            
            DATASET_CONFIG["mp3d"]["parquet_path"],
        )
        self.parquet = io.BytesIO(parquet_data)
        self.scene_frames = self._organize_by_scene_from_parquet()

        self.frame_list = self._create_frame_list()

        if self.use_moge:
            self.transform = SynchronizedTransform_MoGe(
                moge_augmentation=self.moge_augmentation
            )
        elif self.resolutions is not None:
            self.transform = SynchronizedTransformVideoCut3r(seed=777)
        else:
            self.transform = (
                SynchronizedTransform_MP3DVideo(H=resize[0], W=resize[1])
                if transform
                else None
            )

    def _organize_by_scene_from_parquet(self):
        """Organize frames by scene from parquet file and load overlap data."""
        df = pd.read_parquet(self.parquet)
        
        # Sort by scene_name and frame_number to ensure temporal order
        df = df.sort_values(['scene_name', 'frame_number'])
        
        # Group by scene_name (sort=False since we already sorted)
        scene_frames = {}
        overlaps = {}
        
        for scene_name, scene_df in df.groupby('scene_name', sort=False):
            # Convert to dict efficiently - frames are already in temporal order
            frames = scene_df[['frame_number', 'rgb_path', 'depth_path', 
                               'intrinsics', 'extrinsics']].to_dict('records')
            scene_frames[scene_name] = frames
            
            # Load overlap data from local storage
            overlap_path = f"{DATASET_CONFIG['mp3d']['prefix']}/{scene_name}/overlap.npy"
            try:
                overlap_bytes = download_file(
                    overlap_path
                )
                overlap = np.load(io.BytesIO(overlap_bytes))
                overlaps[scene_name] = overlap
            except Exception as e:
                print(f"[mp3d] Warning: Could not load overlap for scene {scene_name}: {e}")
                overlaps[scene_name] = None

        if not scene_frames:
            raise RuntimeError("No valid MP3D entries found in parquet")

        print(f"[mp3d] Loaded {len(scene_frames)} scenes with {sum(len(frames) for frames in scene_frames.values())} total frames")

        self.overlaps = overlaps
        return scene_frames

    def _create_frame_list(self):
        """Create a flat list of all frames for indexing."""
        # Use list comprehension for faster construction
        frame_list = [
            {
                'scene_name': scene_name,
                'frame_idx_in_scene': i,
                'frame_data': frame_data
            }
            for scene_name, frames in self.scene_frames.items()
            for i, frame_data in enumerate(frames)
        ]
        return frame_list

    def __len__(self):
        return len(self.frame_list)

    def _can_sample_sequence(self, scene_name, start_idx, stride, num_frames=None):
        """Check if a valid sequence can be sampled from the given start position."""
        if num_frames is None:
            num_frames = self.T

        frames = self.scene_frames[scene_name]
        if start_idx + (num_frames - 1) * stride >= len(frames):
            return False

        start_frame_num = frames[start_idx]["frame_number"]
        for i in range(num_frames):
            frame_idx = start_idx + i * stride
            expected_frame_num = start_frame_num + i * stride
            actual_frame_num = frames[frame_idx]["frame_number"]
            if actual_frame_num != expected_frame_num:
                return False
        return True

    def _sample_views_by_overlap(self, scene_name, img_idx, num_views, rng, min_overlap=0.05, max_overlap=0.6):
        """
        Sample views based on pre-computed overlap data.
        
        Args:
            scene_name: Name of the scene
            img_idx: Index of the reference frame in the scene
            num_views: Total number of views to sample (including reference)
            rng: Random number generator
            allow_repeat: Whether to allow repeated views
            min_overlap: Minimum overlap threshold
            max_overlap: Maximum overlap threshold
            
        Returns:
            frame_positions: List of frame indices in the scene
            success: Whether sampling was successful
        """
        overlap = self.overlaps.get(scene_name)
        if overlap is None:
            # Fallback to stride-based sampling if no overlap data
            return None, False
        
        num_unique = num_views
        
        # Find overlap entries for this image
        sel_img_idx = np.where(overlap[:, 0] == img_idx)[0]
        if len(sel_img_idx) == 0:
            return None, False
        
        overlap_sel = overlap[sel_img_idx]
        
        # Filter by overlap score
        overlap_sel = overlap_sel[
            (overlap_sel[:, 2] > min_overlap) & (overlap_sel[:, 2] < max_overlap)
        ]
        
        num_views_possible = len(overlap_sel)
        if num_views_possible < num_unique - 1:
            return None, False
        
        # Sample views based on overlap scores as weights
        other_img_indices = overlap_sel[:, 1].astype(np.int64)
        overlap_scores = overlap_sel[:, 2]
        probabilities = overlap_scores / np.sum(overlap_scores)
        
        sampled_indices = rng.choice(
            other_img_indices,
            num_views - 1,
            replace=False,
            p=probabilities,
        )
        
        # Combine reference frame with sampled frames
        frame_positions = [img_idx] + sampled_indices.tolist()
        
        return frame_positions, True

    def _load_single_frame(self, frame_data):
        """Load a single frame from local storage."""
        try:
            # Load RGB from local storage
            rgb_bytes = download_file(
                frame_data["rgb_path"]
            )
            rgb_image = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
            del rgb_bytes  # Free memory immediately

            # Load depth from local storage
            depth_bytes = download_file(
                frame_data["depth_path"]
            )
            depth_array = np.load(io.BytesIO(depth_bytes)).astype(np.float32)
            del depth_bytes  # Free memory immediately
            
            depth_array[~np.isfinite(depth_array)] = 0.0
            depth_image = Image.fromarray(depth_array)
            del depth_array  # Free memory immediately

            # Get camera parameters from parquet
            intrinsics = np.array(frame_data["intrinsics"], dtype=np.float32).reshape(3, 3).copy()
            extrinsics = np.array(frame_data["extrinsics"], dtype=np.float32).reshape(4, 4).copy()

            return rgb_image, depth_image, intrinsics, extrinsics, True

        except Exception as e:
            print(f"[mp3d] Error loading frame {frame_data.get('rgb_path', 'unknown')}: {str(e)}")
            return None, None, None, None, False

    def __getitem__(self, idx):
        try:
            # Handle tuple index for variable num_frames
            if isinstance(idx, tuple):
                len_tuple_idx = len(idx)
                if len_tuple_idx == 2:
                    idx, num_frames = idx
                    resolution_idx = None
                elif len_tuple_idx == 3:
                    idx, num_frames, resolution_idx = idx
                else:
                    raise ValueError(f"Invalid tuple length for idx: {len_tuple_idx}")
            else:
                len_tuple_idx = 1
                num_frames = self.T
                resolution_idx = None

            rng = np.random.default_rng(random.randint(0, 2 ** 32 - 1))

            start_frame_info = self.frame_list[idx]
            scene_name = start_frame_info["scene_name"]
            start_idx = start_frame_info["frame_idx_in_scene"]
            frames = self.scene_frames[scene_name]

            # Frame sampling strategy
            # First try overlap-based sampling if overlap data is available
            frame_positions, is_success = self._sample_views_by_overlap(
                scene_name, start_idx, num_frames, rng
            )
            is_video = False  # MP3D frames are spatially related

            # If sampling failed, try a different starting position
            if not is_success:
                next_idx = (
                    idx
                    + (num_frames - 1)
                    * random.randint(self.stride_range[0], self.stride_range[1])
                ) % self.__len__()
                if len_tuple_idx == 3:
                    return self.__getitem__((next_idx, num_frames, resolution_idx))
                elif len_tuple_idx == 2:
                    return self.__getitem__((next_idx, num_frames))
                else:
                    return self.__getitem__(next_idx)

            # Load frames
            rgb_images = []
            depth_images = []
            intrinsics_list = []
            extrinsics_list = []
            valid_frames = []

            for frame_idx in frame_positions:
                frame_data = frames[frame_idx]
                rgb_img, depth_img, intrinsics, extrinsics, valid = self._load_single_frame(
                    frame_data
                )

                if valid:
                    rgb_images.append(rgb_img)
                    depth_images.append(depth_img)
                    intrinsics_list.append(intrinsics.copy())
                    extrinsics_list.append(extrinsics)
                    valid_frames.append(True)
                else:
                    # Use previous frame if available, otherwise create dummy
                    if rgb_images:
                        rgb_images.append(rgb_images[-1])
                        depth_images.append(depth_images[-1])
                        intrinsics_list.append(intrinsics_list[-1].copy())
                        extrinsics_list.append(extrinsics_list[-1])
                    else:
                        rgb_images.append(Image.new("RGB", self.frame_size[::-1]))
                        depth_images.append(
                            Image.fromarray(
                                np.zeros((self.frame_size[0], self.frame_size[1]), dtype=np.float32)
                            )
                        )
                        intrinsics_list.append(np.eye(3, dtype=np.float32))
                        extrinsics_list.append(np.eye(4, dtype=np.float32))
                    valid_frames.append(False)

            # Apply transforms
            if self.transform is not None:
                if isinstance(self.transform, SynchronizedTransformVideoCut3r):
                    target_resolution = (
                        self.resolutions[resolution_idx]
                        if (resolution_idx is not None and self.resolutions is not None)
                        else (self.frame_size[1], self.frame_size[0])
                    )
                    rgb_tensors, depth_tensors, intrinsics_tensors = self.transform(
                        rgb_images,
                        depth_images,
                        intrinsics_list,
                        resolution=target_resolution,
                    )
                    extrinsics_tensors = [torch.tensor(e) for e in extrinsics_list]
                elif isinstance(self.transform, SynchronizedTransform_MoGe):
                    processed_data = self.transform(
                        rgb_images,
                        depth_images,
                        intrinsics_list,
                        extrinsics_list,
                        valid_frames,
                        self.near_plane,
                        self.far_plane,
                        self.depth_type,
                        resolution_idx,
                        stride=None,
                        rng=rng,
                        no_depth_mask_inf=True,
                    )
                    processed_data['is_metric_scale'] = self.is_metric_scale
                    return processed_data
                else:
                    (
                        rgb_tensors,
                        depth_tensors,
                        intrinsics_tensors,
                        extrinsics_tensors,
                    ) = self.transform(
                        rgb_images,
                        depth_images,
                        intrinsics_list,
                        extrinsics_list,
                    )
            else:
                rgb_tensors = [transforms.ToTensor()(img) for img in rgb_images]
                depth_tensors = [transforms.ToTensor()(img) for img in depth_images]
                intrinsics_tensors = [torch.tensor(intrinsics) for intrinsics in intrinsics_list]
                extrinsics_tensors = [torch.tensor(extrinsics) for extrinsics in extrinsics_list]

            # Process depth and create output
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
                extrinsics_tensor = extrinsics_tensors[t]

                valid_depth_mask = (depth_tensor > self.near_plane) & (
                    depth_tensor < self.far_plane
                )
                inf_depth_mask = depth_tensor >= self.far_plane

                rgb_tensor = rgb_tensor * 2.0 - 1.0

                if valid_depth_mask.any() and valid_frames[t]:
                    flat_depth = depth_tensor[valid_depth_mask].flatten().float()
                    min_depth = torch.quantile(flat_depth, 0.00)
                    max_depth = torch.quantile(flat_depth, 0.98)
                    valid_depth_mask = (depth_tensor > min_depth) & (
                        depth_tensor < max_depth
                    )

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
                processed_extrinsics.append(extrinsics_tensor)

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
                "frame_positions": frame_positions,
                "is_video": is_video,
                "valid": all(valid_frames),
                "depth_type": self.depth_type,
                "scene_name": scene_name,
            }

        except Exception as e:
            print(f"[mp3d] Error processing sample: {str(e)}")
            if num_frames is None:
                num_frames = self.T

            dummy_rgb = torch.zeros(num_frames, 3, self.frame_size[0], self.frame_size[1])
            dummy_depth = torch.zeros(num_frames, 1, self.frame_size[0], self.frame_size[1])
            dummy_mask = torch.zeros(
                num_frames, 1, self.frame_size[0], self.frame_size[1], dtype=torch.bool
            )
            dummy_intrinsics = torch.eye(3).unsqueeze(0).repeat(num_frames, 1, 1)
            dummy_extrinsics = torch.eye(4).unsqueeze(0).repeat(num_frames, 1, 1)

            return {
                "rgb": dummy_rgb,
                "metric_depth": dummy_depth,
                "valid_mask": dummy_mask,
                "intrinsics": dummy_intrinsics,
                "extrinsics": dummy_extrinsics,
                "valid": False,
                "depth_type": self.depth_type,
            }


def generate_parquet_file(
    processed_root: str = "/mnt/localssd/processed_mp3d",
    split: str = "train",
    parquet_local: str | None = None,
):
    """Generate parquet index for MP3D processed data."""

    import tqdm

    processed_root_path = Path(processed_root)
    if not processed_root_path.is_dir():
        raise FileNotFoundError(
            f"Processed MP3D root '{processed_root}' does not exist"
        )

    cfg = DATASET_CONFIG["mp3d"]
    prefix = Path(cfg["prefix"])
    parquet_key = cfg.get("parquet_path", f"parquets/mp3d_{split}.parquet")

    records = []
    for scene_dir in tqdm.tqdm(
        sorted(processed_root_path.iterdir()), desc="Scanning MP3D scenes"
    ):
        if not scene_dir.is_dir():
            continue

        rgb_dir = scene_dir / "rgb"
        depth_dir = scene_dir / "depth"
        cam_dir = scene_dir / "cam"
        
        if not (rgb_dir.is_dir() and depth_dir.is_dir() and cam_dir.is_dir()):
            continue

        for rgb_file in sorted(rgb_dir.glob("*.png")):
            try:
                frame_number = int(rgb_file.stem)
            except ValueError:
                continue

            depth_file = depth_dir / f"{frame_number:06d}.npy"
            cam_file = cam_dir / f"{frame_number:06d}.npz"
            
            if not (depth_file.is_file() and cam_file.is_file()):
                continue

            cam_data = np.load(cam_file)
            intrinsics = cam_data.get("intrinsics")
            extrinsics = cam_data.get("pose")
            
            if intrinsics is None or extrinsics is None:
                print(f"Warning: missing intrinsics/extrinsics in {cam_file}")
                continue

            rgb_rel = rgb_file.relative_to(processed_root_path)
            depth_rel = depth_file.relative_to(processed_root_path)

            records.append(
                {
                    "rgb_path": str(prefix / rgb_rel.as_posix()),
                    "depth_path": str(prefix / depth_rel.as_posix()),
                    "intrinsics": np.asarray(intrinsics, dtype=np.float32).flatten().tolist(),
                    "extrinsics": np.asarray(extrinsics, dtype=np.float32).flatten().tolist(),
                    "scene_name": rgb_rel.parts[0],
                    "frame_number": frame_number,
                }
            )

    if not records:
        raise RuntimeError("No MP3D records were found to build the parquet file")

    df = pd.DataFrame.from_records(
        records,
        columns=[
            "rgb_path",
            "depth_path",
            "intrinsics",
            "extrinsics",
            "scene_name",
            "frame_number",
        ],
    )

    parquet_filename = parquet_local or f"mp3d_{split}.parquet"
    df.to_parquet(parquet_filename)
    print(f"[mp3d] Generated parquet with {len(df)} entries -> {parquet_filename}")

    upload_file(parquet_filename, parquet_key)
    print(f"[mp3d] Saved parquet to {parquet_key}")

    if parquet_local is None:
        os.remove(parquet_filename)

    return df


if __name__ == "__main__":
    import tqdm
    import moviepy.video.io.ffmpeg_writer as video_writer
    import cv2
    from training.dataloaders.batched_sampler import make_sampler
    from torch.utils.data import DataLoader
    import open3d as o3d
    import utils3d
    from third_party.pi3.utils.geometry import homogenize_points, se3_inverse, depth_edge

    os.makedirs("data_vis", exist_ok=True)
    print("\nTesting MP3D dataset...")

    dataset_name = "mp3d_new"
    save_dir = f"data_vis/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    resolutions = [[0, 1]] * 10000
    habitat_dataset = VideoDepthMP3DNew(
        T=10, 
        stride_range=(1, 5),
        transform=True,
        resolutions=resolutions, 
        use_moge=True,
        use_cut3r_frame_sampling=True,
        video_prob=1.0,
        fix_interval_prob=0.6,
        block_shuffle=24,
        moge_augmentation=dict(
            use_flip_augmentation=False,
            center_augmentation=0.10,
            fov_range_absolute_min=60,
            fov_range_absolute_max=120,
            fov_range_relative_min=0.7,
            fov_range_relative_max=1.0,
            image_augmentation=['jittering', 'jpeg_loss', 'blurring'],
            depth_interpolation='nearest',
            clamp_max_depth=1000.0,
            area_range=[250000, 500000],
            aspect_ratio_range=[0.5, 2.0],
        )
    )
    print(f"Habitat dataset length: {len(habitat_dataset)}")

    dataloader_sampler = make_sampler(
        habitat_dataset, 
        batch_size=1, 
        number_of_resolutions=len(resolutions),
        min_num_frames=10, 
        max_num_frames=10, 
        shuffle=True, 
        drop_last=True
    )

    dataloader = DataLoader(
        habitat_dataset,
        batch_sampler=dataloader_sampler,
        num_workers=0,
        pin_memory=False,
    )

    if hasattr(dataloader, "dataset") and hasattr(dataloader.dataset, "set_epoch"):
        dataloader.dataset.set_epoch(0)
    if (
        hasattr(dataloader, "batch_sampler")
        and hasattr(dataloader.batch_sampler, "set_epoch")
    ):
        dataloader.batch_sampler.set_epoch(0)
    
    print(f"Habitat dataset length: {len(habitat_dataset)}")
    
    if len(habitat_dataset) > 0:
        sample_count = 0

        for idx, scene_sample in enumerate(dataloader):
            sample_count += 1
            if sample_count > 5: 
                break

            if isinstance(scene_sample['rgb'], list):
                scene_sample = {k: v[0] if isinstance(v, list) else v for k, v in scene_sample.items()}
            else:
                scene_sample = {k: v[0] if isinstance(v, torch.Tensor) and v.dim() > 3 else v for k, v in scene_sample.items()}

            print(f"\nScene sample shapes:")
            print(f"  RGB: {scene_sample['rgb'].shape}")  # Should be [T, 3, H, W]
            print(f"  Metric depth: {scene_sample['metric_depth'].shape}")  # Should be [T, 1, H, W]
            print(f"  Valid mask: {scene_sample['valid_mask'].shape}")  # Should be [T, 1, H, W]
            print(f"  Fin mask: {scene_sample['fin_mask'].shape}")  # Should be [T, 1, H, W]
            print(f"  Intrinsics: {scene_sample['intrinsics'].shape}")  # Should be [T, 3, 3]
            print(f"  Extrinsics: {scene_sample['extrinsics'].shape}")  # Should be [T, 4, 4]
            print(f"  Valid: {scene_sample['valid']}")
            print(f"  Depth type: {scene_sample['depth_type']}")
            
            # Prepare video writers
            H, W = scene_sample['rgb'].shape[2], scene_sample['rgb'].shape[3]
            
            rgb_writer = video_writer.FFMPEG_VideoWriter(filename=f"{save_dir}/scene_rgb_{idx}.mp4", size=(W, H), fps=2)
            depth_writer = video_writer.FFMPEG_VideoWriter(filename=f"{save_dir}/scene_depth_{idx}.mp4", size=(W, H), fps=2)
            mask_writer = video_writer.FFMPEG_VideoWriter(filename=f"{save_dir}/scene_mask_{idx}.mp4", size=(W, H), fps=2)

            all_gt_pts = []
            all_gt_pts_rgb = []
            print("Saving scene frames...")
            for t in tqdm.tqdm(range(scene_sample['rgb'].shape[0])):
                # Process RGB frame (convert from [-1,1] back to [0,255])
                rgb_frame = ((scene_sample['rgb'][t].numpy().transpose(1, 2, 0) + 1.0) / 2.0 * 255).astype(np.uint8)
                
                # Process depth frame
                depth = scene_sample['metric_depth'][t].squeeze()
                mask = scene_sample['valid_mask'][t].squeeze()

                intrinsics = scene_sample['intrinsics'][t]
                gt_pts = utils3d.torch.depth_to_points(depth, intrinsics=intrinsics)

                poses = scene_sample['extrinsics'][t]
                gt_pts = torch.einsum('ij, hwj -> hwi', poses, homogenize_points(gt_pts))[..., :3]

                gt_pts = gt_pts[::4, ::4][mask[::4, ::4]].reshape(-1, 3)
                gt_pts_rgb = ((scene_sample['rgb'][t].permute(1, 2, 0) + 1.0) / 2.0)[::4, ::4][mask[::4, ::4]].reshape(-1, 3)

                depth = depth.numpy()
                mask = mask.numpy()

                if mask.sum() > 0:
                    depth_max = depth[mask].max()
                    depth_min = depth[mask].min()

                    depth_normalized = depth.copy()
                    depth_normalized[~mask] = depth_min
                    depth_normalized = (depth_normalized - depth_min) / (depth_max - depth_min + 1e-8) * 255
                else:
                    depth_normalized = np.zeros_like(depth)

                depth_normalized = depth_normalized.astype(np.uint8)
                depth_bgr = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)

                valid_mask = (scene_sample['valid_mask'][t].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                mask_bgr = cv2.cvtColor(valid_mask.squeeze(), cv2.COLOR_GRAY2BGR)

                rgb_writer.write_frame(rgb_frame)
                depth_writer.write_frame(depth_bgr)
                mask_writer.write_frame(mask_bgr)

                all_gt_pts.append(gt_pts)
                all_gt_pts_rgb.append(gt_pts_rgb)
                
            rgb_writer.close()
            depth_writer.close()
            mask_writer.close()

            # Save point cloud
            if len(all_gt_pts) > 0:
                all_gt_pts = torch.cat(all_gt_pts, dim=0)
                all_gt_pts_rgb = torch.cat(all_gt_pts_rgb, dim=0)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(all_gt_pts.numpy())
                pcd.colors = o3d.utility.Vector3dVector(all_gt_pts_rgb.numpy())
                save_path = f"{save_dir}/scene_pts_{idx}.ply"
                o3d.io.write_point_cloud(save_path, pcd)
                print(f"Saved pointcloud to {save_path}")
            
            print("Scene test completed successfully!")
    else:
        print("No valid scenes found in dataset.")