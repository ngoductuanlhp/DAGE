import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
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
import shutil
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

class SynchronizedTransform_ArkitScenes:
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


class VideoDepthArkitScenesNew(Dataset):
    """Skeleton video dataset loader for Waymo depth data.

    NOTE: The per-frame loading logic is left as placeholders – you should
    replace those with real parsing of the `*.jpg`, `*.exr`, and `*.npz`
    files found inside each Waymo segment directory.
    """

    depth_type = "sfm"
    is_metric_scale = True

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
        use_cut3r_frame_sampling=False, video_prob=0.6, fix_interval_prob=0.6, block_shuffle=16,
    ):
        parquet_data = download_file(DATASET_CONFIG["arkitscenes"]["parquet_path"])
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

        # Initialize data structures
        self.video_frames = self._organize_by_video()
        self.frame_list = self._create_frame_list()

        if self.use_moge:
            self.transform = SynchronizedTransform_MoGe(moge_augmentation=moge_augmentation)
        elif self.resolutions is not None:
            self.transform = SynchronizedTransformVideoCut3r(seed=777)
        else:
            self.transform = SynchronizedTransform_ArkitScenes(H=resize[0], W=resize[1]) if transform else None

    def _organize_by_video(self):
        """Organize frames by video name.
        Frame numbers are computed from rgb_path timestamps, then sorted.
        """
        df = pd.read_parquet(self.parquet)
        
        # Compute frame_real_number from rgb_path (extract timestamp and convert)
        df['frame_number'] = df['rgb_path'].apply(
            lambda x: int(float(x.replace('.jpg', '').split('/')[-1].split('_')[-1]) * 100)
        )
        
        # Sort by video_name and frame_number to ensure temporal order
        df = df.sort_values(['video_name', 'frame_number'])
        
        # Group by video_name (sort=False since we already sorted)
        video_frames = {}
        for video_name, video_df in df.groupby('video_name', sort=False):
            # Convert to dict efficiently - frames are already in temporal order
            frames = video_df[['frame_number', 'rgb_path', 'depth_path', 
                               'intrinsics', 'extrinsics']].to_dict('records')
            video_frames[video_name] = frames
        
        return video_frames

    def _create_frame_list(self):
        """Create a flat list of all frames with video context."""
        # Use list comprehension for faster construction
        frame_list = [
            {
                'video_name': video_name,
                'frame_idx_in_video': i,
                'frame_data': frame_data
            }
            for video_name, frames in self.video_frames.items()
            for i, frame_data in enumerate(frames)
        ]
        return frame_list

    def __len__(self):
        return len(self.frame_list)
    
    def _can_sample_sequence(self, video_name, start_idx, stride, num_frames=None):

        if num_frames is None:
            num_frames = self.T

        """Check if we can sample a valid sequence from this starting point."""
        frames = self.video_frames[video_name]
        if start_idx + (num_frames - 1) * stride >= len(frames):
            return False
        
        # Check if frame numbers are consecutive with the given stride
        # NOTE valid for this dataset, cause I use timestamp as frame number
        start_frame_num = frames[start_idx]['frame_number']
        end_frame_num = frames[start_idx + (num_frames - 1) * stride]['frame_number']
        if end_frame_num - start_frame_num > 3000: # NOTE just quickly test, if timestamp difference is too large, then it's not a continuous sequence
            return False
        # for i in range(num_frames):
        #     frame_idx = start_idx + i * stride
        #     expected_frame_num = start_frame_num + i * stride
        #     actual_frame_num = frames[frame_idx]['frame_number']
        #     if actual_frame_num != expected_frame_num:
        #         return False
        return True

    def _load_single_frame(self, frame_data):
        """Load a single frame's RGB, depth, and intrinsics for Spring."""
        try:
            # Load RGB
            rgb_data = download_file(frame_data['rgb_path'])
            rgb_image = Image.open(io.BytesIO(rgb_data)).convert('RGB')
            del rgb_data  # Free memory immediately
            
            # Load depth
            depth_data = download_file(frame_data['depth_path'])
            depth_buffer = np.frombuffer(depth_data, np.uint8)
            del depth_data  # Free memory immediately
            
            depth_array = cv2.imdecode(depth_buffer, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
            del depth_buffer  # Free memory immediately
            
            depth_array[~np.isfinite(depth_array)] = 0  # invalid
            depth_image = Image.fromarray(depth_array)
            del depth_array  # Free memory immediately

            intrinsics = np.array(frame_data['intrinsics']).reshape(3, 3).copy()
            extrinsics = np.array(frame_data['extrinsics']).reshape(4, 4).copy()
            
            
            return rgb_image, depth_image, intrinsics, extrinsics, True
            
        except Exception as e:
            print(f"[arkitscenes] Error loading frame: {str(e)}")
            return None, None, None, None, False

    def __getitem__(self, idx):
        try:
            # ------------------------------------------------------------------
            # Flexible `idx` handling (support for sampling custom sequence
            # lengths / multi-resolution à la other loaders)
            # ------------------------------------------------------------------
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

            # ------------------------------------------------------------------
            # Identify starting frame in the flat list
            # ------------------------------------------------------------------
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

            # print(f"ArkitScenes data, scene name: {video_name}, frame positions: {frame_positions}, {self.use_cut3r_frame_sampling}")


            # ------------------------------------------------------------------
            # Load sequence data
            # ------------------------------------------------------------------
            rgb_images: list = []
            depth_images: list = []
            intrinsics_list: list = []
            extrinsics_list: list = []
            valid_frames: list = []

            frame_indices = []

            for frame_idx in frame_positions:

                frame_data = frames[frame_idx]
                rgb_img, depth_img, intrinsics, extrinsics, valid = self._load_single_frame(frame_data)

                if valid:
                    rgb_images.append(rgb_img)
                    depth_images.append(depth_img)
                    intrinsics_list.append(intrinsics.copy())
                    extrinsics_list.append(extrinsics)
                    valid_frames.append(True)
                    frame_indices.append(frame_idx)
                else:
                    # Fallback – replicate previous valid frame or create dummy
                    if rgb_images:
                        rgb_images.append(rgb_images[-1])
                        depth_images.append(depth_images[-1])
                        intrinsics_list.append(intrinsics_list[-1].copy())
                        extrinsics_list.append(extrinsics_list[-1])
                    else:
                        h, w = self.frame_size
                        rgb_images.append(Image.new("RGB", (self.frame_size[1], self.frame_size[0])))
                        depth_images.append(Image.fromarray(np.zeros((h, w), dtype=np.float32)))
                        intrinsics_list.append(np.eye(3, dtype=np.float32))
                        extrinsics_list.append(np.eye(4, dtype=np.float32))
                    valid_frames.append(False)
                    frame_indices.append(frame_indices[-1])

            # ------------------------------------------------------------------
            # Apply spatial transforms (resize / flip / etc.)
            # ------------------------------------------------------------------
            if self.transform is not None:
                if isinstance(self.transform, SynchronizedTransformVideoCut3r):
                    target_resolution = (
                        self.resolutions[resolution_idx]
                        if (resolution_idx is not None and self.resolutions is not None)
                        else (self.frame_size[1], self.frame_size[0])
                    )
                    rgb_tensors, depth_tensors, intrinsics_tensors = self.transform(
                        rgb_images, depth_images, intrinsics_list, resolution=target_resolution
                    )
                elif isinstance(self.transform, SynchronizedTransform_MoGe):
                    processed_data = self.transform(
                        rgb_images, depth_images, intrinsics_list, extrinsics_list, valid_frames, self.near_plane, self.far_plane, self.depth_type, resolution_idx, stride=None, rng=rng
                    )
                    processed_data['is_metric_scale'] = self.is_metric_scale
                    return processed_data # NOTE for moge style, we return the processed data directly
                else:
                    rgb_tensors, depth_tensors, intrinsics_tensors = self.transform(
                        rgb_images, depth_images, intrinsics_list
                    )
            else:
                to_tensor = transforms.ToTensor()
                rgb_tensors = [to_tensor(img) for img in rgb_images]
                depth_tensors = [to_tensor(img) for img in depth_images]
                intrinsics_tensors = [torch.tensor(K) for K in intrinsics_list]

            # ------------------------------------------------------------------
            # Per-frame numerical processing (normalisation / masks)
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

                # Compute validity mask
                valid_depth_mask = (depth_tensor > self.near_plane) & (depth_tensor < self.far_plane)
                inf_depth_mask = depth_tensor >= self.far_plane
                # RGB to [-1,1]
                rgb_tensor = rgb_tensor * 2.0 - 1.0

                # Clamp depth (robust to outliers) similar to other loaders
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
                processed_extrinsics.append(extrinsics_tensor)

            # ------------------------------------------------------------------
            # Stack into tensors with shape [T, ...]
            # ------------------------------------------------------------------
            rgb_sequence = torch.stack(processed_rgb, dim=0)
            metric_depth_sequence = torch.stack(processed_metric_depth, dim=0)
            valid_mask_sequence = torch.stack(processed_valid_mask, dim=0)
            inf_mask_sequence = torch.stack(processed_inf_mask, dim=0)
            intrinsics_sequence = torch.stack(processed_intrinsics, dim=0)
            extrinsics_sequence = torch.stack(processed_extrinsics, dim=0)

            # print(f"Process {video_name} frame indices: {frame_indices}")

            frame_idx_str = "_".join([str(idx) for idx in frame_indices])
            sample_name = f"{video_name}_{frame_idx_str}"

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
                # "sample_name": sample_name,
            }

        except Exception as e:
            # ------------------------------------------------------------------
            # If anything goes wrong, return dummy data so that training can
            # continue without crashing (mirrors behaviour of other loaders)
            # ------------------------------------------------------------------
            print(f"[arkitscenes_video] Error processing frame: {str(e)}")
            if num_frames is None:
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
# Parquet generation – this is the ONLY fully fleshed-out function for now.
# -----------------------------------------------------------------------------

def generate_parquet_file(
    raw_root: str = "/mnt/localssd/arkitscenes/raw",
    pairs_root: str = "./training/dataloaders/datasets/preprocess/arkitscenes_pairs",
    processed_root: str = "/mnt/localssd/arkitscenes/processed",
):
    """Create a parquet index for the processed ARKitScenes dataset.

    The *preprocess_arkitscenes_new.py* script converts the raw ARKitScenes data into
    a directory structure such as::

        processed_root/
            Training/
                <scene_id>/
                    vga_wide/       *.jpg   (RGB images)
                    lowres_depth/   *.png   (metric depth images – 16-bit PNG)
                    scene_metadata.npz       (trajectories, intrinsics, ...)

    This function scans the ``processed_root`` directory, loads each
    ``scene_metadata.npz`` file and constructs a *flat* dataframe where each row
    corresponds to an individual frame with the following columns:

        rgb_path      – path to RGB image (jpg)
        depth_path    – path to depth image (png)
        intrinsics    – Flattened 3x3 intrinsic matrix (list[float] length 9)
        extrinsics    – Flattened 4x4 world matrix  (list[float] length 16)
        video_name    – Scene identifier (string)
        frame_number  – Frame index within the scene (int)

    The dataframe is written to ``arkitscenes_train.parquet`` *locally* and then
    uploaded to the location specified by ``DATASET_CONFIG["arkitscenes"]``.
    """

    import tqdm

    DATA_DIR = processed_root  # local processed directory created by the pre-processing script

    # Resolve the data prefix
    data_path = DATASET_CONFIG["arkitscenes"]["prefix"]

    # Create empty dataframe with the expected schema
    df_columns = ["rgb_path", "depth_path", "intrinsics", "extrinsics", "video_name", "frame_number"]
    records = []

    # Walk through both the *Training* and *Test* splits if they exist.
    # Some users might only have *Training* locally – that is fine.
    # splits = [d for d in ["Training", "Test"] if os.path.isdir(os.path.join(DATA_DIR, d))]
    splits = [d for d in ["Training"] if os.path.isdir(os.path.join(DATA_DIR, d))]

    for split in splits:
        split_dir = os.path.join(DATA_DIR, split)

        # Each folder in *split_dir* represents one video / scene
        scene_dirs = [s for s in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, s))]

        for scene in tqdm.tqdm(scene_dirs, desc=f"[{split}] building parquet ►"):
            scene_dir = os.path.join(split_dir, scene)
            metadata_file = os.path.join(scene_dir, "scene_metadata.npz")

            if not os.path.isfile(metadata_file):
                # Skip scenes that have not been processed correctly
                print(f"[arkitscenes_parquet] Warning: metadata file missing for scene {scene} – skipped.")
                continue

            try:
                with np.load(metadata_file, allow_pickle=True) as meta:
                    images_arr = meta["images"]              # shape (N,) – bytes/str
                    intrinsics_arr = meta["intrinsics"]      # shape (N, 6)
                    trajectories_arr = meta["trajectories"]  # shape (N, 4, 4)
            except Exception as e:
                print(f"[arkitscenes_parquet] Could not load {metadata_file}: {e}")
                continue

            num_frames = images_arr.shape[0]

            # Sanity check
            if not (len(intrinsics_arr) == len(trajectories_arr) == num_frames):
                print(f"[arkitscenes_parquet] Inconsistent metadata lengths for scene {scene} – skipped.")
                continue

            # Build paths once to avoid string allocations inside the tight loop
            rgb_root = os.path.join(scene_dir, "vga_wide")
            depth_root = os.path.join(scene_dir, "lowres_depth")

            for frame_idx in range(num_frames):
                basename_val = images_arr[frame_idx]
                if isinstance(basename_val, bytes):
                    basename = basename_val.decode("utf-8")
                else:
                    basename = str(basename_val)

                if basename.endswith(".png"):
                    rgb_filename = basename.replace(".png", ".jpg")
                else:
                    # Fall-back for unexpected extension
                    rgb_filename = basename

                rgb_local_path = os.path.join(rgb_root, rgb_filename)
                depth_local_path = os.path.join(depth_root, basename)

                # Ensure files actually exist – the preprocessing script should have produced them
                if not (os.path.isfile(rgb_local_path) and os.path.isfile(depth_local_path)):
                    # Skip missing data – occasionally preprocessing could have failed for a frame
                    print(f"[arkitscenes_parquet] Warning: missing data for frame {frame_idx} in scene {scene} – skipped.")
                    continue

                # Build camera intrinsics (3×3) from the 6-tuple (w,h,fx,fy,cx,cy)
                _, _, fx, fy, cx, cy = intrinsics_arr[frame_idx]
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

                # Flatten matrices to python lists so that parquet stores them cleanly
                intrinsics_flat = K.flatten().tolist()
                extrinsics_flat = trajectories_arr[frame_idx].astype(np.float32).flatten().tolist()

                records.append({
                    "rgb_path": rgb_local_path.replace(DATA_DIR, data_path),
                    "depth_path": depth_local_path.replace(DATA_DIR, data_path),
                    "intrinsics": intrinsics_flat,
                    "extrinsics": extrinsics_flat,
                    "video_name": scene,
                    "frame_number": frame_idx
                })

    # Convert the accumulated records into a DataFrame
    df_new = pd.DataFrame.from_records(records, columns=df_columns)

    # Persist dataframe locally
    parquet_filename = "arkitscenes_train.parquet"
    df_new.to_parquet(parquet_filename)

    print(f"[arkitscenes_parquet] Completed – {len(df_new)} frames indexed. Parquet written to {parquet_filename}")

    # Save – we wrap in try/except so local run does not crash if credentials are missing
    try:
        upload_file(parquet_filename, DATASET_CONFIG["arkitscenes"]["parquet_path"])
        print(f"[arkitscenes_parquet] Saved parquet file")
    except Exception as e:
        print(f"[arkitscenes_parquet] Upload failed: {e}")

    # Optionally remove the local parquet file to save space (comment out if you want to keep it)
    # os.remove(parquet_filename)


# -----------------------------------------------------------------------------
# CLI helper – run `python depth_waymo.py` to trigger parquet generation.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import tqdm
    import moviepy.video.io.ffmpeg_writer as video_writer
    import cv2
    from Unip.unip.util.pointmap_util import depth_to_pointmap, save_pointmap_as_ply
    import moviepy.video.io.ffmpeg_writer as video_writer 
    from training.dataloaders.batched_sampler import make_sampler
    # from training.dataloaders.dynamic_batched_sampler import make_dynamic_sampler
    from torch.utils.data import DataLoader
    import open3d as o3d


    import utils3d

    from third_party.pi3.utils.geometry import homogenize_points, se3_inverse, depth_edge
    from third_party.pi3.utils.alignment import align_points_scale
    
    # os.makedirs("data_vis", exist_ok=True)
    # print("\nTesting ARKitScenes video dataset with new sampling...")

    dataset_name = "arkitscenes_new"
    save_dir = f"data_vis/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    resolutions = [[0,1]] * 10000
    video_dataset = VideoDepthArkitScenesNew(
        T=10, 
        stride_range=(1, 8),
        transform=True,
        resolutions=resolutions, 
        use_moge=True,
        use_cut3r_frame_sampling=True,
        video_prob=0.6,
        fix_interval_prob=0.6,
        block_shuffle=16,
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
            area_range=[100000, 255000],
            aspect_ratio_range=[0.5, 2.0],
        )
    )
    print(f"Video dataset length: {len(video_dataset)}")


    dataloader_sampler = make_sampler(
        video_dataset, 
        batch_size=1, 
        number_of_resolutions=len(resolutions),
        min_num_frames=10, 
        max_num_frames=10, 
        shuffle=True, 
        drop_last=True
    )


    dataloader = DataLoader(
        video_dataset,
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
    # if (
    #     hasattr(dataloader, "batch_sampler")
    #     and hasattr(dataloader.batch_sampler, "sampler")
    #     and hasattr(dataloader.batch_sampler.sampler, "set_epoch")
    # ):
    #     dataloader.batch_sampler.sampler.set_epoch(0)
    

    print(f"Video dataset length: {len(video_dataset)}")
    
    if len(video_dataset) > 0:
        sample_count = 0

        for idx, video_sample in enumerate(dataloader):
            # video_sample = video_dataset[idx]


            sample_count +=1
            if sample_count > 20: break

            if isinstance(video_sample['rgb'], list):
                # Handle case where video_sample might be a list of batches
                video_sample = {k: v[0] if isinstance(v, list) else v for k, v in video_sample.items()}
            else:
                # Extract first item from batch dimension
                video_sample = {k: v[0] if isinstance(v, torch.Tensor) and v.dim() > 3 else v for k, v in video_sample.items()}

            print(f"Video sample shapes:")
            print(f"  RGB: {video_sample['rgb'].shape}")  # Should be [T, 3, H, W]
            print(f"  Metric depth: {video_sample['metric_depth'].shape}")  # Should be [T, 1, H, W]
            print(f"  Valid mask: {video_sample['valid_mask'].shape}")  # Should be [T, 1, H, W]
            print(f"  Intrinsics: {video_sample['intrinsics'].shape}")  # Should be [T, 3, 3]
            # print(f"  Extrinsics: {video_sample['extrinsics'].shape}")  # Should be [T, 3, 3]
            print(f"  Valid: {video_sample['valid']}")
            # print(f"  Stride: {video_sample['stride']}")
            
            # Prepare video writers
            H, W = video_sample['rgb'].shape[2], video_sample['rgb'].shape[3]
            
            rgb_writer = video_writer.FFMPEG_VideoWriter(filename=f"{save_dir}/ideo_rgb_{idx}.mp4", size=(W, H), fps=2)
            depth_writer = video_writer.FFMPEG_VideoWriter(filename=f"{save_dir}/video_depth_{idx}.mp4", size=(W, H), fps=2)
            mask_writer = video_writer.FFMPEG_VideoWriter(filename=f"{save_dir}/video_mask_{idx}.mp4", size=(W, H), fps=2)

            # prev_depth_video = load_depth_video(f"data_vis/tartanair_video_depth_{idx}.mp4")
            all_gt_pts = []
            all_gt_pts_rgb = []
            print("Saving video frames...")
            for t in tqdm.tqdm(range(video_sample['rgb'].shape[0])):
                # Process RGB frame
                rgb_frame = (video_sample['rgb'][t].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                
                # Process depth frame
                depth = video_sample['metric_depth'][t].squeeze()
                mask = video_sample['valid_mask'][t].squeeze()

                intrinsics = video_sample['intrinsics'][t]
                gt_pts = utils3d.torch.depth_to_points(depth, intrinsics=intrinsics)

                poses = video_sample['extrinsics'][t]
                # print(f"poses: {poses}")

                # w2c_target = se3_inverse(poses[:, 0])
                # poses = torch.einsum('bik, bnkj -> bnij', w2c_target, poses) # to camera 0
                # breakpoint()
                gt_pts = torch.einsum('ij, hwj -> hwi', poses, homogenize_points(gt_pts))[..., :3]

                gt_pts = gt_pts[::4,::4][mask[::4,::4]].reshape(-1, 3)
                gt_pts_rgb = video_sample['rgb'][t].permute(1, 2, 0)[::4,::4][mask[::4,::4]].reshape(-1,3)

                depth = depth.numpy()
                mask = mask.numpy()

                depth_max = depth[mask].max()
                depth_min = depth[mask].min()

                depth_normalized = depth.copy()
                depth_normalized[~mask] = depth_min
                depth_normalized = (depth_normalized - depth_min) / (depth_max - depth_min) * 255

                depth_normalized = depth_normalized.astype(np.uint8)

                depth_bgr = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)

                valid_mask = (video_sample['valid_mask'][t].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                mask_bgr = cv2.cvtColor(valid_mask.squeeze(), cv2.COLOR_GRAY2BGR)
                

                rgb_writer.write_frame(rgb_frame)
                depth_writer.write_frame(depth_bgr)
                mask_writer.write_frame(mask_bgr)

                all_gt_pts.append(gt_pts)
                all_gt_pts_rgb.append(gt_pts_rgb)
                
            rgb_writer.close()
            depth_writer.close()
            mask_writer.close()

            all_gt_pts = torch.cat(all_gt_pts, dim=0)
            all_gt_pts_rgb = torch.cat(all_gt_pts_rgb, dim=0)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                all_gt_pts
            )
            pcd.colors = o3d.utility.Vector3dVector(
                all_gt_pts_rgb
            )
            save_path = f"{save_dir}/global_pts_{idx}.ply"
            o3d.io.write_point_cloud(
                save_path,
                pcd,
            )

            print(f"Saved pointcloud to {save_path}")
            
            print("Video sequence test completed successfully!")
    else:
        print("No valid video sequences found in dataset.")