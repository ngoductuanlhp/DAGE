import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import io
import glob
import pandas as pd
from torch.utils.data import Dataset
from training.dataloaders.data_io import download_file, upload_file
from training.dataloaders.config import DATASET_CONFIG

import time

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

# NOTE just dummy value
OG_WAYMO_INTRINSICS = np.array([
    [320, 0, 320],
    [0, 320, 240],
    [0, 0, 1]
])

class SynchronizedTransform_WaymoVideo:
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


class VideoDepthWaymoNew(Dataset):
    """Skeleton video dataset loader for Waymo depth data.

    NOTE: The per-frame loading logic is left as placeholders – you should
    replace those with real parsing of the `*.jpg`, `*.exr`, and `*.npz`
    files found inside each Waymo segment directory.
    """
    depth_type = "lidar"
    is_metric_scale = True
    # NOTE original resolution is 512x341

    def __init__(
        self,
        T: int = 5,
        stride_range: tuple = (1, 3),
        transform: bool = True,
        resize: tuple = (320, 480),
        near_plane: float = 1e-5,
        far_plane: float = 80.0,
        resolutions=None,
        use_moge=False,
        moge_augmentation=None,
        use_cut3r_frame_sampling=False, video_prob=0.6, fix_interval_prob=0.6, block_shuffle=16
    ):
        # Lazily load the parquet only if Waymo key exists in config
        parquet_data = download_file(
            
            DATASET_CONFIG["waymo"]["parquet_path"],
        )
        self.parquet = io.BytesIO(parquet_data)

        self.T = T
        self.stride_range = stride_range
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.frame_size = resize
        self.resolutions = resolutions
        self.use_moge = use_moge
        self.moge_augmentation = moge_augmentation  # NOTE for moge style, we need to pass the augmentation type

        # New sophisticated sampling parameters
        self.use_cut3r_frame_sampling = use_cut3r_frame_sampling
        self.video_prob = video_prob
        self.fix_interval_prob = fix_interval_prob
        self.block_shuffle = block_shuffle

        # Organize parquet rows by video name
        self.video_frames = self._organize_by_video()
        self.frame_list = self._create_frame_list()


        if self.use_moge:
            self.transform = SynchronizedTransform_MoGe(moge_augmentation=self.moge_augmentation)
        elif self.resolutions is not None:
            self.transform = SynchronizedTransformVideoCut3r(seed=777)
        else:
            self.transform = SynchronizedTransform_WaymoVideo(H=resize[0], W=resize[1]) if transform else None

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
    
    def _can_sample_sequence(self, video_name, start_idx, stride, num_frames=None):
        """Check if we can sample a valid sequence from this starting point."""

        if num_frames is None:
            num_frames = self.T

        frames = self.video_frames[video_name]
        if start_idx + (num_frames - 1) * stride >= len(frames):
            return False
        
        # Check if frame numbers are consecutive with the given stride
        start_frame_num = frames[start_idx]['frame_number']
        for i in range(num_frames):
            frame_idx = start_idx + i * stride
            expected_frame_num = start_frame_num + i * stride
            actual_frame_num = frames[frame_idx]['frame_number']
            if actual_frame_num != expected_frame_num:
                return False
        return True

    # ------------------------------------------------------------------
    # PLACEHOLDER single-frame loader – FILL THIS IN!
    # ------------------------------------------------------------------
    def _load_single_frame(self, frame_data):
        """Load RGB, depth, intrinsics, extrinsics for one frame.

        TODO: Replace the dummy implementation below with real file I/O that
        reads the `.jpg`, `.exr`, and `.npz` files from local storage
        for the Waymo dataset.
        """
        try:
            # Load RGB
            rgb_data = download_file(frame_data['rgb_path'])
            rgb_image = Image.open(io.BytesIO(rgb_data)).convert('RGB')
            del rgb_data  # Free memory immediately
            
            # Load depth
            depth_data = download_file(frame_data['depth_path'])
            depth_buffer = np.frombuffer(depth_data, np.uint8)
            del depth_data  # Free memory immediately
            
            # Use OpenCV to read .exr file
            depth_array = cv2.imdecode(depth_buffer, cv2.IMREAD_UNCHANGED)
            del depth_buffer  # Free memory immediately
            
            depth_image = Image.fromarray(depth_array)
            del depth_array  # Free memory immediately
            
            # Get intrinsics
            intrinsics = np.array(frame_data['intrinsics']).reshape(3, 3).copy()
            
            # Get extrinsics
            # extrinsics = np.array(frame_data['extrinsics']).reshape(4, 4)[:3, :]
            extrinsics = np.array(frame_data['extrinsics']).reshape(4, 4).copy()

            
            return rgb_image, depth_image, intrinsics, extrinsics, True
            
        except Exception as e:
            print(f"[waymo_video] Error loading frame: {str(e)}")
            return None, None, None, None, False

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
                    raise ValueError(f"Invalid tuple length for idx: {len_tuple_idx}")
            else:
                len_tuple_idx = 1
                num_frames = self.T
                resolution_idx = None

            rng = np.random.default_rng(random.randint(0, 2 ** 32 - 1))

            # Get the starting frame
            start_frame_info = self.frame_list[idx]
            video_name = start_frame_info['video_name']
            start_idx = start_frame_info['frame_idx_in_video']
            
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

            # print(f"Waymo data, scene name: {video_name}, frame positions: {frame_positions}, {self.use_cut3r_frame_sampling}")
            
            # Collect frame data
            rgb_images = []
            depth_images = []
            intrinsics_list = []
            extrinsics_list = []
            valid_frames = []
            
            for frame_idx in frame_positions:
                
                frame_data = frames[frame_idx]
                rgb_img, depth_img, intrinsics, extrinsics, valid = self._load_single_frame(frame_data)
                
                if valid:
                    rgb_images.append(rgb_img)
                    depth_images.append(depth_img)
                    intrinsics_list.append(intrinsics.copy())
                    extrinsics_list.append(extrinsics)
                    valid_frames.append(True)
                else:
                    # Create dummy data for failed frames
                    if rgb_images:  # Use previous frame as fallback
                        rgb_images.append(rgb_images[-1])
                        depth_images.append(depth_images[-1])
                        intrinsics_list.append(intrinsics_list[-1].copy())
                        extrinsics_list.append(extrinsics_list[-1])
                    else:
                        # Create zero tensors if no valid frames yet
                        rgb_images.append(Image.new('RGB', (512, 384)))
                        depth_images.append(Image.fromarray(np.zeros((384, 512), dtype=np.float32)))
                        intrinsics_list.append(OG_WAYMO_INTRINSICS.copy())
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
                        rgb_images, depth_images, intrinsics_list, resolution=target_resolution
                    )
                elif isinstance(self.transform, SynchronizedTransform_MoGe):
                    processed_data = self.transform(
                        rgb_images, depth_images, intrinsics_list, extrinsics_list, valid_frames, self.near_plane, self.far_plane, self.depth_type, resolution_idx, stride=None, no_depth_mask_inf=True, rng=rng
                    )
                    processed_data['is_metric_scale'] = self.is_metric_scale
                    return processed_data # NOTE for moge style, we return the processed data directly
                else:
                    rgb_tensors, depth_tensors, intrinsics_tensors = self.transform(
                        rgb_images, depth_images, intrinsics_list
                    )
            else:
                rgb_tensors = [transforms.ToTensor()(img) for img in rgb_images]
                depth_tensors = [transforms.ToTensor()(img) for img in depth_images]
                intrinsics_tensors = [torch.tensor(intrinsics) for intrinsics in intrinsics_list]

            # Process each frame
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

                # Get valid depth mask
                valid_depth_mask = (depth_tensor > self.near_plane) & (depth_tensor < self.far_plane)
                inf_depth_mask = depth_tensor >= self.far_plane
                # Process RGB 
                rgb_tensor = rgb_tensor * 2.0 - 1.0  # [-1,1]

                # Process depth
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
                processed_extrinsics.append(extrinsics_tensor)  # Only rotation part

            # Stack into [T, C, H, W] tensors
            rgb_sequence = torch.stack(processed_rgb, dim=0)  # [T, 3, H, W]
            metric_depth_sequence = torch.stack(processed_metric_depth, dim=0)  # [T, 1, H, W]
            valid_mask_sequence = torch.stack(processed_valid_mask, dim=0)  # [T, 1, H, W]
            inf_mask_sequence = torch.stack(processed_inf_mask, dim=0)  # [T, 1, H, W]
            intrinsics_sequence = torch.stack(processed_intrinsics, dim=0)  # [T, 3, 3]
            extrinsics_sequence = torch.stack(processed_extrinsics, dim=0)  # [T, 3, 3]

            return {
                'rgb': rgb_sequence,
                'metric_depth': metric_depth_sequence,
                'valid_mask': valid_mask_sequence,
                'inf_mask': inf_mask_sequence,
                'intrinsics': intrinsics_sequence,
                'extrinsics': extrinsics_sequence,
                'frame_positions': frame_positions,  # New: actual frame indices used
                'is_video': is_video,  # New: whether sequence is video-like
                # 'stride': stride,
                'valid': all(valid_frames),
                'depth_type': self.depth_type,
            }
        except Exception as e:
            print(f"[waymo_video] Error processing frame: {str(e)}")
            # Return dummy data with valid=False
            if num_frames is None:
                num_frames = self.T

            dummy_rgb = torch.zeros(num_frames, 3, self.frame_size[0], self.frame_size[1])  # Assuming default resize
            dummy_depth = torch.zeros(num_frames, 1, self.frame_size[0], self.frame_size[1])
            dummy_mask = torch.zeros(num_frames, 1, self.frame_size[0], self.frame_size[1], dtype=torch.bool)
            dummy_intrinsics = torch.eye(3).unsqueeze(0).repeat(num_frames, 1, 1)
            dummy_extrinsics = torch.eye(3).unsqueeze(0).repeat(num_frames, 1, 1)
            
            return {
                'rgb': dummy_rgb,
                'metric_depth': dummy_depth,
                'valid_mask': dummy_mask,
                'intrinsics': dummy_intrinsics,
                'extrinsics': dummy_extrinsics,
                # 'stride': stride,
                'valid': False,
                'depth_type': self.depth_type,
            }


# -----------------------------------------------------------------------------
# Parquet generation – this is the ONLY fully fleshed-out function for now.
# -----------------------------------------------------------------------------

def generate_parquet_file():

    def load_invalid_dict(h5_file_path):
        invalid_dict = {}
        with h5py.File(h5_file_path, "r") as h5f:
            for scene in h5f:
                data = h5f[scene]["invalid_pairs"][:]
                invalid_pairs = set(
                    tuple(pair.decode("utf-8").split("_")) for pair in data
                )
                invalid_dict[scene] = invalid_pairs
        return invalid_dict

    """Scan the locally processed Waymo directory and build a parquet file.

    The produced parquet follows the convention used by other datasets in this
    repo (TartanAir, Point Odyssey, etc.):
        * rgb_path      – path to the RGB `.jpg`
        * depth_path    – path to the depth `.exr`
        * intrinsics    – flattened 3×3 matrix (list of 9 floats)
        * extrinsics    – flattened 4×4 matrix (list of 16 floats)
        * video_name    – unique identifier for a Waymo segment (e.g. segment-…)
        * frame_number  – integer frame index within the segment

    Only the directory traversal and DataFrame construction are implemented
    here – the per-file parsing is left to you.
    """
    from tqdm import tqdm  # Local import avoids extra dependency at module scope

    invalid_dict = load_invalid_dict(
        "training/dataloaders/datasets/video_datasets/invalid_files.h5"
    )

    # ------------------------------------------------------------------
    # CONFIGURATION – adjust these paths / keys to match your setup
    # ------------------------------------------------------------------
    DATA_DIR = "/mnt/localssd/waymo_data_processed"  # Local Waymo root
    data_prefix = DATASET_CONFIG["waymo"]["prefix"]  # location prefix

    # ------------------------------------------------------------------
    # DataFrame we will accumulate rows into
    # ------------------------------------------------------------------
    df_cols = [
        "rgb_path",
        "depth_path",
        "intrinsics",
        "extrinsics",
        "video_name",
        "frame_number",
    ]
    df_new = pd.DataFrame(columns=df_cols)

    # ------------------------------------------------------------------
    # Iterate over every segment directory
    # ------------------------------------------------------------------
    segments = [
        s
        for s in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, s))
    ]

    for segment in tqdm(segments, desc="Waymo segments"):
        segment_dir = os.path.join(DATA_DIR, segment)

        invalid_pairs = invalid_dict.get(segment, set())

        # Gather all RGB files inside this segment (pattern: ######_#.jpg)
        rgb_files = sorted(glob.glob(os.path.join(segment_dir, "*.jpg")))

        # TODO: You may wish to filter by camera ID here.

        for rgb_file in rgb_files:
            base = os.path.basename(rgb_file)
            frame_id_str, cam_id_ext = base.split("_", 1)
            frame_number = int(frame_id_str)
            cam_id, _ = os.path.splitext(cam_id_ext)
            cam_id = int(cam_id)  # noqa: F841 – currently unused

            if cam_id == 5:
                continue
            if (cam_id, frame_number) in invalid_pairs:
                print(f"Skipping invalid file: {cam_id}, {frame_number}, {segment}")
                continue  # Skip invalid files


            # Infer companion depth & metadata paths
            depth_file = rgb_file.replace(".jpg", ".exr")
            meta_file = rgb_file.replace(".jpg", ".npz")

            # Safeguard: skip if any file is missing
            if not (os.path.exists(depth_file) and os.path.exists(meta_file)):
                print(f"Skipping missing file: {cam_id}, {frame_number}, {segment}")
                continue

            # ------------------------------------------------------------------
            # PLACEHOLDER metadata extraction – replace with actual parsing
            # ------------------------------------------------------------------
            # At minimum you should load the npz and extract:
            #   * intrinsics (3×3)
            #   * cam2world  (4×4)
            # The snippet below loads the file and assumes keys named exactly
            # as in the sample inspection ("intrinsics", "cam2world") – please
            # adjust if your files differ.
            try:
                # meta_npz = np.load(meta_file)
                # intrinsics = meta_npz.get("intrinsics", np.eye(3))
                # extrinsics = meta_npz.get("cam2world", np.eye(4))
                camera_params = np.load(meta_file)

                intrinsics = np.float32(camera_params["intrinsics"])
                extrinsics = np.float32(camera_params["cam2world"])
                # breakpoint()
            except Exception:
                # Fallback to identity matrices if the npz cannot be read
                intrinsics = np.eye(3)
                extrinsics = np.eye(4)


            # ------------------------------------------------------------------
            # Convert local paths to paths
            # ------------------------------------------------------------------
            rgb_full_path = rgb_file.replace(DATA_DIR, data_prefix)
            depth_full_path = depth_file.replace(DATA_DIR, data_prefix)

            # breakpoint()

            # Append one row to dataframe
            df_new = pd.concat(
                [
                    df_new,
                    pd.DataFrame(
                        {
                            "rgb_path": [rgb_full_path],
                            "depth_path": [depth_full_path],
                            "intrinsics": [intrinsics.flatten().tolist()],
                            "extrinsics": [extrinsics.flatten().tolist()],
                            "video_name": [f"{segment}_{cam_id}"],
                            "frame_number": [frame_number],
                        }
                    ),
                ],
                ignore_index=True,
            )

    # ------------------------------------------------------------------
    # Write parquet locally, save, and clean up
    # ------------------------------------------------------------------
    parquet_fname = "waymo_train.parquet"
    df_new.to_parquet(parquet_fname)
    print(f"Waymo parquet written with {len(df_new)} rows → {parquet_fname}")

    upload_file(
        parquet_fname,
        DATASET_CONFIG["waymo"]["parquet_path"],
    )
    os.remove(parquet_fname)


# -----------------------------------------------------------------------------
# CLI helper – run `python depth_waymo.py` to trigger parquet generation.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # To avoid accidental long-running jobs, we ask for confirmation.
    # generate_parquet_file()

    # Test video dataset:
    import tqdm
    import cv2
    from Unip.unip.util.pointmap_util import depth_to_pointmap, save_pointmap_as_ply
    import moviepy.video.io.ffmpeg_writer as video_writer
    
    os.makedirs("data_vis", exist_ok=True)
    print("\nTesting video dataset...")
    video_dataset = VideoDepthWaymoNew(T=10, stride_range=(4, 5), transform=True)
    print(f"Video dataset length: {len(video_dataset)}")

    save_dir = "data_vis/waymo"
    os.makedirs(save_dir, exist_ok=True)
    
    if len(video_dataset) > 0:

        for i in np.linspace(0, len(video_dataset)-1000, 10, dtype=int):
            video_sample = video_dataset[i]
            print(f"Video sample shapes:")
            print(f"  RGB: {video_sample['rgb'].shape}")  # Should be [T, 3, H, W]
            print(f"  Metric depth: {video_sample['metric_depth'].shape}")  # Should be [T, 1, H, W]
            print(f"  Valid mask: {video_sample['valid_mask'].shape}")  # Should be [T, 1, H, W]
            print(f"  Intrinsics: {video_sample['intrinsics'].shape}")  # Should be [T, 3, 3]
            print(f"  Extrinsics: {video_sample['extrinsics'].shape}")  # Should be [T, 3, 3]
            print(f"  Valid: {video_sample['valid']}")
            
            # Prepare video writers
            H, W = video_sample['rgb'].shape[2], video_sample['rgb'].shape[3]
            
            rgb_writer = video_writer.FFMPEG_VideoWriter(filename=f"{save_dir}/waymo_video_rgb_{i}.mp4", size=(W, H), fps=2)
            # depth_writer = video_writer.FFMPEG_VideoWriter(filename=f"{save_dir}/waymo_video_depth_{i}.mp4", size=(W, H), fps=2)
            # mask_writer = video_writer.FFMPEG_VideoWriter(filename=f"{save_dir}/waymo_video_mask_{i}.mp4", size=(W, H), fps=2)
            
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
                # depth_writer.write_frame(depth_bgr)
                # mask_writer.write_frame(mask_bgr)
                
                # Save pointcloud for first frame only
                # if t == 0:
                #     intrinsics = video_sample['intrinsics'][t].numpy()
                #     pointmap = depth_to_pointmap(depth, intrinsics)
                #     save_pointmap_as_ply(pointmap, rgb_frame, f"{save_dir}/waymo_video_frame0_pointmap_{i}.ply", far_threshold=1000)

            rgb_writer.close()
            # depth_writer.close()
            # mask_writer.close()
            
            print("Video sequence test completed successfully!")
            print("Saved videos:")
            print(f"  - {save_dir}/waymo_video_rgb_{i}.mp4")
            print(f"  - {save_dir}/waymo_video_depth_{i}.mp4") 
            print(f"  - {save_dir}/waymo_video_mask_{i}.mp4")
            print(f"  - {save_dir}/waymo_video_frame0_pointmap_{i}.ply")
    else:
        print("No valid video sequences found in dataset.")

