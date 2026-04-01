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
from training.dataloaders.datasets.video_datasets_new.synchronized_video_transform import SynchronizedTransformVideoCut3r
from training.dataloaders.datasets.video_datasets_new.synchronized_video_transform_moge import SynchronizedTransform_MoGe

from training.dataloaders.datasets.video_datasets_new.frame_sampling_utils import get_seq_from_start_id

class SynchronizedTransform_MatrixCity:
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


class VideoDepthMatrixCityNew(Dataset):
    """Video dataset loader for MatrixCity depth data."""

    depth_type = "synthetic"

    def __init__(
        self,
        T: int = 5,
        stride_range: tuple = (1, 3),
        transform: bool = True,
        resize: tuple = (1000, 1000),
        near_plane: float = 1e-5,
        far_plane: float = 100.0,
        resolutions=None,
        use_moge=False,
        moge_augmentation=None,
        use_cut3r_frame_sampling=False, video_prob=0.6, fix_interval_prob=0.6, block_shuffle=16
    ):
        # Load the parquet data from local storage
        parquet_data = download_file(
            
            DATASET_CONFIG["matrixcity"]["parquet_path"],
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


        self.use_cut3r_frame_sampling = use_cut3r_frame_sampling
        # New sophisticated sampling parameters
        self.video_prob = video_prob
        self.fix_interval_prob = fix_interval_prob
        self.block_shuffle = block_shuffle
        
        # Initialize data structures
        self.video_frames = self._organize_by_video()
        self.frame_list = self._create_frame_list()

        if self.use_moge:
            self.transform = SynchronizedTransform_MoGe(moge_augmentation=self.moge_augmentation)
        elif self.resolutions is not None:
            self.transform = SynchronizedTransformVideoCut3r(seed=777)
        else:
            self.transform = SynchronizedTransform_MatrixCity(H=resize[0], W=resize[1]) if transform else None

    def _organize_by_video(self):
        """Organize frames by video name and sort by frame number."""
        df = pd.read_parquet(self.parquet)
        video_frames = defaultdict(list)
        
        for _, row in df.iterrows():
            video_name = row['video_name']
            frame_data = {
                'frame_number': row['frame_number'],
                'rgb_path': row['rgb_path'],
                'depth_path': row['depth_path'],
                'intrinsics': row['intrinsics'],
                'extrinsics': row['extrinsics']
            }
            video_frames[video_name].append(frame_data)
        
        # Sort frames by frame number for each video
        for video_name in video_frames:
            video_frames[video_name].sort(key=lambda x: x['frame_number'])
        
        return dict(video_frames)

    def _create_frame_list(self):
        """Create a flat list of all frames with video context."""
        frame_list = []
        for video_name, frames in self.video_frames.items():
            for i, frame_data in enumerate(frames):
                frame_list.append({
                    'video_name': video_name,
                    'frame_idx_in_video': i,
                    'frame_data': frame_data
                })
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

    def _load_single_frame(self, frame_data):
        def convert_rotation_to_opencv(R):
            # Transformation matrix from Unreal to OpenCV coordinate system.
            C = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            # R_cv = C @ R @ C.T
            R_cv = R @ C.T
            return R_cv
        """Load a single frame's RGB, depth, and intrinsics for MatrixCity."""
        try:
            # Load RGB
            rgb_data = download_file(frame_data['rgb_path'])
            rgb_image = Image.open(io.BytesIO(rgb_data)).convert('RGB')
            del rgb_data  # Free memory immediately
            
            # Load depth (EXR format)
            # breakpoint()
            depth_data = download_file(frame_data['depth_path'])
            depth_buffer = np.frombuffer(depth_data, np.uint8)
            del depth_data  # Free memory immediately
            
            # Use OpenCV to read .exr file
            depth_array = cv2.imdecode(depth_buffer, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[...,0]
            del depth_buffer  # Free memory immediately
            
            # depth_array[~np.isfinite(depth_array)] = 0  # invalid
            invalid_depth = (depth_array==65504) # NOTE sky value

            # print(f"invalid ratio: {invalid_depth.sum() / (depth_array.shape[0] * depth_array.shape[1])}")
            if invalid_depth.sum() > depth_array.shape[0] * depth_array.shape[1] * 0.3:
                return None, None, None, None, False # NOTE: if more than 30% of the depth is invalid, return False

            depth_array = depth_array / 100.0 # NOTE: cm to m
            depth_array[invalid_depth] = 0 # NOTE: this is the invalid depth

            depth_image = Image.fromarray(depth_array.astype(np.float32))
            del depth_array  # Free memory immediately

            # Get intrinsics and extrinsics from parquet data
            intrinsics = np.array(frame_data['intrinsics']).reshape(3, 3).copy()
            extrinsics = np.array(frame_data['extrinsics']).reshape(4, 4).copy()
            # breakpoint()

            extrinsics[:3, :3] = convert_rotation_to_opencv(extrinsics[:3, :3])
            extrinsics[:3, 3] = extrinsics[:3, 3] # NOTE: cm to m
            
            return rgb_image, depth_image, intrinsics, extrinsics, True
            
        except Exception as e:
            print(f"[matrixcity] Error loading frame: {str(e)}")
            return None, None, None, None, False

    def __getitem__(self, idx):
        try:
            # Handle flexible idx (tuple for different sequence lengths / resolutions)
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
                print("using cut3r frame sampling")
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

            # Collect frame data
            rgb_images = []
            depth_images = []
            intrinsics_list = []
            extrinsics_list = []
            valid_frames = []
            
            for frame_idx in frame_positions:

                frame_data = frames[frame_idx]
                rgb_img, depth_img, intrinsics, extrinsics, valid = self._load_single_frame(frame_data)
                # print(f"frame_idx: {frame_idx}, valid: {valid}, {extrinsics.shape}")

                if valid:
                    rgb_images.append(rgb_img)
                    depth_images.append(depth_img)
                    intrinsics_list.append(intrinsics.copy())
                    extrinsics_list.append(extrinsics)
                    valid_frames.append(True)
                else:
                    # Fallback – replicate previous valid frame or create dummy
                    # if rgb_images:
                    #     rgb_images.append(rgb_images[-1])
                    #     depth_images.append(depth_images[-1])
                    #     intrinsics_list.append(intrinsics_list[-1].copy())
                    #     extrinsics_list.append(extrinsics_list[-1])
                    # else:
                    #     h, w = self.frame_size
                    #     rgb_images.append(Image.new("RGB", (w, h)))
                    #     depth_images.append(Image.fromarray(np.zeros((h, w), dtype=np.float32)))
                    #     intrinsics_list.append(np.eye(3, dtype=np.float32))
                    #     extrinsics_list.append(np.eye(4, dtype=np.float32))
                    valid_frames.append(False)
                    break
            
            if not all(valid_frames):
                # for some parts, the camera is facing toward the sky, so we need to skip these frames
                next_idx = (idx + 500) % self.__len__()
                if len_tuple_idx == 3:
                    return self.__getitem__((next_idx, num_frames, resolution_idx))
                elif len_tuple_idx == 2:
                    return self.__getitem__((next_idx, num_frames))
                elif len_tuple_idx == 1:
                    return self.__getitem__(next_idx)
                else:
                    raise ValueError(f"Invalid tuple length: {len_tuple_idx}")

            # Apply spatial transforms
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
                        rgb_images, depth_images, intrinsics_list, extrinsics_list, valid_frames, 
                        self.near_plane, self.far_plane, self.depth_type, resolution_idx, stride=None, rng=rng
                    )
                    return processed_data
                else:
                    rgb_tensors, depth_tensors, intrinsics_tensors = self.transform(
                        rgb_images, depth_images, intrinsics_list
                    )
            else:
                to_tensor = transforms.ToTensor()
                rgb_tensors = [to_tensor(img) for img in rgb_images]
                depth_tensors = [to_tensor(img) for img in depth_images]
                intrinsics_tensors = [torch.tensor(K) for K in intrinsics_list]

            # Per-frame numerical processing
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

                # Clamp depth (robust to outliers)
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
                processed_extrinsics.append(extrinsics_tensor[:3, :3])

            # Stack into tensors with shape [T, ...]
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
                "is_video": is_video,
                # "stride": stride,
                "valid": all(valid_frames),
                "depth_type": self.depth_type,
            }

        except Exception as e:
            print(f"[matrixcity_video] Error processing frame: {str(e)}")
            if num_frames is None:
                num_frames = self.T

            dummy_rgb = torch.zeros(num_frames, 3, self.frame_size[0], self.frame_size[1])
            dummy_depth = torch.zeros(num_frames, 1, self.frame_size[0], self.frame_size[1])
            dummy_mask = torch.zeros(num_frames, 1, self.frame_size[0], self.frame_size[1], dtype=torch.bool)
            dummy_intrinsics = torch.eye(3).unsqueeze(0).repeat(num_frames, 1, 1)
            dummy_extrinsics = torch.eye(4).unsqueeze(0).repeat(num_frames, 1, 1)

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


def generate_parquet_file(
    processed_root: str = "/mnt/localssd/matrixcity_processed",
):
    """Create a parquet index for the processed MatrixCity dataset.

    The preprocessing script creates a directory structure like:
        processed_root/
            big_city/street/block_name/
                images/         *.png (RGB images)
                depths/         *.exr (depth images)
                camera_poses.npy    (camera poses for all frames)
    """
    import tqdm
    import json

    DATA_DIR = processed_root
    data_path = DATASET_CONFIG["matrixcity"]["prefix"]

    # Create empty dataframe with expected schema
    df_columns = ["rgb_path", "depth_path", "intrinsics", "extrinsics", "video_name", "frame_number"]
    records = []

    # Default intrinsics for MatrixCity (you may need to adjust these based on transforms.json)
    default_intrinsics = np.array([
        [500, 0, 500],
        [0, 500, 500], 
        [0, 0, 1]
    ], dtype=np.float32)

    for city in ["big_city", "small_city"]:
        city_dir = os.path.join(DATA_DIR, city)
        if not os.path.isdir(city_dir):
            continue
            
        for style in ["street"]:
            style_dir = os.path.join(city_dir, style)
            if not os.path.isdir(style_dir):
                continue
                
            # Get all block directories
            blocks = [b for b in os.listdir(style_dir) if os.path.isdir(os.path.join(style_dir, b))]
            
            for block in tqdm.tqdm(blocks, desc=f"[{city}/{style}] building parquet"):
                block_dir = os.path.join(style_dir, block)
                
                # Paths to data
                images_dir = os.path.join(block_dir, "images")
                depths_dir = os.path.join(block_dir, "depths")
                poses_file = os.path.join(block_dir, "camera_poses.npy")
                
                # Check if all required directories/files exist
                if not all(os.path.exists(p) for p in [images_dir, depths_dir, poses_file]):
                    print(f"[matrixcity_parquet] Warning: missing data for block {block} in {city}/{style} – skipped.")
                    continue
                
                try:
                    # Load camera poses
                    camera_poses = np.load(poses_file)  # Shape: (N, 4, 4)
                    
                    # Get RGB files
                    rgb_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
                    depth_files = sorted([f for f in os.listdir(depths_dir) if f.endswith('.exr')])
                    
                    # Ensure we have matching RGB and depth files
                    min_files = min(len(rgb_files), len(depth_files), len(camera_poses))
                    
                    if min_files == 0:
                        print(f"[matrixcity_parquet] Warning: no valid frames for block {block} – skipped.")
                        continue
                    
                    video_name = f"{city}_{style}_{block}"
                    
                    for frame_idx in range(min_files):
                        rgb_file = rgb_files[frame_idx]
                        depth_file = depth_files[frame_idx]
                        frame_number = int(os.path.basename(rgb_file).split('.')[0])
                        
                        # Build full paths
                        rgb_local_path = os.path.join(images_dir, rgb_file)
                        depth_local_path = os.path.join(depths_dir, depth_file)
                        
                        # Convert to paths
                        rgb_full_path = rgb_local_path.replace(DATA_DIR, data_path)
                        depth_full_path = depth_local_path.replace(DATA_DIR, data_path)
                        
                        # Get camera pose for this frame
                        extrinsics = camera_poses[frame_idx].astype(np.float32)
                        
                        # Use default intrinsics (can be improved by extracting from transforms.json)
                        intrinsics = default_intrinsics.copy()
                        
                        records.append({
                            "rgb_path": rgb_full_path,
                            "depth_path": depth_full_path,
                            "intrinsics": intrinsics.flatten().tolist(),
                            "extrinsics": extrinsics.flatten().tolist(),
                            "video_name": video_name,
                            "frame_number": frame_number
                        })
                        
                except Exception as e:
                    print(f"[matrixcity_parquet] Error processing block {block}: {e}")
                    continue

    # Convert to DataFrame
    df_new = pd.DataFrame.from_records(records, columns=df_columns)
    
    # Save parquet locally
    parquet_filename = "matrixcity_train.parquet"
    df_new.to_parquet(parquet_filename)
    
    print(f"[matrixcity_parquet] Completed – {len(df_new)} frames indexed. Parquet written to {parquet_filename}")
    
    # Save
    try:
        upload_file(parquet_filename, DATASET_CONFIG["matrixcity"]["parquet_path"])
        print(f"[matrixcity_parquet] Saved parquet file")
    except Exception as e:
        print(f"[matrixcity_parquet] Upload failed: {e}")
    
    # Optionally remove local file
    os.remove(parquet_filename)


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
    
    os.makedirs("data_vis", exist_ok=True)
    print("\nTesting MatrixCity video dataset with new sampling...")

    save_dir = "data_vis/matrixcity_new"
    os.makedirs(save_dir, exist_ok=True)
    
    resolutions = [[0,1]] * 10000
    video_dataset = VideoDepthMatrixCityNew(
        T=10, 
        stride_range=(1, 5),
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
            
            rgb_writer = video_writer.FFMPEG_VideoWriter(filename=f"data_vis/matrixcity_new_video_rgb_{idx}.mp4", size=(W, H), fps=2)
            depth_writer = video_writer.FFMPEG_VideoWriter(filename=f"data_vis/matrixcity_new_video_depth_{idx}.mp4", size=(W, H), fps=2)
            mask_writer = video_writer.FFMPEG_VideoWriter(filename=f"data_vis/matrixcity_new_video_mask_{idx}.mp4", size=(W, H), fps=2)

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
                print(f"poses: {poses}")

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
            save_path = f"data_vis/matrixcity_new_global_pts_{idx}.ply"
            o3d.io.write_point_cloud(
                save_path,
                pcd,
            )

            print(f"Saved pointcloud to {save_path}")
            
            print("Video sequence test completed successfully!")
            print("Saved videos:")
            print(f"  - data_vis/matrixcity_new_video_rgb_{idx}.mp4")
            print(f"  - data_vis/matrixcity_new_video_depth_{idx}.mp4") 
            print(f"  - data_vis/matrixcity_new_video_mask_{idx}.mp4")
            print(f"  - data_vis/matrixcity_new_video_frame0_pointmap_{idx}.ply")
    else:
        print("No valid video sequences found in dataset.")