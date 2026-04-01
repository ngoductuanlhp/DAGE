import os
import io
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
import cv2
import json
from collections import defaultdict

from training.dataloaders.datasets.video_datasets_new.synchronized_video_transform import SynchronizedTransformVideoCut3r
from training.dataloaders.datasets.video_datasets_new.synchronized_video_transform_moge import SynchronizedTransform_MoGe

from training.dataloaders.datasets.video_datasets_new.frame_sampling_utils import get_seq_from_start_id

class SynchronizedTransform_CO3DVideo:
    def __init__(self, H, W):
        self.resize          = transforms.Resize((H,W))
        self.resize_depth    = transforms.Resize((H,W), interpolation=Image.NEAREST)
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.to_tensor       = transforms.ToTensor()
        self.W = W
        self.H = H

    def __call__(self, rgb_images, depth_images, intrinsics_list, extrinsics_list=None):
        # Decide on flip for entire sequence
        if extrinsics_list is not None:
            flip = False
        else:
            flip = random.random() > 0.5
        
        rgb_tensors = []
        depth_tensors = []
        intrinsics_tensors = []
        extrinsics_tensors = []
        
        for idx, (rgb_image, depth_image, intrinsics) in enumerate(zip(rgb_images, depth_images, intrinsics_list)):
            # h-flip
            if flip:
                rgb_image = self.horizontal_flip(rgb_image)
                depth_image = self.horizontal_flip(depth_image)

            # resize
            og_width, og_height = rgb_image.size
            scale_w = self.W / og_width
            scale_h = self.H / og_height
            rgb_image   = self.resize(rgb_image)
            depth_image = self.resize_depth(depth_image)
            intrinsics[0, 0] *= scale_w  # fx
            intrinsics[1, 1] *= scale_h  # fy
            intrinsics[0, 2] *= scale_w  # cx
            intrinsics[1, 2] *= scale_h  # cy

            # to tensor
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


class VideoDepthCO3DNew(Dataset):
    depth_type = "lidar"  # CO3D uses rendered depth from multi-view reconstruction, but we mask out background, so consider only global loss
    is_metric_scale = False

    def __init__(self, T=5, stride_range=(1, 3), transform=True, resize=(360, 540), near_plane=0.01, far_plane=5000.0, resolutions=None, use_moge=False, moge_augmentation=None,
        use_cut3r_frame_sampling=False, video_prob=0.6, fix_interval_prob=0.6, block_shuffle=16
    ):
        """
        Video sequence dataset for CO3D depth data.
        
        Args:
            T (int): Number of frames in each sequence
            stride_range (tuple): Range of strides to randomly sample from (min_stride, max_stride)
            transform (bool): Whether to apply transformations
            resize (tuple): Target size (H, W)
            near_plane (float): Near clipping plane for depth
            far_plane (float): Far clipping plane for depth
            resolutions (list): List of resolutions for multi-scale training
        """
        parquet_data = download_file(DATASET_CONFIG["co3d"]["parquet_path"])
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
        
        # Load and organize data by sequence
        self.video_frames = self._organize_by_video()
        self.frame_list = self._create_frame_list()

        if self.use_moge:
            self.transform = SynchronizedTransform_MoGe(moge_augmentation=self.moge_augmentation)
        elif self.resolutions is not None:
            self.transform = SynchronizedTransformVideoCut3r(seed=777)
        else:
            self.transform = SynchronizedTransform_CO3DVideo(H=resize[0], W=resize[1]) if transform else None

    def _organize_by_video(self):
        """Organize frames by sequence name.
        Assumes parquet is pre-sorted by (video_name, frame_number).
        """
        df = pd.read_parquet(self.parquet)
        
        # Group by video_name (much faster than iterrows)
        # sort=False preserves the original row order within each group
        video_frames = {}
        for video_name, video_df in df.groupby('video_name', sort=False):
            # Convert each row to dict efficiently
            # Row order is preserved, so frames are in temporal order
            frames = video_df[['frame_number', 'rgb_path', 'depth_path', 'mask_path', 
                               'maximum_depth', 'intrinsics', 'extrinsics']].to_dict('records')
            
            video_frames[video_name] = frames
        
        return video_frames

    def _create_frame_list(self):
        """Create a flat list of all frames with sequence context."""
        # Use list comprehension for faster construction
        frame_list = [
            {
                'video_name': video_name,
                'frame_idx_in_sequence': i,
                'frame_data': frame_data
            }
            for video_name, frames in self.video_frames.items()
            for i, frame_data in enumerate(frames)
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
        
        # For CO3D, frames within a sequence should be consecutive
        # Check if we can sample the required number of frames
        return True

    def _load_single_frame(self, frame_data):
        """Load a single frame's RGB, depth, mask, and camera data for CO3D."""
        try:
            # Load RGB
            rgb_data = download_file(frame_data['rgb_path'])
            rgb_image = Image.open(io.BytesIO(rgb_data)).convert('RGB')
            del rgb_data  # Free memory immediately
            
            # Load depth (CO3D specific format - 16-bit PNG scaled by max_depth)
            depth_data = download_file(frame_data['depth_path'])
            depth_pil = Image.open(io.BytesIO(depth_data))
            del depth_data  # Free memory immediately
            
            depth_array = np.array(depth_pil)
            del depth_pil  # Free memory immediately
            
            # Load metadata (camera intrinsics, pose, and max_depth)
            # metadata_data = download_file(frame_data['metadata_path'])
            # metadata = np.load(io.BytesIO(metadata_data))
            
            # intrinsics = metadata['camera_intrinsics']
            # extrinsics = metadata['camera_pose']  # This is already camera-to-world
            # max_depth = metadata['maximum_depth']
            maximum_depth = frame_data['maximum_depth']
            intrinsics = np.array(frame_data['intrinsics'], dtype=np.float32).reshape(3, 3).copy()
            extrinsics = np.array(frame_data['extrinsics'], dtype=np.float32).reshape(4, 4).copy()
            
            # Convert depth back to metric units
            depth_array = depth_array.astype(np.float32) / 65535.0 * np.nan_to_num(maximum_depth)
            
            
            mask_out_bg = random.random() > 0.5
            if mask_out_bg:
                # Load mask
                mask_data = download_file(frame_data['mask_path'])
                mask_array = np.array(Image.open(io.BytesIO(mask_data))) / 255.0
                mask_array = (mask_array > 0.1).astype(np.float32)
                depth_array = depth_array * mask_array
                
            # depth_array[mask_array == 0] = 0  # Set masked areas to 0
            depth_image = Image.fromarray(depth_array)
            return rgb_image, depth_image, intrinsics, extrinsics, True
            
        except Exception as e:
            print(f"[co3d_video] Error loading frame: {str(e)}")
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
                len_tuple_idx = 1
                num_frames = self.T
                resolution_idx = None
            
            rng = np.random.default_rng(random.randint(0, 2 ** 32 - 1))

            # Get the starting frame
            start_frame_info = self.frame_list[idx]
            video_name = start_frame_info['video_name']
            start_idx = start_frame_info['frame_idx_in_sequence']
            
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

            
            # print(f"CO3D data, scene name: {video_name}, frame positions: {frame_positions}, {self.use_cut3r_frame_sampling}")
            
            # Collect frame data
            rgb_images = []
            depth_images = []
            intrinsics_list = []
            extrinsics_list = []
            valid_frames = []

            first_frame_reso = None
            
            
            for i, frame_idx in enumerate(frame_positions):
                
                frame_data = frames[frame_idx]
                rgb_img, depth_img, intrinsics, extrinsics, valid = self._load_single_frame(frame_data)
                if valid:

                    if i == 0:
                        first_frame_reso = (rgb_img.size[1], rgb_img.size[0]) # H W
                    else:
                        if rgb_img.size[1] != first_frame_reso[0] or rgb_img.size[0] != first_frame_reso[1]:
                            rgb_img = transforms.Resize(first_frame_reso)(rgb_img)
                            depth_img = transforms.Resize(first_frame_reso, interpolation=Image.NEAREST)(depth_img)
                            intrinsics[0, 0] *= first_frame_reso[1] / rgb_img.size[1]
                            intrinsics[1, 1] *= first_frame_reso[0] / rgb_img.size[0]
                            intrinsics[0, 2] *= first_frame_reso[1] / rgb_img.size[1]
                            intrinsics[1, 2] *= first_frame_reso[0] / rgb_img.size[0]

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
                        rgb_images.append(Image.new('RGB', (512, 512)))
                        depth_images.append(Image.fromarray(np.zeros((512, 512), dtype=np.float32)))
                        intrinsics_list.append(np.eye(3, dtype=np.float32))
                        extrinsics_list.append(np.eye(4, dtype=np.float32))
                    valid_frames.append(False)


            # Apply transforms
            if self.transform is not None:
                if isinstance(self.transform, SynchronizedTransformVideoCut3r):
                    rgb_tensors, depth_tensors, intrinsics_tensors = self.transform(
                        rgb_images, depth_images, intrinsics_list, 
                        resolution=self.resolutions[resolution_idx] if (resolution_idx is not None and self.resolutions is not None) else (self.frame_size[1], self.frame_size[0])
                    )
                elif isinstance(self.transform, SynchronizedTransform_MoGe):
                    processed_data = self.transform(
                        rgb_images, depth_images, intrinsics_list, extrinsics_list, valid_frames, self.near_plane, self.far_plane, self.depth_type, resolution_idx, stride=None, rng=rng, no_depth_mask_inf=True
                    )
                    processed_data['is_metric_scale'] = self.is_metric_scale
                    return processed_data # NOTE for moge style, we return the processed data directly
                else:
                    rgb_tensors, depth_tensors, intrinsics_tensors, extrinsics_tensors = self.transform(
                        rgb_images, depth_images, intrinsics_list, extrinsics_list
                    )
            else:
                rgb_tensors = [transforms.ToTensor()(img) for img in rgb_images]
                depth_tensors = [transforms.ToTensor()(img) for img in depth_images]
                intrinsics_tensors = [torch.tensor(intrinsics) for intrinsics in intrinsics_list]
                extrinsics_tensors = [torch.tensor(extrinsics) for extrinsics in extrinsics_list]

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
                extrinsics_tensor = extrinsics_tensors[t]

                # Get valid depth mask
                valid_depth_mask = (depth_tensor > self.near_plane) & (depth_tensor < self.far_plane)
                inf_depth_mask = torch.zeros_like(depth_tensor).bool() # No inf depth for CO3D

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
                processed_extrinsics.append(extrinsics_tensor)

            # Stack into [T, C, H, W] tensors
            rgb_sequence = torch.stack(processed_rgb, dim=0)  # [T, 3, H, W]
            metric_depth_sequence = torch.stack(processed_metric_depth, dim=0)  # [T, 1, H, W]
            valid_mask_sequence = torch.stack(processed_valid_mask, dim=0)  # [T, 1, H, W]
            inf_mask_sequence = torch.stack(processed_inf_mask, dim=0)  # [T, 1, H, W]
            intrinsics_sequence = torch.stack(processed_intrinsics, dim=0)  # [T, 3, 3]
            extrinsics_sequence = torch.stack(processed_extrinsics, dim=0)  # [T, 4, 4]

            return {
                'rgb': rgb_sequence,
                'metric_depth': metric_depth_sequence,
                'valid_mask': valid_mask_sequence,
                'inf_mask': inf_mask_sequence,
                'intrinsics': intrinsics_sequence,
                'extrinsics': extrinsics_sequence,
                # 'stride': stride,
                'valid': all(valid_frames),
                'depth_type': self.depth_type,
            }
        
        except Exception as e:
            print(f"[co3d_video] Error processing frame: {str(e)}")
            
            if num_frames is None:
                num_frames = self.T
            
            # Return dummy data with valid=False
            dummy_rgb = torch.zeros(num_frames, 3, self.frame_size[0], self.frame_size[1])
            dummy_depth = torch.zeros(num_frames, 1, self.frame_size[0], self.frame_size[1])
            dummy_mask = torch.zeros(num_frames, 1, self.frame_size[0], self.frame_size[1], dtype=torch.bool)
            dummy_inf_mask = torch.zeros(num_frames, 1, self.frame_size[0], self.frame_size[1], dtype=torch.bool)
            dummy_intrinsics = torch.eye(3).unsqueeze(0).repeat(num_frames, 1, 1)
            dummy_extrinsics = torch.eye(4).unsqueeze(0).repeat(num_frames, 1, 1)
            
            return {
                'rgb': dummy_rgb,
                'metric_depth': dummy_depth,
                'valid_mask': dummy_mask,
                'inf_mask': dummy_inf_mask,
                'intrinsics': dummy_intrinsics,
                'extrinsics': dummy_extrinsics,
                # 'stride': 1,
                'valid': False,
                'depth_type': self.depth_type,
            }


def generate_parquet_file():
    """
    Generate parquet file from preprocessed CO3D data.
    Reads the selected sequences JSON files and creates records for all frames.
    """
    import tqdm
    import glob
    
    # Path to preprocessed CO3D data
    DATA_DIR = "/mnt/localssd/co3d_processed"
    data_path = DATASET_CONFIG["co3d"]["prefix"]
    
    # CO3D categories
    CATEGORIES = [
        "apple", "backpack", "ball", "banana", "baseballbat", "baseballglove", "bench",
        "bicycle", "book", "bottle", "bowl", "broccoli", "cake", "car", "carrot",
        "cellphone", "chair", "couch", "cup", "donut", "frisbee", "hairdryer", "handbag",
        "hotdog", "hydrant", "keyboard", "kite", "laptop", "microwave", "motorcycle",
        "mouse", "orange", "parkingmeter", "pizza", "plant", "remote", "sandwich",
        "skateboard", "stopsign", "suitcase", "teddybear", "toaster", "toilet", "toybus",
        "toyplane", "toytrain", "toytruck", "tv", "umbrella", "vase", "wineglass"
    ]
    
    # Load selected sequences for train split
    selected_sequences_path = os.path.join(DATA_DIR, "selected_seqs_train.json")
    
    if not os.path.exists(selected_sequences_path):
        print(f"Selected sequences file not found: {selected_sequences_path}")
        return
    
    with open(selected_sequences_path, 'r') as f:
        selected_sequences = json.load(f)
    
    # Create empty list to accumulate records
    df_columns = ['rgb_path', 'depth_path', 'mask_path', 'metadata_path', 'video_name', 'intrinsics', 'extrinsics', 'maximum_depth', 'frame_number']
    records = []
    
    print("Converting CO3D metadata to parquet format...")
    
    for category in tqdm.tqdm(CATEGORIES):
        if category not in selected_sequences:
            print(f"Warning: No sequences found for category {category}, skipping...")
            continue
            
        category_sequences = selected_sequences[category]
        category_dir = os.path.join(DATA_DIR, category)
        
        if not os.path.exists(category_dir):
            print(f"Warning: Category directory not found: {category_dir}, skipping...")
            continue
        
        for sequence_name, frame_indices in category_sequences.items():
            for frame_idx in frame_indices:
                # Construct file paths based on CO3D structure
                frame_filename = f"frame{frame_idx:06d}"
                
                rgb_path = f"{data_path}/{category}/{sequence_name}/images/{frame_filename}.jpg"
                depth_path = f"{data_path}/{category}/{sequence_name}/depths/{frame_filename}.jpg.geometric.png"
                mask_path = f"{data_path}/{category}/{sequence_name}/masks/{frame_filename}.png"
                metadata_path = f"{data_path}/{category}/{sequence_name}/images/{frame_filename}.npz"
                
                # Verify files exist locally before adding to records
                local_rgb_path = rgb_path.replace(data_path, DATA_DIR)
                local_depth_path = depth_path.replace(data_path, DATA_DIR)
                local_mask_path = mask_path.replace(data_path, DATA_DIR)
                local_metadata_path = metadata_path.replace(data_path, DATA_DIR)

                input_metadata = np.load(local_metadata_path)
                camera_pose = input_metadata["camera_pose"].astype(np.float32)
                intrinsics = input_metadata["camera_intrinsics"].astype(np.float32)
                maximum_depth = float(input_metadata["maximum_depth"])  # Convert numpy scalar to Python float

                intrinsics_flat = intrinsics.flatten().tolist()
                extrinsics_flat = camera_pose.flatten().tolist()
                video_name = f"{category}_{sequence_name}"
                
                if all(os.path.exists(p) for p in [local_rgb_path, local_depth_path, local_mask_path, local_metadata_path]):
                    records.append({
                        'rgb_path': rgb_path,
                        'depth_path': depth_path,
                        'mask_path': mask_path,
                        'metadata_path': metadata_path,
                        'video_name': video_name,
                        'intrinsics': intrinsics_flat,
                        'extrinsics': extrinsics_flat,
                        'maximum_depth': maximum_depth,
                        'frame_number': frame_idx
                    })
                    # breakpoint()
                else:
                    print(f"Warning: Missing files for {category}/{sequence_name}/frame{frame_idx:06d}")

    # Convert accumulated records into DataFrame
    df_new = pd.DataFrame.from_records(records, columns=df_columns)
    
    # Save parquet file
    parquet_filename = "co3d_train.parquet"
    df_new.to_parquet(parquet_filename)
    
    # Read and verify
    df_verify = pd.read_parquet(parquet_filename)
    print(f"Generated parquet file with {len(df_verify)} entries across {len(selected_sequences)} categories")
    
    # Save
    upload_file(parquet_filename, DATASET_CONFIG["co3d"]["parquet_path"])
    
    # Clean up local file
    os.remove(parquet_filename)
    print(f"Saved parquet file and cleaned up local file")


if __name__ == "__main__":
    # Uncomment to generate parquet file from preprocessed data:
    # generate_parquet_file()
    # quit()

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

    dataset_name = "co3d_new"
    save_dir = f"data_vis/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    resolutions = [[0,1]] * 10000
    video_dataset = VideoDepthCO3DNew(
        T=10, 
        stride_range=(1, 2),
        transform=True,
        resolutions=resolutions, 
        use_moge=True,
        use_cut3r_frame_sampling=True,
        video_prob=1.0,
        fix_interval_prob=1.0,
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
                # depth_writer.write_frame(depth_bgr)
                # mask_writer.write_frame(mask_bgr)

                all_gt_pts.append(gt_pts)
                all_gt_pts_rgb.append(gt_pts_rgb)
                
            rgb_writer.close()
            # depth_writer.close()
            # mask_writer.close()

                    # all_gt_pts = torch.cat(all_gt_pts, dim=0)
                    # all_gt_pts_rgb = torch.cat(all_gt_pts_rgb, dim=0)

                    # pcd = o3d.geometry.PointCloud()
                    # pcd.points = o3d.utility.Vector3dVector(
                    #     all_gt_pts
                    # )
                    # pcd.colors = o3d.utility.Vector3dVector(
                    #     all_gt_pts_rgb
                    # )
                    # save_path = f"{save_dir}/global_pts_{idx}.ply"
                    # o3d.io.write_point_cloud(
                    #     save_path,
                    #     pcd,
                    # )

                    # print(f"Saved pointcloud to {save_path}")
            
            print("Video sequence test completed successfully!")
    else:
        print("No valid video sequences found in dataset.")