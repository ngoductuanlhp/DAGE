import os
import io
import tempfile
import pandas as pd
import tqdm
from torch.utils.data import Dataset
from training.dataloaders.data_io import download_file, upload_file
from training.dataloaders.config import DATASET_CONFIG

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import random
import cv2
from collections import defaultdict
from decord import VideoReader, cpu
import einops

from training.dataloaders.datasets.video_datasets_new.synchronized_video_transform import SynchronizedTransformVideoCut3r
from training.dataloaders.datasets.video_datasets_new.synchronized_video_transform_moge import SynchronizedTransform_MoGe

from training.dataloaders.datasets.video_datasets_new.frame_sampling_utils import get_seq_from_start_id


class SynchronizedTransform_Stereo4DVideo:
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


class VideoDepthStereo4DNew(Dataset):
    depth_type = "placeholder"  # Stereo4D has no depth data

    def __init__(self, T=5, stride_range=(1, 10), transform=True, resize=(504, 504), near_plane=1e-5, far_plane=1000.0, resolutions=None, use_moge=False, moge_augmentation=None,
        use_cut3r_frame_sampling=False, video_prob=0.6, fix_interval_prob=0.6, block_shuffle=16
    ):
        """
        Video sequence dataset for Stereo4D data.
        Dataset is organized by video (not individual frames).
        Note: This dataset only has RGB videos, no intrinsics or poses.
        
        Args:
            T (int): Number of frames to sample from each video
            stride_range (tuple): Range of strides to randomly sample from (min_stride, max_stride)
            transform (bool): Whether to apply transformations
            resize (tuple): Target size (H, W)
            near_plane (float): Near clipping plane for depth
            far_plane (float): Far clipping plane for depth
            resolutions (list): List of resolutions for multi-scale training
            use_moge (bool): Whether to use MoGe-style augmentation
            moge_augmentation (dict): MoGe augmentation parameters
            use_cut3r_frame_sampling (bool): Whether to use CUT3R-style frame sampling
            video_prob (float): Probability of sampling sequential video frames
            fix_interval_prob (float): Probability of using fixed interval
            block_shuffle (int): Block size for shuffling
        """
        parquet_data = download_file(DATASET_CONFIG["stereo4d"]["parquet_path"])
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

        # Load video list - each row is one video
        self.video_list = self._load_video_list()

        if self.use_moge:
            self.transform = SynchronizedTransform_MoGe(moge_augmentation=self.moge_augmentation)   
        elif self.resolutions is not None:
            self.transform = SynchronizedTransformVideoCut3r(seed=777)
        else:
            self.transform = SynchronizedTransform_Stereo4DVideo(H=resize[0], W=resize[1]) if transform else None

    def _load_video_list(self):
        """Load the list of videos from parquet. Each row = one video."""
        df = pd.read_parquet(self.parquet)
        video_list = []
        
        for _, row in df.iterrows():
            video_info = {
                'video_name': row['video_name'],
                'rgb_path': row['rgb_path'],
                'fps': row['fps'],
                # Note: Stereo4D doesn't have poses or intrinsics
            }
            video_list.append(video_info)
        
        return video_list

    def __len__(self):
        return len(self.video_list)

    def _load_video_frames(self, video_info, frame_indices):
        """Load specific frames from a video using decord VideoReader."""
        try:
            # Download video file to memory
            video_data = download_file(video_info['rgb_path'])
            
            # Use VideoReader with in-memory video
            # Write to temporary file since decord doesn't support BytesIO directly
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                tmp_file.write(video_data)
                tmp_video_path = tmp_file.name
            del video_data  # Free memory immediately after writing to temp file
            
            try:
                vr = VideoReader(tmp_video_path, ctx=cpu(0))
                num_frames = len(vr)
                original_height, original_width = vr.get_batch([0]).shape[1:3]
                fps = vr.get_avg_fps()
                
                # For Stereo4D, directly use the frame indices (no stride_interval)
                actual_frame_indices = [min(idx, num_frames - 1) for idx in frame_indices]
                frames = vr.get_batch(actual_frame_indices).asnumpy()  # Shape: [T, H, W, 3]
                
                # Convert to list of PIL Images
                rgb_images = [Image.fromarray(frame) for frame in frames]
                del frames  # Free memory immediately
                
                video_metadata = {
                    'original_height': original_height,
                    'original_width': original_width,
                    'fps': fps,
                    'num_frames': num_frames,
                }
                
                return rgb_images, video_metadata, True
            finally:
                # Clean up temporary file
                os.remove(tmp_video_path)
                
        except Exception as e:
            print(f"[stereo4d_video] Error loading video frames: {str(e)}")
            return None, None, False

    def _return_dummy_data(self, num_frames):
        """Return dummy data when loading fails."""
        dummy_rgb = torch.zeros(num_frames, 3, self.frame_size[0], self.frame_size[1])
        dummy_depth = torch.zeros(num_frames, 1, self.frame_size[0], self.frame_size[1])
        dummy_mask = torch.zeros(num_frames, 1, self.frame_size[0], self.frame_size[1], dtype=torch.bool)
        dummy_inf_mask = torch.zeros(num_frames, 1, self.frame_size[0], self.frame_size[1], dtype=torch.bool)
        dummy_fin_mask = torch.zeros(num_frames, 1, self.frame_size[0], self.frame_size[1], dtype=torch.bool)
        dummy_intrinsics = torch.eye(3).unsqueeze(0).repeat(num_frames, 1, 1)
        dummy_extrinsics = torch.eye(4).unsqueeze(0).repeat(num_frames, 1, 1)
        
        return {
            'rgb': dummy_rgb,
            'metric_depth': dummy_depth,
            'valid_mask': dummy_mask,
            'fin_mask': dummy_fin_mask,  # Match MoGe return structure
            'inf_mask': dummy_inf_mask,
            'intrinsics': dummy_intrinsics,
            'extrinsics': dummy_extrinsics,
            'valid': False,
            'depth_type': self.depth_type,
        }

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

            # Get the video info
            video_info = self.video_list[idx]
            
            # For Stereo4D, we need to load the video first to get frame count
            # (since we don't have separate metadata files)
            # We'll do a quick check to get the total number of frames
            try:
                video_data = download_file(video_info['rgb_path'])
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                    tmp_file.write(video_data)
                    tmp_video_path = tmp_file.name
                
                try:
                    vr = VideoReader(tmp_video_path, ctx=cpu(0))
                    num_video_frames = len(vr)
                finally:
                    os.remove(tmp_video_path)
            except Exception as e:
                print(f"[stereo4d_video] Error checking video length: {str(e)}")
                next_idx = (idx + random.randint(1, self.__len__())) % self.__len__()
                return self.__getitem__((next_idx, num_frames, resolution_idx))
            
            # Sample frame indices from the video
            if self.use_cut3r_frame_sampling:
                ids_all = list(range(num_video_frames))
                start_idx = random.randint(0, max(0, num_video_frames - 1 - num_frames * self.stride_range[0]))
                # print(f"total original frames: {num_video_frames}, start_idx: {start_idx}, num_frames: {num_frames}, stride_range: {self.stride_range}")
                    # if start_idx + num_frames * self.stride_range[0] > num_video_frames - 1:
                    #     # print(f"start_idx + num_frames * self.stride_range[0] > num_video_frames - 1, start_idx: {start_idx}, num_frames: {num_frames}, self.stride_range[0]: {self.stride_range[0]}, num_video_frames: {num_video_frames}")

                    #     next_idx = (idx + random.randint(1, self.__len__())) % self.__len__()
                    #     return self.__getitem__((next_idx, num_frames, resolution_idx))

                    
                try:
                    is_success, frame_positions, is_video = get_seq_from_start_id(
                        num_frames, start_idx, ids_all, rng, min_interval=self.stride_range[0], max_interval=self.stride_range[1], 
                        video_prob=self.video_prob, fix_interval_prob=self.fix_interval_prob, block_shuffle=self.block_shuffle
                    )
                except Exception as e:
                    is_success = False
            else:
                is_video = True
                # Randomly sample stride and starting frame
                stride = random.randint(self.stride_range[0], self.stride_range[1])
                
                # Calculate valid starting range
                max_start_idx = max(0, num_video_frames - (num_frames - 1) * stride - 1)
                
                if max_start_idx <= 0:
                    # Video too short, reduce stride
                    stride = max(1, (num_video_frames - 1) // (num_frames - 1))
                    max_start_idx = max(0, num_video_frames - (num_frames - 1) * stride - 1)
                
                start_idx = random.randint(0, max(0, max_start_idx))
                frame_positions = [min(start_idx + i * stride, num_video_frames - 1) for i in range(num_frames)]
                is_success = True

            if not is_success:
                # Try another video
                next_idx = (idx + random.randint(1, self.__len__())) % self.__len__()
                if len_tuple_idx == 3:
                    return self.__getitem__((next_idx, num_frames, resolution_idx))
                elif len_tuple_idx == 2:
                    return self.__getitem__((next_idx, num_frames))
                elif len_tuple_idx == 1:
                    return self.__getitem__(next_idx)
                else:
                    raise ValueError(f"Invalid tuple length: {len_tuple_idx}")
            
            # Load RGB frames from video
            rgb_images, original_video_info, video_load_success = self._load_video_frames(video_info, frame_positions)
            
            if not video_load_success:
                next_idx = (idx + random.randint(1, self.__len__())) % self.__len__()
                return self.__getitem__((next_idx, num_frames, resolution_idx))
            
            # For Stereo4D, we don't have real intrinsics/extrinsics
            # Create dummy camera parameters
            intrinsics_list = []
            extrinsics_list = []
            valid_frames = []
            
            # Create a default intrinsics matrix assuming a reasonable FOV
            # Using a 60-degree horizontal FOV as a starting point
            original_width = original_video_info['original_width']
            original_height = original_video_info['original_height']
            
            # Focal length assuming 60-degree horizontal FOV: f = w / (2 * tan(fov/2))
            fx = original_width / (2 * np.tan(np.radians(60) / 2))
            fy = fx  # Assume square pixels
            cx = original_width / 2.0
            cy = original_height / 2.0

            
            for _ in range(num_frames):
                # Create intrinsics matrix
                intrinsics = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ], dtype=np.float32)
                
                # Create identity extrinsics (all frames at origin)
                # This means we're treating the video as having static camera
                extrinsics = np.eye(4, dtype=np.float32)
                
                intrinsics_list.append(intrinsics)
                extrinsics_list.append(extrinsics)
                valid_frames.append(True)
            
            # Note: Stereo4D doesn't have depth data, we'll create placeholder depth
            # This is for training models that can work with RGB-only data
            depth_images = []
            for rgb_img in rgb_images:
                # Create dummy depth with random values to pass dataloader filters
                w, h = rgb_img.size
                # Random depth to ensure no error in MoGe Transformation
                depth_images.append(Image.fromarray(np.random.randint(1, 100, (h, w)).astype(np.float32)))


            # Apply transforms
            if self.transform is not None:
                if isinstance(self.transform, SynchronizedTransformVideoCut3r):
                    rgb_tensors, depth_tensors, intrinsics_tensors = self.transform(
                        rgb_images, depth_images, intrinsics_list, 
                        resolution=self.resolutions[resolution_idx] if (resolution_idx is not None and self.resolutions is not None) else (self.frame_size[1], self.frame_size[0])
                    )
                elif isinstance(self.transform, SynchronizedTransform_MoGe):
                    processed_data = self.transform(
                        rgb_images, depth_images, intrinsics_list, extrinsics_list, valid_frames, 
                        self.near_plane, self.far_plane, self.depth_type, resolution_idx, stride=None, 
                        no_depth_mask_inf=True, rng=rng, augmentation=False
                    ) # NOTE for this dataset, no augmentation
                    return processed_data  # NOTE for moge style, we return the processed data directly
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

                # Process RGB 
                rgb_tensor = rgb_tensor * 2.0 - 1.0  # [-1,1]

                # For Stereo4D, we don't have real depth, so all depth-related tensors are dummy
                metric_tensor = torch.zeros_like(depth_tensor)
                valid_depth_mask = torch.zeros_like(depth_tensor).bool()
                inf_depth_mask = torch.zeros_like(depth_tensor).bool()

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
            fin_mask_sequence = valid_mask_sequence  # For Stereo4D, fin_mask is same as valid_mask (no depth data)
            intrinsics_sequence = torch.stack(processed_intrinsics, dim=0)  # [T, 3, 3]
            extrinsics_sequence = torch.stack(processed_extrinsics, dim=0)  # [T, 4, 4]

            return {
                'rgb': rgb_sequence,
                'metric_depth': metric_depth_sequence,
                'valid_mask': valid_mask_sequence,
                'fin_mask': fin_mask_sequence,  # Match MoGe return structure
                'inf_mask': inf_mask_sequence,
                'intrinsics': intrinsics_sequence,
                'extrinsics': extrinsics_sequence,
                'valid': all(valid_frames),
                'depth_type': self.depth_type,
            }
        
        except Exception as e:
            print(f"[stereo4d_video] Error processing video: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return self._return_dummy_data(num_frames if 'num_frames' in locals() else self.T)


def generate_parquet_file():
    """
    Generate parquet file for Stereo4D dataset.
    This dataset only has RGB videos (no intrinsics or poses).
    """
    # Constants for data preprocessing
    STEREO4D_BASE_DIR = "/mnt/localssd/stereo4d"
    VIDEO_DIR = os.path.join(STEREO4D_BASE_DIR, "train_dynamic_videos")
    dynamic_videos_file = os.path.join(STEREO4D_BASE_DIR, "dynamic_videos.txt")

    # Read dynamic videos list
    with open(dynamic_videos_file, "r") as f:
        dynamic_videos = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(dynamic_videos)} dynamic videos to process")

    # Define columns for parquet (only what we have for Stereo4D)
    df_columns = ['rgb_path', 'fps', 'video_name']
    records = []

    for video in tqdm.tqdm(dynamic_videos, total=len(dynamic_videos), desc="Processing videos"):
        video_path = os.path.join(VIDEO_DIR, video)
        video_name = video.replace(".mp4", "")

        # Skip if video file is missing
        if not os.path.exists(video_path):
            print(f"Skipping {video} due to missing file")
            continue

        # Create path for the video
        rgb_data_path = os.path.join(DATASET_CONFIG["stereo4d"]["prefix"], "train_dynamic_videos", video)
        
        records.append({
            'rgb_path': rgb_data_path,
            'fps': 30,  # Stereo4D videos are 30 FPS
            'video_name': video_name
        })
    
    # df_new.to_parquet("mvs_synth_train.parquet")
    df_new = pd.DataFrame.from_records(records, columns=df_columns)
    
    # Save parquet file
    parquet_filename = "stereo4d.parquet"
    df_new.to_parquet(parquet_filename)
    print(f"Created parquet file with {len(df_new)} entries")


    
    # Save the parquet file
    upload_file("stereo4d.parquet", DATASET_CONFIG["stereo4d"]["parquet_path"])
    # Remove the local file
    os.remove("stereo4d.parquet")

    return



if __name__ == "__main__":
    """
    Test script for Stereo4D dataset.
    
    To generate the parquet file first:
    1. Uncomment the generate_parquet_file() and quit() lines
    2. Run this script
    3. Save parquet
    4. Comment them out again and run the dataloader test
    
    To test the dataloader:
    - Make sure the parquet file is saved
    - Run this script with generate_parquet_file() commented out
    """
    # Step 1: Generate parquet file (uncomment to run)
    # generate_parquet_file()
    # quit()

    # Step 2: Test the dataloader
    import tqdm
    import moviepy.video.io.ffmpeg_writer as video_writer
    import cv2
    from torch.utils.data import DataLoader

    import utils3d
    from third_party.pi3.utils.geometry import homogenize_points, se3_inverse, depth_edge
    
    os.makedirs("data_vis", exist_ok=True)
    print("\nTesting Stereo4D video dataset...")

    dataset_name = "stereo4d_new"
    save_dir = f"data_vis/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Test without resolutions first (simpler)
    resolutions = [[0,1]] * 10000
    video_dataset = VideoDepthStereo4DNew(
        T=20, 
        stride_range=(10, 20),
        transform=True,
        resize=(504, 504),
        resolutions=resolutions, 
        use_moge=True,
        use_cut3r_frame_sampling=True,  # Use simple sampling first
        video_prob=1.0,
        fix_interval_prob=0.6,
        block_shuffle=32,
        moge_augmentation=dict(
            use_flip_augmentation=False,
            center_augmentation=0.10,
            fov_range_absolute_min=60,
            fov_range_absolute_max=120,
            fov_range_relative_min=0.7,
            fov_range_relative_max=1.0,
            image_augmentation=['jittering', 'jpeg_loss', 'blurring'],
            # image_augmentation=[],
            depth_interpolation='nearest',
            clamp_max_depth=1000.0,
            area_range=[100000, 255000],
            aspect_ratio_range=[0.5, 2.0],
        )
    )
    print(f"Video dataset length: {len(video_dataset)}")

    # Simple dataloader without custom sampler for testing
    dataloader = DataLoader(
        video_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )

    print(f"\nStarting dataloader test...")
    
    if len(video_dataset) > 0:
        sample_count = 0

        for idx, video_sample in enumerate(dataloader):
            sample_count += 1
            if sample_count > 20: break  # Test with just 5 samples

            if isinstance(video_sample['rgb'], list):
                # Handle case where video_sample might be a list of batches
                video_sample = {k: v[0] if isinstance(v, list) else v for k, v in video_sample.items()}
            else:
                # Extract first item from batch dimension
                video_sample = {k: v[0] if isinstance(v, torch.Tensor) and v.dim() > 3 else v for k, v in video_sample.items()}

            print(f"\n[Sample {idx}] Video sample shapes:")
            print(f"  RGB: {video_sample['rgb'].shape}")  # Should be [T, 3, H, W]
            print(f"  Metric depth: {video_sample['metric_depth'].shape}")  # Should be [T, 1, H, W]
            print(f"  Valid mask: {video_sample['valid_mask'].shape}")  # Should be [T, 1, H, W]
            print(f"  Intrinsics: {video_sample['intrinsics'].shape}")  # Should be [T, 3, 3]
            print(f"  Extrinsics: {video_sample['extrinsics'].shape}")  # Should be [T, 4, 4]
            print(f"  Valid: {video_sample['valid']}")
            print(f"  Depth type: {video_sample['depth_type']}")
            
            # Prepare video writers
            H, W = video_sample['rgb'].shape[2], video_sample['rgb'].shape[3]
            
            rgb_writer = video_writer.FFMPEG_VideoWriter(filename=f"{save_dir}/video_rgb_{idx}.mp4", size=(W, H), fps=5)
            all_gt_pts = []
            all_gt_pts_rgb = []
            print("Saving video frames...")
            for t in tqdm.tqdm(range(video_sample['rgb'].shape[0])):
                rgb_frame = (video_sample['rgb'][t].numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                
                

                rgb_writer.write_frame(rgb_frame)
                
            rgb_writer.close()
            
            print("Video sequence test completed successfully!")
    else:
        print("No valid video sequences found in dataset.")

