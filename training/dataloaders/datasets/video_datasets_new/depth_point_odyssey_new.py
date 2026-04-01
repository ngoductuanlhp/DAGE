import os
import io
import pandas as pd
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

from training.dataloaders.datasets.video_datasets_new.synchronized_video_transform import SynchronizedTransformVideoCut3r
from training.dataloaders.datasets.video_datasets_new.synchronized_video_transform_moge import SynchronizedTransform_MoGe

from training.dataloaders.datasets.video_datasets_new.frame_sampling_utils import get_seq_from_start_id

SCENES_TO_USE = [
    # 'cab_h_bench_3rd', 'cab_h_bench_ego1', 'cab_h_bench_ego2',
    "cnb_dlab_0215_3rd",
    "cnb_dlab_0215_ego1",
    "cnb_dlab_0225_3rd",
    "cnb_dlab_0225_ego1",
    "dancing",
    "dancingroom0_3rd",
    "footlab_3rd",
    "footlab_ego1",
    "footlab_ego2",
    "girl",
    "girl_egocentric",
    "human_egocentric",
    "human_in_scene",
    "human_in_scene1",
    "kg",
    "kg_ego1",
    "kg_ego2",
    "kitchen_gfloor",
    "kitchen_gfloor_ego1",
    "kitchen_gfloor_ego2",
    "scene_carb_h_tables",
    "scene_carb_h_tables_ego1",
    "scene_carb_h_tables_ego2",
    "scene_j716_3rd",
    "scene_j716_ego1",
    "scene_j716_ego2",
    "scene_recording_20210910_S05_S06_0_3rd",
    "scene_recording_20210910_S05_S06_0_ego2",
    "scene1_0129",
    "scene1_0129_ego",
    "seminar_h52_3rd",
    "seminar_h52_ego1",
    "seminar_h52_ego2",
]


class SynchronizedTransform_PointOdysseyVideo:
    def __init__(self, H, W):
        self.resize          = transforms.Resize((H,W))
        self.resize_depth    = transforms.Resize((H,W), interpolation=Image.NEAREST)
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.to_tensor       = transforms.ToTensor()
        self.W = W
        self.H = H

    def __call__(self, rgb_images, depth_images, intrinsics_list):
        # Decide on flip for entire sequence
        flip = random.random() > 0.5
        
        rgb_tensors = []
        depth_tensors = []
        intrinsics_tensors = []
        
        for rgb_image, depth_image, intrinsics in zip(rgb_images, depth_images, intrinsics_list):
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

        return rgb_tensors, depth_tensors, intrinsics_tensors


class VideoDepthPointOdysseyNew(Dataset):
    depth_type = "synthetic"
    is_metric_scale = True

    # original reso 540, 960

    def __init__(self, T=5,  stride_range=(1, 3), transform=True, resize=(320, 640), near_plane=1e-5, far_plane=65.0, resolutions=None, use_moge=False, moge_augmentation=None,
        use_cut3r_frame_sampling=False, video_prob=0.6, fix_interval_prob=0.6, block_shuffle=16
    ):
        """
        Video sequence dataset for Point Odyssey depth data.
        
        Args:
            T (int): Number of frames in each sequence
            stride_range (tuple): Range of strides to randomly sample from (min_stride, max_stride)
            transform (bool): Whether to apply transformations
            resize (tuple): Target size (H, W)
            near_plane (float): Near clipping plane for depth
            far_plane (float): Far clipping plane for depth
        """
        parquet_data = download_file(DATASET_CONFIG["point_odyssey"]["parquet_path"])
        self.parquet = io.BytesIO(parquet_data)

        self.T = T

        self.stride_range = stride_range
        self.near_plane = near_plane
        self.far_plane = far_plane
        # self.transform = SynchronizedTransform_PointOdysseyVideo(H=resize[0], W=resize[1]) if transform else None

            
        self.frame_size = resize
        
        # Load and organize data by video
        self.video_frames = self._organize_by_video()
        self.frame_list = self._create_frame_list()

        self.resolutions = resolutions
        self.use_moge = use_moge
        self.moge_augmentation = moge_augmentation  # NOTE for moge style, we need to pass the augmentation type       
        
        # New sophisticated sampling parameters
        self.use_cut3r_frame_sampling = use_cut3r_frame_sampling
        self.video_prob = video_prob
        self.fix_interval_prob = fix_interval_prob
        self.block_shuffle = block_shuffle 

        if self.use_moge:
            self.transform = SynchronizedTransform_MoGe(moge_augmentation=self.moge_augmentation)   
        elif self.resolutions is not None:
            self.transform = SynchronizedTransformVideoCut3r(seed=777)
        else:
            self.transform = SynchronizedTransform_PointOdysseyVideo(H=resize[0], W=resize[1]) if transform else None

    def _organize_by_video(self):
        """Organize frames by video name and sort by frame number."""
        df = pd.read_parquet(self.parquet)
        
        # Filter scenes (following cut3r)
        df = df[df['video_name'].isin(SCENES_TO_USE)]
        
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
        start_frame_num = frames[start_idx]['frame_number']
        for i in range(num_frames):
            frame_idx = start_idx + i * stride
            expected_frame_num = start_frame_num + i * stride
            actual_frame_num = frames[frame_idx]['frame_number']
            if actual_frame_num != expected_frame_num:
                return False
        return True

    def _load_single_frame(self, frame_data):
        """Load a single frame's RGB, depth, and intrinsics."""
        try:
            # Load RGB
            rgb_data = download_file(frame_data['rgb_path'])
            rgb_image = Image.open(io.BytesIO(rgb_data)).convert('RGB')
            del rgb_data  # Free memory immediately
            
            # Load depth
            depth_data = download_file(frame_data['depth_path'])
            depth_buffer = np.frombuffer(depth_data, np.uint8)
            del depth_data  # Free memory immediately
            
            depth_array = cv2.imdecode(depth_buffer, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0 * 1000.0
            del depth_buffer  # Free memory immediately
            
            depth_image = Image.fromarray(depth_array)
            del depth_array  # Free memory immediately
            
            # Get intrinsics and extrinsics
            intrinsics = np.array(frame_data['intrinsics']).reshape(3, 3).copy()
            extrinsics = np.array(frame_data['extrinsics']).reshape(4, 4).copy()

            # print(f"depth_image.size: {depth_image.size}")
            # print(f"rgb_image.size: {rgb_image.size}")

            # NOTE transform from w2c to c2w https://github.com/CUT3R/CUT3R/blob/a0aedf8c7ff2a46c98added83c4d331ce9cd4d50/datasets_preprocess/preprocess_point_odyssey.py#L99
            extrinsics = np.linalg.inv(extrinsics)
            
            return rgb_image, depth_image, intrinsics, extrinsics, True
            
        except Exception as e:
            print(f"[point_odyssey_video] Error loading frame: {str(e)}")
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

            
            # print(f"Point Odyssey data, scene name: {video_name}, frame positions: {frame_positions}, {self.use_cut3r_frame_sampling}")
            
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
                        rgb_images.append(Image.new('RGB', (640, 320)))
                        depth_images.append(Image.fromarray(np.zeros((320, 640), dtype=np.float32)))
                        intrinsics_list.append(np.eye(3, dtype=np.float32))
                        extrinsics_list.append(np.eye(4, dtype=np.float32))
                    valid_frames.append(False)

            # Apply transforms
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
                processed_extrinsics.append(extrinsics_tensor[:3, :3])  # Only rotation part

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
                # 'stride': stride,
                'valid': all(valid_frames),
                'depth_type': self.depth_type,
            }
        
        except Exception as e:
            print(f"[point_odyssey_video] Error processing frame: {str(e)}")

            if num_frames is None:
                num_frames = self.T
            # Return dummy data with valid=False
            dummy_rgb = torch.zeros(num_frames, 3, self.frame_size[0], self.frame_size[1])  # Assuming default resize
            dummy_depth = torch.zeros(num_frames, 1, self.frame_size[0], self.frame_size[1])
            dummy_mask = torch.zeros(num_frames, 1, self.frame_size[0], self.frame_size[1], dtype=torch.bool)
            dummy_inf_mask = torch.zeros(num_frames, 1, self.frame_size[0], self.frame_size[1], dtype=torch.bool)
            dummy_intrinsics = torch.eye(3).unsqueeze(0).repeat(num_frames, 1, 1)
            dummy_extrinsics = torch.eye(3).unsqueeze(0).repeat(num_frames, 1, 1)
            
            return {
                'rgb': dummy_rgb,
                'metric_depth': dummy_depth,
                'valid_mask': dummy_mask,
                'inf_mask': dummy_inf_mask,
                'intrinsics': dummy_intrinsics,
                'extrinsics': dummy_extrinsics,
                # 'stride': stride,
                'valid': False,
                'depth_type': self.depth_type,
            }

def generate_parquet_file():
    import glob
    import tqdm
    # assuming data is downloaded in mnt/localssd/tartanair
    DATA_DIR = "/mnt/localssd/point_odyssey"
    split = "train"
    data_path =  DATASET_CONFIG["point_odyssey"]["prefix"]
    scenes = [s for s in os.listdir(os.path.join(DATA_DIR, split)) if os.path.isdir(os.path.join(DATA_DIR, split, s))]
    df_new = pd.DataFrame(columns=['rgb_path', 'depth_path', 'normal_path', 'intrinsics', 'extrinsics', 'video_name', 'frame_number'])
    for scene in tqdm.tqdm(scenes):
        annotations_file = os.path.join(DATA_DIR, split, scene, 'anno.npz')
        annotations = np.load(annotations_file)
        rgb_dir = os.path.join(DATA_DIR, split, scene, 'rgbs')
        depth_dir = os.path.join(DATA_DIR, split, scene, 'depths')
        normal_dir = os.path.join(DATA_DIR, split, scene, 'normals')
        rgb_files = os.listdir(rgb_dir)
        for rgb_file in rgb_files:
            frame_number = int(rgb_file.split('_')[1].split('.')[0])
            depth_file = rgb_file.replace('rgb', 'depth').replace('jpg', 'png')
            normal_file = rgb_file.replace('rgb', 'normal')
            rgb_path = os.path.join(rgb_dir, rgb_file).replace(DATA_DIR, data_path)
            depth_path = os.path.join(depth_dir, depth_file).replace(DATA_DIR, data_path)
            normal_path = os.path.join(normal_dir, normal_file).replace(DATA_DIR, data_path)
            intrinsics = annotations['intrinsics'][frame_number]
            intrinsics_flat = intrinsics.flatten().tolist()
            extrinsics = annotations['extrinsics'][frame_number]
            extrinsics_flat = extrinsics.flatten().tolist()
            df_new = pd.concat([df_new, pd.DataFrame({
                'rgb_path': [rgb_path],
                'depth_path': [depth_path],
                'normal_path': [normal_path],
                'intrinsics': [intrinsics_flat],
                'extrinsics': [extrinsics_flat],
                'video_name': [scene],
                'frame_number': [frame_number]
            })], ignore_index=True)

    df_new.to_parquet("point_odyssey_train.parquet")
    # read the parquet file
    df_new = pd.read_parquet("point_odyssey_train.parquet")
    # print the length of the dataframe
    print(len(df_new))
    # Save the parquet file
    upload_file("point_odyssey_train.parquet", DATASET_CONFIG["point_odyssey"]["parquet_path"])
    # remove the local file
    # os.remove("point_odyssey_train.parquet")

if __name__ == "__main__":
    # generate_parquet_file()

    # Test video dataset:
    import tqdm
    from Unip.unip.util.pointmap_util import depth_to_pointmap, save_pointmap_as_ply
    import moviepy.video.io.ffmpeg_writer as video_writer
    
    os.makedirs("data_vis", exist_ok=True)
    print("\nTesting video dataset...")
    video_dataset = VideoDepthPointOdysseyNew(T=5, stride_range=(1, 3), transform=True)
    print(f"Video dataset length: {len(video_dataset)}")
    
    if len(video_dataset) > 0:
        video_sample = video_dataset[0]
        print(f"Video sample shapes:")
        print(f"  RGB: {video_sample['rgb'].shape}")  # Should be [T, 3, H, W]
        print(f"  Metric depth: {video_sample['metric_depth'].shape}")  # Should be [T, 1, H, W]
        print(f"  Valid mask: {video_sample['valid_mask'].shape}")  # Should be [T, 1, H, W]
        print(f"  Intrinsics: {video_sample['intrinsics'].shape}")  # Should be [T, 3, 3]
        print(f"  Extrinsics: {video_sample['extrinsics'].shape}")  # Should be [T, 3, 3]
        print(f"  Valid: {video_sample['valid']}")
        print(f"  Stride: {video_sample['stride']}")
        
        # Prepare video writers
        H, W = video_sample['rgb'].shape[2], video_sample['rgb'].shape[3]
        
        rgb_writer = video_writer.FFMPEG_VideoWriter(filename="data_vis/point_odyssey_video_rgb.mp4", size=(W, H), fps=2)
        depth_writer = video_writer.FFMPEG_VideoWriter(filename="data_vis/point_odyssey_video_depth.mp4", size=(W, H), fps=2)
        mask_writer = video_writer.FFMPEG_VideoWriter(filename="data_vis/point_odyssey_video_mask.mp4", size=(W, H), fps=2)
        
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
                save_pointmap_as_ply(pointmap, rgb_frame, "data_vis/point_odyssey_video_frame0_pointmap.ply", far_threshold=1000)

        rgb_writer.close()
        depth_writer.close()
        mask_writer.close()
        
        print("Video sequence test completed successfully!")
        print("Saved videos:")
        print("  - data_vis/point_odyssey_video_rgb.mp4")
        print("  - data_vis/point_odyssey_video_depth.mp4") 
        print("  - data_vis/point_odyssey_video_mask.mp4")
        print("  - data_vis/point_odyssey_video_frame0_pointmap.ply")
    else:
        print("No valid video sequences found in dataset.")
