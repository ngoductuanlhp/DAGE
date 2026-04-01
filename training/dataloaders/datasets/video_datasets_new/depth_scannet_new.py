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
from collections import defaultdict

from training.dataloaders.datasets.video_datasets_new.synchronized_video_transform import SynchronizedTransformVideoCut3r
from training.dataloaders.datasets.video_datasets_new.synchronized_video_transform_moge import SynchronizedTransform_MoGe

from training.dataloaders.datasets.video_datasets_new.frame_sampling_utils import get_seq_from_start_id



class SynchronizedTransform_ScanNetVideo:
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


class VideoDepthScanNetNew(Dataset):
    depth_type = "lidar"
    is_metric_scale = True
    # ScanNet original resolution typically around (1296, 968) or (640, 480); we standardize via resize
    
    def __init__(self, T=5, stride_range=(1, 5), transform=True, resize=(360, 540), near_plane=0.05, far_plane=65.0, 
                 resolutions=None, use_moge=False, moge_augmentation=None, use_cut3r_frame_sampling=False, video_prob=0.6, fix_interval_prob=0.6, block_shuffle=16
                #  min_interval=1, max_interval=25, video_prob=0.5, fix_interval_prob=0.5, 
                #  allow_repeat=True, block_shuffle=None
                 ):
        """
        Video sequence dataset for ScanNet depth data with sophisticated sampling strategy.
        
        Args:
            T (int): Number of frames in each sequence
            transform (bool): Whether to apply transformations
            resize (tuple): Target size (H, W)
            near_plane (float): Near clipping plane for depth
            far_plane (float): Far clipping plane for depth
            resolutions (list): List of resolutions for multi-scale training
            use_moge (bool): Whether to use MoGe transform
            moge_augmentation: MoGe augmentation parameters
            min_interval (int): Minimum interval between frames
            max_interval (int): Maximum interval between frames
            video_prob (float): Probability of sampling as video sequence (vs collection)
            fix_interval_prob (float): Probability of using fixed interval when in video mode
            allow_repeat (bool): Whether to allow repeating frames when insufficient frames
            block_shuffle (int): Block size for shuffling in collection mode
        """
        parquet_data = download_file(DATASET_CONFIG["scannet"]["parquet_path"])
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
        
        # Load and organize data by scene
        self.scene_frames = self._organize_by_scene()
        self.frame_list = self._create_frame_list()

        if self.use_moge:
            self.transform = SynchronizedTransform_MoGe(moge_augmentation=self.moge_augmentation)
        elif self.resolutions is not None:
            self.transform = SynchronizedTransformVideoCut3r(seed=777)
        else:
            self.transform = SynchronizedTransform_ScanNetVideo(H=resize[0], W=resize[1]) if transform else None

    def _organize_by_scene(self):
        """Organize frames by scene name.
        Assumes parquet is pre-sorted by (scene_name, frame_number).
        """
        df = pd.read_parquet(self.parquet)
        
        # Sort by scene_name and frame_number to ensure temporal order
        df = df.sort_values(['scene_name', 'frame_number'])
        
        # Group by scene_name (much faster than iterrows)
        # sort=False preserves the original row order within each group
        scene_frames = {}
        for scene_name, scene_df in df.groupby('scene_name', sort=False):
            # Convert each row to dict efficiently
            # Row order is preserved, so frames are in temporal order
            frames = scene_df[['frame_number', 'rgb_path', 'depth_path', 
                               'intrinsics', 'extrinsics']].to_dict('records')
            scene_frames[scene_name] = frames
        
        return scene_frames

    def _create_frame_list(self):
        """Create a flat list of all frames with scene context."""
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
        """Check if we can sample a valid sequence from this starting point."""
        if num_frames is None:
            num_frames = self.T
        
        frames = self.scene_frames[scene_name]
        if start_idx + (num_frames - 1) * stride >= len(frames):
            return False
        return True

    def _load_single_frame(self, frame_data):
        """Load a single frame's RGB, depth, and intrinsics for ScanNet."""
        try:
            # Load RGB
            rgb_data = download_file(frame_data['rgb_path'])
            rgb_image = Image.open(io.BytesIO(rgb_data)).convert('RGB')
            del rgb_data  # Free memory immediately
            
            # Load depth (ScanNet depth is 16-bit PNG in millimeters; convert to meters)
            depth_data = download_file(frame_data['depth_path'])
            depth_pil = Image.open(io.BytesIO(depth_data))
            del depth_data  # Free memory immediately
            
            depth_array = np.array(depth_pil).astype(np.float32) / 1000.0
            del depth_pil  # Free memory immediately
            
            depth_array[~np.isfinite(depth_array)] = 0
            depth_image = Image.fromarray(depth_array)
            del depth_array  # Free memory immediately

            # Get intrinsics and extrinsics
            intrinsics = np.array(frame_data['intrinsics']).reshape(3, 3).copy()
            extrinsics = np.array(frame_data['extrinsics']).reshape(4, 4).copy()
            
            return rgb_image, depth_image, intrinsics, extrinsics, True
            
        except Exception as e:
            print(f"[scannet_video_new] Error loading frame: {str(e)}")
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
            scene_name = start_frame_info['scene_name']
            start_idx = start_frame_info['frame_idx_in_scene']
            frames = self.scene_frames[scene_name]
            

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
                if not self._can_sample_sequence(scene_name, start_idx, stride, num_frames=num_frames):
                    # Try reducing stride step by step
                    is_success = False
                    while stride > max(self.stride_range[0], 1):
                        stride -= 1
                        if self._can_sample_sequence(scene_name, start_idx, stride, num_frames=num_frames):
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
                        rgb_images.append(Image.new('RGB', (540, 360)))
                        depth_images.append(Image.fromarray(np.zeros((360, 540), dtype=np.float32)))
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
                        rgb_images, depth_images, intrinsics_list, extrinsics_list, valid_frames, 
                        self.near_plane, self.far_plane, self.depth_type, resolution_idx, stride=None, no_depth_mask_inf=True, rng=rng
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
                # No infinity mask for ScanNet sensor depth
                inf_depth_mask = torch.zeros_like(depth_tensor).bool()

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
                'valid': all(valid_frames),
                'depth_type': self.depth_type,
            }
        
        except Exception as e:
            print(f"[scannet_video_new] Error processing frame: {str(e)}")
            
            if num_frames is None:
                num_frames = self.T
            
            # Return dummy data with valid=False
            dummy_rgb = torch.zeros(num_frames, 3, self.frame_size[0], self.frame_size[1])
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
                'frame_positions': list(range(num_frames)),
                'is_video': True,
                'valid': False,
                'depth_type': self.depth_type,
            }


def generate_parquet_file():
    """
    Generate parquet file from preprocessed ScanNet data.
    Assumes preprocess_scannet.py produced directories:
      DATA_DIR/<split>/<scene>/{color,depth,cam}
    where cam/*.npz stores intrinsics (3x3) and pose (4x4).
    """
    import tqdm
    
    DATA_DIR = "/mnt/localssd/scannetv2_processed2"
    data_path = DATASET_CONFIG["scannet"]["prefix"]
    splits = ["scans_train"]

    # Accumulate records
    df_columns = ['rgb_path', 'depth_path', 'intrinsics', 'extrinsics', 'scene_name', 'frame_number']
    records = []

    for split in splits:
        split_dir = os.path.join(DATA_DIR, split)
        if not os.path.exists(split_dir):
            print(f"Warning: split directory missing: {split_dir}")
            continue
        for scene in tqdm.tqdm(sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])):
            scene_dir = os.path.join(split_dir, scene)
            rgb_dir = os.path.join(scene_dir, "color")
            depth_dir = os.path.join(scene_dir, "depth")
            cam_dir = os.path.join(scene_dir, "cam")
            if not all(os.path.exists(d) for d in [rgb_dir, depth_dir, cam_dir]):
                print(f"Warning: missing subdirs for scene {scene}; skipping")
                continue

            rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
            for i, rgb_file in enumerate(rgb_files):
                stem = os.path.splitext(rgb_file)[0]
                depth_file = f"{stem}.png"
                cam_file = f"{stem}.npz"
                
                frame_number = int(stem) // 10

                local_cam_path = os.path.join(cam_dir, cam_file)
                if not os.path.exists(os.path.join(depth_dir, depth_file)) or not os.path.exists(local_cam_path):
                    print(f"Warning: missing depth or cam for {scene}/{stem}; skipping")
                    continue

                # Build paths
                rgb_path = os.path.join(data_path, split, scene, "color", rgb_file)
                depth_path = os.path.join(data_path, split, scene, "depth", depth_file)

                with np.load(local_cam_path) as cam:
                    intrinsics = cam["intrinsics"]
                    pose = cam["pose"]

                intrinsics_flat = intrinsics.flatten().tolist()
                extrinsics_flat = pose.flatten().tolist()

                records.append({
                    'rgb_path': rgb_path,
                    'depth_path': depth_path,
                    'intrinsics': intrinsics_flat,
                    'extrinsics': extrinsics_flat,
                    'scene_name': scene,
                    'frame_number': frame_number
                })
                

    # Save parquet
    df_new = pd.DataFrame.from_records(records, columns=df_columns)
    parquet_filename = "scannet_train.parquet"
    df_new.to_parquet(parquet_filename)

    print(f"length of records: {len(records)}")

    # Save and clean up
    upload_file(parquet_filename, DATASET_CONFIG["scannet"]["parquet_path"])
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
    
    os.makedirs("data_vis", exist_ok=True)
    print("\nTesting ScanNet video dataset with new sampling...")

    save_dir = "data_vis/scannet_new"
    os.makedirs(save_dir, exist_ok=True)
    
    resolutions = [[0,1]] * 10000
    video_dataset = VideoDepthScanNetNew(
        T=10, 
        stride_range=(1, 5),
        transform=True,
        resolutions=resolutions, 
        use_moge=True,
        use_cut3r_frame_sampling=True,
        video_prob=0.6,
        fix_interval_prob=0.6,
        block_shuffle=16,
    )
    print(f"Video dataset length: {len(video_dataset)}")

    # breakpoint()
    # dataloader_sampler = make_dynamic_sampler(
    #     video_dataset, 
    #     batch_size=1, 
    #     number_of_resolutions=len(resolutions),
    #     min_num_frames=8, 
    #     max_num_frames=8, 
    #     shuffle=True, 
    #     drop_last=True
    # )

    # dataloader_sampler = make_dynamic_sampler(
    #     video_dataset, 
    #     target_total_views=40, 
    #     number_of_resolutions=len(resolutions),
    #     min_num_frames=4, 
    #     max_num_frames=40, 
    #     shuffle=True, 
    # )

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
            
            rgb_writer = video_writer.FFMPEG_VideoWriter(filename=f"data_vis/scannet_new_video_rgb_{idx}.mp4", size=(W, H), fps=2)
            depth_writer = video_writer.FFMPEG_VideoWriter(filename=f"data_vis/scannet_new_video_depth_{idx}.mp4", size=(W, H), fps=2)
            mask_writer = video_writer.FFMPEG_VideoWriter(filename=f"data_vis/scannet_new_video_mask_{idx}.mp4", size=(W, H), fps=2)

            # prev_depth_video = load_depth_video(f"data_vis/tartanair_video_depth_{idx}.mp4")
            
            print("Saving video frames...")
            for t in tqdm.tqdm(range(video_sample['rgb'].shape[0])):
                # Process RGB frame
                rgb_frame = (video_sample['rgb'][t].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                
                # Process depth frame
                depth = video_sample['metric_depth'][t].numpy().transpose(1, 2, 0).squeeze()

                mask = video_sample['valid_mask'][t].numpy().transpose(1, 2, 0).squeeze()

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
                
            rgb_writer.close()
            depth_writer.close()
            mask_writer.close()
            
            print("Video sequence test completed successfully!")
            print("Saved videos:")
            print(f"  - data_vis/scannet_new_video_rgb_{idx}.mp4")
            print(f"  - data_vis/scannet_new_video_depth_{idx}.mp4") 
            print(f"  - data_vis/scannet_new_video_mask_{idx}.mp4")
            print(f"  - data_vis/scannet_new_video_frame0_pointmap_{idx}.ply")
    else:
        print("No valid video sequences found in dataset.")