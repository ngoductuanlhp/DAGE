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

class SynchronizedTransform_MVSSynthVideo:
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


class VideoDepthMVSSynthNew(Dataset):
    depth_type = "synthetic"
    is_metric_scale = False
    # NOTE original resolution is 1920x1080

    def __init__(self, T=5, stride_range=(1, 5), transform=True, resize=(360, 640), high_resolution=False, near_plane=1e-5, far_plane=1000.0, resolutions=None, use_moge=False, moge_augmentation=None,
        use_cut3r_frame_sampling=False, video_prob=0.6, fix_interval_prob=0.6, block_shuffle=16
    ):
        """
        Video sequence dataset for MVS Synth depth data.
        
        Args:
            T (int): Number of frames in each sequence
            stride_range (tuple): Range of strides to randomly sample from (min_stride, max_stride)
            transform (bool): Whether to apply transformations
            resize (tuple): Target size (H, W)
            near_plane (float): Near clipping plane for depth
            far_plane (float): Far clipping plane for depth
        """

        if high_resolution:
            parquet_data = download_file(DATASET_CONFIG["mvs_synth_hr"]["parquet_path"])
        else:
            parquet_data = download_file(DATASET_CONFIG["mvs_synth"]["parquet_path"])
        self.parquet = io.BytesIO(parquet_data)

        self.T = T
        self.stride_range = stride_range
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.high_resolution = high_resolution

        if self.high_resolution:
            resize = (1080, 1920)

        
        self.frame_size = resize
        
        # Load and organize data by video
        self.video_frames = self._organize_by_video()
        self.frame_list = self._create_frame_list()

        self.resolutions = resolutions
        self.use_moge = use_moge
        self.moge_augmentation = moge_augmentation


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
            self.transform = SynchronizedTransform_MVSSynthVideo(H=resize[0], W=resize[1]) if transform else None

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
                # 'cam_path': row['cam_path']
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
        """Load a single frame's RGB, depth, and camera data."""
        try:
            # Load RGB
            rgb_data = download_file(frame_data['rgb_path'])
            rgb_image = Image.open(io.BytesIO(rgb_data)).convert('RGB')
            del rgb_data  # Free memory immediately
            
            # Load depth (numpy array from preprocessing)
            depth_data = download_file(frame_data['depth_path'])
            depth_array = np.load(io.BytesIO(depth_data))
            del depth_data  # Free memory immediately

            sky_threshold = (
                np.percentile(depth_array[depth_array > 0], 98)
                if depth_array[depth_array > 0].size > 0
                else 0
            )
            depth_array[depth_array > sky_threshold] = np.inf

            depth_image = Image.fromarray(depth_array)
            del depth_array  # Free memory immediately
            
            # Load camera data (intrinsics and pose from preprocessing)
            # breakpoint()
            extrinsics = np.array(frame_data['extrinsics'], dtype=np.float32).reshape(4, 4).copy()
            intrinsics = np.array(frame_data['intrinsics'], dtype=np.float32).reshape(3, 3).copy()

            
            
            return rgb_image, depth_image, intrinsics, extrinsics, True
            
        except Exception as e:
            print(f"[mvs_synth_video] Error loading frame: {str(e)}")
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
                    intrinsics_list.append(intrinsics)
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
                # 'stride': stride,
                'valid': all(valid_frames),
                'depth_type': self.depth_type,
            }
        
        except Exception as e:
            print(f"[mvs_synth_video] Error processing frame: {str(e)}")

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
                # 'stride': stride,
                'valid': False,
                'depth_type': self.depth_type,
            }


def generate_parquet_file():
    import glob
    import tqdm
    # Assuming processed data is available
    DATA_DIR = "/mnt/localssd/mvs_synth_processed/GTAV_1080"  # Path to processed data
    data_path = DATASET_CONFIG["mvs_synth_hr"]["prefix"]
    
    sequences = sorted([s for s in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, s))])
    # df_new = pd.DataFrame(columns=['rgb_path', 'depth_path', 'cam_path', 'video_name', 'frame_number'])
    df_columns = ['rgb_path', 'depth_path', 'intrinsics', 'extrinsics', 'video_name', 'frame_number']
    records = []
    
    for seq in tqdm.tqdm(sequences):
        rgb_dir = os.path.join(DATA_DIR, seq, 'rgb')
        depth_dir = os.path.join(DATA_DIR, seq, 'depth')
        cam_dir = os.path.join(DATA_DIR, seq, 'cam')
        
        if not all(os.path.exists(d) for d in [rgb_dir, depth_dir, cam_dir]):
            continue
            
        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
        
        for rgb_file in rgb_files:
            basename = rgb_file[:-4]  # Remove .jpg extension
            frame_number = int(basename)  # Assuming basename is frame number
            
            depth_file = basename + '.npy'
            cam_file = basename + '.npz'
            
            rgb_path = os.path.join(rgb_dir, rgb_file).replace(DATA_DIR, data_path)
            depth_path = os.path.join(depth_dir, depth_file).replace(DATA_DIR, data_path)
            cam_path = os.path.join(cam_dir, cam_file).replace(DATA_DIR, data_path)

            camera_data = np.load(os.path.join(cam_dir, cam_file))
            intrinsics = camera_data['intrinsics']
            extrinsics = camera_data['pose']

            intrinsics_flat = intrinsics.flatten().tolist()
            extrinsics_flat = extrinsics.flatten().tolist()
            
            # Check if all files exist
            # if not all(os.path.exists(p.replace(data_path, DATA_DIR)) for p in [rgb_path, depth_path, cam_path]):
            #     continue
            
            # df_new = pd.concat([df_new, pd.DataFrame({
            #     'rgb_path': [rgb_path],
            #     'depth_path': [depth_path],
            #     'cam_path': [cam_path],
            #     'video_name': [seq],
            #     'frame_number': [frame_number]
            # })], ignore_index=True)

            

            records.append({
                'rgb_path': rgb_path,
                'depth_path': depth_path,
                'intrinsics': intrinsics_flat,
                'extrinsics': extrinsics_flat,
                'video_name': seq,
                'frame_number': frame_number
            })

            # breakpoint()

    # df_new.to_parquet("mvs_synth_train.parquet")
    df_new = pd.DataFrame.from_records(records, columns=df_columns)
    
    # Save parquet file
    parquet_filename = "mvs_synth_hr_train.parquet"
    df_new.to_parquet(parquet_filename)
    print(f"Created parquet file with {len(df_new)} entries")


    
    # Save the parquet file
    upload_file("mvs_synth_hr_train.parquet", DATASET_CONFIG["mvs_synth_hr"]["parquet_path"])
    # Remove the local file
    os.remove("mvs_synth_hr_train.parquet")


if __name__ == "__main__":
    # generate_parquet_file()
    # quit()

    # Test video dataset:
    import tqdm
    from Unip.unip.util.pointmap_util import depth_to_pointmap, save_pointmap_as_ply
    import moviepy.video.io.ffmpeg_writer as video_writer
    
    os.makedirs("data_vis", exist_ok=True)
    print("\nTesting MVS Synth video dataset...")
    video_dataset = VideoDepthMVSSynthNew(T=40, stride_range=(1, 1), transform=True, high_resolution=True)
    print(f"Video dataset length: {len(video_dataset)}")

    save_dir = "data_vis/mvs_synth"
    os.makedirs(save_dir, exist_ok=True)
    
    if len(video_dataset) > 0:
        video_sample = video_dataset[0]

        indices = np.linspace(0, len(video_dataset)-10, 10, dtype=int)

        for idx in indices:
            video_sample = video_dataset[idx]

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
            
            rgb_writer = video_writer.FFMPEG_VideoWriter(filename=f"{save_dir}/mvs_synth_video_rgb_{idx}.mp4", size=(W, H), fps=2)
            depth_writer = video_writer.FFMPEG_VideoWriter(filename=f"{save_dir}/mvs_synth_video_depth_{idx}.mp4", size=(W, H), fps=2)
            mask_writer = video_writer.FFMPEG_VideoWriter(filename=f"{save_dir}/mvs_synth_video_mask_{idx}.mp4", size=(W, H), fps=2)
            
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
                
                # # Save pointcloud for first frame only
                # if t == 0:
                #     intrinsics = video_sample['intrinsics'][t].numpy()
                #     pointmap = depth_to_pointmap(depth, intrinsics)
                #     save_pointmap_as_ply(pointmap, rgb_frame, f"{save_dir}/mvs_synth_video_frame0_pointmap_{idx}.ply", far_threshold=1000)

            rgb_writer.close()
            depth_writer.close()
            mask_writer.close()
            
            print("Video sequence test completed successfully!")
            print("Saved videos:")
            print(f"  - {save_dir}/mvs_synth_video_rgb_{idx}.mp4")
            print(f"  - {save_dir}/mvs_synth_video_depth_{idx}.mp4") 
            print(f"  - {save_dir}/mvs_synth_video_mask_{idx}.mp4")
            print(f"  - {save_dir}/mvs_synth_video_frame0_pointmap_{idx}.ply")
        else:
            print("No valid video sequences found in dataset.") 