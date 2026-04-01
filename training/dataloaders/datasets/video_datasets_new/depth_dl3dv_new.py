import os
import io
import json
import pandas as pd
import tqdm
from torch.utils.data import Dataset
from training.dataloaders.data_io import download_file, upload_file, download_file_with_cache
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

from third_party.pi3.utils.geometry import homogenize_points, se3_inverse, depth_edge


class SynchronizedTransform_DL3DVVideo:
    def __init__(self, H, W):
        self.resize = transforms.Resize((H, W))
        self.resize_depth = transforms.Resize((H, W), interpolation=Image.NEAREST)
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.to_tensor = transforms.ToTensor()
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
            rgb_image = self.resize(rgb_image)
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


class VideoDepthDL3DVNew(Dataset):
    depth_type = "placeholder"  # DL3DV has no depth data
    is_metric_scale = False

    def __init__(self, T=5, stride_range=(1, 5), transform=True, resize=(360, 640), near_plane=1e-5, far_plane=200.0, 
                 resolutions=None, use_moge=False, moge_augmentation=None, use_cut3r_frame_sampling=False, 
                 video_prob=0.6, fix_interval_prob=0.6, block_shuffle=25):
        """
        Video sequence dataset for DL3DV data.
        
        Args:
            T (int): Number of frames to sample from each scene
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
        # t1 = time.time()
        parquet_data = download_file(DATASET_CONFIG["dl3dv"]["parquet_path"])
        # parquet_data = download_file_with_cache(DATASET_CONFIG["dl3dv"]["parquet_path"])
        # t2 = time.time()
        # print(f"Time taken to download parquet: {t2 - t1} seconds")
        
        self.parquet = io.BytesIO(parquet_data)


        self.T = T
        self.stride_range = stride_range
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.frame_size = resize

        self.resolutions = resolutions
        self.use_moge = use_moge
        self.moge_augmentation = moge_augmentation

        # Sophisticated sampling parameters
        self.use_cut3r_frame_sampling = use_cut3r_frame_sampling
        self.video_prob = video_prob
        self.fix_interval_prob = fix_interval_prob
        self.block_shuffle = block_shuffle

        # Load scene list - each row is one scene
        self.scene_list = self._load_scene_list()

        if self.use_moge:
            self.transform = SynchronizedTransform_MoGe(moge_augmentation=self.moge_augmentation)   
        elif self.resolutions is not None:
            self.transform = SynchronizedTransformVideoCut3r(seed=777)
        else:
            self.transform = SynchronizedTransform_DL3DVVideo(H=resize[0], W=resize[1]) if transform else None

    def _load_scene_list(self):
        """Load the list of frames from parquet. Each row = one frame.
        Assumes parquet is pre-sorted by (scene_name, frame_number).
        """
        df = pd.read_parquet(self.parquet)
        
        # Group by scene_name (much faster than iterrows)
        # sort=False preserves the original row order within each group
        scene_list = []
        for scene_name, scene_df in df.groupby('scene_name', sort=False):
            # Convert each row to dict efficiently
            # Row order is preserved, so frames are in temporal order
            frames = scene_df[['frame_number', 'rgb_path', 'intrinsics', 'extrinsics']].to_dict('records')
            
            scene_list.append({
                'scene_name': scene_name,
                'frames': frames,
                'num_frames': len(frames)
            })
        
        return scene_list

    def __len__(self):
        return len(self.scene_list)

    def _load_single_frame(self, frame_data):
        """Load a single frame's RGB and camera data."""
        try:
            # Load RGB
            rgb_data = download_file(frame_data['rgb_path'])
            rgb_image = Image.open(io.BytesIO(rgb_data)).convert('RGB')
            del rgb_data  # Free memory immediately
            
            # Load camera parameters (already scaled to 960p in parquet)
            intrinsics = np.array(frame_data['intrinsics']).reshape(3, 3).copy()
            extrinsics = np.array(frame_data['extrinsics']).reshape(4, 4).copy()
            
            return rgb_image, intrinsics, extrinsics, True
            
        except Exception as e:
            print(f"[dl3dv] Error loading frame: {str(e)}")
            return None, None, None, False

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
            'fin_mask': dummy_fin_mask,
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

            # Get the scene info
            scene_info = self.scene_list[idx]
            num_scene_frames = scene_info['num_frames']
            scene_frames = scene_info['frames']
            
            # Sample frame indices from the scene
            if self.use_cut3r_frame_sampling:
                ids_all = list(range(num_scene_frames))
                start_idx = random.randint(0, max(0, num_scene_frames - 1 - num_frames * self.stride_range[0]))
                
                try:
                    is_success, frame_positions, is_video = get_seq_from_start_id(
                        num_frames, start_idx, ids_all, rng, min_interval=self.stride_range[0], 
                        max_interval=self.stride_range[1], video_prob=self.video_prob, 
                        fix_interval_prob=self.fix_interval_prob, block_shuffle=self.block_shuffle
                    )
                except Exception as e:
                    is_success = False
            else:
                is_video = True
                # Randomly sample stride and starting frame
                stride = random.randint(self.stride_range[0], self.stride_range[1])
                
                # Calculate valid starting range
                max_start_idx = max(0, num_scene_frames - (num_frames - 1) * stride - 1)
                
                if max_start_idx <= 0:
                    # Scene too short, reduce stride
                    stride = max(1, (num_scene_frames - 1) // (num_frames - 1))
                    max_start_idx = max(0, num_scene_frames - (num_frames - 1) * stride - 1)
                
                start_idx = random.randint(0, max(0, max_start_idx))
                frame_positions = [min(start_idx + i * stride, num_scene_frames - 1) for i in range(num_frames)]
                is_success = True

            if not is_success:
                # Try another scene
                next_idx = (idx + random.randint(1, self.__len__())) % self.__len__()
                if len_tuple_idx == 3:
                    return self.__getitem__((next_idx, num_frames, resolution_idx))
                elif len_tuple_idx == 2:
                    return self.__getitem__((next_idx, num_frames))
                elif len_tuple_idx == 1:
                    return self.__getitem__(next_idx)
                else:
                    raise ValueError(f"Invalid tuple length: {len_tuple_idx}")
            
            # Load the frames
            rgb_images = []
            intrinsics_list = []
            extrinsics_list = []
            valid_frames = []
            
            for frame_idx in frame_positions:
                frame_data = scene_frames[frame_idx]
                rgb_img, intrinsics, extrinsics, valid = self._load_single_frame(frame_data)
                
                if valid:
                    rgb_images.append(rgb_img)
                    intrinsics_list.append(intrinsics)
                    extrinsics_list.append(extrinsics)
                    valid_frames.append(True)
                else:
                    # Create dummy data for failed frames
                    if rgb_images:  # Use previous frame as fallback
                        rgb_images.append(rgb_images[-1])
                        intrinsics_list.append(intrinsics_list[-1].copy())
                        extrinsics_list.append(extrinsics_list[-1].copy())
                    else:
                        # Try another scene if first frame fails
                        next_idx = (idx + random.randint(1, self.__len__())) % self.__len__()
                        return self.__getitem__((next_idx, num_frames, resolution_idx))
                    valid_frames.append(False)
            
            # Create placeholder depth images
            depth_images = []
            for rgb_img in rgb_images:
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
                        no_depth_mask_inf=True, rng=rng, augmentation=True
                    )
                    processed_data['is_metric_scale'] = self.is_metric_scale
                    return processed_data
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

                # No real depth, so all depth-related tensors are dummy
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
            rgb_sequence = torch.stack(processed_rgb, dim=0)
            metric_depth_sequence = torch.stack(processed_metric_depth, dim=0)
            valid_mask_sequence = torch.stack(processed_valid_mask, dim=0)
            inf_mask_sequence = torch.stack(processed_inf_mask, dim=0)
            fin_mask_sequence = valid_mask_sequence
            intrinsics_sequence = torch.stack(processed_intrinsics, dim=0)
            extrinsics_sequence = torch.stack(processed_extrinsics, dim=0)

            return {
                'rgb': rgb_sequence,
                'metric_depth': metric_depth_sequence,
                'valid_mask': valid_mask_sequence,
                'fin_mask': fin_mask_sequence,
                'inf_mask': inf_mask_sequence,
                'intrinsics': intrinsics_sequence,
                'extrinsics': extrinsics_sequence,
                'valid': all(valid_frames),
                'depth_type': self.depth_type,
            }
        
        except Exception as e:
            print(f"[dl3dv] Error processing scene: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return self._return_dummy_data(num_frames if 'num_frames' in locals() else self.T)


def generate_parquet_file():
    """
    Generate parquet file for DL3DV dataset.
    Scans /mnt/localssd/dl3dv_raw/ directory and creates metadata with per-frame camera data.
    Similar to blendedmvs, stores intrinsics and extrinsics for each frame.
    """
    DL3DV_RAW_DIR = "/mnt/localssd/dl3dv_raw"
    data_prefix = DATASET_CONFIG["dl3dv"]["prefix"]
    
    # Subsets to process
    subsets = ['1K', '2K', '3K', '4K', '5K', '6K', '7K', '8K']
    
    df_columns = ['rgb_path', 'intrinsics', 'extrinsics', 'scene_name', 'frame_number']
    records = []
    
    for subset in subsets:
        subset_dir = os.path.join(DL3DV_RAW_DIR, subset)
        
        if not os.path.exists(subset_dir):
            print(f"Subset directory {subset_dir} does not exist, skipping...")
            continue
        
        # Get all scene directories
        scene_dirs = [d for d in os.listdir(subset_dir) 
                      if os.path.isdir(os.path.join(subset_dir, d))]
        
        print(f"Processing subset {subset} with {len(scene_dirs)} scenes")
        
        for scene_name in tqdm.tqdm(scene_dirs, desc=f"Processing {subset}"):
            scene_dir = os.path.join(subset_dir, scene_name)
            
            # Check for required files
            images_dir = os.path.join(scene_dir, "images_4")
            transform_json_path = os.path.join(scene_dir, "transforms.json")
            
            if not os.path.exists(images_dir) or not os.path.exists(transform_json_path):
                print(f"Skipping {scene_name}: missing images_4 or transform.json")
                continue
            
            # Load transform.json
            try:
                with open(transform_json_path, 'r') as f:
                    transform_data = json.load(f)
                
                # Get image list
                image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
                
                if len(image_files) == 0:
                    print(f"Skipping {scene_name}: no images found")
                    continue
                
                # Get actual image resolution (960p)
                sample_image_path = os.path.join(images_dir, image_files[0])
                with Image.open(sample_image_path) as img:
                    original_width, original_height = img.size
                
                # Get 4K resolution from transform.json
                w_4k = transform_data['w']
                h_4k = transform_data['h']
                fl_x_4k = transform_data['fl_x']
                fl_y_4k = transform_data['fl_y']
                cx_4k = transform_data['cx']
                cy_4k = transform_data['cy']
                
                # Calculate scale factors (4K -> 960p)
                scale_w = original_width / w_4k
                scale_h = original_height / h_4k
                
                # Skip scenes with too few frames
                frames_data = transform_data['frames']
                if len(frames_data) < 10:
                    print(f"Skipping {scene_name}: only {len(frames_data)} frames")
                    continue
                
                # Verify frame count matches images
                num_frames = min(len(frames_data), len(image_files))
                
                # Conversion matrix from OpenGL to OpenCV convention
                flip_yz = np.array([
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                
                # Process each frame
                for frame_idx in range(num_frames):
                    frame_data = frames_data[frame_idx]
                    
                    # Create intrinsics matrix (scaled to 960p)
                    intrinsics = np.eye(3, dtype=np.float32)
                    intrinsics[0, 0] = fl_x_4k * scale_w  # fx
                    intrinsics[1, 1] = fl_y_4k * scale_h  # fy
                    intrinsics[0, 2] = cx_4k * scale_w    # cx
                    intrinsics[1, 2] = cy_4k * scale_h    # cy
                    
                    # Extract transform matrix (c2w in OpenGL/NeRF format)
                    transform_matrix = np.array(frame_data['transform_matrix'], dtype=np.float32)
                    
                    # Convert from OpenGL/NeRF to OpenCV convention
                    extrinsics = transform_matrix @ flip_yz  # c2w in OpenCV convention
                    
                    # Create path for this image
                    image_filename = f"frame_{frame_idx+1:05d}.png"
                    rgb_data_path = os.path.join(data_prefix, subset, scene_name, "images_4", image_filename)
                    
                    # Flatten matrices for storage
                    intrinsics_flat = intrinsics.flatten().tolist()
                    extrinsics_flat = extrinsics.flatten().tolist()
                    
                    records.append({
                        'rgb_path': rgb_data_path,
                        'intrinsics': intrinsics_flat,
                        'extrinsics': extrinsics_flat,
                        'scene_name': f"{subset}_{scene_name}",
                        'frame_number': frame_idx,
                    })
                
            except Exception as e:
                print(f"Error processing {scene_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Create DataFrame
    df_new = pd.DataFrame.from_records(records, columns=df_columns)
    
    # Save parquet file
    parquet_filename = "dl3dv_1k_8k.parquet"
    df_new.to_parquet(parquet_filename)
    print(f"Created parquet file with {len(df_new)} frames")
    print(f"Frames per subset:")
    for subset in subsets:
        subset_frames = df_new[df_new['scene_name'].str.startswith(subset + '_')]
        if len(subset_frames) > 0:
            num_scenes = subset_frames['scene_name'].nunique()
            print(f"  {subset}: {len(subset_frames)} frames from {num_scenes} scenes")
    
    # Save the parquet file
    upload_file(
        parquet_filename,
        DATASET_CONFIG["dl3dv"]["parquet_path"]
    )
    
    # Remove the local file
    os.remove(parquet_filename)
    
    print(f"Saved parquet file")


if __name__ == "__main__":
    # Uncomment to generate parquet file
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
    
    os.makedirs("data_vis", exist_ok=True)
    print("\nTesting spatialhq video dataset with new sampling...")

    dataset_name = "dl3dv"
    save_dir = f"data_vis/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    resolutions = [[0,1]] * 10000
    video_dataset = VideoDepthDL3DVNew(
        T=8, 
        stride_range=(6, 6),
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
            if sample_count > 10: break

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
            # rgb_noaug_writer = video_writer.FFMPEG_VideoWriter(filename=f"{save_dir}/ideo_rgb_noaug_{idx}.mp4", size=(W, H), fps=2)
            # depth_writer = video_writer.FFMPEG_VideoWriter(filename=f"{save_dir}/video_depth_{idx}.mp4", size=(W, H), fps=2)
            # mask_writer = video_writer.FFMPEG_VideoWriter(filename=f"{save_dir}/video_mask_{idx}.mp4", size=(W, H), fps=2)

            # prev_depth_video = load_depth_video(f"data_vis/tartanair_video_depth_{idx}.mp4")
            all_gt_pts = []
            all_gt_pts_rgb = []
            print("Saving video frames...")
            for t in tqdm.tqdm(range(video_sample['rgb'].shape[0])):
                # Process RGB frame
                rgb_frame = (video_sample['rgb_noaug'][t].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                

                rgb_writer.write_frame(rgb_frame)
                
            rgb_writer.close()
            
            print("Video sequence test completed successfully!")
    else:
        print("No valid video sequences found in dataset.")

