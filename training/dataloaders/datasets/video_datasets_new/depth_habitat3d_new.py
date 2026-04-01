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
import einops

from training.dataloaders.datasets.video_datasets_new.synchronized_video_transform import SynchronizedTransformVideoCut3r
from training.dataloaders.datasets.video_datasets_new.synchronized_video_transform_moge import SynchronizedTransform_MoGe

from training.dataloaders.datasets.video_datasets_new.frame_sampling_utils import get_seq_from_start_id

from third_party.pi3.utils.geometry import homogenize_points, se3_inverse, depth_edge


class SynchronizedTransform_Habitat3D:
    def __init__(self, H, W):
        self.resize          = transforms.Resize((H, W))
        self.resize_depth    = transforms.Resize((H, W), interpolation=Image.NEAREST)
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


class VideoDepthHabitat3DNew(Dataset):
    depth_type = "sfm"  # Habitat3D has metric depth data
    dataset_config_key = "habitat3d"  # Override this in child classes

    def __init__(self, T=5, pre_defined_num_frames=None, stride_range=(1, 5), transform=True, resize=(360, 640), near_plane=1e-3, far_plane=50.0, 
                 resolutions=None, use_moge=False, moge_augmentation=None,
                 use_cut3r_frame_sampling=False, video_prob=0.6, fix_interval_prob=0.6, block_shuffle=16):
        """
        Video sequence dataset for Habitat3D data.
        Dataset is organized by scene/sample (not individual frames).
        
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
        parquet_data = download_file(DATASET_CONFIG[self.dataset_config_key]["parquet_path"])
        self.parquet = io.BytesIO(parquet_data)

        self.T = T
        self.pre_defined_num_frames = pre_defined_num_frames
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
            self.transform = SynchronizedTransform_Habitat3D(H=resize[0], W=resize[1]) if transform else None

    def _load_scene_list(self):
        """Load the list of scenes from parquet. Each row = one scene."""
        df = pd.read_parquet(self.parquet)
        scene_list = []
        
        for _, row in df.iterrows():
            scene_info = {
                'scene_id': row['scene_id'],
                'subscene_id': row['subscene_id'],
                'scene_name': row['scene_name'],
                'rgb_dir': row['rgb_dir'],
                'depth_dir': row['depth_dir'],
                'camera_dir': row['camera_dir'],
                'num_views': row['num_views']
            }
            scene_list.append(scene_info)
        
        return scene_list

    def __len__(self):
        return len(self.scene_list)

    def _load_frame_data(self, scene_info, view_idx):
        """Load a single frame (RGB, depth, camera) from local storage."""
        try:
            view_label = f"view{view_idx:05d}"
            
            # Load RGB image
            rgb_path = os.path.join(scene_info['rgb_dir'], f"{view_label}.png")
            rgb_data = download_file(rgb_path)
            rgb_image = Image.open(io.BytesIO(rgb_data)).convert('RGB')
            del rgb_data  # Free memory immediately
            
            # Load depth image (16-bit PNG, need to convert back to meters)
            depth_path = os.path.join(scene_info['depth_dir'], f"{view_label}.png")
            depth_data = download_file(depth_path)
            depth_pil = Image.open(io.BytesIO(depth_data))
            del depth_data  # Free memory immediately
            
            depth_image_16bit = np.array(depth_pil)
            del depth_pil  # Free memory immediately
            
            depth_meters = depth_image_16bit.astype(np.float32) / 1000.0  # Convert back to meters
            del depth_image_16bit  # Free memory immediately
            
            depth_image = Image.fromarray(depth_meters)
            del depth_meters  # Free memory immediately
            
            # Load camera parameters
            camera_path = os.path.join(scene_info['camera_dir'], f"{view_label}.npz")
            camera_data = download_file(camera_path)
            camera_params = np.load(io.BytesIO(camera_data))
            intrinsics = camera_params['intrinsic'].copy()  # Copy to break reference
            extrinsics = camera_params['extrinsic'].copy()  # Copy to break reference
            camera_params.close()  # Close npz file
            del camera_data  # Free memory immediately
            
            return rgb_image, depth_image, intrinsics, extrinsics, True
            
        except Exception as e:
            print(f"[{self.dataset_config_key}] Error loading frame data: {str(e)}")
            return None, None, None, None, False

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

            if self.pre_defined_num_frames is not None:
                num_frames = self.pre_defined_num_frames # NOTE force to use the pre-defined number of frames

            rng = np.random.default_rng(random.randint(0, 2 ** 32 - 1))

            # Get the scene info
            scene_info = self.scene_list[idx]
            scene_name = scene_info['scene_name']
            num_views = scene_info['num_views']
            
            # Sample frame indices from the scene
                    # if self.use_cut3r_frame_sampling:
                    #     ids_all = list(range(num_views))
                    #     start_idx = random.randint(0, max(0, num_views - 1 - num_frames * self.stride_range[0]))
                    #     if start_idx + num_frames * self.stride_range[0] > num_views - 1:
                    #         next_idx = (idx + random.randint(1, self.__len__())) % self.__len__()
                    #         return self.__getitem__((next_idx, num_frames, resolution_idx))
                        
                    #     try:
                    #         is_success, frame_positions, is_video = get_seq_from_start_id(
                    #             num_frames, start_idx, ids_all, rng, min_interval=self.stride_range[0], max_interval=self.stride_range[1], 
                    #             video_prob=self.video_prob, fix_interval_prob=self.fix_interval_prob, block_shuffle=self.block_shuffle
                    #         )
                    #     except Exception as e:
                    #         is_success = False
                    # else:
                    #     is_video = True
                    #     # Randomly sample stride and starting frame
                    #     stride = random.randint(self.stride_range[0], self.stride_range[1])
                        
                    #     # Calculate valid starting range
                    #     max_start_idx = max(0, num_views - (num_frames - 1) * stride - 1)
                        
                    #     if max_start_idx <= 0:
                    #         # Scene too short, reduce stride
                    #         stride = max(1, (num_views - 1) // (num_frames - 1))
                    #         max_start_idx = max(0, num_views - (num_frames - 1) * stride - 1)
                        
                    #     start_idx = random.randint(0, max(0, max_start_idx))
                    #     frame_positions = [min(start_idx + i * stride, num_views - 1) for i in range(num_frames)]
                    #     is_success = True
            # NOTE for habitat3d, we use the first num_views frames for each scene to ensure a continous 3d space
            frame_positions = list(range(num_frames))
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
            
            # Load frames
            rgb_images = []
            depth_images = []
            intrinsics_list = []
            extrinsics_list = []
            valid_frames = []
            
            for frame_idx in frame_positions:
                rgb_img, depth_img, intrinsics, extrinsics, load_success = self._load_frame_data(scene_info, frame_idx)
                
                if not load_success:
                    next_idx = (idx + random.randint(1, self.__len__())) % self.__len__()
                    return self.__getitem__((next_idx, num_frames, resolution_idx))
                
                rgb_images.append(rgb_img)
                depth_images.append(depth_img)
                intrinsics_list.append(intrinsics.copy())
                extrinsics_list.append(extrinsics.copy())
                valid_frames.append(True)

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
                    )
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
            processed_fin_mask = []
            processed_intrinsics = []
            processed_extrinsics = []

            for t in range(num_frames):
                rgb_tensor = rgb_tensors[t]
                depth_tensor = depth_tensors[t]
                intrinsics_tensor = intrinsics_tensors[t]
                extrinsics_tensor = extrinsics_tensors[t]

                # Process RGB 
                rgb_tensor = rgb_tensor * 2.0 - 1.0  # [-1,1]

                # Process depth
                metric_tensor = depth_tensor
                
                # Create valid mask (non-zero depth values)
                valid_depth_mask = (metric_tensor > self.near_plane) & (metric_tensor < self.far_plane)
                
                # For Habitat3D, we don't expect infinite depth values
                inf_depth_mask = torch.zeros_like(depth_tensor).bool()
                fin_mask = valid_depth_mask & ~inf_depth_mask

                processed_rgb.append(rgb_tensor)
                processed_metric_depth.append(metric_tensor)
                processed_valid_mask.append(valid_depth_mask)
                processed_inf_mask.append(inf_depth_mask)
                processed_fin_mask.append(fin_mask)
                processed_intrinsics.append(intrinsics_tensor)
                processed_extrinsics.append(extrinsics_tensor)

            # Stack into [T, C, H, W] tensors
            rgb_sequence = torch.stack(processed_rgb, dim=0)  # [T, 3, H, W]
            metric_depth_sequence = torch.stack(processed_metric_depth, dim=0)  # [T, 1, H, W]
            valid_mask_sequence = torch.stack(processed_valid_mask, dim=0)  # [T, 1, H, W]
            inf_mask_sequence = torch.stack(processed_inf_mask, dim=0)  # [T, 1, H, W]
            fin_mask_sequence = torch.stack(processed_fin_mask, dim=0)  # [T, 1, H, W]
            intrinsics_sequence = torch.stack(processed_intrinsics, dim=0)  # [T, 3, 3]
            extrinsics_sequence = torch.stack(processed_extrinsics, dim=0)  # [T, 4, 4]

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
            print(f"[{self.dataset_config_key}] Error processing scene: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return self._return_dummy_data(num_frames if 'num_frames' in locals() else self.T)


def generate_parquet_file(habitat3d_data_dir, output_parquet_filename="habitat3d.parquet"):
    """
    Generate parquet file for Habitat3D dataset.
    
    Args:
        habitat3d_data_dir: Root directory containing the processed Habitat3D data
                           Expected structure: habitat3d_data_dir/{scene_id:08}/rgb/view{idx:05d}.png
        output_parquet_filename: Name of the output parquet file
    """
    
    df_columns = ['scene_id', 'subscene_id', 'rgb_dir', 'depth_dir', 'camera_dir', 'num_views', 'scene_name']
    records = []
    
    # # List all scene directories (directories with 8-digit names)
    # scene_dirs = sorted([d for d in os.listdir(habitat3d_data_dir) 
    #                     if os.path.isdir(os.path.join(habitat3d_data_dir, d)) and d.isdigit() and len(d) == 8])

    scene_dirs = sorted([d for d in os.listdir(habitat3d_data_dir) if os.path.isdir(os.path.join(habitat3d_data_dir, d))])
    
    print(f"Found {len(scene_dirs)} scene directories")
    
    for scene_id in tqdm.tqdm(scene_dirs):
        scene_path = os.path.join(habitat3d_data_dir, scene_id)
        subscene_dirs = sorted([d for d in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, d))])

        for subscene_id in subscene_dirs:
            subscene_path = os.path.join(scene_path, subscene_id)
            rgb_dir = os.path.join(subscene_path, "rgb")
            depth_dir = os.path.join(subscene_path, "depth")
            camera_dir = os.path.join(subscene_path, "camera")

            if not all(os.path.exists(d) for d in [rgb_dir, depth_dir, camera_dir]):
                print(f"Skipping {scene_id} {subscene_id} due to missing directories")
                continue
        
            # Count number of views
            rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
            depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])
            camera_files = sorted([f for f in os.listdir(camera_dir) if f.endswith('.npz')])
            
            # Verify all have the same number of files
            if not (len(rgb_files) == len(depth_files) == len(camera_files)):
                print(f"Skipping {scene_id} {subscene_id} due to mismatched file counts: "
                    f"rgb={len(rgb_files)}, depth={len(depth_files)}, camera={len(camera_files)}")
                continue
            
            num_views = len(rgb_files)
            
            # Skip scenes with too few views
            if num_views < 5:
                print(f"Skipping {scene_id} {subscene_id} due to insufficient views ({num_views})")
                continue
            
            # Create paths
            data_prefix = DATASET_CONFIG["habitat3d"]["prefix"] if "habitat3d" in DATASET_CONFIG else "habitat3d"
            rgb_dir_path = os.path.join(data_prefix, scene_id, subscene_id, "rgb")
            depth_dir_path = os.path.join(data_prefix, scene_id, subscene_id, "depth")
            camera_dir_path = os.path.join(data_prefix, scene_id, subscene_id, "camera")
            
            records.append({
                'scene_id': scene_id,
                'subscene_id': subscene_id,
                'rgb_dir': rgb_dir_path,
                'depth_dir': depth_dir_path,
                'camera_dir': camera_dir_path,
                'num_views': num_views,
                'scene_name': f"{scene_id}_{subscene_id}",
            })
    
    # Create DataFrame to parquet
    df_new = pd.DataFrame.from_records(records, columns=df_columns)
    df_new.to_parquet(output_parquet_filename)
    print(f"Created parquet file with {len(df_new)} scenes")
    print(f"Total views across all scenes: {df_new['num_views'].sum()}")
    print(f"Average views per scene: {df_new['num_views'].mean():.2f}")
    
    # Save if configured
    if "habitat3d" in DATASET_CONFIG:
        try:
            upload_file(output_parquet_filename,
                       DATASET_CONFIG["habitat3d"]["parquet_path"])
            print(f"Saved parquet file")
            # Remove local file after upload
            os.remove(output_parquet_filename)
        except Exception as e:
            print(f"Failed to save: {str(e)}")
    
    return df_new


if __name__ == "__main__":
    import tqdm
    import moviepy.video.io.ffmpeg_writer as video_writer
    import cv2
    from training.dataloaders.batched_sampler import make_sampler
    from torch.utils.data import DataLoader
    import open3d as o3d
    import utils3d
    
    # Test generate_parquet_file
    # Uncomment and set your data directory to generate parquet
    # habitat3d_data_dir = "/mnt/localssd/hm3d_release"
    # generate_parquet_file(habitat3d_data_dir)
    # quit()
    
    os.makedirs("data_vis", exist_ok=True)
    print("\nTesting Habitat3D dataset...")

    dataset_name = "habitat3d_new"
    save_dir = f"data_vis/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    resolutions = [[0, 1]] * 10000
    habitat_dataset = VideoDepthHabitat3DNew(
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

