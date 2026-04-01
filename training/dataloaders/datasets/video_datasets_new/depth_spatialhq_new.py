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


from third_party.pi3.utils.geometry import homogenize_points, se3_inverse, depth_edge

def quaternion_to_matrix(quaternions, eps: float = 1e-8):
    """
    Convert 4-dimensional quaternions to 3x3 rotation matrices.
    This is adapted from Pytorch3D:
    https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py

    Args:
        quaternions: Quaternion tensor [..., 4] (order: i, j, k, r)
        eps: Small value for numerical stability

    Returns:
        Rotation matrices [..., 3, 3]
    """

    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return einops.rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def pose_from_quaternion(pose):
    """
    Convert pose from quaternion representation to transformation matrix.

    Args:
        pose: Pose tensor [..., 7] where first 3 elements are translation (t)
              and last 4 elements are quaternion rotation (r)

    Returns:
        w2c_matrix: World-to-camera transformation matrices [..., 3, 4]
    """
    # Input is w2c, pose(n,7) or (n,v,7), output is (N,3,4) w2c matrix
    # Tensor format from https://github.com/pointrix-project/Geomotion/blob/6ab0c364f1b44ab4ea190085dbf068f62b42727c/geomotion/model/cameras.py#L6
    if type(pose) == np.ndarray:
        pose = torch.tensor(pose)
    if len(pose.shape) == 1:
        pose = pose[None]
    quat_t = pose[..., :3]  # Translation
    quat_r = pose[..., 3:]  # Quaternion rotation
    w2c_matrix = torch.zeros((*list(pose.shape)[:-1], 3, 4), device=pose.device)
    w2c_matrix[..., :3, 3] = quat_t
    w2c_matrix[..., :3, :3] = quaternion_to_matrix(quat_r)
    return w2c_matrix

def get_intrinsics_matrices(camera_info, original_width, original_height):
    """
    Convert camera parameters to intrinsics matrices
    Args:
        camera_info: np.array of shape (num_frames, 4) containing [fx, fy, cx, cy] in normalized format
        original_width: int, original image width
        original_height: int, original image height
    Returns:
        intrinsics: np.array of shape (num_frames, 3, 3) containing intrinsics matrices
    """
    # num_frames = camera_info.shape[0]
    
    # Convert normalized parameters to pixel coordinates
    fx_pixel = camera_info[..., 0] * original_width   # fx in pixels
    fy_pixel = camera_info[..., 1] * original_height  # fy in pixels
    cx_pixel = camera_info[..., 2] * original_width   # cx in pixels
    cy_pixel = camera_info[..., 3] * original_height  # cy in pixels
    
    # Create empty array for intrinsics matrices
    intrinsics = np.zeros((*list(camera_info.shape)[:-1], 3, 3))
    
    # Fill in the intrinsics matrices
    # Each matrix will be of form:
    # [[fx,  0, cx],
    #  [ 0, fy, cy],
    #  [ 0,  0,  1]]
    
    intrinsics[..., 0, 0] = fx_pixel  # fx
    intrinsics[..., 1, 1] = fy_pixel  # fy
    intrinsics[..., 0, 2] = cx_pixel  # cx
    intrinsics[..., 1, 2] = cy_pixel  # cy
    intrinsics[..., 2, 2] = 1.0       # last element is always 1
    
    return intrinsics

class SynchronizedTransform_SpatialVIDHQVideo:
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


class VideoDepthSpatialVIDHQNew(Dataset):
    depth_type = "placeholder"  # SpatialVID has no depth data

    def __init__(self, T=5, stride_range=(1, 5), transform=True, resize=(360, 640), near_plane=1e-5, far_plane=1000.0, resolutions=None, use_moge=False, moge_augmentation=None,
        use_cut3r_frame_sampling=False, video_prob=0.6, fix_interval_prob=0.6, block_shuffle=16
    ):
        """
        Video sequence dataset for SpatialVID-HQ data.
        Dataset is organized by video (not individual frames).
        
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
        parquet_data = download_file(DATASET_CONFIG["spatialvidhq_videoonly"]["parquet_path"])
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
            self.transform = SynchronizedTransform_SpatialVIDHQVideo(H=resize[0], W=resize[1]) if transform else None

    def _load_video_list(self):
        """Load the list of videos from parquet. Each row = one video."""
        df = pd.read_parquet(self.parquet)
        video_list = []
        
        for _, row in df.iterrows():
            video_info = {
                'video_name': row['video_name'],
                'rgb_path': row['rgb_path'],
                'pose_path': row['pose_path'],
                'intrinsic_path': row['intrinsic_path'],
                'fps': row['fps'],
                'group_id': row['group_id'],
                'video_id': row['video_id']
            }
            video_list.append(video_info)
        
        return video_list

    def __len__(self):
        return len(self.video_list)

    def _load_video_metadata(self, video_info):
        """Load poses and intrinsics for a video."""
        try:
            # Load poses (camera extrinsics for each frame)
            pose_data = download_file(video_info['pose_path'])
            poses = np.load(io.BytesIO(pose_data))  # Shape: [N, 7] (quaternion format: tx,ty,tz,qi,qj,qk,qr)

            
            # Load intrinsics (camera intrinsics for each frame)
            intrinsic_data = download_file(video_info['intrinsic_path'])
            intrinsics = np.load(io.BytesIO(intrinsic_data))  # Shape: [N, 4] (normalized format: fx,fy,cx,cy)
            
            return poses, intrinsics, True
        except Exception as e:
            print(f"[spatialvid_video] Error loading video metadata: {str(e)}")
            return None, None, False

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
                stride_interval = int(fps / 5)
                actual_frame_indices = [min(idx*stride_interval, num_frames - 1) for idx in frame_indices]
                # print(f"frame_indices: {frame_indices}, actual_frame_indices: {actual_frame_indices}")
                frames = vr.get_batch(actual_frame_indices).asnumpy()  # Shape: [T, H, W, 3]
                
                # Convert to list of PIL Images
                rgb_images = [Image.fromarray(frame) for frame in frames]
                del frames  # Free memory immediately
                
                video_info = {
                    'original_height': original_height,
                    'original_width': original_width,
                    'fps': fps,
                    'num_frames': num_frames,
                }
                
                return rgb_images, video_info, True
            finally:
                # Clean up temporary file
                os.remove(tmp_video_path)
                
        except Exception as e:
            print(f"[spatialvid_video] Error loading video frames: {str(e)}")
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
            
            # Load video metadata (poses and intrinsics)
            poses, intrinsics_array, metadata_valid = self._load_video_metadata(video_info)
            
            if not metadata_valid:
                next_idx = (idx + random.randint(1, self.__len__())) % self.__len__()
                return self.__getitem__((next_idx, num_frames, resolution_idx))
            
            num_video_frames = len(poses)
            
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
            
            # Collect corresponding intrinsics and extrinsics
            intrinsics_list = []
            extrinsics_list = []
            valid_frames = []
            
            for frame_idx in frame_positions:
                intrinsics = intrinsics_array[frame_idx].copy()


                intrinsics = get_intrinsics_matrices(intrinsics, original_video_info['original_width'], original_video_info['original_height'])


                pose = poses[frame_idx].copy()

                extrinsics = pose_from_quaternion(pose)  # Returns [3, 4] w2c matrix
                extrinsics = se3_inverse(extrinsics)  # w2c to c2w, returns [4, 4]
                extrinsics = extrinsics.squeeze(0).numpy()  # Remove batch dim and convert to numpy
                
                intrinsics_list.append(intrinsics)
                extrinsics_list.append(extrinsics)
                valid_frames.append(True)
            
            # Note: SpatialVID doesn't have depth data, we'll create placeholder depth
            # This is for training models that can work with pose-only supervision
            depth_images = []
            for rgb_img in rgb_images:
                # Create dummy depth (10.0) to pass the filter of dataloader
                w, h = rgb_img.size
                # depth_images.append(Image.fromarray(10.0 * np.ones((h, w), dtype=np.float32)))
                depth_images.append(Image.fromarray(np.random.randint(1, 100, (h, w)).astype(np.float32))) # random depth to ensure no error in MoGe Transformation


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
            processed_intrinsics = []
            processed_extrinsics = []

            for t in range(num_frames):
                rgb_tensor = rgb_tensors[t]
                depth_tensor = depth_tensors[t]
                intrinsics_tensor = intrinsics_tensors[t]
                extrinsics_tensor = extrinsics_tensors[t]

                # Process RGB 
                rgb_tensor = rgb_tensor * 2.0 - 1.0  # [-1,1]

                # For SpatialVID, we don't have real depth, so all depth-related tensors are dummy
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
            fin_mask_sequence = valid_mask_sequence  # For SpatialVID, fin_mask is same as valid_mask (no inf depth)
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
            print(f"[spatialvid_video] Error processing video: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return self._return_dummy_data(num_frames if 'num_frames' in locals() else self.T)


def generate_parquet_file():



    # Constants for data preprocessing
    SPATIALVID_DATA_DIR = "/mnt/localssd/SpatialVID-HQ_raw"

    # video_name = "00094653-a9c6-5558-8e2a-4119e7d64f36"

    # groupd_id = "group_0001"
    VIDEO_DIR = f"{SPATIALVID_DATA_DIR}/videos/SpatialVid/HQ/videos/"
    DEPTH_DIR = f"{SPATIALVID_DATA_DIR}/depths/SpatialVID/depths/"
    ANNOTATION_DIR = f"{SPATIALVID_DATA_DIR}/annotations/SpatialVID/annotations/"

    metadata = pd.read_csv(f'{SPATIALVID_DATA_DIR}/data/train/SpatialVID_HQ_metadata.csv')

    ocr_cond = (metadata['ocr score'] < 0.01)
    fps_cond = (metadata['fps'] >= 24)
    brightness_cond = ~metadata['brightness'].str.contains(r'dark|night|artifact|unknown|dusk', case=False, na=False)
    # group_cond = (metadata['group id'] == 1)
    weather_cond = metadata['weather'].str.contains(r'clear|sunny|rainy|Frosty|Cloudy|snowy', case=False, na=False)

    filtered_metadata = metadata[ocr_cond & fps_cond & brightness_cond & weather_cond]

    group_lists = [f"group_{i:04d}" for i in range(1, 74+1)]
    # group_lists = ['group_0001']

    # df_new = pd.DataFrame(columns=['rgb_path', 'depth_path', 'cam_path', 'video_name', 'frame_number'])
    df_columns = ['rgb_path', 'pose_path', 'intrinsic_path', 'fps', 'group_id', 'video_id', 'video_name']
    records = []

    for group_list in group_lists:
        group_metadata = filtered_metadata[filtered_metadata['group id'] == int(group_list.replace('group_', ''))]
        print(f"Processing {group_list} with {len(group_metadata)} videos")
        for idx, video_info in tqdm.tqdm(group_metadata.iterrows(), total=len(group_metadata)):
            video_id = video_info['id']
            group_id = f"group_{video_info['group id']:04d}"
            video_fps = int(video_info['fps'])
            stride_interval = int(video_fps / 5)
            num_frames = video_info['num frames']
            if num_frames < 200: 
                # print(f"Skipping {video_id} due to less than 300 frames")
                continue

            video_path = os.path.join(VIDEO_DIR, group_id, f"{video_id}.mp4")
            pose_path = os.path.join(ANNOTATION_DIR, group_id, f"{video_id}", "poses.npy")
            intrinsic_path = os.path.join(ANNOTATION_DIR, group_id, f"{video_id}", "intrinsics.npy")


            # Skip if any file is missing
            if not all(os.path.exists(p) for p in [video_path, pose_path, intrinsic_path]):
                print(f"Skipping {video_id} due to missing files")
                continue

            intrinsic_data = np.load(intrinsic_path)
            if np.any(intrinsic_data < 0):
                print(f"Skipping {video_id} due to negative intrinsics")
                continue

        
            # Get frame indices
            # chosen_frame_idx = list(range(0, num_frames, stride_interval))

            rgb_data_path = os.path.join(DATASET_CONFIG["spatialvidhq_videoonly"]["prefix"], "videos", group_id, f"{video_id}.mp4")
            pose_data_path = os.path.join(DATASET_CONFIG["spatialvidhq_videoonly"]["prefix"], "annotations", group_id, f"{video_id}", "poses.npy")
            intrinsic_data_path = os.path.join(DATASET_CONFIG["spatialvidhq_videoonly"]["prefix"], "annotations", group_id, f"{video_id}", "intrinsics.npy")

            records.append({
                'rgb_path': rgb_data_path,
                'pose_path': pose_data_path,
                'intrinsic_path': intrinsic_data_path,
                'fps': video_fps,
                'group_id': group_id,
                'video_id': video_id,
                'video_name': f"{group_id}_{video_id}",
            })
    
    # breakpoint()
    # df_new.to_parquet("mvs_synth_train.parquet")
    df_new = pd.DataFrame.from_records(records, columns=df_columns)
    
    # Save parquet file
    parquet_filename = "spatialvidhq_videoonly_1_74.parquet"
    df_new.to_parquet(parquet_filename)
    print(f"Created parquet file with {len(df_new)} entries")


    
    # Save the parquet file
    upload_file("spatialvidhq_videoonly_1_74.parquet", DATASET_CONFIG["spatialvidhq_videoonly"]["parquet_path"])
    # Remove the local file
    os.remove("spatialvidhq_videoonly_1_74.parquet")

    return



if __name__ == "__main__":
    # generate_parquet_file()
    # quit()

    import tqdm
    import moviepy.video.io.ffmpeg_writer as video_writer
    import cv2
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

    dataset_name = "spatialhq_new"
    save_dir = f"data_vis/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    resolutions = [[0,1]] * 10000
    video_dataset = VideoDepthSpatialVIDHQNew(
        T=20, 
        stride_range=(5, 20),
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
                rgb_frame = (video_sample['rgb'][t].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                # rgb_noaug_frame = (video_sample['rgb_noaug'][t].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                # # Process depth frame
                # depth = video_sample['metric_depth'][t].squeeze()
                # mask = video_sample['valid_mask'][t].squeeze()

                # intrinsics = video_sample['intrinsics'][t]
                # gt_pts = utils3d.torch.depth_to_points(depth, intrinsics=intrinsics)

                # poses = video_sample['extrinsics'][t]
                # # print(f"poses: {poses}")

                # # w2c_target = se3_inverse(poses[:, 0])
                # # poses = torch.einsum('bik, bnkj -> bnij', w2c_target, poses) # to camera 0
                # # breakpoint()
                # gt_pts = torch.einsum('ij, hwj -> hwi', poses, homogenize_points(gt_pts))[..., :3]

                # gt_pts = gt_pts[::4,::4][mask[::4,::4]].reshape(-1, 3)
                # gt_pts_rgb = video_sample['rgb'][t].permute(1, 2, 0)[::4,::4][mask[::4,::4]].reshape(-1,3)

                # depth = depth.numpy()
                # mask = mask.numpy()

                # depth_max = depth[mask].max()
                # depth_min = depth[mask].min()

                # depth_normalized = depth.copy()
                # depth_normalized[~mask] = depth_min
                # depth_normalized = (depth_normalized - depth_min) / (depth_max - depth_min) * 255

                # depth_normalized = depth_normalized.astype(np.uint8)

                # depth_bgr = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)

                # valid_mask = (video_sample['valid_mask'][t].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                # mask_bgr = cv2.cvtColor(valid_mask.squeeze(), cv2.COLOR_GRAY2BGR)
                

                rgb_writer.write_frame(rgb_frame)
                # rgb_noaug_writer.write_frame(rgb_noaug_frame)
                # depth_writer.write_frame(depth_bgr)
                # mask_writer.write_frame(mask_bgr)

                # all_gt_pts.append(gt_pts)
                # all_gt_pts_rgb.append(gt_pts_rgb)
                
            rgb_writer.close()
            # rgb_noaug_writer.close()
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

