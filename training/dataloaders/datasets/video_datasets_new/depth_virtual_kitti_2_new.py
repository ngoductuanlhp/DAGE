import os
import io
import glob
import pandas as pd
from torch.utils.data import Dataset
from training.dataloaders.data_io import download_file, upload_file, list_files, check_path_exists
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

# -----------------------------------------------------------------------------
# NOTE:
# This file is a FIRST PASS skeleton for supporting the Waymo depth video
# dataset.  It mirrors the structure of the existing TartanAir / Point Odyssey
# loaders in this repository but **does NOT yet contain the final parsing logic**.
# Wherever you see a `TODO` comment, please replace the placeholder with the
# proper implementation that matches your processed Waymo data.
# -----------------------------------------------------------------------------


KB_CROP_HEIGHT = 352
KB_CROP_WIDTH = 1216
# Original dataset parameters
OG_VKITTI_HEIGHT = 375
OG_VKITTI_WIDTH = 1242
OG_VKITTI_FOCAL_LENGTH = 725.0087
OG_VKITTI_INTRINSICS = np.array([
    [725.0087, 0, 620.5],
    [0, 725.0087, 187],
    [0, 0, 1]
])

class SynchronizedTransform_VirtualKitti2:
    """Apply the same random transformation to all frames in a sequence."""

    def __init__(self, H: int, W: int):
        self.resize = transforms.Resize((H, W))
        self.resize_depth = transforms.Resize((H, W), interpolation=Image.NEAREST)
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.to_tensor = transforms.ToTensor()
        self.W = W
        self.H = H

    @staticmethod
    def kitti_benchmark_crop(input_img, intrinsics = None):

        height, width = input_img.shape[-2:]
        top_margin = int(height - KB_CROP_HEIGHT)
        left_margin = int((width - KB_CROP_WIDTH) / 2)
        if 2 == len(input_img.shape):
            out = input_img[
                top_margin : top_margin + KB_CROP_HEIGHT,
                left_margin : left_margin + KB_CROP_WIDTH,
            ]
        elif 3 == len(input_img.shape):
            out = input_img[
                :,
                top_margin : top_margin + KB_CROP_HEIGHT,
                left_margin : left_margin + KB_CROP_WIDTH,
            ]
        if intrinsics is not None:
            intrinsics[0, 2] -= left_margin
            intrinsics[1, 2] -= top_margin
            return out, intrinsics
        
        return out

    def __call__(self, rgb_images, depth_images, intrinsics_list, extrinsics_list=None):
        # Decide on a single horizontal flip for the entire sequence
        if extrinsics_list is not None:
            flip = False
        else:
            flip = random.random() > 0.5

        rgb_tensors = []
        depth_tensors = []
        intrinsics_tensors = []
        extrinsics_tensors = []


        
        for idx, (rgb_image, depth_image, intrinsics) in enumerate(zip(rgb_images, depth_images, intrinsics_list)):
            # Horizontal flip (if selected once per sequence)
            if flip:
                rgb_image = self.horizontal_flip(rgb_image)
                depth_image = self.horizontal_flip(depth_image)

            # to tensor
            rgb_tensor = self.to_tensor(rgb_image)      
            depth_tensor = self.to_tensor(depth_image)  
            intrinsics_tensor = torch.tensor(intrinsics)

            # kitti benchmark crop
            rgb_tensor = self.kitti_benchmark_crop(rgb_tensor)
            depth_tensor, intrinsics_tensor = self.kitti_benchmark_crop(depth_tensor, intrinsics_tensor)

            rgb_tensors.append(rgb_tensor)
            depth_tensors.append(depth_tensor)
            intrinsics_tensors.append(intrinsics_tensor)


            if extrinsics_list is not None:
                extrinsics_tensor = torch.tensor(extrinsics_list[idx])
                extrinsics_tensors.append(extrinsics_tensor)

                # # Resize keeping aspect ratio as handled by torchvision
                # og_width, og_height = rgb_image.size
                # scale_w = self.W / og_width
                # scale_h = self.H / og_height
                # rgb_image = self.resize(rgb_image)
                # depth_image = self.resize_depth(depth_image)

                # # Adjust intrinsics to resized resolution
                # intrinsics[0, 0] *= scale_w  # fx
                # intrinsics[1, 1] *= scale_h  # fy
                # intrinsics[0, 2] *= scale_w  # cx
                # intrinsics[1, 2] *= scale_h  # cy

                # # To tensor
                # rgb_tensors.append(self.to_tensor(rgb_image))
                # depth_tensors.append(self.to_tensor(depth_image))
                # intrinsics_tensors.append(torch.tensor(intrinsics))

        
        if extrinsics_list is not None:
            return rgb_tensors, depth_tensors, intrinsics_tensors, extrinsics_tensors
        else:
            return rgb_tensors, depth_tensors, intrinsics_tensors


class VideoDepthVirtualKitti2New(Dataset):
    """Skeleton video dataset loader for VirtualKitti2 depth data.

    NOTE: The per-frame loading logic is left as placeholders – you should
    replace those with real parsing of the `*.jpg`, `*.exr`, and `*.npz`
    files found inside each DynamicReplica segment directory.
    """
    depth_type = "synthetic"
    is_metric_scale = True
    # NOTE original resolution is 1242x375

    def __init__(
        self,
        T: int = 5,
        stride_range: tuple = (1, 3),
        transform: bool = True,
        resize: tuple = (384, 512),
        near_plane: float = 1e-5,
        far_plane: float = 80.0,
        resolutions=None,
        use_moge=False,
        moge_augmentation=None,
        use_cut3r_frame_sampling=False, video_prob=0.6, fix_interval_prob=0.6, block_shuffle=16
    ):
        # Lazily load the parquet only if VirtualKitti2 key exists in config
        parquet_data = download_file(
            
            DATASET_CONFIG["virtual_kitti_2_video"]["parquet_path"],
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
            self.transform = SynchronizedTransform_VirtualKitti2(H=resize[0], W=resize[1]) if transform else None

    # ------------------------------------------------------------------
    # Helper functions – identical structure to other dataset loaders
    # ------------------------------------------------------------------

    def _organize_by_video(self):
        df = pd.read_parquet(self.parquet)
        video_frames = defaultdict(list)
        for _, row in df.iterrows():
            video_name = row["video_name"]
            frame_data = {
                "frame_number": row["frame_number"],
                "rgb_path": row["rgb_path"],
                "depth_path": row["depth_path"],
                # "intrinsics": row.get("intrinsics", None),
                "extrinsics": row["extrinsics"],
            }
            video_frames[video_name].append(frame_data)

        # Sort each video by frame number
        for v in video_frames:
            video_frames[v].sort(key=lambda x: x["frame_number"])
        return dict(video_frames)

    def _create_frame_list(self):
        frame_list = []
        for video_name, frames in self.video_frames.items():
            for i, _ in enumerate(frames):
                frame_list.append(
                    {
                        "video_name": video_name,
                        "frame_idx_in_video": i,
                    }
                )
        return frame_list

    def __len__(self):
        return len(self.frame_list)

    # ------------------------------------------------------------------
    # PLACEHOLDER single-frame loader – FILL THIS IN!
    # ------------------------------------------------------------------
    def _load_single_frame(self, frame_data):
        """Load RGB image, depth (metric), intrinsics and extrinsics for a single Dynamic-Replica frame.

        The parquet row already stores the paths for the RGB/Depth files **as well as** the per-frame
        intrinsics & extrinsics flattened as 1-D lists.  We therefore only need to download the RGB
        and depth assets – the camera parameters can be reconstructed from the row itself.
        """
        try:
            # ------------------------------------------------------------------
            # RGB image (stored as PNG)
            # ------------------------------------------------------------------
            rgb_bytes = download_file(
                
                frame_data["rgb_path"],
            )
            rgb_image = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
            del rgb_bytes  # Free memory immediately

            # ------------------------------------------------------------------
            # Depth – stored as a `.npy` (float32, metric depth in meters)
            # ------------------------------------------------------------------
            depth_bytes = download_file(
                
                frame_data["depth_path"],
            )
            depth_pil = Image.open(io.BytesIO(depth_bytes))
            del depth_bytes  # Free memory immediately
            
            depth_array = np.array(depth_pil)
            del depth_pil  # Free memory immediately
            
            depth_array = depth_array / 100.0 # cm to meters
            depth_image = Image.fromarray(depth_array.astype(np.float32))
            del depth_array  # Free memory immediately

            intrinsics = OG_VKITTI_INTRINSICS.copy()
            extrinsics = np.array(frame_data['extrinsics'], dtype=np.float32).reshape(4, 4).copy()


            return rgb_image, depth_image, intrinsics, extrinsics, True
        except Exception as e:
            # In case anything goes wrong we return dummy tensors so that the rest of the
            # data-pipeline keeps flowing.
            print(f"[virtual_kitti_2] Error loading frame: {str(e)}")
            rgb_dummy = Image.new("RGB", (self.frame_size[1], self.frame_size[0]))
            depth_dummy = Image.fromarray(np.zeros((self.frame_size[0], self.frame_size[1]), dtype=np.float32))
            return rgb_dummy, depth_dummy, np.eye(3, dtype=np.float32), np.eye(4, dtype=np.float32), False

    # ------------------------------------------------------------------
    # The rest of __getitem__ mirrors other dataset loaders (unchanged).
    # ------------------------------------------------------------------
    def _can_sample_sequence(self, video_name, start_idx, stride, num_frames=None):
        if num_frames is None:
            num_frames = self.T
        frames = self.video_frames[video_name]
        if start_idx + (num_frames - 1) * stride >= len(frames):
            return False
        start_frame_num = frames[start_idx]["frame_number"]
        for i in range(num_frames):
            frame_idx = start_idx + i * stride
            expected = start_frame_num + i * stride
            actual = frames[frame_idx]["frame_number"]
            if actual != expected:
                return False
        return True

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

            # Starting frame information
            start_frame_info = self.frame_list[idx]
            video_name = start_frame_info["video_name"]
            start_idx = start_frame_info["frame_idx_in_video"]

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

            # ------------------------------------------------------------------
            # Collect data for each frame in the sequence
            # ------------------------------------------------------------------
            rgb_images = []
            depth_images = []
            intrinsics_list = []
            extrinsics_list = []
            valid_frames = []

            for frame_idx in frame_positions:
                
                frame_data = frames[frame_idx]  # frames[frame_idx] is already the frame_data dict
                rgb_img, depth_img, intrinsics, extrinsics, valid = self._load_single_frame(frame_data)

                if valid:
                    rgb_images.append(rgb_img)
                    depth_images.append(depth_img)
                    intrinsics_list.append(intrinsics.copy())
                    extrinsics_list.append(extrinsics)
                    valid_frames.append(True)
                else:
                    # Fallback to previous frame (or dummy if none)
                    if rgb_images:
                        rgb_images.append(rgb_images[-1])
                        depth_images.append(depth_images[-1])
                        intrinsics_list.append(intrinsics_list[-1].copy())
                        extrinsics_list.append(extrinsics_list[-1])
                    else:
                        rgb_images.append(Image.new("RGB", (self.frame_size[1], self.frame_size[0])))
                        depth_images.append(Image.fromarray(np.zeros((self.frame_size[0], self.frame_size[1]), dtype=np.float32)))
                        intrinsics_list.append(np.eye(3, dtype=np.float32))
                        extrinsics_list.append(np.eye(4, dtype=np.float32))
                    valid_frames.append(False)

            # ------------------------------------------------------------------
            # Apply the synchronized data-augmentation (resize/h-flip)
            # ------------------------------------------------------------------
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
                intrinsics_tensors = [torch.tensor(k) for k in intrinsics_list]
                extrinsics_tensors = [torch.tensor(k) for k in extrinsics_list]

            # ------------------------------------------------------------------
            # Post-processing & tensor stacking – identical to other loaders
            # ------------------------------------------------------------------
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

                # print(f"depth_tensor.shape: {depth_tensor.shape}, {depth_tensor.min()}, {depth_tensor.max()}")

                # Valid depth mask
                valid_depth_mask = (depth_tensor > self.near_plane) & (depth_tensor < self.far_plane)
                inf_depth_mask = depth_tensor >= self.far_plane
                # Normalise RGB to [-1,1]
                rgb_tensor = rgb_tensor * 2.0 - 1.0

                # Depth clamping similar to other datasets
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
                'frame_positions': frame_positions,  # New: actual frame indices used
                'is_video': is_video,  # New: whether sequence is video-like
                # 'stride': stride,
                "valid": all(valid_frames),
                "depth_type": self.depth_type,
            }

        except Exception as e:
            print(f"[virtual_kitti_2] Error processing item: {str(e)}")
            num_frames = self.T
            dummy_rgb = torch.zeros(num_frames, 3, self.frame_size[0], self.frame_size[1])
            dummy_depth = torch.zeros(num_frames, 1, self.frame_size[0], self.frame_size[1])
            dummy_mask = torch.zeros(num_frames, 1, self.frame_size[0], self.frame_size[1], dtype=torch.bool)
            dummy_intrinsics = torch.eye(3).unsqueeze(0).repeat(num_frames, 1, 1)
            dummy_extrinsics = torch.eye(3).unsqueeze(0).repeat(num_frames, 1, 1)
            return {
                "rgb": dummy_rgb,
                "metric_depth": dummy_depth,
                "valid_mask": dummy_mask,
                "intrinsics": dummy_intrinsics,
                "extrinsics": dummy_extrinsics,
                "stride": 1,
                "valid": False,
                "depth_type": self.depth_type,
            }


# -----------------------------------------------------------------------------
# Parquet generation – scans the *processed* VirtualKitti2 directory and
# builds a dataframe which is saved locally.
# -----------------------------------------------------------------------------

# def generate_parquet_file(split = "train"):
#     from pathlib import Path

#     SCRIPT_DIR = Path(__file__).parent.parent.parent

#     df_new = pd.DataFrame(columns=['rgb_path', 'depth_path', 'normal_path'])

#     scenes = ["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"]
#     weather_conditions = ["morning", "fog", "rain", "sunset", "overcast"]
#     cameras = ["Camera_0", "Camera_1"]
#     vkitti2_rgb_path = os.path.join(DATASET_CONFIG["virtual_kitti_2"]["prefix"], "vkitti_2.0.3_rgb")
#     vkitti2_depth_path =  os.path.join(DATASET_CONFIG["virtual_kitti_2"]["prefix"], "vkitti_2.0.3_depth")
#     vkitti2_normal_path = os.path.join(DATASET_CONFIG["virtual_kitti_2"]["prefix"], "vkitti_DAG_normals")
#     for scene in scenes:
#         for weather in weather_conditions:
#             for camera in cameras:
#                 print(f"Processing {scene}_{weather}_{camera}")
#                 rgb_dir = os.path.join(vkitti2_rgb_path, scene, weather, "frames", "rgb" ,camera)
#                 depth_dir = os.path.join(vkitti2_depth_path, scene, weather, "frames","depth" , camera)
#                 normal_dir = os.path.join(vkitti2_normal_path, scene, weather, "frames", "normal", camera)
#                 if check_path_exists(rgb_dir) and \
#                     check_path_exists(depth_dir):
#                     rgb_files = list_files(rgb_dir)
#                     rgb_files  = [file[3:] for file in rgb_files]
#                     for idx, file in enumerate(rgb_files):
#                         rgb_file = "rgb" + file
#                         depth_file = "depth" + file.replace('.jpg', '.png')
#                         normal_file = "normal" + file.replace('.jpg', '.png')
#                         rgb_path = os.path.join(rgb_dir, rgb_file)
#                         depth_path = os.path.join(depth_dir, depth_file)
#                         normal_path = os.path.join(normal_dir, normal_file)
#                         df_new = pd.concat([df_new, pd.DataFrame({
#                             'rgb_path': [rgb_path],
#                             'depth_path': [depth_path],
#                             'normal_path': [normal_path],
#                             'video_name': [f'{scene}_{weather}_{camera}'],
#                             'frame_number': [idx]
#                         })], ignore_index=True)
#     df_new.to_parquet(os.path.join(SCRIPT_DIR, "virtual_kitti_2_video_{}.parquet".format(split)))
#     # read the parquet file
#     df_new = pd.read_parquet(os.path.join(SCRIPT_DIR, "virtual_kitti_2_video_{}.parquet".format(split)))
#     # print the length of the dataframe
#     print(len(df_new))
#     # Save the parquet file
#     upload_file(os.path.join(SCRIPT_DIR, "virtual_kitti_2_video_{}.parquet".format(split)), DATASET_CONFIG["virtual_kitti_2_video"]["parquet_path"])
#     # remove the local file
#     os.remove(os.path.join(SCRIPT_DIR, "virtual_kitti_2_video_{}.parquet".format(split)))

def generate_parquet_file(split = "train"):

    def _load_timestamps_and_poses(pose_file, camera_id = 0):
        """Load ground truth poses (T_w_cam) and timestamps from file."""
        timestamps = []
        poses = []

        # Read and parse the poses
        with open(pose_file, 'r') as f:
            for line in f.readlines():
                line = line.split()
                if line[0] == 'frame':  # this is the header
                    continue
                timestamps.append(float(line[0]))

                current_camera_id = int(line[1])
                if current_camera_id != camera_id:
                    continue

                # from world to camera
                Tmatrix = np.array([float(x)
                                    for x in line[2:2+16]]).reshape((4, 4))
                # from camera to world
                poses.append(np.linalg.inv(Tmatrix))

        return timestamps, poses

    from pathlib import Path

    # SCRIPT_DIR = Path(__file__).parent.parent.parent

    data_path = DATASET_CONFIG["virtual_kitti_2_video"]["prefix"]

    df_columns = ['rgb_path', 'depth_path', 'intrinsics', 'extrinsics', 'video_name', 'frame_number']
    records = []

    # df_new = pd.DataFrame(columns=['rgb_path', 'depth_path', 'normal_path'])

    scenes = ["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"]
    weather_conditions = ["morning", "fog", "rain", "sunset", "overcast"]
    cameras = ["Camera_0", "Camera_1"]
    # vkitti2_rgb_path = os.path.join(DATASET_CONFIG["virtual_kitti_2"]["prefix"], "vkitti_2.0.3_rgb")
    # vkitti2_depth_path =  os.path.join(DATASET_CONFIG["virtual_kitti_2"]["prefix"], "vkitti_2.0.3_depth")
    # vkitti2_normal_path = os.path.join(DATASET_CONFIG["virtual_kitti_2"]["prefix"], "vkitti_DAG_normals")

    vkitti2_path = "/mnt/localssd/vkitti2"
    for scene in scenes:
        for weather in weather_conditions:
            for camera in cameras:
                print(f"Processing {scene}_{weather}_{camera}")
                rgb_dir = os.path.join(vkitti2_path, scene, weather, "frames", "rgb" ,camera)
                depth_dir = os.path.join(vkitti2_path, scene, weather, "frames","depth" , camera)
                pose_file = os.path.join(vkitti2_path, scene, weather, "extrinsic.txt")

                timestamps, poses = _load_timestamps_and_poses(pose_file, camera_id = 0 if camera == "Camera_0" else 1)

                rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])

                assert len(poses) == len(rgb_files)

                for idx, file in tqdm.tqdm(enumerate(rgb_files)):
                    file_name = file.split(".")[0][3:]
                    rgb_path = os.path.join(rgb_dir, "rgb" + file_name + ".jpg")
                    depth_path = os.path.join(depth_dir, "depth" + file_name + ".png")

                    extrinsics = poses[idx]
                    intrinsics = OG_VKITTI_INTRINSICS.copy()

                    extrinsics_flat = extrinsics.flatten().tolist()
                    intrinsics_flat = intrinsics.flatten().tolist()

                    # df_new = pd.concat([df_new, pd.DataFrame({
                    #     'rgb_path': [rgb_path],
                    #     'depth_path': [depth_path],
                    #     'extrinsics': [extrinsics.flatten().tolist()],
                    #     'video_name': [f'{scene}_{weather}_{camera}'],
                    #     'frame_number': [idx]
                    # })], ignore_index=True)
                    video_name = f'{scene}_{weather}_{camera}'

                    new_rgb_path = f'{data_path}/{scene}/{weather}/frames/rgb/{camera}/rgb{file_name}.jpg'
                    new_depth_path = f'{data_path}/{scene}/{weather}/frames/depth/{camera}/depth{file_name}.png'

                    # breakpoint()

                    records.append({
                        'rgb_path': new_rgb_path,
                        'depth_path': new_depth_path,
                        'intrinsics': intrinsics_flat,
                        'extrinsics': extrinsics_flat,
                        'video_name': video_name,
                        'frame_number': idx  # Frame number within the scene
                    })

                    # breakpoint()


                # normal_dir = os.path.join(vkitti2_normal_path, scene, weather, "frames", "normal", camera)
                # if check_path_exists(rgb_dir) and \
                #     check_path_exists(depth_dir):
                #     rgb_files = list_files(rgb_dir)
                #     rgb_files  = [file[3:] for file in rgb_files]
                #     for idx, file in enumerate(rgb_files):
                #         rgb_file = "rgb" + file
                #         depth_file = "depth" + file.replace('.jpg', '.png')
                #         normal_file = "normal" + file.replace('.jpg', '.png')
                #         rgb_path = os.path.join(rgb_dir, rgb_file)
                #         depth_path = os.path.join(depth_dir, depth_file)
                #         # normal_path = os.path.join(normal_dir, normal_file)
                #         df_new = pd.concat([df_new, pd.DataFrame({
                #             'rgb_path': [rgb_path],
                #             'depth_path': [depth_path],
                #             # 'normal_path': [normal_path],
                #             'video_name': [f'{scene}_{weather}_{camera}'],
                #             'frame_number': [idx]
                #         })], ignore_index=True)
    # df_new.to_parquet(os.path.join(SCRIPT_DIR, "virtual_kitti_2_video_{}.parquet".format(split)))
    # # read the parquet file
    # df_new = pd.read_parquet(os.path.join(SCRIPT_DIR, "virtual_kitti_2_video_{}.parquet".format(split)))
    # print the length of the dataframe
    # breakpoint()
    df_new = pd.DataFrame.from_records(records, columns=df_columns)
    parquet_filename = "virtual_kitti_2_video_train.parquet"
    df_new.to_parquet(parquet_filename)
    print(len(df_new))
    # Save the parquet file
    upload_file(parquet_filename, DATASET_CONFIG["virtual_kitti_2_video"]["parquet_path"])
    # remove the local file
    os.remove(parquet_filename)

# -----------------------------------------------------------------------------
# CLI helper – run `python depth_virtual_kitti_2.py` to preprocess & upload parquet
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # traj = torch.load("/mnt/localssd/dynamic_replica/train/11a245-1_obj_source_left/trajectories/000000.pth")
    # breakpoint()
    # print(traj)
    # exit()
    # import argparse
    # parser = argparse.ArgumentParser(description="Generate parquet for Dynamic Replica (raw → processed)")
    # parser.add_argument("--raw_root", type=str, default="/mnt/localssd/dynamic_replica", help="Path to raw Dynamic Replica root directory")
    # parser.add_argument("--processed_root", type=str, default="/mnt/localssd/dynamic_replica_processed", help="Output directory for processed assets")
    # parser.add_argument("--splits", type=str, nargs="*", default=["train", "valid"], help="Splits to process")
    # args = parser.parse_args()
    # import tqdm
    # generate_parquet_file()

    # quit()

    import tqdm
    import cv2
    from Unip.unip.util.pointmap_util import depth_to_pointmap, save_pointmap_as_ply
    import moviepy.video.io.ffmpeg_writer as video_writer
    

    from training.dataloaders.batched_sampler import make_sampler
    from torch.utils.data import DataLoader

    os.makedirs("data_vis", exist_ok=True)
    print("\nTesting video dataset...")

    # resolutions = [(512, 384), (512, 336), (512, 288), (512, 256), (384, 512), (336, 512), (288, 512), (256, 512)]
    resolutions = [[0,1]] * 10000
    video_dataset = VideoDepthVirtualKitti2New(T=10, stride_range=(1, 5), transform=True, resolutions=resolutions, use_moge=True)

    dataloader_sampler = make_sampler(
        video_dataset, 
        batch_size=1, 
        number_of_resolutions=len(resolutions),
        min_num_frames=8, 
        max_num_frames=16, 
        shuffle=True, 
        drop_last=True
    )

    dataloader = DataLoader(
        video_dataset,
        batch_sampler=dataloader_sampler,
        num_workers=1,
        pin_memory=False,
    )

    if hasattr(dataloader, "dataset") and hasattr(dataloader.dataset, "set_epoch"):
        dataloader.dataset.set_epoch(0)
    if (
        hasattr(dataloader, "batch_sampler")
        and hasattr(dataloader.batch_sampler, "sampler")
        and hasattr(dataloader.batch_sampler.sampler, "set_epoch")
    ):
        dataloader.batch_sampler.sampler.set_epoch(0)
    

    print(f"Video dataset length: {len(video_dataset)}")
    
    if len(video_dataset) > 0:

        # Test with a few samples from the dataloader
        sample_count = 0
        max_samples = 10  # Test with 5 samples instead of 20
        
        for batch_idx, video_sample in enumerate(dataloader):

            # if batch_idx % 20 != 0:
            #     continue
            
            if sample_count >= max_samples:
                break
                
            # Extract the first item from the batch since batch_size_per_gpu=1
            if isinstance(video_sample['rgb'], list):
                # Handle case where video_sample might be a list of batches
                video_sample = {k: v[0] if isinstance(v, list) else v for k, v in video_sample.items()}
            else:
                # Extract first item from batch dimension
                video_sample = {k: v[0] if isinstance(v, torch.Tensor) and v.dim() > 3 else v for k, v in video_sample.items()}

            print(f"Video sample {sample_count} shapes:")
            print(f"  RGB: {video_sample['rgb'].shape}")  # Should be [T, 3, H, W]
            print(f"  Metric depth: {video_sample['metric_depth'].shape}")  # Should be [T, 1, H, W]
            print(f"  Valid mask: {video_sample['valid_mask'].shape}")  # Should be [T, 1, H, W]
            print(f"  Intrinsics: {video_sample['intrinsics'].shape}")  # Should be [T, 3, 3]
            print(f"  Extrinsics: {video_sample['extrinsics'].shape}")  # Should be [T, 3, 3]
            print(f"  Valid: {video_sample['valid']}")
            print(f"  Stride: {video_sample['stride']}")
            
            # Prepare video writers
            H, W = video_sample['rgb'].shape[2], video_sample['rgb'].shape[3]
            
            rgb_writer = video_writer.FFMPEG_VideoWriter(filename=f"data_vis/virtual_kitti_2_video_rgb_{sample_count}.mp4", size=(W, H), fps=2)
            depth_writer = video_writer.FFMPEG_VideoWriter(filename=f"data_vis/virtual_kitti_2_video_depth_{sample_count}.mp4", size=(W, H), fps=2)
            mask_writer = video_writer.FFMPEG_VideoWriter(filename=f"data_vis/virtual_kitti_2_video_mask_{sample_count}.mp4", size=(W, H), fps=2)
            
            print("Saving video frames...")
            for t in tqdm.tqdm(range(video_sample['rgb'].shape[0])):
                # Process RGB frame
                # rgb_frame = ((video_sample['rgb'][t].numpy().transpose(1, 2, 0) + 1) / 2 * 255).astype(np.uint8)

                rgb_frame = (video_sample['rgb'][t].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                
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
                    save_pointmap_as_ply(pointmap, rgb_frame, f"data_vis/virtual_kitti_2_video_frame0_pointmap_{sample_count}.ply", far_threshold=1000)

            rgb_writer.close()
            depth_writer.close()
            mask_writer.close()
            
            print("Video sequence test completed successfully!")
            print("Saved videos:")
            print(f"  - data_vis/virtual_kitti_2_video_rgb_{sample_count}.mp4")
            print(f"  - data_vis/virtual_kitti_2_video_depth_{sample_count}.mp4") 
            print(f"  - data_vis/virtual_kitti_2_video_mask_{sample_count}.mp4")
            print(f"  - data_vis/virtual_kitti_2_video_frame0_pointmap_{sample_count}.ply")
            
            sample_count += 1
    else:
        print("No valid video sequences found in dataset.")
