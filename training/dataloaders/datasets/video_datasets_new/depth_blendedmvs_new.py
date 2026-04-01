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
import h5py
from collections import defaultdict
from tqdm import tqdm

from training.dataloaders.datasets.video_datasets_new.synchronized_video_transform import SynchronizedTransformVideoCut3r
from training.dataloaders.datasets.video_datasets_new.synchronized_video_transform_moge import SynchronizedTransform_MoGe

# from training.dataloaders.datasets.video_datasets_new.frame_sampling_utils import get_seq_from_start_id

class SynchronizedTransform_BlendedMVSVideo:
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


class MultiViewDepthBlendedMVSNew(Dataset):
    depth_type = "lidar"
    is_metric_scale = False

    def __init__(self, stride_range=(1, 3), T=5, transform=True, resize=(384, 512), near_plane=1e-5, far_plane=1000.0,  resolutions=None, use_moge=False, moge_augmentation=None, allow_repeat=False, overlap_threshold=0.2):
        """
        Multi-view dataset for BlendedMVS depth data.
        
        Args:
            T (int): Number of views in each multi-view set
            transform (bool): Whether to apply transformations
            resize (tuple): Target size (H, W) - default matches preprocessing output
            near_plane (float): Near clipping plane for depth
            far_plane (float): Far clipping plane for depth
            allow_repeat (bool): Whether to allow repeating views in multi-view sets
            overlap_threshold (float): Threshold for overlap scores in adjacency matrix
        """
        parquet_data = download_file(DATASET_CONFIG["blendedmvs"]["parquet_path"])
        self.parquet = io.BytesIO(parquet_data)

        # Download overlap matrix data
        overlap_data = download_file(DATASET_CONFIG["blendedmvs"]["overlap_path"])
        self.overlap_h5 = io.BytesIO(overlap_data)

        self.T = T
        self.stride_range = stride_range # NOTE for dummy

        self.near_plane = near_plane
        self.far_plane = far_plane
        self.frame_size = resize
        self.allow_repeat = allow_repeat
        self.overlap_threshold = overlap_threshold
        
        # Load and organize data by scene
        self.scene_data = self._load_overlap_data()
        self.scene_frames = self._organize_by_scene()
        self.all_ref_imgs = self._create_ref_list()

        # print(f"overlap threshold: {self.overlap_threshold}")
        
        # Cache for reachability checks
        self.invalid_scenes = []
        self.is_reachable_cache = {scene: {} for scene in self.scene_data.keys()}

        self.resolutions = resolutions
        self.use_moge = use_moge
        self.moge_augmentation = moge_augmentation

        if self.use_moge:
            self.transform = SynchronizedTransform_MoGe(moge_augmentation=self.moge_augmentation)   
        elif self.resolutions is not None:
            self.transform = SynchronizedTransformVideoCut3r(seed=777)
        else:
            self.transform = SynchronizedTransform_BlendedMVSVideo(H=resize[0], W=resize[1]) if transform else None

    def _load_overlap_data(self):
        
        data_dict = {}
        with h5py.File(self.overlap_h5, "r") as f:
            for scene_dir in f.keys():
                group = f[scene_dir]
                basenames = group["basenames"][:]
                indices = group["indices"][:]
                values = group["values"][:]
                shape = group.attrs["shape"]
                # Reconstruct the sparse matrix
                score_matrix = np.zeros(shape, dtype=np.float32)
                score_matrix[indices[0], indices[1]] = values
                data_dict[scene_dir] = {
                    "basenames": basenames,
                    "score_matrix": self._build_adjacency_list(score_matrix),
                }
        print(f"[blendedmvs_multiview] Loaded {len(data_dict)} scenes")
        return data_dict

    def _build_adjacency_list(self, S):
        """Build adjacency list from overlap matrix."""
        adjacency_list = [[] for _ in range(len(S))]
        S = S - self.overlap_threshold
        S[S < 0] = 0
        rows, cols = np.nonzero(S)
        for i, j in zip(rows, cols):
            adjacency_list[i].append((j, S[i][j]))
        return adjacency_list

    def _organize_by_scene(self):
        """Organize frames by scene name."""
        df = pd.read_parquet(self.parquet)
        
        # Extract frame_name once for all rows using vectorized string operations
        df['frame_name'] = df['rgb_path'].str.split('/').str[-1].str.split('.').str[0]
        
        # Group by scene name (video_name)
        scene_frames = {}
        for scene_name, scene_df in df.groupby('video_name', sort=False):
            # Convert to list of dicts, then reorganize by frame_name
            records = scene_df[['frame_name', 'frame_number', 'rgb_path', 'depth_path', 
                               'cam_path', 'intrinsics', 'extrinsics']].to_dict('records')
            
            # Build dict with frame_name as key
            frames_dict = {
                record['frame_name']: {
                    'frame_number': record['frame_number'],
                    'rgb_path': record['rgb_path'],
                    'depth_path': record['depth_path'],
                    'cam_path': record['cam_path'],
                    'intrinsics': record['intrinsics'],
                    'extrinsics': record['extrinsics']
                }
                for record in records
            }
            scene_frames[scene_name] = frames_dict
        
        return scene_frames

    def _create_ref_list(self):
        """Create a flat list of all images with scene context."""
        # Use list comprehension for faster construction
        all_ref_imgs = [
            (scene_name, b)
            for scene_name in self.scene_data.keys()
            if scene_name in self.scene_frames
            for b in range(len(self.scene_data[scene_name]["basenames"]))
        ]
        return all_ref_imgs

    def __len__(self):
        return len(self.all_ref_imgs)

    @staticmethod
    def _is_reachable(adjacency_list, start_index, k):
        """Check if we can reach k nodes from start_index."""
        visited = set()
        stack = [start_index]
        while stack and len(visited) < k:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                for neighbor in adjacency_list[node]:
                    if neighbor[0] not in visited:
                        stack.append(neighbor[0])
        return len(visited) >= k

    def _random_sequence_no_revisit_with_backtracking(self, adjacency_list, k, start_index, rng):
        """Generate sequence without revisiting nodes using backtracking."""
        path = [start_index]
        visited = set([start_index])
        neighbor_iterators = []
        
        # Initialize the iterator for the start index
        neighbors = adjacency_list[start_index]
        neighbor_idxs = [n[0] for n in neighbors]
        neighbor_weights = [n[1] for n in neighbors]
        if neighbor_weights:
            neighbor_idxs = rng.choice(
                neighbor_idxs,
                size=len(neighbor_idxs),
                replace=False,
                p=np.array(neighbor_weights) / np.sum(neighbor_weights),
            ).tolist()
        neighbor_iterators.append(iter(neighbor_idxs))

        while len(path) < k:
            if not neighbor_iterators:
                return None
            current_iterator = neighbor_iterators[-1]
            try:
                next_index = next(current_iterator)
                if next_index not in visited:
                    path.append(next_index)
                    visited.add(next_index)

                    # Prepare iterator for the next node
                    neighbors = adjacency_list[next_index]
                    neighbor_idxs = [n[0] for n in neighbors]
                    neighbor_weights = [n[1] for n in neighbors]
                    if neighbor_weights:
                        neighbor_idxs = rng.choice(
                            neighbor_idxs,
                            size=len(neighbor_idxs),
                            replace=False,
                            p=np.array(neighbor_weights) / np.sum(neighbor_weights),
                        ).tolist()
                    neighbor_iterators.append(iter(neighbor_idxs))
            except StopIteration:
                neighbor_iterators.pop()
                if path:
                    visited.remove(path.pop())
        return path

    def _random_sequence_with_optional_repeats(self, adjacency_list, k, start_index, rng, max_k=None, max_attempts=100):
        """Generate sequence with optional repeats."""
        if max_k is None:
            max_k = k
        path = [start_index]
        visited = set([start_index])
        current_index = start_index
        attempts = 0

        while len(path) < max_k and attempts < max_attempts:
            attempts += 1
            neighbors = adjacency_list[current_index]
            neighbor_idxs = [n[0] for n in neighbors]
            neighbor_weights = [n[1] for n in neighbors]

            if not neighbor_idxs:
                break

            # Try to find unvisited neighbors
            unvisited_neighbors = [
                (idx, wgt)
                for idx, wgt in zip(neighbor_idxs, neighbor_weights)
                if idx not in visited
            ]
            if unvisited_neighbors:
                unvisited_idxs = [idx for idx, _ in unvisited_neighbors]
                unvisited_weights = [wgt for _, wgt in unvisited_neighbors]
                probabilities = np.array(unvisited_weights) / np.sum(unvisited_weights)
                next_index = rng.choice(unvisited_idxs, p=probabilities)
                visited.add(next_index)
            else:
                probabilities = np.array(neighbor_weights) / np.sum(neighbor_weights)
                next_index = rng.choice(neighbor_idxs, p=probabilities)

            path.append(next_index)
            current_index = next_index

        if len(set(path)) >= k:
            while len(path) < max_k:
                next_index = rng.choice(path)
                path.append(next_index)
            return path
        else:
            return None

    def _generate_sequence(self, scene, adj_list, num_views, start_index, rng):
        """Generate a sequence of views from the scene."""
        cutoff = num_views if not self.allow_repeat else max(num_views // 5, 3)
        
        if start_index in self.is_reachable_cache[scene]:
            if not self.is_reachable_cache[scene][start_index]:
                return None
        else:
            self.is_reachable_cache[scene][start_index] = self._is_reachable(
                adj_list, start_index, cutoff
            )
            if not self.is_reachable_cache[scene][start_index]:
                return None
                
        if not self.allow_repeat:
            sequence = self._random_sequence_no_revisit_with_backtracking(
                adj_list, cutoff, start_index, rng
            )
        else:
            sequence = self._random_sequence_with_optional_repeats(
                adj_list, cutoff, start_index, rng, max_k=num_views
            )
            
        if not sequence:
            self.is_reachable_cache[scene][start_index] = False
            
        return sequence

    def _load_single_frame(self, frame_data):
        """Load a single frame's RGB, depth, and camera data."""
        try:
            # Load RGB
            rgb_data = download_file(frame_data['rgb_path'])
            rgb_image = Image.open(io.BytesIO(rgb_data)).convert('RGB')
            del rgb_data  # Free memory immediately
            
            # Load depth (EXR format from preprocessing)
            depth_data = download_file(frame_data['depth_path'])
            depth_buffer = np.frombuffer(depth_data, np.uint8)
            del depth_data  # Free memory immediately

            depth_array = cv2.imdecode(depth_buffer, cv2.IMREAD_UNCHANGED)
            del depth_buffer  # Free memory immediately
            
            depth_image = Image.fromarray(depth_array)
            del depth_array  # Free memory immediately
            
                # # Convert bytes to numpy array for OpenEXR loading
                # depth_bytes = np.frombuffer(depth_data, dtype=np.uint8)
                # depth_array = cv2.imdecode(depth_bytes, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                
                # # Handle potential issues with EXR loading
                # if depth_array is None:
                #     raise ValueError("Failed to decode EXR depth data")
                
                # # EXR files can have multiple channels, take first channel if needed
                # if len(depth_array.shape) == 3:
                #     depth_array = depth_array[:, :, 0]
                
                # depth_image = Image.fromarray(depth_array.astype(np.float32))
            
            intrinsics = np.array(frame_data['intrinsics']).reshape(3, 3).copy()
            extrinsics = np.array(frame_data['extrinsics']).reshape(4, 4).copy()

            
            return rgb_image, depth_image, intrinsics, extrinsics, True
            
        except Exception as e:
            print(f"[blendedmvs_video] Error loading frame: {str(e)}")
            return None, None, None, None, False

    def __getitem__(self, idx):
        try:
            if isinstance(idx, tuple):
                len_tuple_idx = len(idx)
                if len_tuple_idx == 2:
                    idx, num_views = idx
                    resolution_idx = None   
                elif len_tuple_idx == 3:
                    idx, num_views, resolution_idx = idx
            else:
                len_tuple_idx = 1
                num_views = self.T
                resolution_idx = None

            rng = np.random.default_rng()
            
            # Get the starting reference image
            scene_info, ref_img_idx = self.all_ref_imgs[idx]
            
            # Handle invalid scenes
            invalid_seq = True
            while invalid_seq:
                basenames = self.scene_data[scene_info]["basenames"]
                
                # Check if too many invalid starting points in this scene
                if (sum([(1 - int(x)) for x in list(self.is_reachable_cache[scene_info].values())]) > 
                    len(basenames) - num_views):
                    self.invalid_scenes.append(scene_info)
                    
                # Skip invalid scenes
                while scene_info in self.invalid_scenes:
                    idx = rng.integers(low=0, high=len(self.all_ref_imgs))
                    scene_info, ref_img_idx = self.all_ref_imgs[idx]
                    basenames = self.scene_data[scene_info]["basenames"]

                score_matrix = self.scene_data[scene_info]["score_matrix"]
                imgs_idxs = self._generate_sequence(
                    scene_info, score_matrix, num_views, ref_img_idx, rng
                )

                if imgs_idxs is None:
                    # Try to find alternative starting point
                    random_direction = 2 * rng.choice(2) - 1
                    for offset in range(1, len(basenames)):
                        tentative_im_idx = (ref_img_idx + (random_direction * offset)) % len(basenames)
                        if (tentative_im_idx not in self.is_reachable_cache[scene_info] or 
                            self.is_reachable_cache[scene_info][tentative_im_idx]):
                            ref_img_idx = tentative_im_idx
                            break
                else:
                    invalid_seq = False

            # Load the multi-view data
            rgb_images = []
            depth_images = []
            intrinsics_list = []
            extrinsics_list = []
            valid_frames = []
            
            for view_idx in imgs_idxs:
                basename = basenames[view_idx].decode("utf-8") if isinstance(basenames[view_idx], bytes) else str(basenames[view_idx])
                

                frame_data = self.scene_frames[scene_info][basename]

                
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
                        extrinsics_list.append(extrinsics_list[-1].copy())
                    else:
                        # Create zero tensors if no valid frames yet
                        rgb_images.append(Image.new('RGB', (512, 384)))
                        depth_images.append(Image.fromarray(np.zeros((384, 512), dtype=np.float32)))
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
                        self.near_plane, self.far_plane, self.depth_type, resolution_idx, stride=None, no_depth_mask_inf=True, rng=rng # stride=1 for multi-view
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

            # Process each view
            processed_rgb = []
            processed_metric_depth = []
            processed_valid_mask = []
            processed_inf_mask = []
            processed_intrinsics = []
            processed_extrinsics = []

            for t in range(num_views):
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
            extrinsics_sequence = torch.stack(processed_extrinsics, dim=0)  # [T, 4, 4]

            return {
                'rgb': rgb_sequence,
                'metric_depth': metric_depth_sequence,
                'valid_mask': valid_mask_sequence,
                'inf_mask': inf_mask_sequence,
                'intrinsics': intrinsics_sequence,
                'extrinsics': extrinsics_sequence,
                # 'stride': 1,  # No stride concept for multi-view
                'valid': all(valid_frames),
                'depth_type': self.depth_type,
            }
        
        except Exception as e:
            print(f"[blendedmvs_multiview] Error processing view set: {str(e)}")

            if 'num_views' not in locals():
                num_views = self.T
            # Return dummy data with valid=False
            dummy_rgb = torch.zeros(num_views, 3, self.frame_size[0], self.frame_size[1])
            dummy_depth = torch.zeros(num_views, 1, self.frame_size[0], self.frame_size[1])
            dummy_mask = torch.zeros(num_views, 1, self.frame_size[0], self.frame_size[1], dtype=torch.bool)
            dummy_inf_mask = torch.zeros(num_views, 1, self.frame_size[0], self.frame_size[1], dtype=torch.bool)
            dummy_intrinsics = torch.eye(3).unsqueeze(0).repeat(num_views, 1, 1)
            dummy_extrinsics = torch.eye(4).unsqueeze(0).repeat(num_views, 1, 1)
            
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


# Compatibility alias for backwards compatibility
VideoDepthBlendedMVSNew = MultiViewDepthBlendedMVSNew


def generate_parquet_file():
    import glob
    
    # Path to processed BlendedMVS data
    DATA_DIR = "/mnt/localssd/blendedmvs_processed"  # Path to processed data from preprocessing script
    data_path = DATASET_CONFIG["blendedmvs"]["prefix"]
    
    sequences = sorted([s for s in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, s)) and len(s) == 24])
    df_columns = ['rgb_path', 'depth_path', 'cam_path', 'intrinsics', 'extrinsics', 'video_name', 'frame_number']
    records = []
    
    print(f"Found {len(sequences)} sequences to process")
    
    for seq in tqdm(sequences, desc="Processing sequences"):
        seq_dir = os.path.join(DATA_DIR, seq)
        
        # Get all files in sequence directory
        rgb_files = sorted([f for f in os.listdir(seq_dir) if f.endswith('.jpg')])
        
        for rgb_file in rgb_files:
            basename = rgb_file[:-4]  # Remove .jpg extension
            # frame_number = int(basename, 16) if all(c in '0123456789abcdef' for c in basename) else int(basename)
            frame_number = int(basename)
            
            depth_file = basename + '.exr'
            cam_file = basename + '.npz'
            
            # Check if all files exist
            rgb_path_local = os.path.join(seq_dir, rgb_file)
            depth_path_local = os.path.join(seq_dir, depth_file)
            cam_path_local = os.path.join(seq_dir, cam_file)
            
            if not all(os.path.exists(p) for p in [rgb_path_local, depth_path_local, cam_path_local]):
                continue
            
            # Generate paths
            rgb_path = rgb_path_local.replace(DATA_DIR, data_path)
            depth_path = depth_path_local.replace(DATA_DIR, data_path)
            cam_path = cam_path_local.replace(DATA_DIR, data_path)

            # Load camera data to extract intrinsics and extrinsics
            try:
                camera_data = np.load(cam_path_local)
                intrinsics = camera_data['intrinsics']
                R_cam2world = camera_data['R_cam2world']
                t_cam2world = camera_data['t_cam2world']
                
                # Convert to extrinsics matrix
                extrinsics = np.eye(4, dtype=np.float32)
                extrinsics[:3, :3] = R_cam2world
                extrinsics[:3, 3] = t_cam2world

                intrinsics_flat = intrinsics.flatten().tolist()
                extrinsics_flat = extrinsics.flatten().tolist()
                
                records.append({
                    'rgb_path': rgb_path,
                    'depth_path': depth_path,
                    'cam_path': cam_path,
                    'intrinsics': intrinsics_flat,
                    'extrinsics': extrinsics_flat,
                    'video_name': seq,
                    'frame_number': frame_number
                })

                
            except Exception as e:
                print(f"Error processing {cam_path_local}: {e}")
                continue

    # Create DataFrame
    df_new = pd.DataFrame.from_records(records, columns=df_columns)
    
    # Save parquet file
    parquet_filename = "blendedmvs_train.parquet"
    df_new.to_parquet(parquet_filename)
    print(f"Created parquet file with {len(df_new)} entries")
    
    # Save the parquet file
    upload_file(parquet_filename, DATASET_CONFIG["blendedmvs"]["parquet_path"])
    # Remove the local file
    os.remove(parquet_filename)


if __name__ == "__main__":
    # Uncomment to generate parquet file:
    # generate_parquet_file()
    # quit()

    # Test multi-view dataset:
    import moviepy.video.io.ffmpeg_writer as video_writer
    
    os.makedirs("data_vis", exist_ok=True)
    print("\nTesting BlendedMVS multi-view dataset...")
    video_dataset = MultiViewDepthBlendedMVS(T=10, transform=True, allow_repeat=False)
    print(f"Dataset length: {len(video_dataset)}")

    save_dir = "data_vis/blendedmvs"
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
            
            rgb_writer = video_writer.FFMPEG_VideoWriter(filename=f"{save_dir}/blendedmvs_video_rgb_{idx}.mp4", size=(W, H), fps=2)
            depth_writer = video_writer.FFMPEG_VideoWriter(filename=f"{save_dir}/blendedmvs_video_depth_{idx}.mp4", size=(W, H), fps=2)
            mask_writer = video_writer.FFMPEG_VideoWriter(filename=f"{save_dir}/blendedmvs_video_mask_{idx}.mp4", size=(W, H), fps=2)
            
            print("Saving video frames...")
            for t in tqdm(range(video_sample['rgb'].shape[0])):
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
            print(f"  - {save_dir}/blendedmvs_video_rgb_{idx}.mp4")
            print(f"  - {save_dir}/blendedmvs_video_depth_{idx}.mp4") 
            print(f"  - {save_dir}/blendedmvs_video_mask_{idx}.mp4")
            print(f"  - {save_dir}/blendedmvs_video_frame0_pointmap_{idx}.ply")
        else:
            print("No valid video sequences found in dataset.") 