import os
import os.path as osp
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
from collections import defaultdict

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402

from training.dataloaders.datasets.video_datasets_new.synchronized_video_transform import (
    SynchronizedTransformVideoCut3r,
)
from training.dataloaders.datasets.video_datasets_new.synchronized_video_transform_moge import (
    SynchronizedTransform_MoGe,
)


class SynchronizedTransform_MegaDepthVideo:
    def __init__(self, H, W):
        self.resize = transforms.Resize((H, W))
        self.resize_depth = transforms.Resize((H, W), interpolation=Image.NEAREST)
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.to_tensor = transforms.ToTensor()
        self.W = W
        self.H = H

    def __call__(self, rgb_images, depth_images, intrinsics_list, extrinsics_list=None):
        if extrinsics_list is not None:
            flip = False
        else:
            flip = random.random() > 0.5

        rgb_tensors = []
        depth_tensors = []
        intrinsics_tensors = []
        extrinsics_tensors = []

        for idx, (rgb_image, depth_image, intrinsics) in enumerate(
            zip(rgb_images, depth_images, intrinsics_list)
        ):
            if flip:
                rgb_image = self.horizontal_flip(rgb_image)
                depth_image = self.horizontal_flip(depth_image)

            og_width, og_height = rgb_image.size
            scale_w = self.W / og_width
            scale_h = self.H / og_height

            rgb_image = self.resize(rgb_image)
            depth_image = self.resize_depth(depth_image)

            intrinsics[0, 0] *= scale_w
            intrinsics[1, 1] *= scale_h
            intrinsics[0, 2] *= scale_w
            intrinsics[1, 2] *= scale_h

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


class VideoDepthMegaDepthNew(Dataset):
    depth_type = "sfm"
    is_metric_scale = False

    def __init__(
        self,
        T=5,
        stride_range=(1, 4),
        transform=True,
        resize=(384, 512),
        near_plane=0.05,
        far_plane=1000.0,
        resolutions=None,
        use_moge=False,
        moge_augmentation=None,
        use_cut3r_frame_sampling=False,
        video_prob=0.6,
        fix_interval_prob=0.6,
        block_shuffle=16,
        allow_repeat=False,
    ):
        parquet_data = download_file(
            
            DATASET_CONFIG["megadepth"]["parquet_path"],
        )
        self.parquet = io.BytesIO(parquet_data)

        self.T = T
        self.stride_range = stride_range
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.frame_size = resize

        self.resolutions = resolutions
        self.use_moge = use_moge
        self.moge_augmentation = moge_augmentation

        self.use_cut3r_frame_sampling = use_cut3r_frame_sampling
        self.video_prob = video_prob
        self.fix_interval_prob = fix_interval_prob
        self.block_shuffle = block_shuffle

        self.allow_repeat = allow_repeat

        # self.scene_frames = self._organize_by_scene()
        self._load_precomputed_sets()
        # self.frame_list = list(range(len(self.precomputed_sets)))

        if self.use_moge:
            self.transform = SynchronizedTransform_MoGe(
                moge_augmentation=self.moge_augmentation
            )
        elif self.resolutions is not None:
            self.transform = SynchronizedTransformVideoCut3r(seed=777)
        else:
            self.transform = (
                SynchronizedTransform_MegaDepthVideo(H=resize[0], W=resize[1])
                if transform
                else None
            )


    def _load_precomputed_sets(self):
        default_sets_path = DATASET_CONFIG["megadepth"].get("precomputed_sets_path")

        precomputed_sets_file = download_file(
            
            default_sets_path,
        )

        
        with np.load(io.BytesIO(precomputed_sets_file), allow_pickle=True) as data:
            self.all_scenes = data["scenes"]
            self.all_images = data["images"]
            self.sets = data["sets"]

    def __len__(self):
        return len(self.sets)

    def _can_sample_sequence(self, scene_key, start_idx, stride, num_frames=None):
        if num_frames is None:
            num_frames = self.T

        frames = self.scene_frames[scene_key]
        if start_idx + (num_frames - 1) * stride >= len(frames):
            return False

        return True

    def _load_single_frame(self, scene_path, img_name):
        try:
            rgb_data = download_file(
                os.path.join(scene_path, img_name + ".jpg")
            )
            rgb_image = Image.open(io.BytesIO(rgb_data)).convert("RGB")
            del rgb_data  # Free memory immediately

            depth_data = download_file(
                os.path.join(scene_path, img_name + ".exr")
            )
            depth_buffer = np.frombuffer(depth_data, dtype=np.uint8)
            del depth_data  # Free memory immediately
            
            depth_array = cv2.imdecode(
                depth_buffer, cv2.IMREAD_UNCHANGED
            )
            del depth_buffer  # Free memory immediately
            
            if depth_array is None:
                raise ValueError("Failed to decode EXR depth data")

            if depth_array.ndim == 3:
                depth_array = depth_array[..., 0]

            camera_data = download_file(
                os.path.join(scene_path, img_name + ".npz")
            )
            with np.load(io.BytesIO(camera_data)) as data:
                intrinsics = data["intrinsics"].copy()
                extrinsics = data["cam2world"].copy()
            del camera_data  # Free memory immediately

            # breakpoint()

            depth_array = depth_array.astype(np.float32)
            depth_array[~np.isfinite(depth_array)] = 0.0
            depth_image = Image.fromarray(depth_array)
            del depth_array  # Free memory immediately

            # intrinsics = np.array(frame_data["intrinsics"], dtype=np.float32).reshape(3, 3)
            # extrinsics = np.array(frame_data["extrinsics"], dtype=np.float32).reshape(4, 4)

            return rgb_image, depth_image, intrinsics, extrinsics, True

        except Exception as exc:
            print(f"[megadepth_video] Error loading frame: {exc}")
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
                    raise ValueError(f"Invalid tuple length: {len_tuple_idx}")
            else:
                len_tuple_idx = 1
                num_frames = self.T
                resolution_idx = None

            rng = np.random.default_rng(random.randint(0, 2**32 - 1))


            scene_id = self.sets[idx][0]
            all_image_idxs = self.sets[idx][1:65]
            
            image_idxs = rng.choice(all_image_idxs, num_frames, replace=True)
            scene, subscene = self.all_scenes[scene_id].split()
            scene_path = os.path.join(DATASET_CONFIG["megadepth"]["prefix"], scene, subscene)
            is_video = False
            is_success = True


            if not is_success:
                next_idx = (
                    idx
                    + (num_frames - 1)
                    * random.randint(self.stride_range[0], self.stride_range[1])
                ) % self.__len__()

                if len_tuple_idx == 3:
                    return self.__getitem__((next_idx, num_frames, resolution_idx))
                elif len_tuple_idx == 2:
                    return self.__getitem__((next_idx, num_frames))
                else:
                    return self.__getitem__(next_idx)

            rgb_images = []
            depth_images = []
            intrinsics_list = []
            extrinsics_list = []
            valid_frames = []
            

            first_frame_reso = None

            for i,img_idx in enumerate(image_idxs):
                img_name = self.all_images[img_idx]
                # frame_data = frames[frame_idx]
                rgb_img, depth_img, intrinsics, extrinsics, valid = self._load_single_frame(
                    scene_path, img_name
                )
                # print(f"rgb_img: {rgb_img.size}, depth_img: {depth_img.size}, intrinsics: {intrinsics.shape}, extrinsics: {extrinsics.shape}, valid: {valid}")

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
                    if rgb_images:
                        rgb_images.append(rgb_images[-1])
                        depth_images.append(depth_images[-1])
                        intrinsics_list.append(intrinsics_list[-1].copy())
                        extrinsics_list.append(extrinsics_list[-1])
                    else:
                        rgb_images.append(Image.new("RGB", (self.frame_size[1], self.frame_size[0])))
                        depth_images.append(
                            Image.fromarray(
                                np.zeros(
                                    (self.frame_size[0], self.frame_size[1]),
                                    dtype=np.float32,
                                )
                            )
                        )
                        intrinsics_list.append(np.eye(3, dtype=np.float32))
                        extrinsics_list.append(np.eye(4, dtype=np.float32))
                    valid_frames.append(False)

            if self.transform is not None:
                if isinstance(self.transform, SynchronizedTransformVideoCut3r):
                    target_resolution = (
                        self.resolutions[resolution_idx]
                        if (resolution_idx is not None and self.resolutions is not None)
                        else (self.frame_size[1], self.frame_size[0])
                    )
                    rgb_tensors, depth_tensors, intrinsics_tensors = self.transform(
                        rgb_images,
                        depth_images,
                        intrinsics_list,
                        resolution=target_resolution,
                    )
                    extrinsics_tensors = [torch.tensor(extrinsics) for extrinsics in extrinsics_list]
                elif isinstance(self.transform, SynchronizedTransform_MoGe):
                    processed_data = self.transform(
                        rgb_images,
                        depth_images,
                        intrinsics_list,
                        extrinsics_list,
                        valid_frames,
                        self.near_plane,
                        self.far_plane,
                        self.depth_type,
                        resolution_idx,
                        stride=None,
                        rng=rng,
                        no_depth_mask_inf=True,
                    )
                    processed_data['is_metric_scale'] = self.is_metric_scale
                    return processed_data
                else:
                    (
                        rgb_tensors,
                        depth_tensors,
                        intrinsics_tensors,
                        extrinsics_tensors,
                    ) = self.transform(rgb_images, depth_images, intrinsics_list, extrinsics_list)
            else:
                rgb_tensors = [transforms.ToTensor()(img) for img in rgb_images]
                depth_tensors = [transforms.ToTensor()(img) for img in depth_images]
                intrinsics_tensors = [torch.tensor(intrinsics) for intrinsics in intrinsics_list]
                extrinsics_tensors = [torch.tensor(extrinsics) for extrinsics in extrinsics_list]

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

                depth_tensor = depth_tensor.float()

                valid_depth_mask = (
                    torch.isfinite(depth_tensor)
                    & (depth_tensor > self.near_plane)
                    & (depth_tensor < self.far_plane)
                )
                inf_depth_mask = depth_tensor >= self.far_plane

                rgb_tensor = rgb_tensor * 2.0 - 1.0

                if valid_depth_mask.any() and valid_frames[t]:
                    flat_depth = depth_tensor[valid_depth_mask].flatten()
                    min_depth = torch.quantile(flat_depth, 0.0)
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
                "frame_positions": image_idxs,
                "is_video": is_video,
                "valid": all(valid_frames),
                "depth_type": self.depth_type,
            }

        except Exception as exc:
            print(f"[megadepth_video] Error processing frame sequence: {exc}")

            if num_frames is None:
                num_frames = self.T

            dummy_rgb = torch.zeros(num_frames, 3, self.frame_size[0], self.frame_size[1])
            dummy_depth = torch.zeros(num_frames, 1, self.frame_size[0], self.frame_size[1])
            dummy_mask = torch.zeros(
                num_frames, 1, self.frame_size[0], self.frame_size[1], dtype=torch.bool
            )
            dummy_inf_mask = torch.zeros_like(dummy_mask)
            dummy_intrinsics = torch.eye(3).unsqueeze(0).repeat(num_frames, 1, 1)
            dummy_extrinsics = torch.eye(4).unsqueeze(0).repeat(num_frames, 1, 1)

            return {
                "rgb": dummy_rgb,
                "metric_depth": dummy_depth,
                "valid_mask": dummy_mask,
                "inf_mask": dummy_inf_mask,
                "intrinsics": dummy_intrinsics,
                "extrinsics": dummy_extrinsics,
                "valid": False,
                "depth_type": self.depth_type,
            }


def generate_parquet_file():
    import tqdm

    data_dir = "/mnt/localssd/processed_megadepth"
    data_path = DATASET_CONFIG["megadepth"]["prefix"]
    precomputed_sets_local = "/mnt/localssd/megadepth_sets_64.npz"
    precomputed_sets_loc = DATASET_CONFIG["megadepth"].get("precomputed_sets_path")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"MegaDepth processed directory not found: {data_dir}")

    if not os.path.isfile(precomputed_sets_local):
        raise FileNotFoundError(
            f"MegaDepth precomputed sets not found: {precomputed_sets_local}"
        )

    with np.load(
        osp.join(precomputed_sets_local), allow_pickle=True
    ) as data:
        all_scenes = data["scenes"]
        all_images = data["images"]
        sets = data["sets"]


    scene_dirs = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    )

    # breakpoint()
    df_columns = [
        "rgb_path",
        "depth_path",
        "intrinsics",
        "extrinsics",
        "scene_name",
        "subscene_name",
        "image_name",
        "frame_number",
    ]
    records = []

    for scene_name in tqdm.tqdm(scene_dirs, desc="Processing MegaDepth scenes"):
        scene_path = os.path.join(data_dir, scene_name)
        subscene_dirs = sorted(
            [d for d in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, d))]
        )

        for subscene_name in subscene_dirs:
            subscene_path = os.path.join(scene_path, subscene_name)
            npz_files = sorted(
                [f for f in os.listdir(subscene_path) if f.endswith(".npz")]
            )

            for idx, meta_file in enumerate(npz_files):
                stem = meta_file[:-4]

                rgb_path_local = os.path.join(subscene_path, f"{stem}.jpg")
                depth_path_local = os.path.join(subscene_path, f"{stem}.exr")
                meta_path_local = os.path.join(subscene_path, meta_file)

                if not all(
                    os.path.exists(path)
                    for path in [rgb_path_local, depth_path_local, meta_path_local]
                ):
                    continue

                with np.load(meta_path_local) as meta:
                    intrinsics = meta["intrinsics"]
                    cam2world = meta["cam2world"]

                rgb_path = rgb_path_local.replace(data_dir, data_path)
                depth_path = depth_path_local.replace(data_dir, data_path)
                camera_path = meta_path_local.replace(data_dir, data_path)
                records.append(
                    {
                        "rgb_path": rgb_path,
                        "depth_path": depth_path,
                        "camera_path": camera_path,
                        "intrinsics": intrinsics.flatten().tolist(),
                        "extrinsics": cam2world.flatten().tolist(),
                        "scene_name": scene_name,
                        "subscene_name": subscene_name,
                        "image_name": stem,
                        "frame_number": idx,
                    }
                )

    df_new = pd.DataFrame.from_records(records, columns=df_columns)

    parquet_filename = "megadepth_train.parquet"
    df_new.to_parquet(parquet_filename)

    upload_file(
        parquet_filename,
        DATASET_CONFIG["megadepth"]["parquet_path"],
    )

    if precomputed_sets_loc is not None:
        upload_file(
            precomputed_sets_local,
            precomputed_sets_loc,
        )

    os.remove(parquet_filename)


if __name__ == "__main__":
    # Uncomment to generate parquet file after preprocessing MegaDepth:
    # generate_parquet_file()
    # pass

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
    print("\nTesting megadepth video dataset with new sampling...")

    dataset_name = "megadepth_new"
    save_dir = f"data_vis/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    resolutions = [[0,1]] * 10000
    video_dataset = VideoDepthMegaDepthNew(
        T=10, 
        stride_range=(1, 1),
        transform=True,
        resolutions=resolutions, 
        use_moge=True,
        use_cut3r_frame_sampling=True,
        video_prob=0.6,
        fix_interval_prob=0.6,
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
                depth_writer.write_frame(depth_bgr)
                mask_writer.write_frame(mask_bgr)

                all_gt_pts.append(gt_pts)
                all_gt_pts_rgb.append(gt_pts_rgb)
                
            rgb_writer.close()
            depth_writer.close()
            mask_writer.close()

            all_gt_pts = torch.cat(all_gt_pts, dim=0)
            all_gt_pts_rgb = torch.cat(all_gt_pts_rgb, dim=0)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                all_gt_pts
            )
            pcd.colors = o3d.utility.Vector3dVector(
                all_gt_pts_rgb
            )
            save_path = f"{save_dir}/global_pts_{idx}.ply"
            o3d.io.write_point_cloud(
                save_path,
                pcd,
            )

            print(f"Saved pointcloud to {save_path}")
            
            print("Video sequence test completed successfully!")
    else:
        print("No valid video sequences found in dataset.")

