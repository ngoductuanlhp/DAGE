import os
import io
import pandas as pd
import tqdm
import numpy as np
from training.dataloaders.data_io import download_file, upload_file
from training.dataloaders.config import DATASET_CONFIG
from PIL import Image

from training.dataloaders.datasets.video_datasets_new.depth_habitat3d_new import VideoDepthHabitat3DNew


class VideoDepthGibsonNew(VideoDepthHabitat3DNew):
    """
    Video sequence dataset for Gibson data.
    Inherits from Habitat3D dataset as they share identical structure.
    Simply override the dataset_config_key to use Gibson configuration.
    """
    depth_type = "sfm"  # Gibson has metric depth data
    dataset_config_key = "gibson"  # Override to use Gibson config


def generate_parquet_file(gibson_data_dir, output_parquet_filename="gibson.parquet"):
    """
    Generate parquet file for Gibson dataset.
    
    Args:
        gibson_data_dir: Root directory containing the processed Gibson data
                        Expected structure: gibson_data_dir/{scene_id:08}/rgb/view{idx:05d}.png
        output_parquet_filename: Name of the output parquet file
    """
    
    df_columns = ['scene_id', 'subscene_id', 'rgb_dir', 'depth_dir', 'camera_dir', 'num_views', 'scene_name']
    records = []
    
    # List all scene directories (directories with 8-digit names)
    scene_dirs = sorted([d for d in os.listdir(gibson_data_dir) 
                        if os.path.isdir(os.path.join(gibson_data_dir, d))])
    
    print(f"Found {len(scene_dirs)} scene directories for Gibson")
    
    for scene_id in tqdm.tqdm(scene_dirs):
        scene_path = os.path.join(gibson_data_dir, scene_id)
        subscene_dirs = sorted([d for d in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, d))])

        for subscene_id in subscene_dirs:
            subscene_path = os.path.join(scene_path, subscene_id)

            rgb_dir = os.path.join(subscene_path, "rgb")
            depth_dir = os.path.join(subscene_path, "depth")
            camera_dir = os.path.join(subscene_path, "camera")
            
            # Check if all required directories exist
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
            data_prefix = DATASET_CONFIG["gibson"]["prefix"] if "gibson" in DATASET_CONFIG else "gibson"
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
    print(f"Created Gibson parquet file with {len(df_new)} scenes")
    print(f"Total views across all scenes: {df_new['num_views'].sum()}")
    print(f"Average views per scene: {df_new['num_views'].mean():.2f}")
    
    # Save if configured
    if "gibson" in DATASET_CONFIG:
        try:
            upload_file(output_parquet_filename,
                       DATASET_CONFIG["gibson"]["parquet_path"])
            print(f"Saved parquet file")
            # Remove local file after upload
            os.remove(output_parquet_filename)
        except Exception as e:
            print(f"Failed to save: {str(e)}")
    
    return df_new


if __name__ == "__main__":
    import torch
    import tqdm
    import moviepy.video.io.ffmpeg_writer as video_writer
    import cv2
    from training.dataloaders.batched_sampler import make_sampler
    from torch.utils.data import DataLoader
    import open3d as o3d
    import utils3d
    from third_party.pi3.utils.geometry import homogenize_points
    
    # Test generate_parquet_file
    # Uncomment and set your data directory to generate parquet
    # gibson_data_dir = "/mnt/localssd/gibson_release/"
    # generate_parquet_file(gibson_data_dir)
    # quit()
    
    os.makedirs("data_vis", exist_ok=True)
    print("\nTesting Gibson dataset...")

    dataset_name = "gibson_new"
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

