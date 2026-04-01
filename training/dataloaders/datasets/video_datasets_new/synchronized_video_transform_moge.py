import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import cv2
import utils3d
import random
from collections import defaultdict

import time

from evaluation.moge.utils.geometry_numpy import (
    mask_aware_nearest_resize_numpy,
    harmonic_mean_numpy,
    norm3d,
    depth_occlusion_edge_numpy,
    depth_of_field,
)

class SynchronizedTransform_MoGe:
    def __init__(self, moge_augmentation: dict = None):
        if moge_augmentation is not None:
            self.use_flip_augmentation = moge_augmentation.get('use_flip_augmentation', True)
            self.center_augmentation = moge_augmentation['center_augmentation']
            self.fov_range_absolute_min = moge_augmentation['fov_range_absolute_min']
            self.fov_range_absolute_max = moge_augmentation['fov_range_absolute_max']
            self.fov_range_relative_min = moge_augmentation['fov_range_relative_min']
            self.fov_range_relative_max = moge_augmentation['fov_range_relative_max']
            self.image_augmentation = moge_augmentation['image_augmentation']
            self.depth_interpolation = moge_augmentation['depth_interpolation']
            self.clamp_max_depth = moge_augmentation['clamp_max_depth']
            self.area_range = moge_augmentation['area_range']
            self.aspect_ratio_range = moge_augmentation['aspect_ratio_range']

        else:
        
            # NOTE default params from MoGe
            # self.center_augmentation = 0.25
            # self.fov_range_absolute_min = 30
            # self.fov_range_absolute_max = 150
            # self.fov_range_relative_min = 0.5
            # self.fov_range_relative_max = 1.0
            # self.image_augmentation = ['jittering', 'jpeg_loss', 'blurring']
            # self.depth_interpolation = 'nearest'
            # self.clamp_max_depth = 1000.0

            # # self.area_range = [250000, 600000]
            # self.area_range = [196000, 313600]
            # self.aspect_ratio_range = [0.5, 2.0]

            # NOTE reduced augmentation

            # self.center_augmentation = 0.10
            # self.fov_range_absolute_min = 60
            # self.fov_range_absolute_max = 120
            # self.fov_range_relative_min = 0.7
            # self.fov_range_relative_max = 1.0
            # self.image_augmentation = ['jittering', 'jpeg_loss', 'blurring']
            # self.depth_interpolation = 'nearest'
            # self.clamp_max_depth = 1000.0

            # NOTE only color aug
            self.use_flip_augmentation = True
            self.center_augmentation = 0.0
            self.fov_range_absolute_min = 60
            self.fov_range_absolute_max = 120
            self.fov_range_relative_min = 1.0
            self.fov_range_relative_max = 1.0
            self.image_augmentation = ['jittering', 'jpeg_loss', 'blurring']
            self.depth_interpolation = 'nearest'
            self.clamp_max_depth = 1000.0

            # self.area_range = [250000, 600000]
            self.area_range = [196000, 313600]
            self.aspect_ratio_range = [0.7, 1.5]

        # print(f"Use flip augmentation: {self.use_flip_augmentation}")

    def __call__(self, rgb_images, depth_images, raw_intrinsics_list, extrinsics_list, valid_frames, near_plane, far_plane, depth_type, resolution_idx, stride=None, no_depth_mask_inf=False, rng=None, augmentation=True):

        if augmentation:
            return self.transform_with_augmentation(rgb_images, depth_images, raw_intrinsics_list, extrinsics_list, valid_frames, near_plane, far_plane, depth_type, resolution_idx, stride, no_depth_mask_inf, rng)
        else:
            return self.transform_no_augmentation(rgb_images, depth_images, raw_intrinsics_list, extrinsics_list, valid_frames, near_plane, far_plane, depth_type, resolution_idx, stride, no_depth_mask_inf, rng)


    def transform_with_augmentation(self, rgb_images, depth_images, raw_intrinsics_list, extrinsics_list, valid_frames, near_plane, far_plane, depth_type, resolution_idx, stride=None, no_depth_mask_inf=False, rng=None):

        
        try:
            
            center_augmentation = self.center_augmentation
            fov_range_absolute_min = self.fov_range_absolute_min
            fov_range_absolute_max = self.fov_range_absolute_max
            fov_range_relative_min = self.fov_range_relative_min
            fov_range_relative_max = self.fov_range_relative_max
            image_augmentation = self.image_augmentation
            depth_interpolation = self.depth_interpolation
            clamp_max_depth = self.clamp_max_depth

            area_range = self.area_range
            aspect_ratio_range = self.aspect_ratio_range

            # NOTE the trick is use resolution_idx as random seed so all samples in the same batch have the same resolution
            resolution_rng = np.random.default_rng(resolution_idx)
            area = resolution_rng.uniform(area_range[0], area_range[1])
            aspect_ratio = resolution_rng.uniform(aspect_ratio_range[0], aspect_ratio_range[1])
            tgt_width, tgt_height = int((area * aspect_ratio) ** 0.5), int((area / aspect_ratio) ** 0.5)

            # NOTE ensure the target width and height are divisible by the patch size
            patch_size = 14
            tgt_width = tgt_width // patch_size * patch_size
            tgt_height = tgt_height // patch_size * patch_size
            # ------------------------------------------------------------

            num_frames = len(rgb_images)

            # Initialize variables that will be used across all frames
            if rng is None:
                rng = np.random.default_rng(random.randint(0, 2 ** 32 - 1))

            image_seq = []
            image_noaug_seq = []
            depth_seq = []
            depth_mask_seq = []
            depth_mask_inf_seq = []
            depth_mask_fin_seq = []
            intrinsics_seq = []
            extrinsics_seq = []
            
            rgb_images_np = []
            depth_images_np = []
            intrinsics_list_np = []
            extrinsics_list_np = []
            depth_masks_np = []
            depth_masks_inf_np = []
                
            
            for idx, (image, depth, raw_intrinsics, extrinsics) in enumerate(zip(rgb_images, depth_images, raw_intrinsics_list, extrinsics_list)):
                # Convert PIL Image to numpy array if needed
                if isinstance(image, Image.Image):
                    image = np.array(image)
                if isinstance(depth, Image.Image):
                    depth = np.array(depth)

                raw_height, raw_width = image.shape[:2]
                
                # Create depth masks
                depth_mask_nan = depth < near_plane
                depth_mask_inf = depth > far_plane

                depth[depth_mask_nan] = np.nan
                depth[depth_mask_inf] = np.inf
                
                depth_mask = np.isfinite(depth)
                depth_mask_inf = np.zeros_like(depth_mask) if no_depth_mask_inf else np.isinf(depth)
                depth = np.nan_to_num(depth, nan=1, posinf=1, neginf=1)

                
                intrinsics = raw_intrinsics.copy()
                intrinsics[0, 0] = intrinsics[0, 0] / raw_width
                intrinsics[0, 2] = intrinsics[0, 2] / raw_width
                intrinsics[1, 1] = intrinsics[1, 1] / raw_height
                intrinsics[1, 2] = intrinsics[1, 2] / raw_height

                rgb_images_np.append(image)
                depth_images_np.append(depth)
                intrinsics_list_np.append(intrinsics)
                depth_masks_np.append(depth_mask)
                depth_masks_inf_np.append(depth_mask_inf)
                extrinsics_list_np.append(extrinsics)

            rgb_images_np = np.stack(rgb_images_np, axis=0)
            depth_images_np = np.stack(depth_images_np, axis=0)
            intrinsics_list_np = np.stack(intrinsics_list_np, axis=0)
            depth_masks_np = np.stack(depth_masks_np, axis=0)
            depth_masks_inf_np = np.stack(depth_masks_inf_np, axis=0)
            extrinsics_list_np = np.stack(extrinsics_list_np, axis=0)
            ############################################

            # NOTE get transformation parameters for the first frame
            raw_horizontal, raw_vertical = abs(1.0 / intrinsics_list_np[0][0, 0]), abs(1.0 / intrinsics_list_np[0][1, 1])
            raw_fov_x, raw_fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics_list_np[0])
            raw_pixel_w, raw_pixel_h = raw_horizontal / raw_width, raw_vertical / raw_height

            tgt_aspect = tgt_width / tgt_height
            
            tgt_fov_x_min = min(float(fov_range_relative_min * raw_fov_x), float(fov_range_relative_min * utils3d.focal_to_fov(utils3d.fov_to_focal(raw_fov_y) / tgt_aspect)))
            tgt_fov_x_max = min(float(fov_range_relative_max * raw_fov_x), float(fov_range_relative_max * utils3d.focal_to_fov(utils3d.fov_to_focal(raw_fov_y) / tgt_aspect)))
            tgt_fov_x_min, tgt_fov_max = max(np.deg2rad(fov_range_absolute_min), float(tgt_fov_x_min)), min(np.deg2rad(fov_range_absolute_max), float(tgt_fov_x_max))
            tgt_fov_x = rng.uniform(min(tgt_fov_x_min, tgt_fov_x_max), tgt_fov_x_max)
            tgt_fov_y = utils3d.focal_to_fov(utils3d.numpy.fov_to_focal(tgt_fov_x) * tgt_aspect)

            # 2. set target image center (principal point) and the corresponding z-direction in raw camera space
            center_dtheta = center_augmentation * rng.uniform(-0.5, 0.5) * (raw_fov_x - tgt_fov_x)
            center_dphi = center_augmentation * rng.uniform(-0.5, 0.5) * (raw_fov_y - tgt_fov_y)
            cu, cv = 0.5 + 0.5 * np.tan(center_dtheta) / np.tan(raw_fov_x / 2), 0.5 + 0.5 *  np.tan(center_dphi) / np.tan(raw_fov_y / 2)
            direction = utils3d.unproject_cv(np.array([[cu, cv]], dtype=np.float32), np.array([1.0], dtype=np.float32), intrinsics=intrinsics_list_np[0])[0]

            # 3. obtain the rotation matrix for homography warping
            R = utils3d.rotation_matrix_from_vectors(direction, np.array([0, 0, 1], dtype=np.float32))
            # print(f"rotation matrix: {R}")

            # 4. shrink the target view to fit into the warped image
            corners = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32)
            corners = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1) @ (np.linalg.inv(intrinsics_list_np[0]).T @ R.T)   # corners in viewport's camera plane
            corners = corners[:, :2] / corners[:, 2:3]
            tgt_horizontal, tgt_vertical = np.tan(tgt_fov_x / 2) * 2, np.tan(tgt_fov_y / 2) * 2
            warp_horizontal, warp_vertical = float('inf'), float('inf')
            for i in range(4):
                intersection, _ = utils3d.numpy.ray_intersection(
                    np.array([0., 0.]), np.array([[tgt_aspect, 1.0], [tgt_aspect, -1.0]]),
                    corners[i - 1], corners[i] - corners[i - 1],
                )
                warp_horizontal, warp_vertical = min(warp_horizontal, 2 * np.abs(intersection[:, 0]).min()), min(warp_vertical, 2 * np.abs(intersection[:, 1]).min())
            tgt_horizontal, tgt_vertical = min(tgt_horizontal, warp_horizontal), min(tgt_vertical, warp_vertical)
            
            # 5. obtain the target intrinsics
            fx, fy = 1 / tgt_horizontal, 1 / tgt_vertical
            tgt_intrinsics = utils3d.numpy.intrinsics_from_focal_center(fx, fy, 0.5, 0.5).astype(np.float32)
            
            tgt_pixel_w, tgt_pixel_h = tgt_horizontal / tgt_width, tgt_vertical / tgt_height        # (should be exactly the same for x and y axes)
            rescaled_w, rescaled_h = int(raw_width * raw_pixel_w / tgt_pixel_w), int(raw_height * raw_pixel_h / tgt_pixel_h)

            # 6.2 calculate homography warping
            transform = intrinsics_list_np[0] @ np.linalg.inv(R) @ np.linalg.inv(tgt_intrinsics)
            uv_tgt = utils3d.numpy.image_uv(width=tgt_width, height=tgt_height)
            pts = np.concatenate([uv_tgt, np.ones((tgt_height, tgt_width, 1), dtype=np.float32)], axis=-1) @ transform.T
            uv_remap = pts[:, :, :2] / (pts[:, :, 2:3] + 1e-12)
            pixel_remap = utils3d.numpy.uv_to_pixel(uv_remap, width=rescaled_w, height=rescaled_h).astype(np.float32)
            tgt_ray_length = norm3d(utils3d.numpy.unproject_cv(uv_tgt, np.ones_like(uv_tgt[:, :, 0]), intrinsics=tgt_intrinsics))

            # NOTE precompute heavy part
            # print(f"Rescaled width: {rescaled_w}, rescaled height: {rescaled_h}, depth masks shape: {depth_masks_np.shape}, intrinsics: {intrinsics_list_np[0]}")
            _, depth_masks_nearest, resize_indices = mask_aware_nearest_resize_numpy(None, depth_masks_np, (rescaled_w, rescaled_h), return_index=True)
            depths_nearest = depth_images_np[resize_indices]
            distances_nearest = norm3d(utils3d.numpy.depth_to_points(depths_nearest, intrinsics=intrinsics_list_np))

            # depth_masks_nearest, depths_nearest, distances_nearest = [], [], []
            # for t in range(num_frames):
            #     _, depth_masks_nearest_t, resize_indices_t = mask_aware_nearest_resize_numpy(None, depth_masks_np[t], (rescaled_w, rescaled_h), return_index=True)
            #     depths_nearest_t = depth_images_np[t][resize_indices_t]
            #     distances_nearest_t = norm3d(utils3d.numpy.depth_to_points(depths_nearest_t, intrinsics=intrinsics_list_np[t]))

            #     depth_masks_nearest.append(depth_masks_nearest_t)
            #     depths_nearest.append(depths_nearest_t)
            #     distances_nearest.append(distances_nearest_t)

            #############################################
            
            # Sample augmentation parameters once for the entire video sequence
            aug_params = {}
            aug_params['flip_augmentation'] = rng.choice([True, False])
            
            if 'jittering' in image_augmentation:
                aug_params['brightness'] = rng.uniform(0.7, 1.3)
                aug_params['contrast'] = rng.uniform(0.7, 1.3)
                aug_params['saturation'] = rng.uniform(0.7, 1.3)
                aug_params['hue'] = rng.uniform(-0.1, 0.1)
                aug_params['gamma'] = rng.uniform(0.7, 1.3)
            
            if 'dof' in image_augmentation:
                aug_params['apply_dof'] = rng.uniform() < 0.5
                if aug_params['apply_dof']:
                    aug_params['dof_strength'] = rng.integers(12)
            
            if 'shot_noise' in image_augmentation:
                aug_params['apply_shot_noise'] = rng.uniform() < 0.5
                if aug_params['apply_shot_noise']:
                    aug_params['shot_noise_k'] = np.exp(rng.uniform(np.log(100), np.log(10000))) / 255
            
            if 'jpeg_loss' in image_augmentation:
                aug_params['apply_jpeg_loss'] = rng.uniform() < 0.5
                if aug_params['apply_jpeg_loss']:
                    aug_params['jpeg_quality'] = rng.integers(20, 100)
            
            if 'blurring' in image_augmentation:
                aug_params['apply_blurring'] = rng.uniform() < 0.5
                if aug_params['apply_blurring']:
                    aug_params['blur_ratio'] = rng.uniform(0.25, 1)
                    aug_params['blur_interpolation'] = rng.choice([cv2.INTER_LINEAR_EXACT, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])

            #############################################

            


            for idx in range(num_frames):
                image = rgb_images_np[idx]
                depth = depth_images_np[idx]
                intrinsics = intrinsics_list_np[idx]
                depth_mask = depth_masks_np[idx]
                depth_mask_inf = depth_masks_inf_np[idx]

                depth_mask_nearest = depth_masks_nearest[idx]
                # depth_nearest = depths_nearest[idx]
                distance_nearest = distances_nearest[idx]
                extrinsics = extrinsics_list_np[idx]

                
                # 6. do homogeneous transformation 
                # 6.1 The image and depth are resized first to approximately the same pixel size as the target image with PIL's antialiasing resampling
                image = np.array(Image.fromarray(image).resize((rescaled_w, rescaled_h), Image.Resampling.LANCZOS))

                depth_mask_inf = cv2.resize(depth_mask_inf.astype(np.uint8), (rescaled_w, rescaled_h), interpolation=cv2.INTER_NEAREST) > 0

                
                
                tgt_image = cv2.remap(image, pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_LANCZOS4)
                tgt_depth_mask_nearest = cv2.remap(depth_mask_nearest.astype(np.uint8), pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_NEAREST) > 0
                tgt_depth_nearest = cv2.remap(distance_nearest, pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_NEAREST) / tgt_ray_length
                # tgt_edge_mask = cv2.remap(edge_mask.astype(np.uint8), pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_NEAREST) > 0
                if depth_interpolation == 'bilinear':
                    raise NotImplementedError("Bilinear interpolation is not implemented due to heavy computation")
                else:
                    tgt_depth = tgt_depth_nearest
                tgt_depth_mask = tgt_depth_mask_nearest
                
                tgt_depth_mask_inf = cv2.remap(depth_mask_inf.astype(np.uint8), pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_NEAREST) > 0
                

                extrinsics_updated = extrinsics.copy()
                R_4x4 = np.eye(4, dtype=np.float32)
                R_4x4[:3, :3] = R.T  # R^-1 (since R is orthogonal, R^-1 = R^T)
                extrinsics_updated = extrinsics @ R_4x4
                
                # always make sure that mask is not empty
                if tgt_depth_mask.sum() / tgt_depth_mask.size < 0.001:
                    tgt_depth_mask = np.ones_like(tgt_depth_mask)
                    tgt_depth = np.ones_like(tgt_depth)
                    valid_frames[idx] = False


                
                # Flip augmentation
                if aug_params.get('flip_augmentation', False) and self.use_flip_augmentation:
                    tgt_image = np.flip(tgt_image, axis=1).copy()
                    tgt_depth = np.flip(tgt_depth, axis=1).copy()
                    tgt_depth_mask = np.flip(tgt_depth_mask, axis=1).copy()
                    tgt_depth_mask_inf = np.flip(tgt_depth_mask_inf, axis=1).copy()


                
                tgt_image_noaug = tgt_image.copy() # NOTE copy image no aug from here, meaning we skipp blurring, noise augmentation


                # Apply consistent augmentation to each frame
                if 'jittering' in image_augmentation:
                    tgt_image = torch.from_numpy(tgt_image).permute(2, 0, 1)
                    tgt_image = TF.adjust_brightness(tgt_image, aug_params['brightness'])
                    tgt_image = TF.adjust_contrast(tgt_image, aug_params['contrast'])
                    tgt_image = TF.adjust_saturation(tgt_image, aug_params['saturation'])
                    tgt_image = TF.adjust_hue(tgt_image, aug_params['hue'])
                    tgt_image = TF.adjust_gamma(tgt_image, aug_params['gamma'])
                    tgt_image = tgt_image.permute(1, 2, 0).numpy()
                



                if 'dof' in image_augmentation and aug_params.get('apply_dof', False):
                    tgt_disp = np.where(tgt_depth_mask_inf, 0, 1 / tgt_depth)
                    disp_min, disp_max = tgt_disp[tgt_depth_mask].min(), tgt_disp[tgt_depth_mask].max()
                    tgt_disp = cv2.inpaint(tgt_disp, (~tgt_depth_mask & ~tgt_depth_mask_inf).astype(np.uint8), 3, cv2.INPAINT_TELEA).clip(disp_min, disp_max)
                    dof_focus = rng.uniform(disp_min, disp_max)  # This can vary per frame for dof effect
                    tgt_image = depth_of_field(tgt_image, tgt_disp, dof_focus, aug_params['dof_strength'])
                if 'shot_noise' in image_augmentation and aug_params.get('apply_shot_noise', False):
                    tgt_image = (rng.poisson(tgt_image * aug_params['shot_noise_k']) / aug_params['shot_noise_k']).clip(0, 255).astype(np.uint8)
                if 'jpeg_loss' in image_augmentation and aug_params.get('apply_jpeg_loss', False):
                    tgt_image = cv2.imdecode(cv2.imencode('.jpg', tgt_image, [cv2.IMWRITE_JPEG_QUALITY, aug_params['jpeg_quality']])[1], cv2.IMREAD_COLOR)
                if 'blurring' in image_augmentation and aug_params.get('apply_blurring', False):
                    tgt_image = cv2.resize(cv2.resize(tgt_image, (int(tgt_width * aug_params['blur_ratio']), int(tgt_height * aug_params['blur_ratio'])), interpolation=cv2.INTER_AREA), (tgt_width, tgt_height), interpolation=aug_params['blur_interpolation'])

                
                # clamp depth maximum values
                max_depth = np.nanquantile(np.where(tgt_depth_mask, tgt_depth, np.nan), 0.01) * clamp_max_depth
                tgt_depth = np.clip(tgt_depth, 0, max_depth)
                tgt_depth = np.nan_to_num(tgt_depth, nan=1.0)

                tgt_depth_mask_fin = tgt_depth_mask

                image_seq.append(tgt_image)
                image_noaug_seq.append(tgt_image_noaug)
                depth_seq.append(tgt_depth)
                depth_mask_seq.append(tgt_depth_mask)
                depth_mask_inf_seq.append(tgt_depth_mask_inf)
                depth_mask_fin_seq.append(tgt_depth_mask_fin)
                intrinsics_seq.append(tgt_intrinsics)
                # extrinsics_seq.append(extrinsics)
                extrinsics_seq.append(extrinsics_updated)


            
            image_seq = np.stack(image_seq, axis=0).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
            image_noaug_seq = np.stack(image_noaug_seq, axis=0).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
            depth_seq = np.stack(depth_seq, axis=0)[:, None]
            depth_mask_seq = np.stack(depth_mask_seq, axis=0)[:, None] # T, 1, H, W
            depth_mask_inf_seq = np.stack(depth_mask_inf_seq, axis=0)[:, None] # T, 1, H, W
            depth_mask_fin_seq = np.stack(depth_mask_fin_seq, axis=0)[:, None] # T, 1, H, W
            intrinsics_seq = np.stack(intrinsics_seq, axis=0)
            extrinsics_seq = np.stack(extrinsics_seq, axis=0)


            return {
                'rgb': torch.from_numpy(image_seq).float(),
                'rgb_noaug': torch.from_numpy(image_noaug_seq).float(),
                'metric_depth': torch.from_numpy(depth_seq).float(),
                'valid_mask': torch.from_numpy(depth_mask_seq).bool(),
                'fin_mask': torch.from_numpy(depth_mask_fin_seq).bool(),
                'inf_mask': torch.from_numpy(depth_mask_inf_seq).bool(),
                'intrinsics': torch.from_numpy(intrinsics_seq).float(),
                'extrinsics': torch.from_numpy(extrinsics_seq).float(),
                'depth_type': depth_type,
                'valid': all(valid_frames),
                # 'stride': stride,
            }
            
        except Exception as e:
            print(f"Error in SynchronizedTransform_MoGe: {e}")

            dummy_dict = {
                'rgb': torch.zeros((num_frames, 3, tgt_height, tgt_width)).float(),
                'rgb_noaug': torch.zeros((num_frames, 3, tgt_height, tgt_width)).float(),
                'metric_depth': torch.zeros((num_frames, 1, tgt_height, tgt_width)).float(),
                'valid_mask': torch.zeros((num_frames, 1, tgt_height, tgt_width)).bool(),
                'fin_mask': torch.zeros((num_frames, 1, tgt_height, tgt_width)).bool(),
                'inf_mask': torch.zeros((num_frames, 1, tgt_height, tgt_width)).bool(),
                'intrinsics': torch.eye(3)[None].repeat(num_frames, 1, 1).float(),
                'extrinsics': torch.eye(4)[None].repeat(num_frames, 1, 1).float(),
                'depth_type': depth_type,
                'valid': False,
                # 'stride': stride,
            }

            return dummy_dict

    def transform_no_augmentation(self, rgb_images, depth_images, raw_intrinsics_list, extrinsics_list, valid_frames, near_plane, far_plane, depth_type, resolution_idx, stride=None, no_depth_mask_inf=False, rng=None):

        try:
            
            # center_augmentation = self.center_augmentation
            # fov_range_absolute_min = self.fov_range_absolute_min
            # fov_range_absolute_max = self.fov_range_absolute_max
            # fov_range_relative_min = self.fov_range_relative_min
            # fov_range_relative_max = self.fov_range_relative_max
            # image_augmentation = self.image_augmentation
            depth_interpolation = self.depth_interpolation
            clamp_max_depth = self.clamp_max_depth

            # area_range = self.area_range
            # aspect_ratio_range = self.aspect_ratio_range

            # NOTE the trick is use resolution_idx as random seed so all samples in the same batch have the same resolution
            # resolution_rng = np.random.default_rng(resolution_idx)
            # area = resolution_rng.uniform(area_range[0], area_range[1])
            # aspect_ratio = resolution_rng.uniform(aspect_ratio_range[0], aspect_ratio_range[1])
            # tgt_width, tgt_height = int((area * aspect_ratio) ** 0.5), int((area / aspect_ratio) ** 0.5)
            
            if isinstance(rgb_images[0], Image.Image):
                tgt_width, tgt_height = rgb_images[0].size
            else:
                tgt_width, tgt_height = rgb_images[0].shape[-1], rgb_images[0].shape[-2]

            num_frames = len(rgb_images)

            # Initialize variables that will be used across all frames
            # rng = np.random.default_rng(random.randint(0, 2 ** 32 - 1))
            image_seq = []
            depth_seq = []
            depth_mask_seq = []
            depth_mask_inf_seq = []
            depth_mask_fin_seq = []
            intrinsics_seq = []
            extrinsics_seq = []
            
            rgb_images_np = []
            depth_images_np = []
            intrinsics_list_np = []
            extrinsics_list_np = []
            depth_masks_np = []
            depth_masks_inf_np = []
                
            
            for idx, (image, depth, raw_intrinsics, extrinsics) in enumerate(zip(rgb_images, depth_images, raw_intrinsics_list, extrinsics_list)):
                # Convert PIL Image to numpy array if needed
                if isinstance(image, Image.Image):
                    image = np.array(image)
                if isinstance(depth, Image.Image):
                    depth = np.array(depth)

                raw_height, raw_width = image.shape[:2]
                
                # Create depth masks
                depth_mask_nan = depth < near_plane
                depth_mask_inf = depth > far_plane

                depth[depth_mask_nan] = np.nan
                depth[depth_mask_inf] = np.inf
                
                depth_mask = np.isfinite(depth)
                depth_mask_inf = np.zeros_like(depth_mask) if no_depth_mask_inf else np.isinf(depth)
                depth = np.nan_to_num(depth, nan=1, posinf=1, neginf=1)

                
                intrinsics = raw_intrinsics.copy()
                intrinsics[0, 0] = intrinsics[0, 0] / raw_width
                intrinsics[0, 2] = intrinsics[0, 2] / raw_width
                intrinsics[1, 1] = intrinsics[1, 1] / raw_height
                intrinsics[1, 2] = intrinsics[1, 2] / raw_height

                rgb_images_np.append(image)
                depth_images_np.append(depth)
                intrinsics_list_np.append(intrinsics)
                depth_masks_np.append(depth_mask)
                depth_masks_inf_np.append(depth_mask_inf)
                extrinsics_list_np.append(extrinsics)

            rgb_images_np = np.stack(rgb_images_np, axis=0)
            depth_images_np = np.stack(depth_images_np, axis=0)
            intrinsics_list_np = np.stack(intrinsics_list_np, axis=0)
            depth_masks_np = np.stack(depth_masks_np, axis=0)
            depth_masks_inf_np = np.stack(depth_masks_inf_np, axis=0)
            extrinsics_list_np = np.stack(extrinsics_list_np, axis=0)
            ############################################

            # NOTE get transformation parameters for the first frame
            raw_horizontal, raw_vertical = abs(1.0 / intrinsics_list_np[0][0, 0]), abs(1.0 / intrinsics_list_np[0][1, 1])
            raw_fov_x, raw_fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics_list_np[0])
            raw_pixel_w, raw_pixel_h = raw_horizontal / raw_width, raw_vertical / raw_height
            tgt_aspect = tgt_width / tgt_height
            
            # tgt_fov_x_min = min(float(fov_range_relative_min * raw_fov_x), float(fov_range_relative_min * utils3d.focal_to_fov(utils3d.fov_to_focal(raw_fov_y) / tgt_aspect)))
            # tgt_fov_x_max = min(float(fov_range_relative_max * raw_fov_x), float(fov_range_relative_max * utils3d.focal_to_fov(utils3d.fov_to_focal(raw_fov_y) / tgt_aspect)))
            # tgt_fov_x_min, tgt_fov_max = max(np.deg2rad(fov_range_absolute_min), float(tgt_fov_x_min)), min(np.deg2rad(fov_range_absolute_max), float(tgt_fov_x_max))
            # tgt_fov_x = rng.uniform(min(tgt_fov_x_min, tgt_fov_x_max), tgt_fov_x_max)
            # tgt_fov_y = utils3d.focal_to_fov(utils3d.numpy.fov_to_focal(tgt_fov_x) * tgt_aspect)

            # 2. set target image center (principal point) and the corresponding z-direction in raw camera space
            # center_dtheta = center_augmentation * rng.uniform(-0.5, 0.5) * (raw_fov_x - tgt_fov_x)
            # center_dphi = center_augmentation * rng.uniform(-0.5, 0.5) * (raw_fov_y - tgt_fov_y)
            # cu, cv = 0.5 + 0.5 * np.tan(center_dtheta) / np.tan(raw_fov_x / 2), 0.5 + 0.5 *  np.tan(center_dphi) / np.tan(raw_fov_y / 2)
            # direction = utils3d.unproject_cv(np.array([[cu, cv]], dtype=np.float32), np.array([1.0], dtype=np.float32), intrinsics=intrinsics_list_np[0])[0]

            # # 3. obtain the rotation matrix for homography warping
            # R = utils3d.rotation_matrix_from_vectors(direction, np.array([0, 0, 1], dtype=np.float32))

            # set target view direction
            cu, cv = 0.5, 0.5
            direction = utils3d.numpy.unproject_cv(np.array([[cu, cv]], dtype=np.float32), np.array([1.0], dtype=np.float32), intrinsics=intrinsics_list_np[0])[0]
            R = utils3d.numpy.rotation_matrix_from_vectors(direction, np.array([0, 0, 1], dtype=np.float32))

            # 4. shrink the target view to fit into the warped image
            corners = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32)
            corners = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1) @ (np.linalg.inv(intrinsics_list_np[0]).T @ R.T)   # corners in viewport's camera plane
            corners = corners[:, :2] / corners[:, 2:3]

            # tgt_horizontal, tgt_vertical = np.tan(tgt_fov_x / 2) * 2, np.tan(tgt_fov_y / 2) * 2
            # warp_horizontal, warp_vertical = float('inf'), float('inf')

            tgt_horizontal = min(raw_horizontal, raw_vertical * tgt_aspect)
            tgt_vertical = tgt_horizontal / tgt_aspect
            warp_horizontal, warp_vertical = abs(1.0 / intrinsics_list_np[0][0, 0]), abs(1.0 / intrinsics_list_np[0][1, 1])

            for i in range(4):
                intersection, _ = utils3d.numpy.ray_intersection(
                    np.array([0., 0.]), np.array([[tgt_aspect, 1.0], [tgt_aspect, -1.0]]),
                    corners[i - 1], corners[i] - corners[i - 1],
                )
                warp_horizontal, warp_vertical = min(warp_horizontal, 2 * np.abs(intersection[:, 0]).min()), min(warp_vertical, 2 * np.abs(intersection[:, 1]).min())
            tgt_horizontal, tgt_vertical = min(tgt_horizontal, warp_horizontal), min(tgt_vertical, warp_vertical)
            
            # 5. obtain the target intrinsics
            fx, fy = 1 / tgt_horizontal, 1 / tgt_vertical
            tgt_intrinsics = utils3d.numpy.intrinsics_from_focal_center(fx, fy, 0.5, 0.5).astype(np.float32)
            
            tgt_pixel_w, tgt_pixel_h = tgt_horizontal / tgt_width, tgt_vertical / tgt_height        # (should be exactly the same for x and y axes)
            rescaled_w, rescaled_h = int(raw_width * raw_pixel_w / tgt_pixel_w), int(raw_height * raw_pixel_h / tgt_pixel_h)

            # 6.2 calculate homography warping
            transform = intrinsics_list_np[0] @ np.linalg.inv(R) @ np.linalg.inv(tgt_intrinsics)
            uv_tgt = utils3d.numpy.image_uv(width=tgt_width, height=tgt_height)
            pts = np.concatenate([uv_tgt, np.ones((tgt_height, tgt_width, 1), dtype=np.float32)], axis=-1) @ transform.T
            uv_remap = pts[:, :, :2] / (pts[:, :, 2:3] + 1e-12)
            pixel_remap = utils3d.numpy.uv_to_pixel(uv_remap, width=rescaled_w, height=rescaled_h).astype(np.float32)
            tgt_ray_length = norm3d(utils3d.numpy.unproject_cv(uv_tgt, np.ones_like(uv_tgt[:, :, 0]), intrinsics=tgt_intrinsics))

            # NOTE precompute heavy part
            _, depth_masks_nearest, resize_indices = mask_aware_nearest_resize_numpy(None, depth_masks_np, (rescaled_w, rescaled_h), return_index=True)
            depths_nearest = depth_images_np[resize_indices]
            distances_nearest = norm3d(utils3d.numpy.depth_to_points(depths_nearest, intrinsics=intrinsics_list_np))
            #############################################
            

            for idx in range(num_frames):
                image = rgb_images_np[idx]
                depth = depth_images_np[idx]
                intrinsics = intrinsics_list_np[idx]
                depth_mask = depth_masks_np[idx]
                depth_mask_inf = depth_masks_inf_np[idx]

                depth_mask_nearest = depth_masks_nearest[idx]
                depth_nearest = depths_nearest[idx]
                distance_nearest = distances_nearest[idx]
                extrinsics = extrinsics_list_np[idx]

                
                # 6. do homogeneous transformation 
                # 6.1 The image and depth are resized first to approximately the same pixel size as the target image with PIL's antialiasing resampling
                image = np.array(Image.fromarray(image).resize((rescaled_w, rescaled_h), Image.Resampling.LANCZOS))

                depth_mask_inf = cv2.resize(depth_mask_inf.astype(np.uint8), (rescaled_w, rescaled_h), interpolation=cv2.INTER_NEAREST) > 0

                
                
                tgt_image = cv2.remap(image, pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_LANCZOS4)
                tgt_depth_mask_nearest = cv2.remap(depth_mask_nearest.astype(np.uint8), pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_NEAREST) > 0
                tgt_depth_nearest = cv2.remap(distance_nearest, pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_NEAREST) / tgt_ray_length
                # tgt_edge_mask = cv2.remap(edge_mask.astype(np.uint8), pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_NEAREST) > 0
                if depth_interpolation == 'bilinear':
                    raise NotImplementedError("Bilinear interpolation is not implemented due to heavy computation")
                else:
                    tgt_depth = tgt_depth_nearest
                tgt_depth_mask = tgt_depth_mask_nearest
                
                tgt_depth_mask_inf = cv2.remap(depth_mask_inf.astype(np.uint8), pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_NEAREST) > 0

                
                # always make sure that mask is not empty
                if tgt_depth_mask.sum() / tgt_depth_mask.size < 0.001:
                    tgt_depth_mask = np.ones_like(tgt_depth_mask)
                    tgt_depth = np.ones_like(tgt_depth)
                    valid_frames[idx] = False

                
                # clamp depth maximum values
                max_depth = np.nanquantile(np.where(tgt_depth_mask, tgt_depth, np.nan), 0.01) * clamp_max_depth
                tgt_depth = np.clip(tgt_depth, 0, max_depth)
                tgt_depth = np.nan_to_num(tgt_depth, nan=1.0)

                tgt_depth_mask_fin = tgt_depth_mask

                image_seq.append(tgt_image)
                depth_seq.append(tgt_depth)
                depth_mask_seq.append(tgt_depth_mask)
                depth_mask_inf_seq.append(tgt_depth_mask_inf)
                depth_mask_fin_seq.append(tgt_depth_mask_fin)
                intrinsics_seq.append(tgt_intrinsics)
                extrinsics_seq.append(extrinsics)


            
            image_seq = np.stack(image_seq, axis=0).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
            image_seq_noaug = image_seq.copy()
            depth_seq = np.stack(depth_seq, axis=0)[:, None]
            depth_mask_seq = np.stack(depth_mask_seq, axis=0)[:, None] # T, 1, H, W
            depth_mask_inf_seq = np.stack(depth_mask_inf_seq, axis=0)[:, None] # T, 1, H, W
            depth_mask_fin_seq = np.stack(depth_mask_fin_seq, axis=0)[:, None] # T, 1, H, W
            intrinsics_seq = np.stack(intrinsics_seq, axis=0)
            extrinsics_seq = np.stack(extrinsics_seq, axis=0)


            return {
                'rgb': torch.from_numpy(image_seq).float(),
                'rgb_noaug': torch.from_numpy(image_seq_noaug).float(),
                'metric_depth': torch.from_numpy(depth_seq).float(),
                'valid_mask': torch.from_numpy(depth_mask_seq).bool(),
                'fin_mask': torch.from_numpy(depth_mask_fin_seq).bool(),
                'inf_mask': torch.from_numpy(depth_mask_inf_seq).bool(),
                'intrinsics': torch.from_numpy(intrinsics_seq).float(),
                'extrinsics': torch.from_numpy(extrinsics_seq).float(),
                'depth_type': depth_type,
                'valid': all(valid_frames),
                # 'stride': stride,
            }
            
        except Exception as e:
            print(f"Error in SynchronizedTransform_MoGe: {e}")

            dummy_dict = {
                'rgb': torch.zeros((num_frames, 3, tgt_height, tgt_width)).float(),
                'rgb_noaug': torch.zeros((num_frames, 3, tgt_height, tgt_width)).float(),
                'metric_depth': torch.zeros((num_frames, 1, tgt_height, tgt_width)).float(),
                'valid_mask': torch.zeros((num_frames, 1, tgt_height, tgt_width)).bool(),
                'fin_mask': torch.zeros((num_frames, 1, tgt_height, tgt_width)).bool(),
                'inf_mask': torch.zeros((num_frames, 1, tgt_height, tgt_width)).bool(),
                'intrinsics': torch.eye(3)[None].repeat(num_frames, 1, 1).float(),
                'depth_type': depth_type,
                'valid': False,
                # 'stride': stride,
            }

            return dummy_dict