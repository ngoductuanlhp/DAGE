import os
import io
import pandas as pd
from torch.utils.data import Dataset
from training.dataloaders.datasets.video_datasets_new.utils import cropping
from training.dataloaders.datasets.video_datasets_new.utils.transform import SeqColorJitter, ImgNorm

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import random
import cv2
from collections import defaultdict
import PIL

class SynchronizedTransformVideoCut3r:
    def __init__(self, seed=777, aug_crop=16):
        # self.resize          = transforms.Resize((H,W))
        # self.resize_depth    = transforms.Resize((H,W), interpolation=Image.NEAREST)
        # self.horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.to_tensor       = transforms.ToTensor()
        # self.W = W
        # self.H = H

        self.seed = seed

        self.aug_crop = aug_crop
        self.seq_aug_crop = False


    def _crop_resize_if_necessary(
        self, image, depthmap, intrinsics, resolution, rng=None
    ):
        """This function:
        - first downsizes the image with LANCZOS inteprolation,
          which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # prit
        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)
        assert min_margin_x > W / 5, f"Bad principal point"
        assert min_margin_y > H / 5, f"Bad principal point"
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, depthmap, intrinsics = cropping.crop_image_depthmap(
            image, depthmap, intrinsics, crop_bbox
        )

        # transpose the resolution if necessary
        W, H = image.size  # new size

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_resolution += (
                rng.integers(0, self.aug_crop)
                if not self.seq_aug_crop
                else self.delta_target_resolution
            )
        image, depthmap, intrinsics = cropping.rescale_image_depthmap(
            image, depthmap, intrinsics, target_resolution
        )

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(
            intrinsics, image.size, resolution, offset_factor=0.5
        )
        crop_bbox = cropping.bbox_from_intrinsics_in_out(
            intrinsics, intrinsics2, resolution
        )
        image, depthmap, intrinsics2 = cropping.crop_image_depthmap(
            image, depthmap, intrinsics, crop_bbox
        )

        return image, depthmap, intrinsics2

    def __call__(self, rgb_images, depth_images, intrinsics_list, resolution=(640,480), idx=0):
        # Decide on flip for entire sequence

        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, "_rng"):
            seed = torch.randint(0, 2**32, (1,)).item()
            self._rng = np.random.default_rng(seed=seed)

        if self.aug_crop > 1 and self.seq_aug_crop:
            self.delta_target_resolution = self._rng.integers(0, self.aug_crop)


        transform = SeqColorJitter()
        
        rgb_tensors = []
        depth_tensors = []
        intrinsics_tensors = []

        
        for rgb_image, depth_image, intrinsics in zip(rgb_images, depth_images, intrinsics_list):

            if isinstance(depth_image, PIL.Image.Image):
                depth_image = np.array(depth_image)

            rgb_image, depth_image, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depth_image, intrinsics, resolution, rng=self._rng
            )

            rgb_tensor = transform(rgb_image)
            depth_tensor = self.to_tensor(depth_image)
            intrinsics_tensor = torch.tensor(intrinsics)

            # resize
            # og_width, og_height = rgb_image.size
            # scale_w = self.W / og_width
            # scale_h = self.H / og_height
            # rgb_image   = self.resize(rgb_image)
            # depth_image = self.resize_depth(depth_image)
            # intrinsics[0, 0] *= scale_w  # fx
            # intrinsics[1, 1] *= scale_h  # fy
            # intrinsics[0, 2] *= scale_w  # cx
            # intrinsics[1, 2] *= scale_h  # cy

            # to tensor
            # rgb_tensor = self.to_tensor(rgb_image)
            # depth_tensor = self.to_tensor(depth_image)
            # intrinsics_tensor = torch.tensor(intrinsics)
            
            rgb_tensors.append(rgb_tensor)
            depth_tensors.append(depth_tensor)
            intrinsics_tensors.append(intrinsics_tensor)

        return rgb_tensors, depth_tensors, intrinsics_tensors
    
    # def __call__(self, rgb_images, depth_images, intrinsics_list):
    #     # Decide on flip for entire sequence
    #     flip = random.random() > 0.5
        
    #     rgb_tensors = []
    #     depth_tensors = []
    #     intrinsics_tensors = []
        
    #     for rgb_image, depth_image, intrinsics in zip(rgb_images, depth_images, intrinsics_list):
    #         # h-flip
    #         if flip:
    #             rgb_image = self.horizontal_flip(rgb_image)
    #             depth_image = self.horizontal_flip(depth_image)

    #         # resize
    #         og_width, og_height = rgb_image.size
    #         scale_w = self.W / og_width
    #         scale_h = self.H / og_height
    #         rgb_image   = self.resize(rgb_image)
    #         depth_image = self.resize_depth(depth_image)
    #         intrinsics[0, 0] *= scale_w  # fx
    #         intrinsics[1, 1] *= scale_h  # fy
    #         intrinsics[0, 2] *= scale_w  # cx
    #         intrinsics[1, 2] *= scale_h  # cy

    #         # to tensor
    #         rgb_tensor = self.to_tensor(rgb_image)
    #         depth_tensor = self.to_tensor(depth_image)
    #         intrinsics_tensor = torch.tensor(intrinsics)
            
    #         rgb_tensors.append(rgb_tensor)
    #         depth_tensors.append(depth_tensor)
    #         intrinsics_tensors.append(intrinsics_tensor)

    #     return rgb_tensors, depth_tensors, intrinsics_tensors