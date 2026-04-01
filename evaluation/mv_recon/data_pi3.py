import os
import cv2
import json
import numpy as np
import os.path as osp
from collections import deque
from PIL import Image, ImageFile
import random
import imageio.v2
from typing import Optional, Union, List
from evaluation.mv_recon.base import BaseStereoViewDataset
# from dust3r.utils.image import imread_cv2
import evaluation.mv_recon.dataset_utils.cropping as cropping
from third_party.pi3.utils.geometry import homogenize_points, se3_inverse, depth_edge

from evaluation.mv_recon.dataset_utils.cropping import resize_image_depth_and_intrinsic, resize_image

def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """Open an image or a depthmap with opencv-python."""
    if path.endswith((".exr", "EXR")):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f"Could not load image={path} with {options=}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img



def shuffle_deque(dq, seed=None):
    # Set the random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Convert deque to list, shuffle, and convert back
    shuffled_list = list(dq)
    random.shuffle(shuffled_list)
    return deque(shuffled_list)


class SevenScenes(BaseStereoViewDataset):
    def __init__(
        self,
        num_seq=1,
        num_frames=5,
        min_thresh=10,
        max_thresh=100,
        test_id=None,
        full_video=False,
        tuple_list=None,
        seq_id=None,
        rebuttal=False,
        shuffle_seed=-1,
        kf_every=1,
        load_img_size=518,
        patch_size=14,
        *args,
        ROOT,
        **kwargs,
    ):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.num_frames = num_frames
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every
        self.seq_id = seq_id
        self.rebuttal = rebuttal
        self.shuffle_seed = shuffle_seed

        self.load_img_size = load_img_size
        self.patch_size = patch_size
        # load all scenes
        self.load_all_tuples(tuple_list)
        self.load_all_scenes(ROOT)

    def __len__(self):
        if self.tuple_list is not None:
            return len(self.tuple_list)
        return len(self.scene_list) * self.num_seq

    def load_all_tuples(self, tuple_list):
        if tuple_list is not None:
            self.tuple_list = tuple_list
            # with open(tuple_path) as f:
            #     self.tuple_list = f.read().splitlines()

        else:
            self.tuple_list = None

    def load_all_scenes(self, base_dir):

        if self.tuple_list is not None:
            # Use pre-defined simplerecon scene_ids
            self.scene_list = [
                "stairs/seq-06",
                "stairs/seq-02",
                "pumpkin/seq-06",
                "chess/seq-01",
                "heads/seq-02",
                "fire/seq-02",
                "office/seq-03",
                "pumpkin/seq-03",
                "redkitchen/seq-07",
                "chess/seq-02",
                "office/seq-01",
                "redkitchen/seq-01",
                "fire/seq-01",
            ]
            print(f"Found {len(self.scene_list)} sequences in split {self.split}")
            return

        scenes = os.listdir(base_dir)

        file_split = {"train": "TrainSplit.txt", "test": "TestSplit.txt"}[self.split]

        self.scene_list = []
        for scene in scenes:
            if self.test_id is not None and scene != self.test_id:
                continue
            # read file split
            with open(osp.join(base_dir, scene, file_split)) as f:
                seq_ids = f.read().splitlines()

                for seq_id in seq_ids:
                    # seq is string, take the int part and make it 01, 02, 03
                    # seq_id = 'seq-{:2d}'.format(int(seq_id))
                    num_part = "".join(filter(str.isdigit, seq_id))
                    seq_id = f"seq-{num_part.zfill(2)}"
                    if self.seq_id is not None and seq_id != self.seq_id:
                        continue
                    self.scene_list.append(f"{scene}/{seq_id}")

        print(f"Found {len(self.scene_list)} sequences in split {self.split}")

    def _get_views(self, idx, resolution, rng):

        if self.tuple_list is not None:
            line = self.tuple_list[idx].split(" ")
            scene_id = line[0]
            img_idxs = line[1:]

        else:
            scene_id = self.scene_list[idx // self.num_seq]
            seq_id = idx % self.num_seq

            data_path = osp.join(self.ROOT, scene_id)
            num_files = len([name for name in os.listdir(data_path) if "color" in name])
            img_idxs = [f"{i:06d}" for i in range(num_files)]
            img_idxs = img_idxs[:: self.kf_every]

        # Intrinsics used in SimpleRecon
        fx, fy, cx, cy = 525, 525, 320, 240
        intrinsics_ = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        views = []
        imgs_idxs = deque(img_idxs)
        if self.shuffle_seed >= 0:
            imgs_idxs = shuffle_deque(imgs_idxs)

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()
            impath = osp.join(self.ROOT, scene_id, f"frame-{im_idx}.color.png")
            depthpath = osp.join(self.ROOT, scene_id, f"frame-{im_idx}.depth.proj.png")
            posepath = osp.join(self.ROOT, scene_id, f"frame-{im_idx}.pose.txt")

            # rgb_image = imread_cv2(impath)
            # depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            # rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))


            rgb_image: Image.Image = Image.open(impath)
            depthmap: np.ndarray   = imageio.v2.imread(depthpath)
            assert depthmap.shape == (480, 640), f"Depth map shape {depthmap.shape} does not match expected (480, 640)"
            rgb_image: Image.Image = resize_image(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

            depthmap[depthmap == 65535] = 0
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0
            depthmap[depthmap > 10] = 0
            depthmap[depthmap < 1e-3] = 0

            camera_pose = np.loadtxt(posepath).astype(np.float32)

            # if resolution != (224, 224) or self.rebuttal:
            #     rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
            #         rgb_image, depthmap, intrinsics_, resolution, rng=rng, info=impath
            #     )
            # else:
            #     rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
            #         rgb_image, depthmap, intrinsics_, (512, 384), rng=rng, info=impath
            #     )
            #     W, H = rgb_image.size
            #     cx = W // 2
            #     cy = H // 2
            #     l, t = cx - 112, cy - 112
            #     r, b = cx + 112, cy + 112
            #     crop_bbox = (l, t, r, b)
            #     rgb_image, depthmap, intrinsics = cropping.crop_image_depthmap(
            #         rgb_image, depthmap, intrinsics, crop_bbox
            #     )

            rgb_image, depthmap, intrinsics = resize_image_depth_and_intrinsic(
                image=rgb_image,
                depth_map=depthmap,
                intrinsic=intrinsics_.copy(),
                output_width=self.load_img_size, # finally width = 518, height = 388
                patch_size=self.patch_size,
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset="7scenes",
                    label=osp.join(scene_id, im_idx),
                    instance=impath,
                )
            )
        return views


class DTU(BaseStereoViewDataset):
    def __init__(
        self,
        num_seq=49,
        num_frames=5,
        min_thresh=10,
        max_thresh=30,
        test_id=None,
        full_video=False,
        sample_pairs=False,
        kf_every=1,
        load_img_size=518,
        *args,
        ROOT,
        **kwargs,
    ):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)

        self.num_seq = num_seq
        self.num_frames = num_frames
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every
        self.sample_pairs = sample_pairs


        self.load_img_size = load_img_size

        # load all scenes
        self.load_all_scenes(ROOT)

    def __len__(self):
        return len(self.scene_list) * self.num_seq

    def load_all_scenes(self, base_dir):

        if self.test_id is None:
            self.scene_list = os.listdir(osp.join(base_dir))
            print(f"Found {len(self.scene_list)} scenes in split {self.split}")

        else:
            if isinstance(self.test_id, list):
                self.scene_list = self.test_id
            else:
                self.scene_list = [self.test_id]

            print(f"Test_id: {self.test_id}")

    def load_cam_mvsnet(self, file, interval_scale=1):
        """read camera txt file"""
        cam = np.zeros((2, 4, 4))
        words = file.read().split()
        # read extrinsic
        for i in range(0, 4):
            for j in range(0, 4):
                extrinsic_index = 4 * i + j + 1
                cam[0][i][j] = words[extrinsic_index]

        # read intrinsic
        for i in range(0, 3):
            for j in range(0, 3):
                intrinsic_index = 3 * i + j + 18
                cam[1][i][j] = words[intrinsic_index]

        if len(words) == 29:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = 192
            cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
        elif len(words) == 30:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = words[29]
            cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
        elif len(words) == 31:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = words[29]
            cam[1][3][3] = words[30]
        else:
            cam[1][3][0] = 0
            cam[1][3][1] = 0
            cam[1][3][2] = 0
            cam[1][3][3] = 0

        extrinsic = cam[0].astype(np.float32)
        intrinsic = cam[1].astype(np.float32)

        return intrinsic, extrinsic

    def _get_views(self, idx, resolution, rng):
        scene_id = self.scene_list[idx // self.num_seq]
        seq_id = idx % self.num_seq

        print("Scene ID:", scene_id)

        image_path = osp.join(self.ROOT, scene_id, "images")
        depth_path = osp.join(self.ROOT, scene_id, "depths")
        mask_path = osp.join(self.ROOT, scene_id, "binary_masks")
        cam_path = osp.join(self.ROOT, scene_id, "cams")
        pairs_path = osp.join(self.ROOT, scene_id, "pair.txt")

        if not self.full_video:
            img_idxs = self.sample_pairs(pairs_path, seq_id)
        else:
            img_idxs = sorted(os.listdir(image_path))
            img_idxs = img_idxs[:: self.kf_every]

        views = []
        imgs_idxs = deque(img_idxs)

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.pop()
            impath = osp.join(image_path, im_idx)
            depthpath = osp.join(depth_path, im_idx.replace(".jpg", ".npy"))
            campath = osp.join(cam_path, im_idx.replace(".jpg", "_cam.txt"))
            maskpath = osp.join(mask_path, im_idx.replace(".jpg", ".png"))

            rgb_image = Image.open(impath)
            depthmap = np.load(depthpath)
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0)

            mask = imread_cv2(maskpath, cv2.IMREAD_UNCHANGED) / 255.0
            mask = mask.astype(np.float32)

            mask[mask > 0.5] = 1.0
            mask[mask < 0.5] = 0.0

            mask = cv2.resize(
                mask,
                (depthmap.shape[1], depthmap.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            kernel = np.ones((10, 10), np.uint8)  # Define the erosion kernel
            mask = cv2.erode(mask, kernel, iterations=1)
            depthmap = depthmap * mask

            cur_intrinsics, camera_pose = self.load_cam_mvsnet(open(campath, "r"))
            intrinsics = cur_intrinsics[:3, :3]
            camera_pose = np.linalg.inv(camera_pose)

            print(f"rgb_image shape: {rgb_image.size}, depthmap shape: {depthmap.shape}, intrinsics shape: {intrinsics.shape}")

            
            rgb_image, depthmap, intrinsics = resize_image_depth_and_intrinsic(
                image=rgb_image,
                depth_map=depthmap,
                intrinsic=intrinsics.copy(),
                output_width=self.load_img_size, # finally width = 518, height = 388
            )

            
            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset="dtu",
                    label=osp.join(scene_id, im_idx),
                    instance=impath,
                )
            )

        return views


class NRGBD(BaseStereoViewDataset):
    def __init__(
        self,
        num_seq=1,
        num_frames=5,
        min_thresh=10,
        max_thresh=100,
        test_id=None,
        full_video=False,
        tuple_list=None,
        seq_id=None,
        rebuttal=False,
        shuffle_seed=-1,
        kf_every=1,
        load_img_size=518,
        patch_size=14,
        *args,
        ROOT,
        **kwargs,
    ):

        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.num_frames = num_frames
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every
        self.seq_id = seq_id
        self.rebuttal = rebuttal
        self.shuffle_seed = shuffle_seed

        self.load_img_size = load_img_size
        self.patch_size = patch_size

        # load all scenes
        self.load_all_tuples(tuple_list)
        self.load_all_scenes(ROOT)

    def __len__(self):
        if self.tuple_list is not None:
            return len(self.tuple_list)
        return len(self.scene_list) * self.num_seq

    def load_all_tuples(self, tuple_list):
        if tuple_list is not None:
            self.tuple_list = tuple_list
            # with open(tuple_path) as f:
            #     self.tuple_list = f.read().splitlines()

        else:
            self.tuple_list = None

    def load_all_scenes(self, base_dir):

        scenes = [
            d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
        ]

        if self.test_id is not None:
            self.scene_list = [self.test_id]

        else:
            self.scene_list = scenes

        print(f"Found {len(self.scene_list)} sequences in split {self.split}")

    def load_poses(self, path):
        file = open(path, "r")
        lines = file.readlines()
        file.close()
        poses = []
        valid = []
        lines_per_matrix = 4
        for i in range(0, len(lines), lines_per_matrix):
            if "nan" in lines[i]:
                valid.append(False)
                poses.append(np.eye(4, 4, dtype=np.float32).tolist())
            else:
                valid.append(True)
                pose_floats = [
                    [float(x) for x in line.split()]
                    for line in lines[i : i + lines_per_matrix]
                ]
                poses.append(pose_floats)

        return np.array(poses, dtype=np.float32), valid

    def _get_views(self, idx, resolution, rng):

        if self.tuple_list is not None:
            line = self.tuple_list[idx].split(" ")
            scene_id = line[0]
            img_idxs = line[1:]

        else:
            scene_id = self.scene_list[idx // self.num_seq]

            num_files = len(os.listdir(os.path.join(self.ROOT, scene_id, "images")))
            img_idxs = [f"{i}" for i in range(num_files)]
            img_idxs = img_idxs[:: min(self.kf_every, len(img_idxs) // 2)]

        fx, fy, cx, cy = 554.2562584220408, 554.2562584220408, 320, 240
        intrinsics_ = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        posepath = osp.join(self.ROOT, scene_id, f"poses.txt")
        camera_poses, valids = self.load_poses(posepath)

        imgs_idxs = deque(img_idxs)
        if self.shuffle_seed >= 0:
            imgs_idxs = shuffle_deque(imgs_idxs)
        views = []

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()

            impath = osp.join(self.ROOT, scene_id, "images", f"img{im_idx}.png")
            depthpath = osp.join(self.ROOT, scene_id, "depth", f"depth{im_idx}.png")

            # rgb_image = imread_cv2(impath)
            # depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            rgb_image: Image.Image = Image.open(impath)
            depthmap: np.ndarray   = imageio.v2.imread(depthpath)
            assert depthmap.shape == (480, 640), f"Depth map shape {depthmap.shape} does not match expected (480, 640)"
            rgb_image: Image.Image = resize_image(rgb_image, (depthmap.shape[1], depthmap.shape[0]))


            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0
            depthmap[depthmap > 10] = 0
            depthmap[depthmap < 1e-3] = 0

            # rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

            camera_pose = camera_poses[int(im_idx)]
            # gl to cv
            camera_pose[:, 1:3] *= -1.0

            # if resolution != (224, 224) or self.rebuttal:
            #     rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
            #         rgb_image, depthmap, intrinsics_, resolution, rng=rng, info=impath
            #     )
            # else:
            #     rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
            #         rgb_image, depthmap, intrinsics_, (512, 384), rng=rng, info=impath
            #     )
            #     W, H = rgb_image.size
            #     cx = W // 2
            #     cy = H // 2
            #     l, t = cx - 112, cy - 112
            #     r, b = cx + 112, cy + 112
            #     crop_bbox = (l, t, r, b)
            #     rgb_image, depthmap, intrinsics = cropping.crop_image_depthmap(
            #         rgb_image, depthmap, intrinsics, crop_bbox
            #     )

            rgb_image, depthmap, intrinsics = resize_image_depth_and_intrinsic(
                image=rgb_image,
                depth_map=depthmap,
                intrinsic=intrinsics_.copy(),
                output_width=self.load_img_size, # finally width = 518, height = 388
                patch_size=self.patch_size,
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset="nrgbd",
                    label=osp.join(scene_id, im_idx),
                    instance=impath,
                )
            )

        return views


class ETH3D(BaseStereoViewDataset):
    def __init__(
        self,
        ROOT: str,
        load_img_size: int = 518,
        kf_every: int = 5,
        *args,
        **kwargs,
    ):
        
        self.ROOT = ROOT
        if ROOT == None:
            raise NotImplementedError
        print(f"ETH3D_DIR is {ROOT}")


        super().__init__(*args, **kwargs)

        self.sequence_list = [seq for seq in os.listdir(ROOT) if os.path.isdir(osp.join(ROOT, seq))]
        self.sequence_list = sorted(self.sequence_list)

        self.metadata = {}
        for seq in self.sequence_list:
            seq_image_root = osp.join(ROOT, seq, 'images', 'custom_undistorted')
            image_list = [imgname for imgname in os.listdir(seq_image_root) if imgname.endswith('.JPG')]
            image_list = sorted(image_list)

            self.metadata[seq] = image_list

        self.load_img_size = load_img_size
        self.kf_every = kf_every
        print(f"[ETH3D] Data size: {len(self)}")

    def __len__(self):
        return len(self.sequence_list)

    def get_seq_framenum(self, index: Optional[int] = None, sequence_name: Optional[str] = None):
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name = self.sequence_list[index]
        return len(self.metadata[sequence_name])

    # def __getitem__(self, idx_N):
    #     """Fetch item by index and a dynamic variable n_per_seq."""

    #     # Different from most pytorch datasets,
    #     # here we not only get index, but also a dynamic variable n_per_seq
    #     # supported by DynamicBatchSampler

    #     index, n_per_seq = idx_N
    #     sequence_name = self.sequence_list[index]
    #     metadata = self.metadata[sequence_name]
    #     ids = np.random.choice(len(metadata), n_per_seq, replace=False)
    #     return self.get_data(index=index, ids=ids)

    # def _get_views(self, idx, resolution, rng):
    #     sequence_name = self.sequence_list[idx]
    #     metadata = self.metadata[sequence_name]
    #     # ids = np.random.choice(len(metadata), self.num_frames, replace=False)
    #     ids = np.arange(len(metadata)).tolist()[:: self.kf_every]
    #     return self.get_data(index=idx, ids=ids)

    def _get_views(self, idx, resolution, rng):

        sequence_name = self.sequence_list[idx]
        metadata = self.metadata[sequence_name]
        # ids = np.random.choice(len(metadata), self.num_frames, replace=False)
        ids = np.arange(len(metadata)).tolist()[:: self.kf_every]

        if sequence_name is None:
            if idx is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name: str = self.sequence_list[idx]
        image_list: list = self.metadata[sequence_name]
        seq_len: int     = len(image_list)

        if ids is None:
            ids = np.arange(seq_len).tolist()
        elif isinstance(ids, np.ndarray):
            assert ids.ndim == 1, f"ids should be a 1D array, but got {ids.ndim}D"
            ids = ids.tolist()


        views = []

        for id_index, id in enumerate(ids):
            img_name = image_list[id]
            impath = os.path.join(self.ROOT, sequence_name, 'images', 'custom_undistorted', img_name)
            depthpath = os.path.join(self.ROOT, sequence_name, 'ground_truth_depth', 'custom_undistorted', img_name)
            cam_path = os.path.join(self.ROOT, sequence_name,  'custom_undistorted_cam', img_name.replace('JPG', 'npz'))

            cam = np.load(cam_path)
            intrinsic = cam['intrinsics'].astype(np.float32)
            extrinsic = cam['extrinsics'].astype(np.float32)
            extrinsic = se3_inverse(extrinsic) # w2c to c2w

            # load image and depth
            rgb_image: Image.Image = Image.open(impath)
            width, height          = rgb_image.size
            depthmap: np.ndarray   = np.fromfile(depthpath, dtype=np.float32).reshape(height, width)
            depthmap[~np.isfinite(depthmap)] = -1

            rgb_image, depthmap, intrinsic = resize_image_depth_and_intrinsic(
                image=rgb_image,
                depth_map=depthmap,
                intrinsic=intrinsic.copy(),
                output_width=self.load_img_size, # finally width = 518, height = 388
            )

            # image_paths[id_index] = impath
            # images[id_index]      = to_tensor(rgb_image)
            # depths[id_index]      = depthmap
            # intrinsics[id_index]  = intrinsic
            # extrinsics[id_index]  = extrinsic[:3, :]

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=extrinsic,
                    camera_intrinsics=intrinsic,
                    dataset="eth3d",
                    label=osp.join(sequence_name, f"{id:06d}"),
                    instance=impath,
                )
            )

        return views


