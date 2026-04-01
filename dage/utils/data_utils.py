

import os
import json
import re

import numpy as np
import torch
import torch.nn.functional as F

import cv2
from decord import VideoReader, cpu


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def read_video_from_path(data_path):
    if os.path.isdir(data_path):
        rgb_frames = sorted([f for f in os.listdir(data_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))], key=natural_sort_key)

        video = []

        for rgb_frame in rgb_frames:
            img = cv2.imread(os.path.join(data_path, rgb_frame))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video.append(img)

        video = np.stack(video)
        original_height, original_width = video.shape[1:3]
        fps = 1  # For image sequences, fps is not applicable


    elif os.path.isfile(data_path) and data_path.endswith(".mp4"):
        vid = VideoReader(data_path, ctx=cpu(0))
        original_height, original_width = vid.get_batch([0]).shape[1:3]
        frames_idx = list(range(0, len(vid), 1))
        video = vid.get_batch(frames_idx).asnumpy().astype(np.float32)
        fps = vid.get_avg_fps()
    
    else:
        raise ValueError(f"Invalid data path: {data_path}")
    
    return video, original_height, original_width, fps

def read_long_video_from_path(data_path, stride=1, max_frames=500, force_num_frames=None, return_frames_idx=False):

    assert any(data_path.endswith(x) for x in [".mp4", ".MOV"])

    vid = VideoReader(data_path, ctx=cpu(0))
    original_height, original_width = vid.get_batch([0]).shape[1:3]
    if force_num_frames is not None:
        frames_idx = list(np.linspace(0, len(vid)-1, force_num_frames, dtype=int).astype(int))
    else:
        frames_idx = list(range(0, min(max_frames, len(vid)), stride))
    video = vid.get_batch(frames_idx).asnumpy().astype(np.float32)
    fps = vid.get_avg_fps()

    if return_frames_idx:
        return video, original_height, original_width, fps, frames_idx
    return video, original_height, original_width, fps


def read_video_from_folder(data_path, stride=1, max_frames=500, force_num_frames=None, return_frames_idx=False):
    rgb_frames = sorted([f for f in os.listdir(data_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))], key=natural_sort_key)

    if force_num_frames is not None:
        if force_num_frames > len(rgb_frames):
            force_num_frames = len(rgb_frames)
        frames_idx = list(np.linspace(0, len(rgb_frames)-1, force_num_frames, dtype=int).astype(int))
    else:
        frames_idx = list(range(0, min(max_frames, len(rgb_frames)), stride))

    video = []

    for idx in frames_idx:
        rgb_frame = rgb_frames[idx]
        img = cv2.imread(os.path.join(data_path, rgb_frame))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video.append(img)

    video = np.stack(video)
    original_height, original_width = video.shape[1:3]
    fps = 1  # For image sequences, fps is not applicable

    if return_frames_idx:
        return video, original_height, original_width, fps, frames_idx
    return video, original_height, original_width, fps


def read_image_from_path(data_path):
    img = cv2.imread(data_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    original_height, original_width = img.shape[:2]
    return img, original_height, original_width


def resize_to_max_side(video, max_side, patch_size=14, interpolation=cv2.INTER_LINEAR):
    """
    Resize a video/image sequence so its longer side is capped at `max_side` while
    keeping the aspect ratio and ensuring both output dimensions are divisible by
    `patch_size`.

    Args:
        video: np.ndarray of shape (T, H, W, C)
        max_side: target maximum size for the longer side
        patch_size: enforce output dimensions as multiples of this value
        interpolation: OpenCV interpolation mode

    Returns:
        resized_video: np.ndarray of shape (T, H_out, W_out, C)
        out_height: resized height
        out_width: resized width
    """
    original_height, original_width = video.shape[1:3]
    aspect_ratio = original_width / original_height
    max_side = (max_side // patch_size) * patch_size

    if original_width > original_height:
        out_width = min(max_side, original_width // patch_size * patch_size)
        out_height = int((out_width / aspect_ratio) // patch_size * patch_size)
    else:
        out_height = min(max_side, original_height // patch_size * patch_size)
        out_width = int((out_height * aspect_ratio) // patch_size * patch_size)

    resized_video = np.stack(
        [
            cv2.resize(frame, (out_width, out_height), interpolation=interpolation)
            for frame in video
        ],
        axis=0,
    )
    return resized_video, out_height, out_width


def read_video(data_path, stride=1, max_frames=500, force_num_frames=None, return_frames_idx=False):
    """
    Args:
        return_frames_idx: If True, also return the frame indices used.
    """
    if any(data_path.endswith(x) for x in [".mp4", ".MOV"]):
        return read_long_video_from_path(data_path, stride, max_frames, force_num_frames, return_frames_idx)
    else:
        return read_video_from_folder(data_path, stride, max_frames, force_num_frames, return_frames_idx)