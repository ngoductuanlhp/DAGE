from typing import *
from numbers import Number
from functools import partial
from pathlib import Path
import importlib
import warnings
import json
import yaml
import os

from tqdm import tqdm
from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint
import torch.version


from .dpt_block import DPTOutputAdapter  # noqa
from .camera import pose_encoding_to_camera
from .pose_head import PoseDecoder
from .blocks import ConditionModulationBlock
from torch.utils.checkpoint import checkpoint
from .pos_embed import RoPE2D


inf = float("inf")


def reg_dense_depth(xyz, mode, pos_z=False):
    """
    extract 3D points from prediction head output
    """
    mode, vmin, vmax = mode

    no_bounds = (vmin == -float("inf")) and (vmax == float("inf"))
    assert no_bounds

    if mode == "linear":
        if no_bounds:
            return xyz  # [-inf, +inf]
        return xyz.clip(min=vmin, max=vmax)

    if pos_z:
        sign = torch.sign(xyz[..., -1:])
        xyz *= sign
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)

    if mode == "square":
        return xyz * d.square()

    if mode == "exp":
        return xyz * torch.expm1(d)

    raise ValueError(f"bad {mode=}")


def reg_dense_conf(x, mode):
    """
    extract confidence from prediction head output
    """
    mode, vmin, vmax = mode
    if mode == "exp":
        return vmin + x.exp().clip(max=vmax - vmin)
    if mode == "sigmoid":
        return (vmax - vmin) * torch.sigmoid(x) + vmin
    raise ValueError(f"bad {mode=}")


def postprocess(out, depth_mode, conf_mode, pos_z=False):
    """
    extract 3D points/confidence from prediction head output
    """
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,3
    res = dict(pts3d=reg_dense_depth(fmap[:, :, :, 0:3], mode=depth_mode, pos_z=pos_z))

    if conf_mode is not None:
        res["conf"] = reg_dense_conf(fmap[:, :, :, 3], mode=conf_mode)
    return res

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    quaternions = F.normalize(quaternions, p=2, dim=-1)
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def postprocess_pose(out, mode, inverse=False):
    """
    extract pose from prediction head output
    """
    mode, vmin, vmax = mode

    no_bounds = (vmin == -float("inf")) and (vmax == float("inf"))
    assert no_bounds
    trans = out[..., 0:3]
    quats = out[..., 3:7]

    if mode == "linear":
        if no_bounds:
            return trans  # [-inf, +inf]
        return trans.clip(min=vmin, max=vmax)

    d = trans.norm(dim=-1, keepdim=True)

    if mode == "square":
        if inverse:
            scale = d / d.square().clip(min=1e-8)
        else:
            scale = d.square() / d.clip(min=1e-8)

    if mode == "exp":
        if inverse:
            scale = d / torch.expm1(d).clip(min=1e-8)
        else:
            scale = torch.expm1(d) / d.clip(min=1e-8)

    trans = trans * scale
    quats = standardize_quaternion(quats)

    return torch.cat([trans, quats], dim=-1)


class ResidualConvBlock(nn.Module):  
    def __init__(self, in_channels: int, out_channels: int = None, hidden_channels: int = None, padding_mode: str = 'replicate', activation: Literal['relu', 'leaky_relu', 'silu', 'elu'] = 'relu', norm: Literal['group_norm', 'layer_norm'] = 'group_norm'):  
        super(ResidualConvBlock, self).__init__()  
        if out_channels is None:  
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels

        if activation =='relu':
            activation_cls = lambda: nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            activation_cls = lambda: nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation =='silu':
            activation_cls = lambda: nn.SiLU(inplace=True)
        elif activation == 'elu':
            activation_cls = lambda: nn.ELU(inplace=True)
        else:
            raise ValueError(f'Unsupported activation function: {activation}')

        self.layers = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            activation_cls(),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.GroupNorm(hidden_channels // 32 if norm == 'group_norm' else 1, hidden_channels),
            activation_cls(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode)
        )
        
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0) if in_channels != out_channels else nn.Identity()  
  
    def forward(self, x):  
        skip = self.skip_connection(x)  
        x = self.layers(x)
        x = x + skip
        return x  


class MoGeHead(nn.Module):
    def __init__(
        self, 
        num_features: int,
        dim_in: int, 
        dim_out: List[int], 
        dim_proj: int = 512,
        dim_upsample: List[int] = [256, 128, 128],
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: int = 1,
        res_block_norm: Literal['group_norm', 'layer_norm'] = 'group_norm',
        last_res_blocks: int = 0,
        last_conv_channels: int = 32,
        last_conv_size: int = 1,
    ):
        super().__init__()

        
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels=dim_in, out_channels=dim_proj, kernel_size=1, stride=1, padding=0,) for _ in range(num_features)
        ])

        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                self._make_upsampler(in_ch, out_ch),
                *(ResidualConvBlock(out_ch, out_ch, dim_times_res_block_hidden * out_ch, activation="relu", norm=res_block_norm) for _ in range(num_res_blocks))
            ) for in_ch, out_ch in zip([dim_proj] + dim_upsample[:-1], dim_upsample)
        ])

        if isinstance(dim_out, list):
            self.output_block = nn.ModuleList([
                self._make_output_block(
                    dim_upsample[-1], dim_out_, dim_times_res_block_hidden, last_res_blocks, last_conv_channels, last_conv_size, res_block_norm,
                ) for dim_out_ in dim_out
            ])
        else:
            self.output_block = self._make_output_block(
                dim_upsample[-1], dim_out, dim_times_res_block_hidden, last_res_blocks, last_conv_channels, last_conv_size, res_block_norm,
            )

    
    def _make_upsampler(self, in_channels: int, out_channels: int):
        upsampler = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        )
        upsampler[0].weight.data[:] = upsampler[0].weight.data[:, :, :1, :1]
        return upsampler

    def _make_output_block(self, dim_in: int, dim_out: int, dim_times_res_block_hidden: int, last_res_blocks: int, last_conv_channels: int, last_conv_size: int, res_block_norm: Literal['group_norm', 'layer_norm']):
        return nn.Sequential(
            nn.Conv2d(dim_in, last_conv_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            *(ResidualConvBlock(last_conv_channels, last_conv_channels, dim_times_res_block_hidden * last_conv_channels, activation='relu', norm=res_block_norm) for _ in range(last_res_blocks)),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_conv_channels, dim_out, kernel_size=last_conv_size, stride=1, padding=last_conv_size // 2, padding_mode='replicate'),
        )



            
    def forward(self, hidden_states: torch.Tensor, image: torch.Tensor):
        img_h, img_w = image.shape[-2:]
        patch_h, patch_w = img_h // 14, img_w // 14


        # Process the hidden states
        x = torch.stack([
            proj(feat.permute(0, 2, 1).unflatten(2, (patch_h, patch_w)).contiguous())
                for proj, feat in zip(self.projects, hidden_states)
        ], dim=1).sum(dim=1)
        
        # Upsample stage
        # (patch_h, patch_w) -> (patch_h * 2, patch_w * 2) -> (patch_h * 4, patch_w * 4) -> (patch_h * 8, patch_w * 8)
        for i, block in enumerate(self.upsample_blocks):
            for layer in block:
                if x.shape[0] > 32:
                    x = torch.cat([torch.utils.checkpoint.checkpoint(layer, x_chunk, use_reentrant=False) for x_chunk in x.chunk(x.shape[0] // 32)], dim=0)
                else:
                    x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)


        if x.shape[0] > 32:
            x = torch.cat([F.interpolate(x_chunk, (img_h, img_w), mode="bilinear", align_corners=False) for x_chunk in x.chunk(x.shape[0] // 32)], dim=0)
        else:
            x = F.interpolate(x, (img_h, img_w), mode="bilinear", align_corners=False)

        if isinstance(self.output_block, nn.ModuleList):
            output = []
            for block in self.output_block:
                if x.shape[0] > 32:
                    output.append(torch.cat([torch.utils.checkpoint.checkpoint(block, x_chunk, use_reentrant=False) for x_chunk in x.chunk(x.shape[0] // 32)], dim=0))
                else:
                    output.append(torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False))
        else:
            if x.shape[0] > 32:
                output = torch.cat([torch.utils.checkpoint.checkpoint(self.output_block, x_chunk, use_reentrant=False) for x_chunk in x.chunk(x.shape[0] // 32)], dim=0)
            else:
                output = torch.utils.checkpoint.checkpoint(self.output_block, x, use_reentrant=False)
        
        return output


class MoGeHeadPts3dPose(nn.Module):
    def __init__(self, 
            depth_mode=('exp', -inf, inf), 
            conf_mode=None, 
            pose_mode=('exp', -inf, inf), 
            enc_embed_dim=1024, 
            dec_embed_dim=768, 
            dec_num_heads=16, 
            has_conf=True, 
            has_pose=True
        ):
        super(MoGeHeadPts3dPose, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode

        self.has_conf = has_conf
        self.has_pose = has_pose

        # pts_channels = 3 + has_conf
        # rgb_channels = has_rgb * 3
        # feature_dim = 256
        # last_dim = feature_dim // 2
        # ed = enc_embed_dim
        # dd = dec_embed_dim

        # pts_dpt_args = dict(
        #     output_width_ratio=output_width_ratio,
        #     num_channels=pts_channels,
        #     feature_dim=feature_dim,
        #     last_dim=last_dim,
        #     dim_tokens=dim_tokens,
        #     hooks_idx=hooks_idx,
        #     head_type=head_type,
        # )
        # rgb_dpt_args = dict(
        #     output_width_ratio=output_width_ratio,
        #     num_channels=rgb_channels,
        #     feature_dim=feature_dim,
        #     last_dim=last_dim,
        #     dim_tokens=dim_tokens,
        #     hooks_idx=hooks_idx,
        #     head_type=head_type,
        # )
        # if hooks_idx is not None:
        #     pts_dpt_args.update(hooks=hooks_idx)
            # rgb_dpt_args.update(hooks=hooks_idx)

        # self.dpt_self = DPTOutputAdapter_fix(**pts_dpt_args)
        # dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
        # self.dpt_self.init(**dpt_init_args)

        self.final_transform = nn.ModuleList(
            [
                ConditionModulationBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    rope=RoPE2D(freq=100.0),
                )
                for _ in range(2)
            ]
        )

        # total_params_final_transform = sum(p.numel() for p in self.final_transform.parameters())
        # print(f"Total parameters in self.final_transform: {total_params_final_transform:,}")

        # self.dpt_cross = DPTOutputAdapter_fix(**pts_dpt_args)
        # dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
        # self.dpt_cross.init(**dpt_init_args)

        self.dpt_cross = MoGeHead(
            num_features=4,
            dim_in=dec_embed_dim,
            dim_out=3,
            dim_proj= 512,
            dim_upsample=[256, 128, 64],
            dim_times_res_block_hidden=2,
            num_res_blocks=2,
            res_block_norm='group_norm',
            last_res_blocks=0,
            last_conv_channels=32,
            last_conv_size=1
        )

        # total_params_dpt_cross = sum(p.numel() for p in self.dpt_cross.parameters())
        # print(f"Total parameters in self.dpt_cross: {total_params_dpt_cross:,}")

        # if has_rgb:
        #     self.dpt_rgb = DPTOutputAdapter_fix(**rgb_dpt_args)
        #     dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
        #     self.dpt_rgb.init(**dpt_init_args)

        if has_pose:
            in_dim = dec_embed_dim
            self.pose_head = PoseDecoder(hidden_size=in_dim)

    def forward(self, x, pose_token, image, **kwargs):
        if self.has_pose:
            # pose_token = x[-1][:, 0].clone()
            # token = x[-1][:, 1:]
            # token = x[-1]

            # breakpoint()

            with torch.cuda.amp.autocast(enabled=False):
                pose = self.pose_head(pose_token)

            token_cross = x[-1].clone()

            for blk in self.final_transform:
                token_cross = blk(token_cross, pose_token, kwargs.get("pos"))

            # x = x[:-1] + [token]
            x_cross = x[:-1] + [token_cross]

        final_output = {}
        with torch.cuda.amp.autocast(enabled=False):
            # self_out = checkpoint(
            #     self.dpt_self,
            #     x,
            #     image_size=(img_info[0], img_info[1]),
            #     use_reentrant=False,
            # )

            # final_output = postprocess(self_out, self.depth_mode, self.conf_mode)
            # final_output["pts3d_in_self_view"] = final_output.pop("pts3d")
            # final_output["conf_self"] = final_output.pop("conf")

            # if self.has_rgb:
            #     rgb_out = checkpoint(
            #         self.dpt_rgb,
            #         x,
            #         image_size=(img_info[0], img_info[1]),
            #         use_reentrant=False,
            #     )
            #     rgb_output = postprocess_rgb(rgb_out)
            #     final_output.update(rgb_output)

            if self.has_pose:
                pose = postprocess_pose(pose, self.pose_mode)
                final_output["camera_pose"] = pose  # B,7
                # cross_out = checkpoint(
                #     self.dpt_cross,
                #     x_cross,
                #     image,
                #     use_reentrant=False,
                # )
                # breakpoint()
                cross_out = self.dpt_cross(x_cross, image)
                tmp = postprocess(cross_out, self.depth_mode, self.conf_mode)

                final_output["pts3d_in_other_view"] = tmp.pop("pts3d")
                # final_output["conf"] = tmp.pop("conf")
        return final_output


class MoGeHeadPts3d(nn.Module):
    def __init__(self, 
            depth_mode=('exp', -inf, inf), 
            conf_mode=None, 
            enc_embed_dim=1024, 
            dec_embed_dim=768, 
            dec_num_heads=16, 
            has_conf=True, 
        ):
        super(MoGeHeadPts3d, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode

        self.has_conf = has_conf



        self.dpt_cross = MoGeHead(
            num_features=4,
            dim_in=dec_embed_dim,
            dim_out=3,
            dim_proj= 512,
            dim_upsample=[256, 128, 64],
            dim_times_res_block_hidden=2,
            num_res_blocks=2,
            res_block_norm='group_norm',
            last_res_blocks=0,
            last_conv_channels=32,
            last_conv_size=1
        )

    def forward(self, x, image,**kwargs):

        final_output = {}
        with torch.cuda.amp.autocast(enabled=False):

            out = self.dpt_cross(x, image)
            tmp = postprocess(out, self.depth_mode, self.conf_mode)
            final_output["pts3d"] = tmp.pop("pts3d")
            # final_output["conf"] = tmp.pop("conf")

            # if self.has_pose:
            #     pose = postprocess_pose(pose, self.pose_mode)
            #     final_output["camera_pose"] = pose  # B,7
            #     # cross_out = checkpoint(
            #     #     self.dpt_cross,
            #     #     x_cross,
            #     #     image,
            #     #     use_reentrant=False,
            #     # )

            #     cross_out = self.dpt_cross(x_cross, image)
            #     tmp = postprocess(cross_out, self.depth_mode, self.conf_mode)

            #     final_output["pts3d_in_other_view"] = tmp.pop("pts3d")
            #     # final_output["conf"] = tmp.pop("conf")
        return final_output