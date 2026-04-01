# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# from vggt.layers import Mlp
# from vggt.layers.block import Block
# from vggt.heads.head_act import activate_pose

from .cotracker_blocks import Mlp
from .vggt_module.block import Block


class ScaleHead(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        # pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        target_dim: int = 1,
        # trans_act: str = "linear",
        # quat_act: str = "linear",
        # fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()

        # if pose_encoding_type == "absT_quaR_FoV":
        #     self.target_dim = 9
        # else:
        #     raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        # self.trans_act = trans_act
        # self.quat_act = quat_act
        # self.fl_act = fl_act
        self.trunk_depth = trunk_depth
        self.target_dim = target_dim

        # Build the trunk using a sequence of transformer blocks.
        self.trunk = nn.Sequential(
            *[
                Block(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values)
                for _ in range(trunk_depth)
            ]
        )

        # Normalizations for camera token and trunk output.
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        # Learnable empty camera pose token.
        self.empty_scale_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_scale = nn.Linear(self.target_dim, dim_in)

        # Module for producing modulation parameters: shift, scale, and a gate.
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # Adaptive layer normalization without affine parameters.
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.scale_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

    def forward(self, scale_tokens: torch.Tensor, num_iterations: int = 4) -> list:
        """
        Forward pass to predict camera parameters.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.

        Returns:
            list: A list of predicted camera encodings (post-activation) from each iteration.
        """
        # Use tokens from the last block for camera prediction.
        # tokens = aggregated_tokens_list[-1]

        # Extract the camera tokens
        # scale_tokens = tokens[:, :, 0]
        scale_tokens = self.token_norm(scale_tokens)

        pred_scale_enc_list = self.trunk_fn(scale_tokens, num_iterations)
        return pred_scale_enc_list

    def trunk_fn(self, scale_tokens: torch.Tensor, num_iterations: int) -> list:
        """
        Iteratively refine camera pose predictions.

        Args:
            pose_tokens (torch.Tensor): Normalized camera tokens with shape [B, 1, C].
            num_iterations (int): Number of refinement iterations.

        Returns:
            list: List of activated camera encodings from each iteration.
        """
        B, S, C = scale_tokens.shape  # S is expected to be 1.
        pred_scale_enc = None
        pred_scale_enc_list = []

        for _ in range(num_iterations):
            # Use a learned empty pose for the first iteration.
            if pred_scale_enc is None:
                module_input = self.embed_scale(self.empty_scale_tokens.expand(B, S, -1))
            else:
                # Detach the previous prediction to avoid backprop through time.
                pred_scale_enc = pred_scale_enc.detach()
                module_input = self.embed_scale(pred_scale_enc)

            # Generate modulation parameters and split them into shift, scale, and gate components.
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

            # Adaptive layer normalization and modulation.
            scale_tokens_modulated = gate_msa * modulate(self.adaln_norm(scale_tokens), shift_msa, scale_msa)
            scale_tokens_modulated = scale_tokens_modulated + scale_tokens

            scale_tokens_modulated = self.trunk(scale_tokens_modulated)
            # Compute the delta update for the pose encoding.
            pred_scale_enc_delta = self.scale_branch(self.trunk_norm(scale_tokens_modulated))

            # breakpoint()

            if pred_scale_enc is None:
                pred_scale_enc = pred_scale_enc_delta
            else:
                pred_scale_enc = pred_scale_enc + pred_scale_enc_delta

            # Apply final activation functions for translation, quaternion, and field-of-view.
            # activated_scale = activate_scale(
            #     pred_scale_enc, trans_act=self.trans_act, quat_act=self.quat_act, fl_act=self.fl_act
            # )
            pred_scale_enc_list.append(pred_scale_enc)

        return pred_scale_enc_list
    

# NOTE add learnable parameters for ref and src tokens
class ScaleHead2(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        # pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        target_dim: int = 1,
        # trans_act: str = "linear",
        # quat_act: str = "linear",
        # fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()

        # if pose_encoding_type == "absT_quaR_FoV":
        #     self.target_dim = 9
        # else:
        #     raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        # self.trans_act = trans_act
        # self.quat_act = quat_act
        # self.fl_act = fl_act
        self.trunk_depth = trunk_depth
        self.target_dim = target_dim

        # Build the trunk using a sequence of transformer blocks.
        self.trunk = nn.Sequential(
            *[
                Block(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values)
                for _ in range(trunk_depth)
            ]
        )

        # Normalizations for camera token and trunk output.
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        # Learnable empty camera pose token.
        # self.empty_scale_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        # self.embed_scale = nn.Linear(self.target_dim, dim_in)

        self.ref_tokens = nn.Parameter(torch.zeros(1, 1, dim_in))
        self.src_tokens = nn.Parameter(torch.zeros(1, 1, dim_in))

        # Module for producing modulation parameters: shift, scale, and a gate.
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # Adaptive layer normalization without affine parameters.
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.scale_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

    def forward(self, scale_tokens: torch.Tensor, num_iterations: int = 4) -> list:
        """
        Forward pass to predict camera parameters.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.

        Returns:
            list: A list of predicted camera encodings (post-activation) from each iteration.
        """
        # Use tokens from the last block for camera prediction.
        # tokens = aggregated_tokens_list[-1]

        # Extract the camera tokens
        # scale_tokens = tokens[:, :, 0]
        scale_tokens = self.token_norm(scale_tokens)

        pred_scale_enc_list = self.trunk_fn(scale_tokens, num_iterations)
        return pred_scale_enc_list

    def trunk_fn(self, scale_tokens: torch.Tensor, num_iterations: int) -> list:
        """
        Iteratively refine camera pose predictions.

        Args:
            pose_tokens (torch.Tensor): Normalized camera tokens with shape [B, 1, C].
            num_iterations (int): Number of refinement iterations.

        Returns:
            list: List of activated camera encodings from each iteration.
        """
        B, S, C = scale_tokens.shape  # S is expected to be 1.
        pred_scale_enc = None
        pred_scale_enc_list = []

        for _ in range(num_iterations):
            # Use a learned empty pose for the first iteration.
            # if pred_scale_enc is None:
            #     module_input = self.embed_scale(self.empty_scale_tokens.expand(B, S, -1))
            # else:
            #     # Detach the previous prediction to avoid backprop through time.
            #     pred_scale_enc = pred_scale_enc.detach()
            #     module_input = self.embed_scale(pred_scale_enc)
            
            module_input = torch.cat([self.ref_tokens.repeat(B, 1, 1), self.src_tokens.repeat(B, S-1, 1)], dim=1) # B S C

            # Generate modulation parameters and split them into shift, scale, and gate components.
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

            # Adaptive layer normalization and modulation.
            scale_tokens_modulated = gate_msa * modulate(self.adaln_norm(scale_tokens), shift_msa, scale_msa)
            scale_tokens_modulated = scale_tokens_modulated + scale_tokens

            scale_tokens_modulated = self.trunk(scale_tokens_modulated)
            
            # ref_tokens_modulated = scale_tokens_modulated[:, :1, :]
            src_tokens_modulated = scale_tokens_modulated[:, 1:, :]

            pred_src_scale_enc = self.scale_branch(self.trunk_norm(src_tokens_modulated))
            pred_scale_enc = torch.cat([torch.zeros_like(pred_src_scale_enc[:, :1]), pred_src_scale_enc], dim=1)

            # # update the ref and src tokens
            # self.ref_tokens = ref_tokens_modulated
            # self.src_tokens = src_tokens_modulated

            # # Compute the delta update for the pose encoding.
            # pred_scale_enc_delta = self.scale_branch(self.trunk_norm(scale_tokens_modulated))

            # # breakpoint()

            # if pred_scale_enc is None:
            #     pred_scale_enc = pred_scale_enc_delta
            # else:
            #     pred_scale_enc = pred_scale_enc + pred_scale_enc_delta

            # Apply final activation functions for translation, quaternion, and field-of-view.
            # activated_scale = activate_scale(
            #     pred_scale_enc, trans_act=self.trans_act, quat_act=self.quat_act, fl_act=self.fl_act
            # )
            pred_scale_enc_list.append(pred_scale_enc)

        return pred_scale_enc_list
    

class ScaleHeadSequential(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        # pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        target_dim: int = 1,
        # trans_act: str = "linear",
        # quat_act: str = "linear",
        # fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()

        # if pose_encoding_type == "absT_quaR_FoV":
        #     self.target_dim = 9
        # else:
        #     raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        # self.trans_act = trans_act
        # self.quat_act = quat_act
        # self.fl_act = fl_act
        self.trunk_depth = trunk_depth
        self.target_dim = target_dim

        # Build the trunk using a sequence of transformer blocks.
        self.trunk = nn.Sequential(
            *[
                Block(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values)
                for _ in range(trunk_depth)
            ]
        )

        # Normalizations for camera token and trunk output.
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        # Learnable empty camera pose token.
        self.empty_scale_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_scale = nn.Linear(self.target_dim, dim_in)

        # Module for producing modulation parameters: shift, scale, and a gate.
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # Adaptive layer normalization without affine parameters.
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.scale_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

        # register tokens
        self.num_register_tokens = 4
        self.register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, dim_in))


        self.register_norm = nn.LayerNorm(dim_in)
        self.register_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=dim_in, drop=0)

    def forward(self, scale_tokens: torch.Tensor, num_iterations: int = 4) -> list:
        """
        Forward pass to predict camera parameters.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.

        Returns:
            list: A list of predicted camera encodings (post-activation) from each iteration.
        """
        # Use tokens from the last block for camera prediction.
        # tokens = aggregated_tokens_list[-1]

        # Extract the camera tokens
        # scale_tokens = tokens[:, :, 0]
        scale_tokens = self.token_norm(scale_tokens)

        pred_scale_enc_list = self.trunk_fn_sequential(scale_tokens, num_iterations)
        return pred_scale_enc_list
    
    def trunk_fn_sequential(self, scale_tokens: torch.Tensor, num_iterations: int) -> list:
        """
        Iteratively refine camera pose predictions.

        Args:
            pose_tokens (torch.Tensor): Normalized camera tokens with shape [B, 1, C].
            num_iterations (int): Number of refinement iterations.

        Returns:
            list: List of activated camera encodings from each iteration.
        """
        B, S, C = scale_tokens.shape  # S is expected to be 1.

        register_tokens = self.register_tokens.repeat(B, 1, 1)

        window_size = 8
        sliding_window_stride = 4
        if S <= window_size:
            num_windows = 1
        else:
            if (S - window_size) % sliding_window_stride == 0:
                num_windows = (S - window_size) // sliding_window_stride + 1
            else:
                num_windows = (S - window_size) // sliding_window_stride + 2

        # win_pred_scale_enc = None
        final_pred_scale_enc = torch.zeros(B, S, self.target_dim, device=scale_tokens.device, dtype=scale_tokens.dtype)
        all_pred_scale_enc = []
        # pred_scale_enc_list = []

        for window_idx in range(num_windows):
            s = window_idx * sliding_window_stride
            e = min(s + window_size, S)
            current_window_size = e - s

            overlap_size = window_size - sliding_window_stride

            win_scale_tokens = scale_tokens[:, s:e, :]

            if window_idx == 0:
                module_input = self.embed_scale(self.empty_scale_tokens.expand(B, current_window_size, -1))
            else:
                last_win_pred_scale_enc = win_pred_scale_enc[:, -overlap_size:, :].detach()
                module_input = self.embed_scale(torch.cat([last_win_pred_scale_enc, self.empty_scale_tokens.expand(B, current_window_size - overlap_size, -1)], dim=1))

            # Generate modulation parameters and split them into shift, scale, and gate components.
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

            # Adaptive layer normalization and modulation.
            scale_tokens_modulated = gate_msa * modulate(self.adaln_norm(win_scale_tokens), shift_msa, scale_msa)
            scale_tokens_modulated = scale_tokens_modulated + win_scale_tokens

            full_tokens = torch.cat([register_tokens, scale_tokens_modulated], dim=1)
            full_tokens = self.trunk(full_tokens)

            register_tokens = full_tokens[:, :self.num_register_tokens, :]
            scale_tokens_modulated = full_tokens[:, self.num_register_tokens:, :]

            register_tokens = self.register_branch(self.register_norm(register_tokens))


            # Compute the delta update for the pose encoding.
            win_pred_scale_enc = self.scale_branch(self.trunk_norm(scale_tokens_modulated))

            final_pred_scale_enc[:, s:e, :] = win_pred_scale_enc
            all_pred_scale_enc.append(win_pred_scale_enc)

        return final_pred_scale_enc, all_pred_scale_enc


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift
