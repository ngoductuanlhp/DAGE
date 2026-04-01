# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

from torch import Tensor
from torch import nn
import torch

from torch.nn.functional import scaled_dot_product_attention

# if torch.__version__ >= "2.2.0":
#     from torch.nn.attention import SDPBackend
# else:
# from torch.nn.functional import scaled_dot_product_attention as sdpa_kernel
from torch._C import _SDPBackend as SDPBackend
from torch.backends.cuda import sdp_kernel as sdpa_kernel


from ..merging.merge import (
    token_merge_bipartite2d,
)



XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        # warnings.warn("xFormers is available (Attention)")
    else:
        # warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    # warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        # q, k, v = unbind(qkv, 2)
        q, k, v = [qkv[:,:,i] for i in range(3)]

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


    
class FlashAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)

        # q, k, v = unbind(qkv, 2)
        q, k, v = [qkv[:,:,i] for i in range(3)]

        if q.dtype == torch.bfloat16:
            # with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            with sdpa_kernel(enable_flash=True):
                x = scaled_dot_product_attention(q, k, v)
        else:
            # with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
            with sdpa_kernel(enable_math=True, enable_mem_efficient=True):
                x = scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


"""
Following is written by GPT-4o
"""
class CrossAttentionRope(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
        rope=None,
        kv_in_dim: int = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # Separate projection layers for query, key, and value
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias) if kv_in_dim is None else nn.Linear(kv_in_dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias) if kv_in_dim is None else nn.Linear(kv_in_dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = rope

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_bias=None, qpos=None, kpos=None) -> Tensor:
        """
        Args:
            query: Tensor of shape (B, N, C), input query
            key: Tensor of shape (B, M, C), input key
            value: Tensor of shape (B, M, C), input value
            attn_bias: Optional tensor for attention bias
        Returns:
            Tensor of shape (B, N, C), output of cross-attention
        """
        B, N, C = query.shape
        _, M, _ = key.shape

        # Project query, key, and value
        q = self.q_proj(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)

        # Scale query
        q = q * self.scale

        # Compute attention scores
        attn = q @ k.transpose(-2, -1)  # (B, num_heads, N, M)
        if attn_bias is not None:
            attn = attn + attn_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Compute attention output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)

        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffCrossAttentionRope(CrossAttentionRope):
    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_bias=None, qpos=None, kpos=None) -> Tensor:
        """
        Args:
            query: Tensor of shape (B, N, C), input query
            key: Tensor of shape (B, M, C), input key
            value: Tensor of shape (B, M, C), input value
            attn_bias: Optional tensor for attention bias
        Returns:
            Tensor of shape (B, N, C), output of cross-attention
        """
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(query, key, value, attn_bias)

        B, N, C = query.shape
        _, M, _ = key.shape

        # Project query, key, and value
        q = self.q_proj(query).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(key).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(value).reshape(B, M, self.num_heads, C // self.num_heads)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # Compute memory-efficient attention
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape(B, N, C)

        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FlashCrossAttentionRope(CrossAttentionRope):
    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_bias=None, qpos=None, kpos=None) -> Tensor:
        B, N, C = query.shape
        _, M, _ = key.shape

        # 1. 投射 query, key, value 并调整维度为 (B, num_heads, Seq_Len, head_dim)
        q = self.q_proj(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)
        
        dropout_p = self.attn_drop.p if self.training else 0.0
        
        if q.dtype == torch.bfloat16:
            # with sdpa_kernel(SDPBackend.FLASH_ATTENTION):

            with sdpa_kernel(enable_flash=True):
                x = scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_bias, dropout_p=dropout_p
                )
        else:
            # with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
            with sdpa_kernel(enable_math=True, enable_mem_efficient=True):
                x = scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_bias, dropout_p=dropout_p
                )

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AttentionRope(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
        rope=None,
        t_rope=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()

        self.rope = rope
        self.t_rope = t_rope

    def forward(self, x: Tensor, attn_bias=None, xpos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)
        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttentionRope(AttentionRope):
    def forward(self, x: Tensor, attn_bias=None, xpos=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        
        qkv = qkv.transpose(1, 3)
        # q, k, v = unbind(qkv, 2)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        # score_matrix = (q.permute(0, 2, 1, 3) * self.scale @ k.permute(0, 2, 1, 3).transpose(-2, -1)).sum(dim=1).reshape(frame_num, 261, frame_num, 261).mean(dim=[1, 3]).sum(1)         # for frame attention matrix
        # global_valid_id = torch.where(score_matrix > 0)
        # score_matrix = (q.permute(0, 2, 1, 3) * self.scale @ k.permute(0, 2, 1, 3).transpose(-2, -1)).sum(dim=1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    
class FlashAttentionRope(AttentionRope):
    def forward(self, x: Tensor, attn_bias=None, xpos=None, return_attn_weight=False) -> Tensor:

        if return_attn_weight:

            return self.forward_return_attn_weight(x, xpos=xpos)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)

        # q, k, v = unbind(qkv, 2)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        if q.dtype == torch.bfloat16:
            # with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            with sdpa_kernel(enable_flash=True):
                x = scaled_dot_product_attention(q, k, v)
        else:
            # with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
            with sdpa_kernel(enable_math=True, enable_mem_efficient=True):
                x = scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_return_attn_weight(self, x: Tensor, xpos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)

        # q, k, v = unbind(qkv, 2)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype) # k has shape of (B, head, seq_len, C)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        q = q * self.scale
        k = k.transpose(-2, -1)
        original_device = q.device
        original_dtype = q.dtype

        q = q.cpu().float()
        k = k.cpu().float()
        v = v.cpu().float()

        max_chunk_size = 4096
        x_list = []
        attn_weight_list = []
        for i in range(0, N, max_chunk_size):
            # print(f"Processing chunk {i} of {N}")
            j = min(i + max_chunk_size, N)
            attn_chunk = q[:,:, i:j, :] @ k
            attn_chunk = attn_chunk.softmax(dim=-1)
            x_chunk = attn_chunk @ v
            x_list.append(x_chunk)
            attn_weight_list.append(attn_chunk.mean(dim=1))

            del attn_chunk
            del x_chunk

        x = torch.cat(x_list, dim=2).to(original_device).to(original_dtype)
        attn_weight = torch.cat(attn_weight_list, dim=1)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)


        # print("attn_weight", attn_weight.shape)
        # attn_weight = torch.mean(attn_weight, dim=1) # (B, seq_len, seq_len)
        return x, attn_weight
class FlashAttentionRopeMasked(AttentionRope):
    """
    Masked version of FlashAttentionRope that supports sparse attention via attention mask.
    """
    def forward(self, x: Tensor, attn_bias=None, xpos=None, attn_mask=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)

        # q, k, v = unbind(qkv, 2)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        # Process attention mask if provided
        # attn_mask expected shape: [B, N, N] (batch, seq_len, seq_len)
        # Need to convert to shape compatible with multi-head attention: [B, num_heads, N, N]
        # Convention: True = keep (attend to), False = mask out (ignore)
        if attn_mask is not None:
            if attn_mask.dim() == 3:  # [B, N, N]
                # Expand to [B, num_heads, N, N]
                attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        if q.dtype == torch.bfloat16:
            # with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            with sdpa_kernel(enable_flash=True):
                x = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            # with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
            with sdpa_kernel(enable_math=True, enable_mem_efficient=True):
                x = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        x = x.transpose(1, 2).reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FlashAttentionRopeAttnWeight(AttentionRope):
    def forward(self, x: Tensor, attn_bias=None, xpos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)

        # q, k, v = unbind(qkv, 2)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype) # k has shape of (B, head, seq_len, C)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        q = q * self.scale
        k = k.transpose(-2, -1)
        original_device = q.device
        original_dtype = q.dtype

        q = q.cpu().float()
        k = k.cpu().float()
        v = v.cpu().float()

        max_chunk_size = 4096
        x_list = []
        attn_weight_list = []
        for i in range(0, N, max_chunk_size):
            # print(f"Processing chunk {i} of {N}")
            j = min(i + max_chunk_size, N)
            attn_chunk = q[:,:, i:j, :] @ k
            attn_chunk = attn_chunk.softmax(dim=-1)
            x_chunk = attn_chunk @ v
            x_list.append(x_chunk)
            attn_weight_list.append(attn_chunk.mean(dim=1))

            del attn_chunk
            del x_chunk

        x = torch.cat(x_list, dim=2).to(original_device).to(original_dtype)
        attn_weight = torch.cat(attn_weight_list, dim=1)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)


        # print("attn_weight", attn_weight.shape)
        # attn_weight = torch.mean(attn_weight, dim=1) # (B, seq_len, seq_len)
        return x, attn_weight

        # length_of_k = k.shape[-2]
        # eye_matrix_of_k = torch.eye(length_of_k).to(k).expand(k.shape[0], k.shape[1], -1, -1)
        # concat_v = torch.cat([v, eye_matrix_of_k], dim=-1)


        # # sdpa_out = torch.nn.functional.scaled_dot_product_attention(
        # #     q, k, concat_v, is_causal=False, attn_mask=None
        # # )
        # # sdpa_out, sdpa_weight = sdpa_out.split([v.shape[-1], length_of_k], dim=-1)

        # if q.dtype == torch.bfloat16:
        #     # with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        #     with sdpa_kernel(enable_flash=True):
        #         # x = scaled_dot_product_attention(q, k, v)
        #         x = scaled_dot_product_attention(q, k, concat_v)
        #         x, attn_weight = x.split([v.shape[-1], length_of_k], dim=-1)
        # else:
        #     # with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
        #     with sdpa_kernel(enable_math=True, enable_mem_efficient=True):
        #         # x = scaled_dot_product_attention(q, k, v)
        #         x = scaled_dot_product_attention(q, k, concat_v)
        #         x, attn_weight = x.split([v.shape[-1], length_of_k], dim=-1) # attn_weight of the shape of (B, head, seq_len, seq_len)

        # x = x.transpose(1, 2).reshape([B, N, C])

        # x = self.proj(x)
        # x = self.proj_drop(x)

        # attn_weight = torch.mean(attn_weight, dim=1) # (B, seq_len, seq_len)
        # return x, attn_weight

class FlashMergeAttentionRope(AttentionRope):
    def forward(self, x: Tensor, attn_bias=None, xpos=None, global_merging=None, patch_height=None, patch_width=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)

        # q, k, v = unbind(qkv, 2)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)


        if global_merging is not None:
            generator = torch.Generator(device=x.device)
            generator.manual_seed(33)


            merge_ratio = 0.9
            r = int(x.shape[1] * merge_ratio)

            m, u = token_merge_bipartite2d(
                x,
                patch_width,
                patch_height,
                2,
                2,
                r,
                False,
                generator,
                enable_protection=True,
            )

            m_a, u_a = (m, u)

            B_q, H_q, N_q, D_q = q.shape

            q_merge_in = q.permute(0, 2, 1, 3).reshape(B_q, N_q, H_q * D_q)
            k_merge_in = k.permute(0, 2, 1, 3).reshape(B_q, N_q, H_q * D_q)
            v_merge_in = v.permute(0, 2, 1, 3).reshape(B_q, N_q, H_q * D_q)

            q_out, k_out, v_out = m_a(
                q_merge_in,
                mode="mean",
                extra_tensors=k_merge_in,
                extra_tensors_2=v_merge_in,
            )

            del q_merge_in, k_merge_in, v_merge_in

            N_m = q_out.shape[1]
            q = q_out.reshape(B_q, N_m, H_q, D_q).permute(0, 2, 1, 3)
            k = k_out.reshape(B_q, N_m, H_q, D_q).permute(0, 2, 1, 3)
            v = v_out.reshape(B_q, N_m, H_q, D_q).permute(0, 2, 1, 3)

            del q_out, k_out, v_out

            N = N_m

        if q.dtype == torch.bfloat16:
            # with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            with sdpa_kernel(enable_flash=True):
                x = scaled_dot_product_attention(q, k, v)
        else:
            # with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
            with sdpa_kernel(enable_math=True, enable_mem_efficient=True):
                x = scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        if global_merging is not None:
            x = u_a(x)

        return x

class FlashAttentionMultiRope(AttentionRope):
    # def __init__(
    #     self,
    #     t_rope=None,
    #     **kwargs
    # ) -> None:
    #     super().__init__(**kwargs)
    #     self.t_rope = t_rope

    def forward(self, x: Tensor, attn_bias=None, xpos=None, tpos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)

        # q, k, v = unbind(qkv, 2)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        if self.t_rope is not None:
            q = self.t_rope(q, tpos)
            k = self.t_rope(k, tpos)

        if q.dtype == torch.bfloat16:
            # with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            with sdpa_kernel(enable_flash=True):
                x = scaled_dot_product_attention(q, k, v)
        else:
            # with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
            with sdpa_kernel(enable_math=True, enable_mem_efficient=True):
                x = scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def get_attn_score(blk_class, x, frame_num, token_length, xpos=None):
    x = blk_class.norm1(x)
    
    B, N, C = x.shape
    qkv = blk_class.attn.qkv(x).reshape(B, N, 3, blk_class.attn.num_heads, C // blk_class.attn.num_heads)
    
    qkv = qkv.transpose(1, 3)
    # q, k, v = unbind(qkv, 2)
    q, k, v = [qkv[:,:,i] for i in range(3)]
    q, k = blk_class.attn.q_norm(q).to(v.dtype), blk_class.attn.k_norm(k).to(v.dtype)

    if blk_class.attn.rope is not None:
        q = blk_class.attn.rope(q, xpos)
        k = blk_class.attn.rope(k, xpos)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    score = (q.permute(0, 2, 1, 3) * blk_class.attn.scale @ k.permute(0, 2, 1, 3).transpose(-2, -1)).sum(dim=1).reshape(B, frame_num, token_length, frame_num, token_length).mean(dim=[2, 4]).sum(-1)

    return score