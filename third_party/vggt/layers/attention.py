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

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None, attention_mask=None, return_attn_weight=False) -> Tensor:
        if return_attn_weight:
            return self.forward_return_attn_weight(x, pos=pos, attention_mask=attention_mask)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)


        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, attn_mask=attention_mask)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attention_mask is not None:
                if attention_mask.dtype == torch.bool:
                    attention_mask = attention_mask.masked_fill(attention_mask.logical_not(), float("-inf"))

                attn = attn + attention_mask

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_return_attn_weight(self, x: Tensor, pos=None, attention_mask=None) -> Tensor:
            
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # length_of_k = k.shape[-2]
        # eye_matrix_of_k = torch.eye(length_of_k).to(k).expand(k.shape[0], k.shape[1], -1, -1)
        # concat_v = torch.cat([v, eye_matrix_of_k], dim=-1)

        # if self.fused_attn:
        #     x = F.scaled_dot_product_attention(q, k, concat_v, dropout_p=self.attn_drop.p if self.training else 0.0, attn_mask=attention_mask)
        # else:
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
            print(f"Processing chunk {i} of {N}")
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


        print("attn_weight", attn_weight.shape)
        # attn_weight = torch.mean(attn_weight, dim=1) # (B, seq_len, seq_len)
        return x, attn_weight

        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        attn_weight = attn
        # x, attn_weight = x.split([v.shape[-1], length_of_k], dim=-1)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)


        return x, attn_weight

class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None, attention_mask=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
