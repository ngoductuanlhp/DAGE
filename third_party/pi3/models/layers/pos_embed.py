# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------



import numpy as np

import torch

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, n_cls_token=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [n_cls_token+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if n_cls_token>0:
        pos_embed = np.concatenate([np.zeros([n_cls_token, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


#----------------------------------------------------------
# RoPE2D: RoPE implementation in 2D
#----------------------------------------------------------

try:
    from models.curope import cuRoPE2D
    # from third_party.moge.moge.model.dust3r.curope import cuRoPE2D
    RoPE2D = cuRoPE2D
except ImportError:
    print('Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead')

    class RoPE2D(torch.nn.Module):
        
        def __init__(self, freq=100.0, F0=1.0):
            super().__init__()
            self.base = freq 
            self.F0 = F0
            self.cache = {}

        def get_cos_sin(self, D, seq_len, device, dtype):
            if (D,seq_len,device,dtype) not in self.cache:
                inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
                t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
                freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
                freqs = torch.cat((freqs, freqs), dim=-1)
                cos = freqs.cos() # (Seq, Dim)
                sin = freqs.sin()
                self.cache[D,seq_len,device,dtype] = (cos,sin)
            return self.cache[D,seq_len,device,dtype]
            
        @staticmethod
        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
            
        def apply_rope1d(self, tokens, pos1d, cos, sin):
            assert pos1d.ndim==2
            cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
            sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
            return (tokens * cos) + (self.rotate_half(tokens) * sin)
            
        def forward(self, tokens, positions):
            """
            input:
                * tokens: batch_size x nheads x ntokens x dim
                * positions: batch_size x ntokens x 2 (y and x position of each token)
            output:
                * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
            """
            assert tokens.size(3)%2==0, "number of dimensions should be a multiple of two"
            D = tokens.size(3) // 2
            assert positions.ndim==3 and positions.shape[-1] == 2 # Batch, Seq, 2
            cos, sin = self.get_cos_sin(D, int(positions.max())+1, tokens.device, tokens.dtype)
            # split features into two along the feature dimension, and apply rope1d on each half
            y, x = tokens.chunk(2, dim=-1)
            y = self.apply_rope1d(y, positions[:,:,0], cos, sin)
            x = self.apply_rope1d(x, positions[:,:,1], cos, sin)
            tokens = torch.cat((y, x), dim=-1)
            return tokens


    
    
            # #----------------------------------------------------------
            # # RoPE3D: 3D RoPE with interleaved (x, y) low dims and temporal high dims
            # #   - positions: (batch, seq, 3) with channels [y, x, t] to match PositionGetter + Temporal
            # #   - dim allocation (default): x=384, y=384, t=256 (total=1024)
            # #   - first (x,y) occupy lower dims with pair-wise interleaving across frequency pairs
            # #   - temporal (t) occupies higher dims as a contiguous block
            # #----------------------------------------------------------
            # class RoPE3D(torch.nn.Module):
                
            #     def __init__(self, freq=100.0, F0=1.0, dim_x=384, dim_y=384, dim_t=256):
            #         super().__init__()
            #         self.base = freq
            #         self.F0 = F0
            #         self.dim_x = dim_x
            #         self.dim_y = dim_y
            #         self.dim_t = dim_t
            #         self.cache = {}

            #         assert self.dim_x % 2 == 0 and self.dim_y % 2 == 0 and self.dim_t % 2 == 0, \
            #             "RoPE3D requires even feature dimensions for x, y, and t"

            #     def get_cos_sin_pairs(self, D, seq_len, device, dtype):
            #         """Pair-wise cos/sin for RoPE where rotation is applied on (even, odd) channel pairs.
            #         Returns cos/sin of shape (seq_len, D) with values repeated for each pair.
            #         """
            #         key = ("pairs", D, seq_len, device, dtype)
            #         if key not in self.cache:
            #             # frequencies for D/2 pairs; note 2*i/D progression like standard RoPE
            #             inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            #             t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            #             freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)  # (seq_len, D/2)
            #             # repeat each frequency twice to match pair layout [cos, cos, ...]
            #             freqs = torch.repeat_interleave(freqs, repeats=2, dim=-1)  # (seq_len, D)
            #             cos = freqs.cos()
            #             sin = freqs.sin()
            #             self.cache[key] = (cos, sin)
            #         return self.cache[key]

            #     @staticmethod
            #     def rotate_pairs(x):
            #         """Rotate features in pair-wise manner: (even, odd) -> (-odd, even)."""
            #         x_even = x[..., ::2]
            #         x_odd = x[..., 1::2]
            #         x_rot = torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)
            #         return x_rot

            #     def apply_rope1d_pairs(self, tokens, pos1d, cos, sin):
            #         assert pos1d.ndim == 2
            #         cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
            #         sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
            #         return (tokens * cos) + (self.rotate_pairs(tokens) * sin)

            #     def forward(self, tokens, positions):
            #         """
            #         input:
            #             * tokens: (batch, nheads, ntokens, dim_total)
            #             * positions: (batch, ntokens, 3) with order [y, x, t]
            #         behavior:
            #             - lower dims (dim_x + dim_y) are interleaved between x and y at pair level
            #             - higher dims (dim_t) are assigned to temporal t
            #         """
            #         b, h, n, Dtot = tokens.shape
            #         assert positions.ndim == 3 and positions.shape[-1] == 3
            #         assert self.dim_x + self.dim_y + self.dim_t == Dtot, \
            #             f"RoPE3D dims (x+y+t) must sum to token dim. Got {self.dim_x+self.dim_y+self.dim_t} vs {Dtot}"

            #         # positions follow PositionGetter (y, x) + temporal t
            #         y_pos = positions[:, :, 0]
            #         x_pos = positions[:, :, 1]
            #         t_pos = positions[:, :, 2]

            #         # allocate regions
            #         xy_dim = self.dim_x + self.dim_y
            #         t_dim = self.dim_t

            #         tokens_xy = tokens[..., :xy_dim]
            #         tokens_t = tokens[..., xy_dim:xy_dim + t_dim]

            #         # Interleave x and y at pair level across the first xy_dim channels
            #         num_pairs_xy = xy_dim // 2
            #         num_pairs_x = self.dim_x // 2
            #         num_pairs_y = self.dim_y // 2
            #         assert num_pairs_x + num_pairs_y == num_pairs_xy

            #         device = tokens.device
            #         # Pair indices within the xy block [0 .. num_pairs_xy-1]
            #         pair_idx_x = torch.arange(0, num_pairs_xy, 2, device=device)  # even pairs
            #         pair_idx_x = pair_idx_x[:num_pairs_x]  # keep exactly dim_x/2 pairs
            #         pair_idx_y = torch.arange(1, num_pairs_xy, 2, device=device)  # odd pairs
            #         pair_idx_y = pair_idx_y[:num_pairs_y]

            #         # Convert pair indices to channel indices within xy block
            #         def pair_to_channels(pair_idx):
            #             return torch.stack((2 * pair_idx, 2 * pair_idx + 1), dim=-1).view(-1)

            #         ch_idx_x = pair_to_channels(pair_idx_x)
            #         ch_idx_y = pair_to_channels(pair_idx_y)

            #         # Gather slices for x and y
            #         tokens_x = tokens_xy.index_select(dim=-1, index=ch_idx_x)
            #         tokens_y = tokens_xy.index_select(dim=-1, index=ch_idx_y)

            #         # Build cos/sin for each axis
            #         cos_x, sin_x = self.get_cos_sin_pairs(self.dim_x, int(x_pos.max()) + 1, device, tokens.dtype)
            #         cos_y, sin_y = self.get_cos_sin_pairs(self.dim_y, int(y_pos.max()) + 1, device, tokens.dtype)
            #         cos_t, sin_t = self.get_cos_sin_pairs(self.dim_t, int(t_pos.max()) + 1, device, tokens.dtype)

            #         # Apply RoPE for each axis
            #         tokens_x = self.apply_rope1d_pairs(tokens_x, x_pos, cos_x, sin_x)
            #         tokens_y = self.apply_rope1d_pairs(tokens_y, y_pos, cos_y, sin_y)
            #         tokens_t = self.apply_rope1d_pairs(tokens_t, t_pos, cos_t, sin_t)

            #         # Scatter x and y back into the xy block
            #         out_xy = tokens_xy.clone()
            #         out_xy[..., ch_idx_x] = tokens_x
            #         out_xy[..., ch_idx_y] = tokens_y

            #         # Concatenate with temporal block
            #         out = torch.cat((out_xy, tokens_t), dim=-1)
            #         return out


#----------------------------------------------------------
# RoPE1D: RoPE implementation in 1D
#----------------------------------------------------------

class RoPE1D(torch.nn.Module):
    
    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq 
        self.F0 = F0
        self.cache = {}

    def get_cos_sin(self, D, seq_len, device, dtype):
        if (D, seq_len, device, dtype) not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos()  # (Seq, Dim)
            sin = freqs.sin()
            self.cache[D, seq_len, device, dtype] = (cos, sin)
        return self.cache[D, seq_len, device, dtype]
        
    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    def forward(self, tokens, positions):
        """
        input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens x 1 (1D position of each token)
        output:
            * tokens after applying RoPE1D (batch_size x nheads x ntokens x dim)
        """
        assert tokens.size(3) % 2 == 0, "number of dimensions should be a multiple of two"
        D = tokens.size(3)
        assert positions.ndim == 3 and positions.shape[-1] == 1  # Batch, Seq, 1
        positions_1d = positions.squeeze(-1)  # (Batch, Seq)
        cos, sin = self.get_cos_sin(D, int(positions_1d.max()) + 1, tokens.device, tokens.dtype)
        # apply rope1d to the entire feature dimension
        cos = torch.nn.functional.embedding(positions_1d, cos)[:, None, :, :]
        sin = torch.nn.functional.embedding(positions_1d, sin)[:, None, :, :]
        tokens = (tokens * cos) + (self.rotate_half(tokens) * sin)
        return tokens


#----------------------------------------------------------
# RoPE1D with Positional Interpolation
# Based on "Extending Context Window of Large Language Models via Positional Interpolation"
# Paper: https://arxiv.org/abs/2306.15595
#----------------------------------------------------------

class RoPE1DInterpolated(torch.nn.Module):
    """
    RoPE1D with Positional Interpolation for extending context window.
    
    Key idea: Scale down position indices by L/L' factor where:
    - L is the original/pretrained context window length
    - L' is the extended context window length
    
    This allows the model to handle longer sequences by interpolating positions
    to fall within the trained range, rather than extrapolating beyond it.
    
    Example:
        # Original context: 2048, want to extend to 4096
        rope = RoPE1DInterpolated(freq=100.0, original_max_seq_len=2048)
        # positions will be scaled: m -> m * (2048/4096) = m * 0.5
    """
    
    def __init__(self, freq=100.0, F0=1.0, original_max_seq_len=2048):
        """
        Args:
            freq: Base frequency for RoPE (default: 100.0)
            F0: Scaling factor (default: 1.0)
            original_max_seq_len: The original/pretrained context window length (L)
                This is used to compute the interpolation scale when longer sequences are encountered.
        """
        super().__init__()
        self.base = freq 
        self.F0 = F0
        self.original_max_seq_len = original_max_seq_len
        self.cache = {}

    def get_cos_sin(self, D, seq_len, device, dtype, interpolation_scale=1.0):
        """
        Generate cos/sin embeddings with optional positional interpolation.
        
        Args:
            D: Dimension of the embeddings
            seq_len: Maximum sequence length for this batch
            device: Device to create tensors on
            dtype: Data type for tensors
            interpolation_scale: Scale factor (L/L') for positional interpolation
        """
        cache_key = (D, seq_len, device, dtype, interpolation_scale)
        if cache_key not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            # Apply positional interpolation by scaling the position indices
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            if interpolation_scale != 1.0:
                t = t * interpolation_scale  # Scale positions: m -> m * (L/L')
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos()  # (Seq, Dim)
            sin = freqs.sin()
            self.cache[cache_key] = (cos, sin)
        return self.cache[cache_key]
        
    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    def forward(self, tokens, positions):
        """
        Apply RoPE1D with positional interpolation.
        
        Input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens x 1 (1D position of each token)
        Output:
            * tokens after applying RoPE1D with interpolation (batch_size x nheads x ntokens x dim)
        """
        assert tokens.size(3) % 2 == 0, "number of dimensions should be a multiple of two"
        D = tokens.size(3)
        assert positions.ndim == 3 and positions.shape[-1] == 1  # Batch, Seq, 1
        positions_1d = positions.squeeze(-1)  # (Batch, Seq)
        
        # Compute interpolation scale: L/L'
        # If current max position exceeds original training length, apply interpolation
        current_max_pos = int(positions_1d.max()) + 1
        if current_max_pos > self.original_max_seq_len:
            # L/L' - scale down to fit within trained range
            interpolation_scale = self.original_max_seq_len / current_max_pos
        else:
            interpolation_scale = 1.0
        
        cos, sin = self.get_cos_sin(D, current_max_pos, tokens.device, tokens.dtype, interpolation_scale)
        
        # Apply rope1d to the entire feature dimension
        cos = torch.nn.functional.embedding(positions_1d, cos)[:, None, :, :]
        sin = torch.nn.functional.embedding(positions_1d, sin)[:, None, :, :]
        tokens = (tokens * cos) + (self.rotate_half(tokens) * sin)
        return tokens
     

class RoPE2DInterpolated(torch.nn.Module):
    """
    RoPE2D with Positional Interpolation for extending context window in 2D.
    
    Key idea: Scale down position indices by L/L' factor where:
    - L is the original/pretrained spatial resolution (height/width)
    - L' is the extended spatial resolution
    
    This allows the model to handle higher resolution images by interpolating positions
    to fall within the trained range, rather than extrapolating beyond it.
    
    Example:
        # Original trained on 32x32 patches, want to extend to 64x64
        rope = RoPE2DInterpolated(freq=100.0, original_max_h=32, original_max_w=32)
        # positions will be scaled: m_h -> m_h * (32/64) = m_h * 0.5
    """
    
    def __init__(self, freq=100.0, F0=1.0, original_max_h=32, original_max_w=32):
        """
        Args:
            freq: Base frequency for RoPE (default: 100.0)
            F0: Scaling factor (default: 1.0)
            original_max_h: The original/pretrained height resolution (L_h)
            original_max_w: The original/pretrained width resolution (L_w)
        """
        super().__init__()
        self.base = freq 
        self.F0 = F0
        self.original_max_h = original_max_h
        self.original_max_w = original_max_w
        self.cache = {}

    def get_cos_sin(self, D, seq_len, device, dtype, interpolation_scale=1.0):
        """
        Generate cos/sin embeddings with optional positional interpolation.
        
        Args:
            D: Dimension of the embeddings
            seq_len: Maximum sequence length for this dimension
            device: Device to create tensors on
            dtype: Data type for tensors
            interpolation_scale: Scale factor (L/L') for positional interpolation
        """
        cache_key = (D, seq_len, device, dtype, interpolation_scale)
        if cache_key not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            # Apply positional interpolation by scaling the position indices
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            if interpolation_scale != 1.0:
                t = t * interpolation_scale  # Scale positions: m -> m * (L/L')
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos()  # (Seq, Dim)
            sin = freqs.sin()
            self.cache[cache_key] = (cos, sin)
        return self.cache[cache_key]
        
    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    def apply_rope1d(self, tokens, pos1d, cos, sin):
        """Apply 1D RoPE to tokens using precomputed cos/sin."""
        assert pos1d.ndim == 2
        cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
        sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
        return (tokens * cos) + (self.rotate_half(tokens) * sin)
        
    def forward(self, tokens, positions):
        """
        Apply RoPE2D with positional interpolation.
        
        Input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens x 2 (y and x position of each token)
        Output:
            * tokens after applying RoPE2D with interpolation (batch_size x nheads x ntokens x dim)
        """
        assert tokens.size(3) % 2 == 0, "number of dimensions should be a multiple of two"
        D = tokens.size(3) // 2
        assert positions.ndim == 3 and positions.shape[-1] == 2  # Batch, Seq, 2
        
        # Compute interpolation scales for y and x independently
        max_y = int(positions[:, :, 0].max()) + 1
        max_x = int(positions[:, :, 1].max()) + 1
        
        # Calculate interpolation scales: L/L'
        # If current max position exceeds original training resolution, apply interpolation
        if max_y > self.original_max_h:
            interpolation_scale_y = self.original_max_h / max_y
        else:
            interpolation_scale_y = 1.0
            
        if max_x > self.original_max_w:
            interpolation_scale_x = self.original_max_w / max_x
        else:
            interpolation_scale_x = 1.0
        
        # Get cos/sin for both dimensions with their respective interpolation scales
        cos_y, sin_y = self.get_cos_sin(D, max_y, tokens.device, tokens.dtype, interpolation_scale_y)
        cos_x, sin_x = self.get_cos_sin(D, max_x, tokens.device, tokens.dtype, interpolation_scale_x)
        
        # Split features into two along the feature dimension, and apply rope1d on each half
        y, x = tokens.chunk(2, dim=-1)
        y = self.apply_rope1d(y, positions[:, :, 0], cos_y, sin_y)
        x = self.apply_rope1d(x, positions[:, :, 1], cos_x, sin_x)
        
        # Concatenate back
        tokens = torch.cat((y, x), dim=-1)
        return tokens


class RoPE3D(torch.nn.Module):
        
        def __init__(self, freq=100.0, F0=1.0, original_max_seq_len=48):
            super().__init__()
            self.rope2d = RoPE2D(freq=freq, F0=F0)
            self.rope1d = RoPE1DInterpolated(freq=freq, F0=F0, original_max_seq_len=original_max_seq_len)
            
        def forward(self, tokens, positions):
            """
            input:
                * tokens: batch_size x nheads x ntokens x dim
                * positions: batch_size x ntokens x 2 (y and x position of each token)
            output:
                * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
            """

            spatial_channel = 384
            temporal_channel = 256


            spatial_tokens = self.rope2d(tokens[..., 0:spatial_channel*2], positions[..., :2])
            temporal_tokens = self.rope1d(tokens[..., spatial_channel*2:], positions[..., 2:3])

            tokens = torch.cat((spatial_tokens, temporal_tokens), dim=-1)
            return tokens

# patch embedding
class PositionGetter(object):
    """ return positions of patches """

    def __init__(self):
        self.cache_positions = {}
        
    def __call__(self, b, h, w, device):
        if not (h,w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            self.cache_positions[h,w] = torch.cartesian_prod(y, x) # (h, w, 2)
        pos = self.cache_positions[h,w].view(1, h*w, 2).expand(b, -1, 2).clone()
        return pos


class PositionGetterTemporal(object):
    """ return positions for temporal/time dimension """

    def __init__(self):
        self.cache_positions = {}
        
    def __call__(self, b, t, device):
        """
        Args:
            b: batch size
            t: number of time steps
            device: torch device
        
        Returns:
            positions: tensor of shape (b, t, 1) with values [0, 1, 2, ..., t-1]
        """
        if t not in self.cache_positions:
            self.cache_positions[t] = torch.arange(t, device=device).unsqueeze(-1)  # (t, 1)
        pos = self.cache_positions[t].view(1, t, 1).expand(b, -1, -1).clone()
        return pos