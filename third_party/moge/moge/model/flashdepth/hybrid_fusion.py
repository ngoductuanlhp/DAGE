import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math

from third_party.moge.moge.model.dinov2.layers.attention import CrossAttention
from third_party.vggt.heads.utils import create_uv_grid, position_grid_to_embed

class Block(nn.Module):
    def __init__(self, d_model, num_heads=2, mlp_expand=4, proj_dim_in=1024, in_proj=False, out_proj=False):
        super().__init__()
        
        '''
        in_proj means first layer of the block
        out_proj means last layer of the block; do projection accordingly
        '''

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = CrossAttention(dim=d_model, num_heads=num_heads, zero_init=True)
        self.context_norm = nn.LayerNorm(d_model)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_expand),
            nn.GELU(),
            nn.Linear(d_model * mlp_expand, d_model)
        )

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        nn.init.ones_(self.norm1.weight)
        nn.init.zeros_(self.norm1.bias)
        nn.init.ones_(self.norm2.weight)
        nn.init.zeros_(self.norm2.bias)

        # only using proj_dim_in for context projection, if needed
        if proj_dim_in != d_model:
            self.context_proj = nn.Linear(proj_dim_in, d_model)
        else:
            self.context_proj = nn.Identity()


    def _with_pos_embed(self, x, pos_embed):
        if pos_embed is not None:
            x = x + pos_embed
        return x

    def forward(self, x, context, x_pos_embed = None, context_pos_embed = None):

        residual = x
        x = self.norm1(x)
        context = self.context_norm(context)

        x = self.attention(
            self._with_pos_embed(x, x_pos_embed), 
            self._with_pos_embed(context, context_pos_embed)
        )
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        
        return x

def apply_pos_embed(x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
    """
    Apply positional embedding to tensor x.
    """
    patch_w = x.shape[-1]
    patch_h = x.shape[-2]
    pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
    pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
    pos_embed = pos_embed * ratio
    pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
    return x + pos_embed

def get_pos_embed(x: torch.Tensor, W: int, H: int, ratio: float = 0.1, embed_dim: int = None) -> torch.Tensor:
    """
    Apply positional embedding to tensor x.
    """

    if embed_dim is None:
        embed_dim = x.shape[1]

    patch_w = x.shape[-1]
    patch_h = x.shape[-2]
    pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
    pos_embed = position_grid_to_embed(pos_embed, embed_dim)
    pos_embed = pos_embed * ratio
    pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
    return pos_embed

class HybridFusion(nn.Module):
    def __init__(
        self, 
        d_model, # from DPT-S output which is 64
        num_blocks=4, # number of blocks per DPT-layer
        num_heads=2,
        mlp_expand=2,
        patch_size=14,
        proj_dim_in=256,
        use_pos_embed=False,
        **kwargs
    ):
        super(HybridFusion, self).__init__()
        '''
        pass
        '''        

        self.patch_size = patch_size
        self.use_pos_embed = use_pos_embed

        # self.blocks = nn.ModuleList([
        #     nn.ModuleList([Block(d_model, num_heads=num_heads, mlp_expand=mlp_expand, proj_dim_in=256, in_proj=i==0, out_proj=i==num_blocks-1) for i in range(num_blocks)])
        #     for _ in range( 4-len(layers_to_skip) ) 
        # ])

        self.fusion_blocks = nn.ModuleList(
            [
                Block(d_model, num_heads=num_heads, mlp_expand=mlp_expand, proj_dim_in=proj_dim_in, in_proj=i==0, out_proj=i==num_blocks-1) for i in range(num_blocks)
            ]
        )



    def forward(self, x, context):

        # if path_idx in self.layers_to_skip:
        #     return x

        assert x.ndim ==4 and context.ndim ==4, "x and context must be (B,C,h,w)"
        
        B,C,h,w = x.shape

        if self.use_pos_embed:
            x_h, x_w = x.shape[-2:]

            x_pos_embed = get_pos_embed(x, W=x_w, H=x_h) # b c h w
            context_pos_embed = get_pos_embed(context, W=x_w, H=x_h, embed_dim=x.shape[1]) # b c h w

            x_pos_embed = self.reshape_to_spatial(x_pos_embed, spatial_to_sequence=True)
            context_pos_embed = self.reshape_to_spatial(context_pos_embed, spatial_to_sequence=True)
        else:
            x_pos_embed = None
            context_pos_embed = None

        # TODO: might be worth trying interpolating context to same resolution, then adding positional embedding
        # currently using original resolution and relying on positional embedding from ViT encoders
        # context = F.interpolate(context, size=(h,w), mode='bilinear', align_corners=False)

        x = self.reshape_to_spatial(x, spatial_to_sequence=True)
        context = self.reshape_to_spatial(context, spatial_to_sequence=True)

    
        for idx, block in enumerate(self.fusion_blocks):
            if idx == 0:
                context = block.context_proj(context)
            x = block(
                x, 
                context,
                x_pos_embed,
                context_pos_embed
            )

        x = self.reshape_to_spatial(x, h, w)
        assert x.shape == (B,C,h,w), "x shape must be (B,D,h,w)"

        return x
        
    

    def reshape_to_spatial(self, x, patch_h=None, patch_w=None, spatial_to_sequence=False):
        if spatial_to_sequence:
            B,C,h,w = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, -1, C)
        else:
            B,L,D = x.shape
            x = x.permute(0, 2, 1).reshape(B, D, patch_h, patch_w)
        return x


class HybridFusion2(nn.Module):
    def __init__(
        self, 
        d_model, # from DPT-S output which is 64
        num_blocks=4, # number of blocks per DPT-layer
        num_heads=2,
        mlp_expand=2,
        patch_size=14,
        proj_dim_in=256,
        use_pos_embed=False,
        **kwargs
    ):
        super(HybridFusion2, self).__init__()
        '''
        pass
        '''        

        self.patch_size = patch_size
        self.use_pos_embed = use_pos_embed

        # self.blocks = nn.ModuleList([
        #     nn.ModuleList([Block(d_model, num_heads=num_heads, mlp_expand=mlp_expand, proj_dim_in=256, in_proj=i==0, out_proj=i==num_blocks-1) for i in range(num_blocks)])
        #     for _ in range( 4-len(layers_to_skip) ) 
        # ])

        self.context_norm = nn.LayerNorm(proj_dim_in)
        self.context_proj = nn.Linear(proj_dim_in, d_model)

        self.fusion_blocks = nn.ModuleList(
            [
                Block(d_model, num_heads=num_heads, mlp_expand=mlp_expand, proj_dim_in=d_model, in_proj=i==0, out_proj=i==num_blocks-1) for i in range(num_blocks)
            ]
        )

    def get_pos_embed(self, patch_h, patch_w, H, W, ratio: float = 0.1, embed_dim: int = None, dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
        """
        Apply positional embedding to tensor x.
        """

        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=dtype, device=device)
        pos_embed = position_grid_to_embed(pos_embed, embed_dim)
        pos_embed = pos_embed * ratio

        pos_embed = pos_embed.reshape(1, -1, embed_dim) # 1 (h*w) embed_dim
        return pos_embed

    def forward(self, x, context, x_h, x_w, context_h, context_w):

        # if path_idx in self.layers_to_skip:
        #     return x

        # assert x.ndim ==4 and context.ndim ==4, "x and context must be (B,C,h,w)"
        
        # B,C,h,w = x.shape

        B, N, C1 = x.shape
        B, M, C2 = context.shape

        assert N == x_h*x_w, "x shape must be (B,h*w,C)"
        assert M == context_h*context_w, "context shape must be (B,h*w,C)"
        if self.use_pos_embed:

            x_pos_embed = self.get_pos_embed(x_h, x_w, H=x_h, W=x_w, embed_dim=x.shape[-1], dtype=x.dtype, device=x.device) # 1 N C1
            context_pos_embed = self.get_pos_embed(context_h, context_w, H=x_h, W=x_w, embed_dim=x.shape[-1], dtype=x.dtype, device=x.device) # 1 M C2
        else:
            x_pos_embed = None
            context_pos_embed = None

        # TODO: might be worth trying interpolating context to same resolution, then adding positional embedding
        # currently using original resolution and relying on positional embedding from ViT encoders
        # context = F.interpolate(context, size=(h,w), mode='bilinear', align_corners=False)

        context = self.context_norm(context)
        context = self.context_proj(context)

        for idx, block in enumerate(self.fusion_blocks):
            x = block(
                x, 
                context,
                x_pos_embed,
                context_pos_embed
            )

        return x