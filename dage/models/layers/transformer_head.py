from .attention import FlashAttentionRope, FlashCrossAttentionRope
from .block import BlockRope, CrossBlockRope, CrossOnlyBlockRope
from ..dinov2.layers import Mlp
import torch.nn as nn
from functools import partial
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import torch
from einops import rearrange
from .pos_embed import RoPE2D, RoPE2DInterpolated, PositionGetter
   
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4,
        rope=None,
        need_project=True,
        use_checkpoint=False,
    ):
        super().__init__()

        self.projects = nn.Linear(in_dim, dec_embed_dim) if need_project else nn.Identity()
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
                qk_norm=False,
                # attn_class=MemEffAttentionRope,
                attn_class=FlashAttentionRope,
                rope=rope
            ) for _ in range(depth)])

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def zero_init(self):
        nn.init.zeros_(self.linear_out.weight)
        nn.init.zeros_(self.linear_out.bias)

    def forward(self, hidden, xpos=None):
        hidden = self.projects(hidden)
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                hidden = checkpoint(blk, hidden, xpos=xpos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=xpos)
        out = self.linear_out(hidden)
        return out

class LinearPts3d (nn.Module):
    """ 
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(self, patch_size, dec_embed_dim, output_dim=3,):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Linear(dec_embed_dim, (output_dim)*self.patch_size**2)

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape

        # extract 3D points
        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W

        # permute + norm depth
        return feat.permute(0, 2, 3, 1)
    

class ContextTransformerDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4,
        rope=None,
        use_checkpoint=False,
        cross_first=False,
        y_in_dim=None,
        use_pe=False,
        use_rope=False,
        use_rope_interpolated=False,
        norm_input=False,
        use_cls_token=False
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.use_pe = use_pe
        self.norm_input = norm_input
        self.use_cls_token = use_cls_token
        self.use_rope = use_rope
        self.use_rope_interpolated = use_rope_interpolated

        if self.use_pe:
            n_freqs = 16
            self.lr_pe = ImplicitFeaturizer(color_feats=False, n_freqs=n_freqs, learn_bias=True)
            self.hr_pe = ImplicitFeaturizer(color_feats=False, n_freqs=n_freqs, learn_bias=True)

            dec_embed_dim = dec_embed_dim + n_freqs*2*2

            self.projects_x = nn.Linear(in_dim, dec_embed_dim-n_freqs*2*2)
            self.projects_y = nn.Linear(in_dim, dec_embed_dim-n_freqs*2*2) if y_in_dim is None else nn.Linear(y_in_dim, dec_embed_dim-n_freqs*2*2)

        else:

            self.projects_x = nn.Linear(in_dim, dec_embed_dim)
            self.projects_y = nn.Linear(in_dim, dec_embed_dim) if y_in_dim is None else nn.Linear(y_in_dim, dec_embed_dim)

        if self.norm_input:
            self.norm_x = nn.LayerNorm(in_dim, eps=1e-6)
            self.norm_y = nn.LayerNorm(in_dim, eps=1e-6) if y_in_dim is None else nn.LayerNorm(y_in_dim, eps=1e-6)

        if use_rope:
            self.rope = RoPE2D(freq=100.0)
            self.position_getter = PositionGetter()
        else:
            self.rope = None

        if self.use_rope_interpolated:
            self.rope_interpolated = RoPE2DInterpolated(freq=100.0, original_max_h=32, original_max_w=32)
        else:
            self.rope_interpolated = None


        self.blocks = nn.ModuleList([
            CrossBlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
                qk_norm=False,
                attn_class=FlashAttentionRope, 
                cross_attn_class=FlashCrossAttentionRope,
                rope=self.rope,
                rope_interpolated=self.rope_interpolated,
                cross_first=cross_first
            ) for _ in range(depth)])

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)
        if self.use_cls_token:
            self.linear_out_cls = nn.Linear(dec_embed_dim, out_dim)


    def zero_init(self):

        nn.init.zeros_(self.linear_out.weight)
        nn.init.zeros_(self.linear_out.bias)

        if self.use_cls_token:
            nn.init.zeros_(self.linear_out_cls.weight)
            nn.init.zeros_(self.linear_out_cls.bias)

    def forward(self, hidden, context, xpos=None, ypos=None, hidden_shape=None, context_shape=None, context_patch_start_idx=0, hidden_cls_token=None):
        BN = hidden.shape[0]
        if self.norm_input:
            hidden = self.norm_x(hidden)
            context = self.norm_y(context)

        hidden = self.projects_x(hidden)
        context = self.projects_y(context)

        if self.use_pe:
            assert hidden_shape is not None and context_shape is not None, "hidden_shape and context_shape must be provided"
            lr_pe = self.lr_pe(
                rearrange(context[:, context_patch_start_idx:, :], 'b (h w) c -> b c h w', h=context_shape[0], w=context_shape[1]),
            )
            lr_pe = rearrange(lr_pe, 'b c h w -> b (h w) c')
            if context_patch_start_idx > 0:
                lr_pe_special = torch.zeros((context.shape[0], context_patch_start_idx, 64), dtype=context.dtype, device=context.device)
                lr_pe = torch.cat([lr_pe_special, lr_pe], dim=1)

            hr_pe = self.hr_pe(
                rearrange(hidden, 'b (h w) c -> b c h w', h=hidden_shape[0], w=hidden_shape[1]),
            )
            hr_pe = rearrange(hr_pe, 'b c h w -> b (h w) c')

            hidden = torch.cat([hidden, hr_pe], dim=2)
            context = torch.cat([context, lr_pe], dim=2)

        if self.use_rope:
            real_hidden_pos = self.position_getter(BN, hidden_shape[0], hidden_shape[1], hidden.device)
            context_pos = self.position_getter(BN, context_shape[0], context_shape[1], context.device)

            context_pos = rearrange(context_pos, 'b (h w) c -> b c h w', h=context_shape[0], w=context_shape[1]).float()
            interp_context_pos = F.interpolate(context_pos, size=(hidden_shape[0], hidden_shape[1]), mode='nearest') # (BN, 2, hidden_shape[0], hidden_shape[1])
            context_pos = rearrange(context_pos, 'b c h w -> b (h w) c').long()
            interp_context_pos = rearrange(interp_context_pos, 'b c h w -> b (h w) c').long()


            hidden_pos = interp_context_pos
            hidden_pos = hidden_pos + 1
            real_hidden_pos = real_hidden_pos + 1
            context_pos = context_pos + 1

            context_pos_special = torch.zeros((BN, context_patch_start_idx, 2), dtype=context_pos.dtype, device=context.device)
            context_pos = torch.cat([context_pos_special, context_pos], dim=1)

        if self.use_cls_token:
            hidden = torch.cat([hidden_cls_token[:, None], hidden], dim=1)
            
            if self.use_rope:
                hidden_pos_special = torch.zeros((BN, 1, 2), dtype=hidden_pos.dtype, device=hidden.device)
                hidden_pos = torch.cat([hidden_pos_special, hidden_pos], dim=1)
                real_hidden_pos = torch.cat([hidden_pos_special, real_hidden_pos], dim=1)
        
        if self.use_rope:
            for i, blk in enumerate(self.blocks):
                if (
                    self.use_checkpoint
                    and self.training
                    and torch.is_grad_enabled()
                ):
                    hidden = checkpoint(blk, hidden, context, xpos=hidden_pos, ypos=context_pos, real_xpos=real_hidden_pos, use_reentrant=False)
                else:
                    hidden = blk(hidden, context, xpos=hidden_pos, ypos=context_pos, real_xpos=real_hidden_pos)

        else:
            for i, blk in enumerate(self.blocks):
                if (
                    self.use_checkpoint
                    and self.training
                    and torch.is_grad_enabled()
                ):
                    hidden = checkpoint(blk, hidden, context, xpos=None, ypos=None, use_reentrant=False)
                else:
                    hidden = blk(hidden, context, xpos=None, ypos=None)
        
        if self.use_cls_token:
            hidden_class_token, hidden = hidden[:, 0], hidden[:, 1:]
            out = self.linear_out(hidden)
            cls_out = self.linear_out_cls(hidden_class_token)

            return out, cls_out
        else:
            out = self.linear_out(hidden)
            return out


class ClsTransformerDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=3,
        dec_num_heads=8,
        mlp_ratio=4,
        use_checkpoint=False,
        y_in_dim=None,
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.depth = depth


        self.projects_x = nn.Linear(in_dim, dec_embed_dim)
        self.projects_y = nn.Linear(in_dim, dec_embed_dim) if y_in_dim is None else nn.Linear(y_in_dim, dec_embed_dim)



        self.cross_blocks = nn.ModuleList([
            CrossOnlyBlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                act_layer=nn.GELU,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                cross_attn_class=FlashCrossAttentionRope,
                ffn_layer=Mlp,
                init_values=None,
                qk_norm=False,
                rope=None,
            ) for _ in range(depth)])

        self.self_blocks = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                act_layer=nn.GELU,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=None
            ) for _ in range(depth)])

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)


    def zero_init(self):

        nn.init.zeros_(self.linear_out.weight)
        nn.init.zeros_(self.linear_out.bias)


    def forward(self, hidden, context, num_frames):
        '''
        hidden: (BN, 1, D)
        context: (BN, N, D)
        '''
        BN = hidden.shape[0]
        assert BN % num_frames == 0, "BN must be divisible by num_frames"
        batch_size = BN // num_frames

        hidden = self.projects_x(hidden)
        context = self.projects_y(context)

        for i in range(self.depth):
            if (
                self.use_checkpoint
                and self.training
                and torch.is_grad_enabled()
            ):
                
                hidden = checkpoint(self.cross_blocks[i], hidden, context, use_reentrant=False)
                hidden = rearrange(hidden, '(b t) 1 d -> b t d', t=num_frames)
                hidden = checkpoint(self.self_blocks[i], hidden, use_reentrant=False)
                hidden = rearrange(hidden, 'b t d -> (b t) 1 d')

            else:
                hidden = self.cross_blocks[i](hidden, context)
                hidden = rearrange(hidden, '(b t) 1 d -> b t d', t=num_frames)
                hidden = self.self_blocks[i](hidden)
                hidden = rearrange(hidden, 'b t d -> (b t) 1 d')

        out = self.linear_out(hidden)
        return out

# NOTE for ablation only, rope for cross-attention is disabled
class ContextTransformerDecoderAblNoCrossRope(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4,
        rope=None,
        use_checkpoint=False,
        cross_first=False,
        y_in_dim=None,
        use_pe=False,
        use_rope=False,
        use_rope_interpolated=False,
        norm_input=False,
        use_cls_token=False
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.use_pe = use_pe
        self.norm_input = norm_input
        self.use_cls_token = use_cls_token
        self.use_rope = use_rope
        self.use_rope_interpolated = use_rope_interpolated

        if self.use_pe:
            n_freqs = 16
            self.lr_pe = ImplicitFeaturizer(color_feats=False, n_freqs=n_freqs, learn_bias=True)
            self.hr_pe = ImplicitFeaturizer(color_feats=False, n_freqs=n_freqs, learn_bias=True)

            dec_embed_dim = dec_embed_dim + n_freqs*2*2

            self.projects_x = nn.Linear(in_dim, dec_embed_dim-n_freqs*2*2)
            self.projects_y = nn.Linear(in_dim, dec_embed_dim-n_freqs*2*2) if y_in_dim is None else nn.Linear(y_in_dim, dec_embed_dim-n_freqs*2*2)

        else:

            self.projects_x = nn.Linear(in_dim, dec_embed_dim)
            self.projects_y = nn.Linear(in_dim, dec_embed_dim) if y_in_dim is None else nn.Linear(y_in_dim, dec_embed_dim)

        if self.norm_input:
            self.norm_x = nn.LayerNorm(in_dim, eps=1e-6)
            self.norm_y = nn.LayerNorm(in_dim, eps=1e-6) if y_in_dim is None else nn.LayerNorm(y_in_dim, eps=1e-6)

        if use_rope:
            self.rope = RoPE2D(freq=100.0)
            self.position_getter = PositionGetter()
        else:
            self.rope = None

        if self.use_rope_interpolated:
            self.rope_interpolated = RoPE2DInterpolated(freq=100.0, original_max_h=32, original_max_w=32)
        else:
            self.rope_interpolated = None


        self.blocks = nn.ModuleList([
            CrossBlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
                qk_norm=False,
                attn_class=FlashAttentionRope, 
                cross_attn_class=FlashCrossAttentionRope,
                rope=None,
                rope_interpolated=self.rope_interpolated,
                cross_first=cross_first
            ) for _ in range(depth)])

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)
        if self.use_cls_token:
            self.linear_out_cls = nn.Linear(dec_embed_dim, out_dim)


    def zero_init(self):

        nn.init.zeros_(self.linear_out.weight)
        nn.init.zeros_(self.linear_out.bias)

        if self.use_cls_token:
            nn.init.zeros_(self.linear_out_cls.weight)
            nn.init.zeros_(self.linear_out_cls.bias)

    def forward(self, hidden, context, xpos=None, ypos=None, hidden_shape=None, context_shape=None, context_patch_start_idx=0, hidden_cls_token=None):
        BN = hidden.shape[0]
        if self.norm_input:
            hidden = self.norm_x(hidden)
            context = self.norm_y(context)

        hidden = self.projects_x(hidden)
        context = self.projects_y(context)

        if self.use_pe:
            assert hidden_shape is not None and context_shape is not None, "hidden_shape and context_shape must be provided"
            lr_pe = self.lr_pe(
                rearrange(context[:, context_patch_start_idx:, :], 'b (h w) c -> b c h w', h=context_shape[0], w=context_shape[1]),
            )
            lr_pe = rearrange(lr_pe, 'b c h w -> b (h w) c')
            if context_patch_start_idx > 0:
                lr_pe_special = torch.zeros((context.shape[0], context_patch_start_idx, 64), dtype=context.dtype, device=context.device)
                lr_pe = torch.cat([lr_pe_special, lr_pe], dim=1)

            hr_pe = self.hr_pe(
                rearrange(hidden, 'b (h w) c -> b c h w', h=hidden_shape[0], w=hidden_shape[1]),
            )
            hr_pe = rearrange(hr_pe, 'b c h w -> b (h w) c')

            hidden = torch.cat([hidden, hr_pe], dim=2)
            context = torch.cat([context, lr_pe], dim=2)

        if self.use_rope:
            real_hidden_pos = self.position_getter(BN, hidden_shape[0], hidden_shape[1], hidden.device)
            context_pos = self.position_getter(BN, context_shape[0], context_shape[1], context.device)

            context_pos = rearrange(context_pos, 'b (h w) c -> b c h w', h=context_shape[0], w=context_shape[1]).float()
            interp_context_pos = F.interpolate(context_pos, size=(hidden_shape[0], hidden_shape[1]), mode='nearest') # (BN, 2, hidden_shape[0], hidden_shape[1])
            context_pos = rearrange(context_pos, 'b c h w -> b (h w) c').long()
            interp_context_pos = rearrange(interp_context_pos, 'b c h w -> b (h w) c').long()


            hidden_pos = interp_context_pos
            hidden_pos = hidden_pos + 1
            real_hidden_pos = real_hidden_pos + 1
            context_pos = context_pos + 1

            context_pos_special = torch.zeros((BN, context_patch_start_idx, 2), dtype=context_pos.dtype, device=context.device)
            context_pos = torch.cat([context_pos_special, context_pos], dim=1)

        if self.use_cls_token:
            hidden = torch.cat([hidden_cls_token[:, None], hidden], dim=1)
            
            if self.use_rope:
                hidden_pos_special = torch.zeros((BN, 1, 2), dtype=hidden_pos.dtype, device=hidden.device)
                hidden_pos = torch.cat([hidden_pos_special, hidden_pos], dim=1)
                real_hidden_pos = torch.cat([hidden_pos_special, real_hidden_pos], dim=1)
        
        if self.use_rope:
            for i, blk in enumerate(self.blocks):
                if (
                    self.use_checkpoint
                    and self.training
                    and torch.is_grad_enabled()
                ):
                    hidden = checkpoint(blk, hidden, context, xpos=hidden_pos, ypos=context_pos, real_xpos=real_hidden_pos, use_reentrant=False)
                else:
                    hidden = blk(hidden, context, xpos=hidden_pos, ypos=context_pos, real_xpos=real_hidden_pos)

        else:
            for i, blk in enumerate(self.blocks):
                if (
                    self.use_checkpoint
                    and self.training
                    and torch.is_grad_enabled()
                ):
                    hidden = checkpoint(blk, hidden, context, xpos=None, ypos=None, use_reentrant=False)
                else:
                    hidden = blk(hidden, context, xpos=None, ypos=None)
        
        if self.use_cls_token:
            hidden_class_token, hidden = hidden[:, 0], hidden[:, 1:]
            out = self.linear_out(hidden)
            cls_out = self.linear_out_cls(hidden_class_token)

            return out, cls_out
        else:
            out = self.linear_out(hidden)
            return out


class ContextGlobalTransformerDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4,
        rope=None,
        use_checkpoint=False,
        cross_first=False,
        y_in_dim=None,
        use_pe=False,
        norm_input=False
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.use_pe = use_pe
        self.norm_input = norm_input


        if self.use_pe:
            n_freqs = 16
            self.lr_pe = ImplicitFeaturizer(color_feats=False, n_freqs=n_freqs, learn_bias=True)
            self.hr_pe = ImplicitFeaturizer(color_feats=False, n_freqs=n_freqs, learn_bias=True)

            dec_embed_dim = dec_embed_dim + n_freqs*2*2

            self.projects_x = nn.Linear(in_dim, dec_embed_dim-n_freqs*2*2)
            self.projects_y = nn.Linear(in_dim, dec_embed_dim-n_freqs*2*2) if y_in_dim is None else nn.Linear(y_in_dim, dec_embed_dim-n_freqs*2*2)

        else:

            self.projects_x = nn.Linear(in_dim, dec_embed_dim)
            self.projects_y = nn.Linear(in_dim, dec_embed_dim) if y_in_dim is None else nn.Linear(y_in_dim, dec_embed_dim)

        if self.norm_input:
            self.norm_x = nn.LayerNorm(in_dim, eps=1e-6)
            self.norm_y = nn.LayerNorm(in_dim, eps=1e-6) if y_in_dim is None else nn.LayerNorm(y_in_dim, eps=1e-6)


        self.blocks = nn.ModuleList([
            CrossBlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
                qk_norm=False,
                attn_class=FlashAttentionRope, 
                cross_attn_class=FlashCrossAttentionRope,
                rope=rope,
                cross_first=cross_first
            ) for _ in range(depth)])

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

        self.global_blocks = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=rope
            ) for _ in range(6)])
        self.global_linear_out = nn.Linear(dec_embed_dim, out_dim)


    def zero_init(self):

        nn.init.zeros_(self.linear_out.weight)
        nn.init.zeros_(self.linear_out.bias)

        nn.init.zeros_(self.global_linear_out.weight)
        nn.init.zeros_(self.global_linear_out.bias)

    def forward(self, hidden, context, xpos=None, ypos=None, hidden_shape=None, context_shape=None, context_patch_start_idx=0, B=None, N=None):

        if self.norm_input:
            hidden = self.norm_x(hidden)
            context = self.norm_y(context)

        hidden = self.projects_x(hidden)
        context = self.projects_y(context)

        if self.use_pe:
            assert hidden_shape is not None and context_shape is not None, "hidden_shape and context_shape must be provided"
            lr_pe = self.lr_pe(
                rearrange(context[:, context_patch_start_idx:, :], 'b (h w) c -> b c h w', h=context_shape[0], w=context_shape[1]),
            )
            lr_pe = rearrange(lr_pe, 'b c h w -> b (h w) c')
            if context_patch_start_idx > 0:
                lr_pe_special = torch.zeros((context.shape[0], context_patch_start_idx, 64), dtype=context.dtype, device=context.device)
                lr_pe = torch.cat([lr_pe_special, lr_pe], dim=1)

            hr_pe = self.hr_pe(
                rearrange(hidden, 'b (h w) c -> b c h w', h=hidden_shape[0], w=hidden_shape[1]),
            )
            hr_pe = rearrange(hr_pe, 'b c h w -> b (h w) c')

            hidden = torch.cat([hidden, hr_pe], dim=2)
            context = torch.cat([context, lr_pe], dim=2)

        

        for i, blk in enumerate(self.blocks):
            if (
                self.use_checkpoint
                and self.training
                and torch.is_grad_enabled()
            ):
                hidden = checkpoint(blk, hidden, context, xpos=xpos, ypos=ypos, use_reentrant=False)
            else:
                hidden = blk(hidden, context, xpos=xpos, ypos=ypos)

        out = self.linear_out(hidden) # (BN) (H W) C
        

        context_register_tokens = context[:, :context_patch_start_idx]
        global_hidden = torch.cat([context_register_tokens, hidden], dim=1) # (BN) (n_regis + H W) C

        
        num_tokens = global_hidden.shape[1]
        for i, blk in enumerate(self.global_blocks):

            if i % 2 == 0:
                global_hidden = global_hidden.reshape(B, N*num_tokens, -1)
            else:
                global_hidden = global_hidden.reshape(B*N, num_tokens, -1)

            if (
                self.use_checkpoint
                and self.training
                and torch.is_grad_enabled()
            ):
                global_hidden = checkpoint(blk, global_hidden, use_reentrant=False)
            else:
                global_hidden = blk(global_hidden)
        # global_hidden = global_hidden[:, 
        global_hidden = global_hidden.reshape(B*N, num_tokens, -1)[:, context_patch_start_idx:, :]
        global_out = self.global_linear_out(global_hidden)

        out = out + global_out

        return out



class ContextGlobalTransformerDecoder2(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4,
        rope=None,
        use_checkpoint=False,
        cross_first=False,
        y_in_dim=None,
        use_pe=False,
        norm_input=False,
        global_depth=6
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.use_pe = use_pe
        self.norm_input = norm_input


        if self.use_pe:
            n_freqs = 16
            self.lr_pe = ImplicitFeaturizer(color_feats=False, n_freqs=n_freqs, learn_bias=True)
            self.hr_pe = ImplicitFeaturizer(color_feats=False, n_freqs=n_freqs, learn_bias=True)

            dec_embed_dim = dec_embed_dim + n_freqs*2*2

            self.projects_x = nn.Linear(in_dim, dec_embed_dim-n_freqs*2*2)
            self.projects_y = nn.Linear(in_dim, dec_embed_dim-n_freqs*2*2) if y_in_dim is None else nn.Linear(y_in_dim, dec_embed_dim-n_freqs*2*2)

        else:

            self.projects_x = nn.Linear(in_dim, dec_embed_dim)
            self.projects_y = nn.Linear(in_dim, dec_embed_dim) if y_in_dim is None else nn.Linear(y_in_dim, dec_embed_dim)

        if self.norm_input:
            self.norm_x = nn.LayerNorm(in_dim, eps=1e-6)
            self.norm_y = nn.LayerNorm(in_dim, eps=1e-6) if y_in_dim is None else nn.LayerNorm(y_in_dim, eps=1e-6)


        self.blocks = nn.ModuleList([
            CrossBlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
                qk_norm=False,
                attn_class=FlashAttentionRope, 
                cross_attn_class=FlashCrossAttentionRope,
                rope=rope,
                cross_first=cross_first
            ) for _ in range(depth)])

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

        self.global_rope = RoPE2DInterpolated(freq=100.0, original_max_h=80, original_max_w=80)
        self.position_getter = PositionGetter()

        self.global_blocks = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.global_rope
            ) for _ in range(global_depth)])
        # self.global_linear_out = nn.Linear(dec_embed_dim, out_dim)


    def zero_init(self):

        nn.init.zeros_(self.linear_out.weight)
        nn.init.zeros_(self.linear_out.bias)
    
    def decode(self, hidden, hidden_shape, num_tokens, context_patch_start_idx, B, N):
        pos = self.position_getter(B * N, hidden_shape[0], hidden_shape[1], hidden.device)
        pos = pos + 1
        pos_special = torch.zeros(B * N, context_patch_start_idx, 2).to(hidden.device).to(pos.dtype)
        pos = torch.cat([pos_special, pos], dim=1)


        final_output = []
        for i, blk in enumerate(self.global_blocks):

            if i % 2 == 0:
                pos = pos.reshape(B, N*num_tokens, -1)
                hidden = hidden.reshape(B, N*num_tokens, -1)
            else:
                pos = pos.reshape(B*N, num_tokens, -1)
                hidden = hidden.reshape(B*N, num_tokens, -1)

            if (
                self.use_checkpoint
                and self.training
                and torch.is_grad_enabled()
            ):
                hidden = checkpoint(blk, hidden, xpos=pos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=pos)

            # if i+1 in [len(self.global_blocks)-1, len(self.global_blocks)]:
            #     final_output.append(hidden.reshape(B*N, num_tokens, -1))
        final_output = hidden.reshape(B*N, num_tokens, -1)
        final_output = final_output[:, context_patch_start_idx:, :]
        final_output = self.linear_out(final_output)

        return final_output


    def forward(self, hidden, context, xpos=None, ypos=None, hidden_shape=None, context_shape=None, context_patch_start_idx=0, B=None, N=None):

        if self.norm_input:
            hidden = self.norm_x(hidden)
            context = self.norm_y(context)

        hidden = self.projects_x(hidden)
        context = self.projects_y(context)

        if self.use_pe:
            assert hidden_shape is not None and context_shape is not None, "hidden_shape and context_shape must be provided"
            lr_pe = self.lr_pe(
                rearrange(context[:, context_patch_start_idx:, :], 'b (h w) c -> b c h w', h=context_shape[0], w=context_shape[1]),
            )
            lr_pe = rearrange(lr_pe, 'b c h w -> b (h w) c')
            if context_patch_start_idx > 0:
                lr_pe_special = torch.zeros((context.shape[0], context_patch_start_idx, 64), dtype=context.dtype, device=context.device)
                lr_pe = torch.cat([lr_pe_special, lr_pe], dim=1)

            hr_pe = self.hr_pe(
                rearrange(hidden, 'b (h w) c -> b c h w', h=hidden_shape[0], w=hidden_shape[1]),
            )
            hr_pe = rearrange(hr_pe, 'b c h w -> b (h w) c')

            hidden = torch.cat([hidden, hr_pe], dim=2)
            context = torch.cat([context, lr_pe], dim=2)

        context_register_tokens = context[:, :context_patch_start_idx]
        hidden = torch.cat([context_register_tokens, hidden], dim=1) # (BN) (n_regis + H W) C
        num_tokens = hidden.shape[1]

        for i, blk in enumerate(self.blocks):
            if (
                self.use_checkpoint
                and self.training
                and torch.is_grad_enabled()
            ):
                hidden = checkpoint(blk, hidden, context, xpos=xpos, ypos=ypos, use_reentrant=False)
            else:
                hidden = blk(hidden, context, xpos=xpos, ypos=ypos)

        out = self.decode(hidden, hidden_shape, num_tokens, context_patch_start_idx, B, N)

        return out


class CrossOnlyTransformerDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4,
        rope=None,
        use_checkpoint=False,
        y_in_dim=None,
        use_pe=False,
        norm_input=False
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.use_pe = use_pe
        self.norm_input = norm_input


        if self.use_pe:
            n_freqs = 16
            self.lr_pe = ImplicitFeaturizer(color_feats=False, n_freqs=n_freqs, learn_bias=True)
            self.hr_pe = ImplicitFeaturizer(color_feats=False, n_freqs=n_freqs, learn_bias=True)

            dec_embed_dim = dec_embed_dim + n_freqs*2*2

            self.projects_x = nn.Linear(in_dim, dec_embed_dim-n_freqs*2*2)
            self.projects_y = nn.Linear(in_dim, dec_embed_dim-n_freqs*2*2) if y_in_dim is None else nn.Linear(y_in_dim, dec_embed_dim-n_freqs*2*2)

        else:

            self.projects_x = nn.Linear(in_dim, dec_embed_dim)
            self.projects_y = nn.Linear(in_dim, dec_embed_dim) if y_in_dim is None else nn.Linear(y_in_dim, dec_embed_dim)

        if self.norm_input:
            self.norm_x = nn.LayerNorm(in_dim, eps=1e-6)
            self.norm_y = nn.LayerNorm(in_dim, eps=1e-6) if y_in_dim is None else nn.LayerNorm(y_in_dim, eps=1e-6)


        self.blocks = nn.ModuleList([
            # CrossBlockRope(
            #     dim=dec_embed_dim,
            #     num_heads=dec_num_heads,
            #     mlp_ratio=mlp_ratio,
            #     qkv_bias=True,
            #     proj_bias=True,
            #     ffn_bias=True,
            #     norm_layer=partial(nn.LayerNorm, eps=1e-6),
            #     act_layer=nn.GELU,
            #     ffn_layer=Mlp,
            #     init_values=None,
            #     qk_norm=False,
            #     attn_class=FlashAttentionRope, 
            #     cross_attn_class=FlashCrossAttentionRope,
            #     rope=rope,
            #     cross_first=cross_first
            # ) for _ in range(depth)
            CrossOnlyBlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
                qk_norm=False,
                cross_attn_class=FlashCrossAttentionRope,
                rope=rope,
            ) for _ in range(depth)
        ])

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)


    def zero_init(self):

        nn.init.zeros_(self.linear_out.weight)
        nn.init.zeros_(self.linear_out.bias)

    def forward(self, hidden, context, xpos=None, ypos=None, hidden_shape=None, context_shape=None, context_patch_start_idx=0):

        if self.norm_input:
            hidden = self.norm_x(hidden)
            context = self.norm_y(context)

        hidden = self.projects_x(hidden)
        context = self.projects_y(context)

        if self.use_pe:
            assert hidden_shape is not None and context_shape is not None, "hidden_shape and context_shape must be provided"
            lr_pe = self.lr_pe(
                rearrange(context[:, context_patch_start_idx:, :], 'b (h w) c -> b c h w', h=context_shape[0], w=context_shape[1]),
            )
            lr_pe = rearrange(lr_pe, 'b c h w -> b (h w) c')
            if context_patch_start_idx > 0:
                lr_pe_special = torch.zeros((context.shape[0], context_patch_start_idx, 64), dtype=context.dtype, device=context.device)
                lr_pe = torch.cat([lr_pe_special, lr_pe], dim=1)

            hr_pe = self.hr_pe(
                rearrange(hidden, 'b (h w) c -> b c h w', h=hidden_shape[0], w=hidden_shape[1]),
            )
            hr_pe = rearrange(hr_pe, 'b c h w -> b (h w) c')

            hidden = torch.cat([hidden, hr_pe], dim=2)
            context = torch.cat([context, lr_pe], dim=2)

        

        for i, blk in enumerate(self.blocks):
            if (
                self.use_checkpoint
                and self.training
                and torch.is_grad_enabled()
            ):
                hidden = checkpoint(blk, hidden, context, xpos=xpos, ypos=ypos, use_reentrant=False)
            else:
                hidden = blk(hidden, context, xpos=xpos, ypos=ypos)

        out = self.linear_out(hidden)

        return out



class ImplicitFeaturizer(nn.Module):

    def __init__(self, color_feats=True, n_freqs=10, learn_bias=False, time_feats=False, lr_feats=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color_feats = color_feats
        self.time_feats = time_feats
        self.n_freqs = n_freqs
        self.learn_bias = learn_bias

        self.dim_multiplier = 2

        if self.color_feats:
            self.dim_multiplier += 3

        if self.time_feats:
            self.dim_multiplier += 1

        if self.learn_bias:
            self.biases = torch.nn.Parameter(torch.randn(2, self.dim_multiplier, n_freqs).to(torch.float32))
        
        self.low_res_feat = lr_feats

    def forward(self, original_image):
        b, c, h, w = original_image.shape
        grid_h = torch.linspace(-1, 1, h, device=original_image.device)
        grid_w = torch.linspace(-1, 1, w, device=original_image.device)
        feats = torch.cat([t.unsqueeze(0) for t in torch.meshgrid([grid_h, grid_w])]).unsqueeze(0) # torch.Size([1, 2, 224, 224])
        feats = torch.broadcast_to(feats, (b, feats.shape[1], h, w))

        if self.color_feats:
            feat_list = [feats, original_image]
        else:
            feat_list = [feats]

        feats = torch.cat(feat_list, dim=1).unsqueeze(1)
        freqs = torch.exp(torch.linspace(-2, 10, self.n_freqs, device=original_image.device)) \
            .reshape(1, self.n_freqs, 1, 1, 1) # torch.Size([1, 30, 1, 1, 1])
        feats = (feats * freqs) # torch.Size([1, 30, 5, 224, 224])

        if self.learn_bias:
            sin_feats = feats + self.biases[0].reshape(1, self.n_freqs, self.dim_multiplier, 1, 1) # torch.Size([1, 30, 5, 224, 224])
            cos_feats = feats + self.biases[1].reshape(1, self.n_freqs, self.dim_multiplier, 1, 1) # torch.Size([1, 30, 5, 224, 224])
        else:
            sin_feats = feats
            cos_feats = feats

        sin_feats = sin_feats.reshape(b, self.n_freqs * self.dim_multiplier, h, w) # torch.Size([1, 150, 224, 224])
        cos_feats = cos_feats.reshape(b, self.n_freqs * self.dim_multiplier, h, w) # torch.Size([1, 150, 224, 224])

        if self.color_feats:
            all_feats = [torch.sin(sin_feats), torch.cos(cos_feats), original_image]
        else:
            all_feats = [torch.sin(sin_feats), torch.cos(cos_feats)]

        if self.low_res_feat is not None:
            upsampled_feats = F.interpolate(self.low_res_feat, size=(h, w), mode='bilinear', align_corners=False)
            all_feats.append(upsampled_feats)

        return torch.cat(all_feats, dim=1)