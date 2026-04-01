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
from torch.utils.checkpoint import checkpoint

import torch.utils.checkpoint
from huggingface_hub import hf_hub_download
from einops import rearrange

# from ..utils.geometry_torch import normalized_view_plane_uv, recover_focal_shift, recover_focal_shift_same, gaussian_blur_2d
# from .utils import wrap_dinov2_attention_with_sdpa, wrap_module_with_gradient_checkpointing, unwrap_module_with_gradient_checkpointing
from ..utils.tools import timeit

from .cotracker_blocks import CrossAttnBlock, AttnBlock, Attention
from safetensors.torch import load_file as safe_load_file

from .embeddings import get_2d_sincos_pos_embed, get_2d_embedding, get_1d_sincos_pos_embed_from_grid

from ..utils.sliding_window_utils import get_interpolate_frames_points, compute_scale_and_shift, get_interpolate_frames


from .dust3r.local_memory import LocalMemory
from .dust3r.pose_head import PoseDecoder, postprocess_pose
from .dust3r.camera import pose_encoding_to_camera
from .dust3r.blocks import Block, DecoderBlock, PositionGetter, PatchEmbed
from .dust3r.pos_embed import RoPE2D
from .dust3r.patch_embed import ManyAR_PatchEmbed
from .dust3r.dpt_head import DPTPts3dPose



class Cut3r(nn.Module):
    image_mean: torch.Tensor
    image_std: torch.Tensor

    def __init__(self, 
        # pose_mode=("exp", -float("inf"), float("inf")),
        state_size=768,
        local_mem_size=256,
        state_pe="2d",
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        enc_embed_dim=1024,
        enc_num_heads=16,
        enc_depth=24,
        dec_embed_dim=768,
        dec_num_heads=12,
        dec_depth=12,
        **deprecated_kwargs
    ):
        super().__init__()
        if deprecated_kwargs:
            warnings.warn(f"The following deprecated/invalid arguments are ignored: {deprecated_kwargs}")

        # self.output_mode = output_mode
        # self.head_type = head_type
        # self.depth_mode = depth_mode
        # self.conf_mode = conf_mode
        self.pose_mode = ("exp", -float("inf"), float("inf"))
        # self.freeze = freeze
        # self.patch_embed_cls = patch_embed_cls
        self.state_size = state_size
        self.state_pe = state_pe
        self.dec_num_heads = dec_num_heads
        self.local_mem_size = local_mem_size
        # self.depth_head = depth_head
        # self.rgb_head = rgb_head
        # self.pose_conf_head = pose_conf_head
        # self.pose_head = pose_head
        self.dec_depth = dec_depth
        self.enc_depth = enc_depth
        self.mlp_ratio = mlp_ratio
        self.enc_embed_dim = enc_embed_dim
        self.dec_embed_dim = dec_embed_dim
        self.enc_num_heads = enc_num_heads

        if isinstance(norm_layer, str):
            norm_layer = eval(norm_layer)

        # breakpoint()
        self.gradient_checkpointing = False

        if RoPE2D is None:
            raise ImportError(
                "Cannot find cuRoPE2D, please install it following the README instructions"
            )
        self.rope = RoPE2D(freq=100.0)

        self.position_getter = PositionGetter()

        self.patch_embed = ManyAR_PatchEmbed(
            img_size=(512,512),
            patch_size=16,
            in_chans=3,
            embed_dim=self.enc_embed_dim,
            norm_layer=None,
        )
        self.enc_blocks = nn.ModuleList(
            [
                Block(
                    self.enc_embed_dim,
                    self.enc_num_heads,
                    self.mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    rope=self.rope,
                )
                for i in range(self.enc_depth)
            ]
        )
        self.enc_norm = norm_layer(self.enc_embed_dim)


        self.register_tokens = nn.Embedding(self.state_size, self.enc_embed_dim)

        self.decoder_embed = nn.Linear(self.enc_embed_dim, self.dec_embed_dim, bias=True)
        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    self.dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=True,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm = norm_layer(self.dec_embed_dim)

        self.decoder_embed_state = nn.Linear(self.enc_embed_dim, self.dec_embed_dim, bias=True)
        self.dec_blocks_state = nn.ModuleList(
            [
                DecoderBlock(
                    self.dec_embed_dim,
                    16,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=True,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm_state = norm_layer(self.dec_embed_dim)

        self.pose_token = nn.Parameter(
            torch.randn(1, 1, self.dec_embed_dim) * 0.02, requires_grad=True
        )
        self.pose_retriever = LocalMemory(
            size=self.local_mem_size,
            k_dim=self.enc_embed_dim,
            v_dim=self.dec_embed_dim,
            num_heads=self.dec_num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            attn_drop=0.0,
            norm_layer=norm_layer,
            rope=None,
        )

        # self.pose_decoder = PoseDecoder(
        #     hidden_size=self.dec_embed_dim,
        #     mlp_ratio=4,
        #     pose_encoding_type="absT_quaR",
        # )

        self.downstream_head = DPTPts3dPose()
        

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path, IO[bytes]], model_kwargs: Optional[Dict[str, Any]] = None, strict: bool = True, model_config: Optional[Dict[str, Any]] = None, **hf_kwargs) -> 'MoGeModelTemporal14':
        """
        Load a model from a checkpoint file.

        ### Parameters:
        - `pretrained_model_name_or_path`: path to the checkpoint file or repo id.
        - `model_kwargs`: additional keyword arguments to override the parameters in the checkpoint.
        - `hf_kwargs`: additional keyword arguments to pass to the `hf_hub_download` function. Ignored if `pretrained_model_name_or_path` is a local path.

        ### Returns:
        - A new instance of `MoGe` with the parameters loaded from the checkpoint.
        """
        if Path(pretrained_model_name_or_path).exists():
            if pretrained_model_name_or_path.endswith('.safetensors'):
                checkpoint = safe_load_file(pretrained_model_name_or_path)
            else:
                try:
                    checkpoint = torch.load(pretrained_model_name_or_path, map_location='cpu', weights_only=True)
                except:
                    checkpoint = torch.load(pretrained_model_name_or_path, map_location='cpu')
        else:
            cached_checkpoint_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                repo_type="model",
                filename="model.pt",
                **hf_kwargs
            )
            checkpoint = torch.load(cached_checkpoint_path, map_location='cpu', weights_only=True)

        if model_config is None:
            if 'model_config' in checkpoint:
                model_config = checkpoint['model_config']
            elif 'config' in checkpoint:
                model_config = checkpoint['config']
            else:
                model_config = yaml.load(open('model_config_moge.yaml', 'r'), Loader=yaml.FullLoader)
        
        if model_kwargs is not None:
            model_config.update(model_kwargs)

        model = cls(**model_config)

        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
        elif 'ema_model' in checkpoint:
            checkpoint_model = checkpoint['ema_model']
        else:
            checkpoint_model = checkpoint

        new_checkpoint_model = {}
        for k, v in checkpoint_model.items():
            if k.startswith('module.'):
                new_k = k[7:]
            else:
                new_k = k
            new_checkpoint_model[new_k] = v

        
        print(model.load_state_dict(new_checkpoint_model, strict=strict))
        return model


    # # NOTE for only train head    
    # def train(self, mode: bool = True, train_all: bool = False):
    #     """
    #     Override train method to always set whole model to eval first, 
    #     then only set head to train mode when mode=True.
    #     """
    #     # Always set the whole model to eval first

    #     super().train(False)
    #     self.head.train(mode=mode)
    #     return self

    def _encode_image(self, image, true_shape):
        x, pos = self.patch_embed(image, true_shape=true_shape)
        # assert self.enc_pos_embed is None
        for blk in self.enc_blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                x = blk(x, pos)
        x = self.enc_norm(x)
        return x, pos, None

    def _encode_state(self, image_tokens):
        batch_size = image_tokens.shape[0]
        device, dtype = image_tokens.device, image_tokens.dtype
        state_feat = self.register_tokens(
            torch.arange(self.state_size, device=device)
        )
        if self.state_pe == "1d":
            state_pos = (
                torch.tensor(
                    [[i, i] for i in range(self.state_size)],
                    dtype=dtype,
                    device=device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )  # .long()
        elif self.state_pe == "2d":
            width = int(self.state_size**0.5)
            width = width + 1 if width % 2 == 1 else width
            state_pos = (
                torch.tensor(
                    [[i // width, i % width] for i in range(self.state_size)],
                    dtype=dtype,
                    device=device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )
        elif self.state_pe == "none":
            state_pos = None
        state_feat = state_feat[None].expand(batch_size, -1, -1)
        return state_feat, state_pos, None

    def _init_state(self, image_tokens):
        """
        Current Version: input the first frame img feature and pose to initialize the state feature and pose
        """
        state_feat, state_pos, _ = self._encode_state(image_tokens)
        state_feat = self.decoder_embed_state(state_feat)
        return state_feat, state_pos

    def _decoder(self, f_state, pos_state, f_img, pos_img, f_pose, pos_pose):
        final_output = [(f_state, f_img)]  # before projection
        assert f_state.shape[-1] == self.dec_embed_dim
        f_img = self.decoder_embed(f_img)
        # if self.pose_head_flag:
        assert f_pose is not None and pos_pose is not None
        f_img = torch.cat([f_pose, f_img], dim=1)
        pos_img = torch.cat([pos_pose, pos_img], dim=1)
        final_output.append((f_state, f_img))
        for blk_state, blk_img in zip(self.dec_blocks_state, self.dec_blocks):
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                f_state, _ = torch.utils.checkpoint.checkpoint(
                    blk_state,
                    *final_output[-1][::+1],
                    pos_state,
                    pos_img,
                    use_reentrant=False,
                )
                f_img, _ = torch.utils.checkpoint.checkpoint(
                    blk_img,
                    *final_output[-1][::-1],
                    pos_img,
                    pos_state,
                    use_reentrant=False,
                )
            else:
                f_state, _ = blk_state(*final_output[-1][::+1], pos_state, pos_img)
                f_img, _ = blk_img(*final_output[-1][::-1], pos_img, pos_state)
            final_output.append((f_state, f_img))
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = (
            self.dec_norm_state(final_output[-1][0]),
            self.dec_norm(final_output[-1][1]),
        )
        return zip(*final_output)

    def _get_img_level_feat(self, feat):
        return torch.mean(feat, dim=1, keepdim=True)

    def forward(
        self, 
        video: torch.Tensor, 
    ) -> Dict[str, torch.Tensor]:

        batch_size, num_frames = video.shape[:2]
        
        video_reshaped = rearrange(video, "b t c h w -> (b t) c h w")
        true_shape = torch.tensor([[video.shape[-2], video.shape[-1]]], device=video.device).repeat(batch_size*num_frames, 1)
        video_tokens, video_tokens_pos, _ = self._encode_image(video_reshaped, true_shape)

        video_tokens = video_tokens.reshape(batch_size, num_frames, -1, self.enc_embed_dim)
        video_tokens_pos = video_tokens_pos.reshape(batch_size, num_frames, -1, 2)
        true_shape = true_shape.reshape(batch_size, num_frames, 2)


        first_frame_tokens = video_tokens[:, 0] # B, N, C

        state_feat, state_pos = self._init_state(first_frame_tokens)
        state_pos = state_pos.long()

        mem = self.pose_retriever.mem.expand(batch_size, -1, -1)

        pose_out = []
        global_points_out = []
        global_points_conf_out = []

        local_points_out = []
        local_points_conf_out = []


        hidden_out = []
        pos_out = []

        for i in range(num_frames):
            feat_i = video_tokens[:, i]
            pos_i = video_tokens_pos[:, i]
            # pointmap_i = video_pointmaps[:, i]

            global_img_feat_i = self._get_img_level_feat(feat_i)
            if i == 0:
                pose_feat_i = self.pose_token.expand(batch_size, -1, -1)
            else:
                pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)

            pose_pos_i = -torch.ones(
                batch_size, 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )
            new_state_feat, dec = self._decoder(
                state_feat, state_pos, feat_i, pos_i, pose_feat_i, pose_pos_i
            )
            new_state_feat = new_state_feat[-1]

            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )

            hidden = dec[-1].float()
            pos = pos_i

            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]

            # breakpoint()
            true_shape_i = (video.shape[-2], video.shape[-1])
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
                res = self.downstream_head(head_input, true_shape_i, pos=pos_i)

            pose = res["camera_pose"]
            global_point = res["pts3d_in_other_view"] # B, H, W, 3
            global_point_conf = res["conf"]

            local_point = res["pts3d_in_self_view"]
            local_point_conf = res["conf_self"]


            global_points_out.append(global_point)
            global_points_conf_out.append(global_point_conf)

            local_points_out.append(local_point)
            local_points_conf_out.append(local_point_conf)

            pose_out.append(pose)


            hidden_out.append(hidden)
            pos_out.append(pos)

            state_feat = new_state_feat  # Always update since update=True
            mem = new_mem                # Always update since update=True

        
        hidden_out = torch.stack(hidden_out, dim=1)
        pos_out = torch.stack(pos_out, dim=1)


        pose_out = torch.stack(pose_out, dim=1)
        global_points_out = torch.stack(global_points_out, dim=1)
        global_points_conf_out = torch.stack(global_points_conf_out, dim=1)

        local_points_out = torch.stack(local_points_out, dim=1)
        local_points_conf_out = torch.stack(local_points_conf_out, dim=1)

        camera_poses = pose_out.clone().detach()
        camera_poses_shape = camera_poses.shape[:-1]
        camera_poses = camera_poses.reshape(-1, 7)
        camera_poses = pose_encoding_to_camera(camera_poses)
        camera_poses = camera_poses.reshape(*camera_poses_shape, 4, 4)

        output = {
            "hidden": hidden_out,
            "pos": pos_out,
            "poses": pose_out,
            "camera_poses": camera_poses,
            "points": global_points_out,
            "points_conf": global_points_conf_out,
            "local_points": local_points_out,
            "local_points_conf": local_points_conf_out,
            "patch_start_idx": 1,
        }
        return output

    def forward_features(
        self, 
        video: torch.Tensor, 
    ) -> Dict[str, torch.Tensor|int]:

        batch_size, num_frames = video.shape[:2]
        
        video_reshaped = rearrange(video, "b t c h w -> (b t) c h w")
        true_shape = torch.tensor([[video.shape[-2], video.shape[-1]]], device=video.device).repeat(batch_size*num_frames, 1)

        video_tokens, video_tokens_pos, _ = self._encode_image(video_reshaped, true_shape)

        video_tokens = video_tokens.reshape(batch_size, num_frames, -1, self.enc_embed_dim)
        video_tokens_pos = video_tokens_pos.reshape(batch_size, num_frames, -1, 2)
        true_shape = true_shape.reshape(batch_size, num_frames, 2)


        first_frame_tokens = video_tokens[:, 0] # B, N, C

        state_feat, state_pos = self._init_state(first_frame_tokens)
        state_pos = state_pos.long()

        mem = self.pose_retriever.mem.expand(batch_size, -1, -1)


        hidden_out = []
        pos_out = []

        for i in range(num_frames):
            feat_i = video_tokens[:, i]
            pos_i = video_tokens_pos[:, i]
            # pointmap_i = video_pointmaps[:, i]

            global_img_feat_i = self._get_img_level_feat(feat_i)
            if i == 0:
                pose_feat_i = self.pose_token.expand(batch_size, -1, -1)
            else:
                pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)

            pose_pos_i = -torch.ones(
                batch_size, 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )
            new_state_feat, dec = self._decoder(
                state_feat, state_pos, feat_i, pos_i, pose_feat_i, pose_pos_i
            )
            new_state_feat = new_state_feat[-1]

            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )

            hidden = dec[-1].float()

            hidden_out.append(hidden)
            pos_out.append(pos_i)

            state_feat = new_state_feat  # Always update since update=True
            mem = new_mem                # Always update since update=True

        hidden_out = torch.stack(hidden_out, dim=1)
        pos_out = torch.stack(pos_out, dim=1)

        output = {
            "hidden": hidden_out,
            "pos": pos_out,
            "patch_start_idx": 1,
        }
        return output


    @torch.inference_mode()
    def infer(self, imgs, max_size=None):

        batch_size, num_frames, _, original_height, original_width = imgs.shape
        original_aspect_ratio = original_width / original_height
        patch_size = 16

        if max_size is None:
            max_size = max(original_height//patch_size*patch_size, original_width//patch_size*patch_size)

        if original_width > original_height:
            new_width = min(max_size, original_width // patch_size * patch_size) # NOTE hardcode here 
            new_height = int((new_width / original_aspect_ratio) // patch_size * patch_size)
        else:
            new_height = min(max_size, original_height // patch_size * patch_size) # NOTE hardcode here 
            new_width = int((new_height * original_aspect_ratio) // patch_size * patch_size)
            
        imgs = F.interpolate(imgs.reshape(batch_size*num_frames, _, original_height, original_width), (new_height, new_width), mode='bilinear', antialias=True).clamp(0, 1)
        imgs = imgs.reshape(batch_size, num_frames, _, new_height, new_width)

        # with torch.amp.autocast(enabled=True, dtype=torch.bfloat16, device_type='cuda'):
        output = self.forward(imgs)

        output["conf"] = output["local_points_conf"].unsqueeze(-1)

        for k in ["points", "local_points", "conf"]:
            if k not in output:
                continue
            output[k] = F.interpolate(rearrange(output[k], 'b t h w c -> (b t) c h w'), (original_height, original_width), mode='bilinear', antialias=True)
            output[k] = rearrange(output[k], '(b t) c h w -> b t h w c', b=batch_size, t=num_frames)

        if "mask" not in output:
            output["mask"] = output["conf"].squeeze(-1) > 1.5 

        for k in output.keys():
            if isinstance(output[k], torch.Tensor):
                output[k] = output[k].squeeze(0)

        return output