import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from copy import deepcopy

from typing import *
from numbers import Number
from functools import partial
from pathlib import Path
import warnings
import yaml
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safe_load_file

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points
from .layers.pos_embed import RoPE2D, RoPE2DInterpolated, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from torch.utils.checkpoint import checkpoint
from huggingface_hub import PyTorchModelHubMixin

from einops import rearrange

class Pi3Teacher(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
            **kwargs,
        ):
        super().__init__()


        self.use_checkpoint = True

        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope = None
        if self.pos_type.startswith('rope'): # eg rope100 
            if self.pos_type.endswith('_interpolated'):
                freq = float(self.pos_type[len('rope'):].replace('_interpolated', ''))
                self.rope = RoPE2DInterpolated(freq=freq, original_max_h=32, original_max_w=32)
            else:
                freq = 100.0
                self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError
        

        # ----------------------
        #        Decoder
        # ----------------------
        enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features        # 1024
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
        self.decoder = nn.ModuleList([
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
                rope=self.rope
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
            use_checkpoint=self.use_checkpoint
        )
        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # ----------------------
        #     Conf Decoder
        # ----------------------
        self.conf_decoder = deepcopy(self.point_decoder)
        self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=512,
            rope=self.rope,
            use_checkpoint=self.use_checkpoint
        )
        self.camera_head = CameraHead(dim=512)

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path, IO[bytes]], model_kwargs: Optional[Dict[str, Any]] = None, strict: bool = True, model_config: Optional[Dict[str, Any]] = None, **hf_kwargs) -> 'MoGeModelV2PoseV1':
        """
        Load a model from a checkpoint file.

        ### Parameters:
        - `pretrained_model_name_or_path`: path to the checkpoint file or repo id.
        - `compiled`
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
            return super().from_pretrained(pretrained_model_name_or_path)

        #     checkpoint_path = hf_hub_download(
        #         repo_id=pretrained_model_name_or_path,
        #         repo_type="model",
        #         filename="model.pt",
        #         **hf_kwargs
        #     )
        # checkpoint_debug = torch.load("model-finetuned/train_moge_v2_pose_v1_memdec_v4_15datasets_80000_scaleinvariant_fp16_newsampling/checkpoint-ema-26000.pt", map_location='cpu')
        # checkpoint['config'] = checkpoint_debug['config']

        if model_config is None:
            if 'model_config' in checkpoint:
                model_config = checkpoint['model_config']
            elif 'config' in checkpoint:
                model_config = checkpoint['config']
            else:
                model_config = yaml.load(open('model_config_mogev2.yaml', 'r'), Loader=yaml.FullLoader)
        
        if model_kwargs is not None:
            model_config.update(model_kwargs)

        if hasattr(model_config, '_content'):  # Check if it's an OmegaConf object
            model_config = OmegaConf.to_container(model_config, resolve=True)


        model = cls(**model_config)
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        elif 'ema_model' in checkpoint:
            checkpoint = checkpoint['ema_model']

        filtered_checkpoint = dict()
        current_model_state_dict = model.state_dict()
        for k, v in checkpoint.items():
            if k not in current_model_state_dict:
                print(f"Skipping {k} because it is not in the model state dict")
                continue
            if v.shape != current_model_state_dict[k].shape:
                print(f"Skipping {k} because the shape does not match")
                continue
            filtered_checkpoint[k] = v
        
        print(model.load_state_dict(filtered_checkpoint, strict=strict))
        
        return model


    def decode(self, hidden, N, H, W):
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []
        
        hidden = hidden.reshape(B*N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
       
        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)
            else:
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)

            
            if (
                self.use_checkpoint
                and self.training
                and torch.is_grad_enabled()
            ):
                hidden = checkpoint(blk, hidden, xpos=pos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=pos)

            if i+1 in [len(self.decoder)-1, len(self.decoder)]:
                final_output.append(hidden.reshape(B*N, hw, -1))


        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1)
    
    def forward(self, imgs, return_precomputed_hidden=False, prior_max_size=None, prior_patch_size=None, prior_resolution=None):
        original_imgs = imgs.clone()
        _, num_frames, _, original_height, original_width = original_imgs.shape

        original_aspect_ratio = original_width / original_height

        
        if prior_resolution is None:
            if prior_max_size is None:
                prior_max_size = 518
            if prior_patch_size is None:
                prior_patch_size = 14

            if original_width > original_height:
                prior_width = min(prior_max_size, original_width // prior_patch_size * prior_patch_size) # NOTE hardcode here 
                prior_height = int((prior_width / original_aspect_ratio) // prior_patch_size * prior_patch_size)
            else:
                prior_height = min(prior_max_size, original_height // prior_patch_size * prior_patch_size) # NOTE hardcode here 
                prior_width = int((prior_height * original_aspect_ratio) // prior_patch_size * prior_patch_size)
            prior_patch_h, prior_patch_w = prior_height // prior_patch_size, prior_width // prior_patch_size
        else:
            prior_height, prior_width = prior_resolution
        
        if imgs.shape[-2] != prior_height or imgs.shape[-1] != prior_width:
            imgs = F.interpolate(
                rearrange(imgs, 'b t c h w -> (b t) c h w'), (prior_height, prior_width), mode='bilinear', antialias=True
            ).clamp(0, 1)
            imgs = rearrange(imgs, '(b t) c h w -> b t c h w', t=num_frames)

        imgs = (imgs - self.image_mean) / self.image_std



        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14
        
        # encode by dinov2
        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        if return_precomputed_hidden:
            precomputed_hidden = hidden.clone().detach()
        else:
            precomputed_hidden = None

        hidden, pos = self.decode(hidden, N, H, W)

        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # local points
            point_hidden = point_hidden.float()
            ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)

            ret = F.interpolate(
                rearrange(ret, 'b t h w c -> (b t) c h w'),
                (original_height, original_width), 
                mode='bilinear', align_corners=False, antialias=False
            )
            ret = rearrange(ret, '(b t) c h w -> b t h w c', b=B, t=N)

            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # confidence
            conf_hidden = conf_hidden.float()
            conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            conf = F.interpolate(
                rearrange(conf, 'b t h w c -> (b t) c h w'),
                (original_height, original_width), 
                mode='bilinear', align_corners=False, antialias=False
            )
            conf = rearrange(conf, '(b t) c h w -> b t h w c', b=B, t=N)

            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

            # unproject local points using camera poses
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        return dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
            hidden=hidden.reshape(B, N, self.patch_start_idx+patch_h*patch_w, -1),
            pos=pos.reshape(B, N, self.patch_start_idx+patch_h*patch_w, 2),
            patch_start_idx=self.patch_start_idx,
            precomputed_hidden=precomputed_hidden,
        )

    def forward_features(self, imgs):
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14
        
        # encode by dinov2
        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        hidden, pos = self.decode(hidden, N, H, W)

        return dict(
            hidden=hidden.reshape(B, N, self.patch_start_idx+patch_h*patch_w, -1),
            pos=pos.reshape(B, N, self.patch_start_idx+patch_h*patch_w, 2),
            patch_start_idx=self.patch_start_idx,
        )

    @torch.inference_mode()
    def infer(self, imgs, max_size=None, conf_threshold=0.2):

        batch_size, num_frames, _, original_height, original_width = imgs.shape
        original_aspect_ratio = original_width / original_height
        patch_size = 14

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

        with torch.amp.autocast(enabled=True, dtype=torch.bfloat16, device_type='cuda'):
            output = self.forward(imgs, prior_resolution=(new_height, new_width))

        for k in ["points", "local_points", "conf"]:
            if k not in output:
                continue
            output[k] = F.interpolate(rearrange(output[k], 'b t h w c -> (b t) c h w'), (original_height, original_width), mode='bilinear', antialias=True)
            output[k] = rearrange(output[k], '(b t) c h w -> b t h w c', b=batch_size, t=num_frames)

        if "mask" not in output:
            output["mask"] = torch.sigmoid(output["conf"]).squeeze(-1) > conf_threshold

        for k in output.keys():
            if isinstance(output[k], torch.Tensor):
                output[k] = output[k].squeeze(0)

        return output

    def forward_long(self, imgs, original_resolution):
        # original_imgs = imgs.clone()
        num_frames = imgs.shape[1]
        original_height, original_width = original_resolution
        original_aspect_ratio = original_width / original_height

        
        # if prior_resolution is None:
        #     if prior_max_size is None:
        #         prior_max_size = 518
        #     if prior_patch_size is None:
        #         prior_patch_size = 14

        #     if original_width > original_height:
        #         prior_width = min(prior_max_size, original_width // prior_patch_size * prior_patch_size) # NOTE hardcode here 
        #         prior_height = int((prior_width / original_aspect_ratio) // prior_patch_size * prior_patch_size)
        #     else:
        #         prior_height = min(prior_max_size, original_height // prior_patch_size * prior_patch_size) # NOTE hardcode here 
        #         prior_width = int((prior_height * original_aspect_ratio) // prior_patch_size * prior_patch_size)
        #     prior_patch_h, prior_patch_w = prior_height // prior_patch_size, prior_width // prior_patch_size
        # else:
        #     prior_height, prior_width = prior_resolution
        
        # if imgs.shape[-2] != prior_height or imgs.shape[-1] != prior_width:
        #     imgs = F.interpolate(
        #         rearrange(imgs, 'b t c h w -> (b t) c h w'), (prior_height, prior_width), mode='bilinear', antialias=True
        #     ).clamp(0, 1)
        #     imgs = rearrange(imgs, '(b t) c h w -> b t c h w', t=num_frames)

        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14
        
        # encode by dinov2
        imgs = imgs.reshape(B*N, _, H, W)

        chunk_size = 1

        hidden_list = []
        for c_idx in range(0, imgs.shape[0], chunk_size):
            chunk_start = c_idx
            chunk_end = min(chunk_start + chunk_size, imgs.shape[0])
            chunk_imgs = imgs[chunk_start:chunk_end]
            chunk_hidden = self.encoder(chunk_imgs, is_training=True)
            if isinstance(chunk_hidden, dict):
                chunk_hidden = chunk_hidden["x_norm_patchtokens"]
            hidden_list.append(chunk_hidden)
        hidden = torch.cat(hidden_list, dim=0)

        del imgs

        

        hidden, pos = self.decode(hidden, N, H, W)

        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        del hidden

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # local points
            point_hidden = point_hidden.float()
            ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)

            ret = F.interpolate(
                rearrange(ret, 'b t h w c -> (b t) c h w'),
                (original_height, original_width), 
                mode='bilinear', align_corners=False, antialias=False
            )
            ret = rearrange(ret, '(b t) c h w -> b t h w c', b=B, t=N)

            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # confidence
            conf_hidden = conf_hidden.float()
            conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            conf = F.interpolate(
                rearrange(conf, 'b t h w c -> (b t) c h w'),
                (original_height, original_width), 
                mode='bilinear', align_corners=False, antialias=False
            )
            conf = rearrange(conf, '(b t) c h w -> b t h w c', b=B, t=N)

            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

            # unproject local points using camera poses
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        del point_hidden, conf_hidden, camera_hidden

        return dict(
            points=points[0],
            local_points=local_points[0],
            conf=conf[0],
            camera_poses=camera_poses[0],
            # hidden=hidden.reshape(B, N, self.patch_start_idx+patch_h*patch_w, -1),
            # pos=pos.reshape(B, N, self.patch_start_idx+patch_h*patch_w, 2),
            # patch_start_idx=self.patch_start_idx,
            # precomputed_hidden=precomputed_hidden,
        )