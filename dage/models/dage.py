from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from functools import partial
from copy import deepcopy
from einops import rearrange
from collections import defaultdict
import utils3d

from typing import *
from functools import partial
from pathlib import Path
import yaml
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safe_load_file


from .dinov2.layers import Mlp
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import (
    TransformerDecoder,
    LinearPts3d,
    ContextTransformerDecoder,
    ClsTransformerDecoder,
)
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14_reg
from .token_splitter import TokenSplitter

# from safetensors.torch import load_file
from .moge.model.modules import DINOv2Encoder, MLP, ConvStack

from dage.utils.geometry import homogenize_points, normalized_view_plane_uv
from dage.models.moge.utils.geometry_torch import recover_focal_shift


class DAGE(nn.Module):
    def __init__(
        self,
        pos_type="rope100",
        hr_encoder: Dict[str, Any] = None,
        hr_neck: Dict[str, Any] = None,
        hr_points_head: Dict[str, Any] = None,
        hr_mask_head: Dict[str, Any] = None,
        hr_scale_head: Dict[str, Any] = None,
        adapter: Dict[str, Any] = None,
        remap_output: Literal["linear", "sinh", "exp", "sinh_exp"] = "exp",
        **deprecated_kwargs,
    ):
        super().__init__()
        if deprecated_kwargs:
            print(f"The following deprecated/invalid arguments are ignored: {deprecated_kwargs}")

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
        self.pos_type = pos_type if pos_type is not None else "none"
        self.rope = None
        if self.pos_type.startswith("rope"):  # eg rope100
            if RoPE2D is None:
                raise ImportError(
                    "Cannot find cuRoPE2D, please install it following the README instructions"
                )
            freq = float(self.pos_type[len("rope") :])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError

        dec_embed_dim = 1024
        dec_num_heads = 16
        mlp_ratio = 4
        dec_depth = 36

        self.decoder = nn.ModuleList(
            [
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
                    rope=self.rope,
                )
                for _ in range(dec_depth)
            ]
        )
        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(
            torch.randn(1, 1, num_register_tokens, self.dec_embed_dim)
        )
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
            use_checkpoint=self.use_checkpoint,
        )
        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,  # 8
            out_dim=512,
            rope=self.rope,
            use_checkpoint=self.use_checkpoint,
        )
        self.camera_head = CameraHead(dim=512)

        # ----------------------
        #  HR Stream
        # ----------------------

        self.hr_encoder = DINOv2Encoder(**hr_encoder)
        self.hr_points_head = ConvStack(**hr_points_head)
        self.hr_mask_head = ConvStack(**hr_mask_head)
        self.hr_neck = ConvStack(**hr_neck)

        if hr_scale_head is not None:
            self.hr_scale_head = MLP(**hr_scale_head)
        else:
            self.hr_scale_head = None

        self.token_splitter = TokenSplitter(self.dec_embed_dim, (2, 2))

        self.remap_output = remap_output

        self.hr_encoder.enable_pytorch_native_sdpa()

        if adapter is not None:
            adapter_name = adapter.pop("model_name")
            print(f"Use fused decoder: {adapter_name}")
            self.adapter = eval(adapter_name)(**adapter)
        else:
            self.adapter = ContextTransformerDecoder(
                in_dim=1024,
                out_dim=1024,
                dec_embed_dim=1024,
                depth=5,
                dec_num_heads=16,
                use_checkpoint=True,
                cross_first=True,
                y_in_dim=2 * 1024,
                rope=None,
                use_pe=False,
            )
        self.adapter.zero_init()

        self.cls_decoder = ClsTransformerDecoder(
            in_dim=1024,
            out_dim=1024,
            dec_embed_dim=1024,
            depth=3,
            dec_num_heads=16,
            use_checkpoint=True,
            y_in_dim=1024,
        )
        self.cls_decoder.zero_init()

        if self.use_checkpoint:
            self.hr_encoder.enable_gradient_checkpointing()
            self.hr_neck.enable_gradient_checkpointing()
            self.hr_points_head.enable_gradient_checkpointing()
            self.hr_mask_head.enable_gradient_checkpointing()

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path, IO[bytes]],
        model_kwargs: Optional[Dict[str, Any]] = None,
        strict: bool = True,
        model_config: Optional[Dict[str, Any]] = None,
        **hf_kwargs,
    ) -> "DAGE":
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
            if pretrained_model_name_or_path.endswith(".safetensors"):
                checkpoint = safe_load_file(pretrained_model_name_or_path)
            else:
                try:
                    checkpoint = torch.load(
                        pretrained_model_name_or_path,
                        map_location="cpu",
                        weights_only=True,
                    )
                except:
                    checkpoint = torch.load(
                        pretrained_model_name_or_path, map_location="cpu"
                    )
        else:
            requested_filename = hf_kwargs.pop("filename", None)
            candidate_filenames = [
                requested_filename,
                "model.pt",
                "model.safetensors",
                "pytorch_model.bin",
                "checkpoint.pt",
            ]
            candidate_filenames = [
                filename for i, filename in enumerate(candidate_filenames)
                if filename is not None and filename not in candidate_filenames[:i]
            ]

            cached_checkpoint_path = None
            last_error = None
            for filename in candidate_filenames:
                try:
                    cached_checkpoint_path = hf_hub_download(
                        repo_id=pretrained_model_name_or_path,
                        repo_type="model",
                        filename=filename,
                        **hf_kwargs,
                    )
                    break
                except Exception as error:
                    last_error = error

            if cached_checkpoint_path is None:
                raise FileNotFoundError(
                    f"Could not find a checkpoint file in Hugging Face repo "
                    f"{pretrained_model_name_or_path}. Tried: {candidate_filenames}"
                ) from last_error

            if cached_checkpoint_path.endswith(".safetensors"):
                checkpoint = safe_load_file(cached_checkpoint_path)
            else:
                try:
                    checkpoint = torch.load(
                        cached_checkpoint_path, map_location="cpu", weights_only=True
                    )
                except:
                    checkpoint = torch.load(cached_checkpoint_path, map_location="cpu")

        if model_config is None:
            if "model_config" in checkpoint:
                model_config = checkpoint["model_config"]
            elif "config" in checkpoint:
                model_config = checkpoint["config"]
            else:
                print(f"No model config found in checkpoint {pretrained_model_name_or_path}, using default config")
                model_config = yaml.load(
                    open("configs/model_config_dage.yaml", "r"), Loader=yaml.FullLoader
                )["model"]["config"]

        if model_kwargs is not None:
            model_config.update(model_kwargs)

        if hasattr(model_config, "_content"):  # Check if it's an OmegaConf object
            model_config = OmegaConf.to_container(model_config, resolve=True)

        model = cls(**model_config)
        if "ema_model" in checkpoint:
            print(f"Loading EMA model state dict")
            checkpoint = checkpoint["ema_model"]
        elif "model" in checkpoint:
            checkpoint = checkpoint["model"]

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

    def decode(
        self, hidden: torch.Tensor, N: int, H: int, W: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden: (B*N, H//patch_size * W//patch_size, C) - encoded features
            N: int - number of frames
            H: int - height of input image
            W: int - width of input image
        Returns:
            hidden: (B*N, n_register + H//patch_size * W//patch_size, 2*C) - decoded features
            pos: (B*N, n_register + H//patch_size * W//patch_size, 2) - position encoding
            student_feat_for_distill: (B*N, 2*H//patch_size * 2*W//patch_size, C) or None
        """
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []
        student_feat_for_distill = None

        hidden = hidden.reshape(B * N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(
            B * N, *self.register_token.shape[-2:]
        )

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith("rope"):
            pos = self.position_getter(
                B * N, H // self.patch_size, W // self.patch_size, hidden.device
            )

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = (
                torch.zeros(B * N, self.patch_start_idx, 2)
                .to(hidden.device)
                .to(pos.dtype)
            )
            pos = torch.cat([pos_special, pos], dim=1)

        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B * N, hw, -1)
                hidden = hidden.reshape(B * N, hw, -1)
            else:
                pos = pos.reshape(B, N * hw, -1)
                hidden = hidden.reshape(B, N * hw, -1)

            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                hidden = checkpoint(blk, hidden, xpos=pos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=pos)

            if i + 1 in [len(self.decoder) - 1, len(self.decoder)]:
                final_output.append(hidden.reshape(B * N, hw, -1))

            if (
                i + 1 == len(self.decoder) - 2
                and self.training
                and torch.is_grad_enabled()
            ):
                student_feat = hidden.reshape(B * N, hw, -1)
                student_feat_for_distill = self.token_splitter(
                    student_feat,
                    patch_start_idx=self.patch_start_idx,
                    patch_hw=(H // self.patch_size, W // self.patch_size),
                    B=B,
                    N=N,
                )

        return (
            torch.cat([final_output[0], final_output[1]], dim=-1),
            pos.reshape(B * N, hw, -1),
            student_feat_for_distill,
        )

    def hr_stream(
        self,
        hr_video: torch.Tensor,
        hr_max_size: Optional[int] = None,
        hr_num_tokens: Optional[int] = None,
        # hr_resolution_level: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hr_x: (B, N, 3, H, W) - high resolution input images
            hr_num_tokens: int or None - target number of tokens
            hr_resolution_level: int - resolution level (0-9)
        Returns:
            features: (B*N, C, base_h, base_w) - high resolution features
            cls_token: (B*N, C) - classification token
        """
        batch_size, num_frames, _, img_h, img_w = hr_video.shape
        aspect_ratio = img_w / img_h
        # device, dtype = hr_x.device, hr_x.dtype


        pseudo_image = hr_video.reshape(batch_size * num_frames, 3, img_h, img_w)
        
        if hr_max_size is None:
            if hr_num_tokens is None:
                hr_num_tokens = 3600
                # min_tokens, max_tokens = 1200, 3600
                # if hr_resolution_level is None:
                #     hr_resolution_level = 9
                # hr_num_tokens = int(
                #     min_tokens + (hr_resolution_level / 9) * (max_tokens - min_tokens)
                # )
            base_h, base_w = int((hr_num_tokens / aspect_ratio) ** 0.5), int(
                (hr_num_tokens * aspect_ratio) ** 0.5
            )
        else:
            hr_max_size = (hr_max_size // self.patch_size) * self.patch_size
            if img_w > img_h:
                base_w = hr_max_size // self.patch_size
                base_h = int(base_w * aspect_ratio)
            else:
                base_h = hr_max_size // self.patch_size
                base_w = int(base_h / aspect_ratio)

        print(f"HR stream: {base_h*14}x{base_w*14} with num_tokens {base_h*base_w}")

        # NOTE Backbones encoding ######################################
        features, cls_token = self.hr_encoder(
            image=pseudo_image, token_rows=base_h, token_cols=base_w, return_class_token=True
        )

        return features, cls_token

    def _remap_points(self, points: torch.Tensor) -> torch.Tensor:
        if self.remap_output == "linear":
            pass
        elif self.remap_output == "sinh":
            points = torch.sinh(points)
        elif self.remap_output == "exp":
            xy, z = points.split([2, 1], dim=-1)
            z = torch.exp(z)
            points = torch.cat([xy * z, z], dim=-1)
        elif self.remap_output == "sinh_exp":
            xy, z = points.split([2, 1], dim=-1)
            points = torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)
        else:
            raise ValueError(f"Invalid remap output type: {self.remap_output}")
        return points

    def _process_hr_features(
        self,
        features: List[Optional[torch.Tensor]],
        base_h: int,
        base_w: int,
        batch_size: int,
        aspect_ratio: float,
        original_shape: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper function to process HR features through neck and heads.

        Args:
            features: List of 5 feature maps at different scales (or None)
            base_h: int - base height in tokens
            base_w: int - base width in tokens
            batch_size: int - batch size (B*N for chunks)
            aspect_ratio: float - aspect ratio of original image
            original_shape: (H, W) - original image shape
            device: torch device
            dtype: torch dtype
        Returns:
            points: (batch_size, H, W, 3) - predicted 3D points
            conf: (batch_size, 1, H, W) - confidence map
        """
        original_height, original_width = original_shape

        # Concat UVs for aspect ratio input
        for level in range(5):
            uv = normalized_view_plane_uv(
                width=base_w * 2**level,
                height=base_h * 2**level,
                aspect_ratio=aspect_ratio,
                dtype=dtype,
                device=device,
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            if features[level] is None:
                features[level] = uv
            else:
                features[level] = torch.concat([features[level], uv], dim=1)

        features = self.hr_neck(features)

        points = self.hr_points_head(features)[-1]
        points = F.interpolate(
            points,
            (original_height, original_width),
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )

        conf = self.hr_mask_head(features)[-1]
        conf = F.interpolate(
            conf,
            (original_height, original_width),
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )

        return points, conf

    def hr_head(
        self,
        features: torch.Tensor,
        token_shape: Tuple[int, int],
        aspect_ratio: float,
        original_shape: Tuple[int, int],
        force_chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B*N, H_hr*W_hr, C) - fused high resolution features
            pos: (B*N, H_hr*W_hr, 2) - position encoding
            token_shape: (H_hr, W_hr) - token shape
            aspect_ratio: float - aspect ratio of original image
            original_shape: (H, W) - original image shape
            force_chunk_size: int or None - force specific chunk size
        Returns:
            points: (B*N, H, W, 3) - predicted 3D points in local camera space
            conf: (B*N, 1, H, W) - confidence map
        """
        BN = features.shape[0]
        base_h, base_w = token_shape
        original_height, original_width = original_shape
        device = features.device
        dtype = features.dtype

        features = rearrange(features, "b (h w) c -> b c h w", h=base_h, w=base_w)

        features = [features, None, None, None, None]
        with torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=not self.training
        ):

            if force_chunk_size is not None:
                chunk_size = force_chunk_size
            else:
                if base_w * base_h > 12000:
                    chunk_size = 1
                elif base_w * base_h > 5000:
                    chunk_size = 8
                else:
                    chunk_size = 16

            if BN >= chunk_size and not self.training:
                num_chunks = (
                    (BN) // chunk_size
                    if (BN) % chunk_size == 0
                    else (BN) // chunk_size + 1
                )
                points_list, conf_list = [], []
                for i in range(num_chunks):
                    start_frame = i * chunk_size
                    end_frame = min(start_frame + chunk_size, BN)

                    # Extract chunk features
                    chunk_features = []
                    for f in features:
                        chunk_features.append(
                            f[start_frame:end_frame] if f is not None else None
                        )

                    # Process chunk through helper function
                    chunk_points, chunk_conf = self._process_hr_features(
                        chunk_features,
                        base_h,
                        base_w,
                        end_frame - start_frame,
                        aspect_ratio,
                        original_shape,
                        device,
                        dtype,
                    )
                    points_list.append(chunk_points)
                    conf_list.append(chunk_conf)

                points = torch.cat(points_list, dim=0)
                conf = torch.cat(conf_list, dim=0)

            else:
                # Process all features at once (non-chunked)
                points, conf = self._process_hr_features(
                    features,
                    base_h,
                    base_w,
                    BN,
                    aspect_ratio,
                    original_shape,
                    device,
                    dtype,
                )

            # Remap output
            points = points.permute(0, 2, 3, 1)
            points = self._remap_points(
                points
            )  # slightly improves the performance in case of very large output values

            return points, conf

    def lr_head(
        self,
        lr_hidden: torch.Tensor,
        lr_pos: torch.Tensor,
        lr_token_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Args:
            lr_hidden: (B*N, n_register + H_lr*W_lr, 2*C) - low resolution features
            lr_pos: (B*N, n_register + H_lr*W_lr, 2) - position encoding
            lr_token_shape: (H_lr, W_lr) - low resolution token shape
        Returns:
            camera_poses: (B*N, 4, 4) - predicted camera poses
        """
        BN = lr_hidden.shape[0]
        patch_h, patch_w = lr_token_shape
        camera_hidden = self.camera_decoder(lr_hidden, xpos=lr_pos)

        with torch.amp.autocast(device_type="cuda", enabled=False):

            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(
                camera_hidden[:, self.patch_start_idx :], patch_h, patch_w
            ).reshape(BN, 4, 4)

        return camera_poses

    def fuse(
        self,
        hr_features: torch.Tensor,
        hr_token_shape: Tuple[int, int],
        lr_features: torch.Tensor,
        lr_token_shape: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hr_features: (B*N, H_hr*W_hr, C) - high resolution features
            hr_token_shape: (H_hr, W_hr) - high resolution token shape
            lr_features: (B*N, n_register + H_lr*W_lr, 2*C) - low resolution features
            lr_token_shape: (H_lr, W_lr) - low resolution token shape
        Returns:
            fused_features: (B*N, H_hr*W_hr, C) - fused features
            hr_feature_pos: (B*N, H_hr*W_hr, 2) - high resolution position encoding
        """
        BN = hr_features.shape[0]
        device = hr_features.device

        fused_features = self.adapter(
            hidden=hr_features,
            context=lr_features,
            xpos=None,
            ypos=None,
            hidden_shape=hr_token_shape,
            context_shape=lr_token_shape,
            context_patch_start_idx=self.patch_start_idx,
            hidden_cls_token=None,
        )

        # NOTE residual connection
        out_features = hr_features + fused_features  # (B T) (H_hr W_hr ) C

        return out_features

    def metric_head(
        self,
        fused_features: torch.Tensor,
        moge_cls_token: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """
        Args:
            fused_features: (B*N, H_hr*W_hr, C) - fused features
            moge_cls_token: (B*N, C) - classification token from MoGe
            num_frames: int - number of frames
        Returns:
            metric_scale: (B, 1) - metric scale for converting relative depth to absolute
        """
        moge_cls_token = moge_cls_token.detach()[:, None]
        fused_features = fused_features.detach()
        fused_cls_token = self.cls_decoder(
            moge_cls_token, fused_features, num_frames=num_frames
        )
        fused_cls_token = fused_cls_token + moge_cls_token
        fused_cls_token = rearrange(fused_cls_token, "(b t) 1 c -> b c t", t=num_frames)
        fused_cls_token = F.adaptive_avg_pool1d(fused_cls_token, 1).squeeze(
            -1
        )  # (B, C)

        metric_scale = self.hr_scale_head(fused_cls_token)
        metric_scale = metric_scale.exp()  # (B, 1)
        return metric_scale

    def downsample_input(
        self,
        lr_video: torch.Tensor,
        lr_max_size: Optional[int] = None,
        lr_resolution: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, 3, H, W) - input images
            lr_max_size: int or None - maximum size for low resolution
            lr_resolution: (H_lr, W_lr) or None - explicit low resolution size
        Returns:
            lr_x: (B, N, 3, H_lr, W_lr) - downsampled images
        """
        batch_size, num_frames, _, original_height, original_width = lr_video.shape
        original_aspect_ratio = original_width / original_height

        if lr_resolution is None:
            if lr_max_size is None:
                lr_max_size = 518
            else:
                lr_max_size = (lr_max_size // self.patch_size) * self.patch_size

            if original_width > original_height:
                lr_width = min(
                    lr_max_size, original_width // self.patch_size * self.patch_size
                )  # NOTE hardcode here
                lr_height = int(
                    (lr_width / original_aspect_ratio)
                    // self.patch_size
                    * self.patch_size
                )
            else:
                lr_height = min(
                    lr_max_size, original_height // self.patch_size * self.patch_size
                )  # NOTE hardcode here
                lr_width = int(
                    (lr_height * original_aspect_ratio)
                    // self.patch_size
                    * self.patch_size
                )

        else:
            lr_height, lr_width = lr_resolution

        if lr_video.shape[-2] != lr_height or lr_video.shape[-1] != lr_width:
            lr_x = F.interpolate(
                rearrange(lr_video, "b t c h w -> (b t) c h w"),
                (lr_height, lr_width),
                mode="bilinear",
                antialias=True,
            ).clamp(0, 1)
            lr_x = rearrange(lr_x, "(b t) c h w -> b t c h w", t=num_frames)
        else:
            lr_x = lr_video

        return lr_x

    def lr_stream(
        self, lr_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            lr_x: (B, N, 3, H_lr, W_lr) - low resolution input images
        Returns:
            lr_hidden: (B*N, n_register + H_lr//patch_size * W_lr//patch_size, 2*C) - decoded features
            lr_pos: (B*N, n_register + H_lr//patch_size * W_lr//patch_size, 2) - position encoding
            lr_student_feat_for_distill: (B*N, 4*H_lr//patch_size * 4*W_lr//patch_size, C) or None
        """
        batch_size, num_frames, _, lr_height, lr_width = lr_x.shape

        lr_x = lr_x.reshape(batch_size * num_frames, 3, lr_height, lr_width)

        print(f"LR stream: {lr_height}x{lr_width}")
        lr_hidden = self.encoder(lr_x, is_training=True)
        if isinstance(lr_hidden, dict):
            lr_hidden = lr_hidden["x_norm_patchtokens"]

        lr_hidden, lr_pos, lr_student_feat_for_distill = self.decode(
            lr_hidden, num_frames, lr_height, lr_width
        )

        return lr_hidden, lr_pos, lr_student_feat_for_distill

    def forward(
        self,
        hr_video: torch.Tensor,
        lr_video: Optional[torch.Tensor] = None,
        lr_max_size: Optional[int] = None,
        lr_resolution: Optional[Tuple[int, int]] = None,
        hr_max_size: Optional[int] = None,
        hr_num_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            x: (B, N, 3, H, W) - input images
            lr_max_size: int or None - maximum size for low resolution stream (default: 518)
            lr_resolution: (H_lr, W_lr) or None - explicit low resolution size
            hr_max_size: int or None - maximum size for high resolution stream
            hr_num_tokens: int or None - target number of tokens for high resolution stream
        Returns:
            output_dict: dict containing:
                - local_points: (B, N, H, W, 3) - 3D points in local camera space
                - conf: (B, N, H, W, 1) - confidence map
                - metric_scale: (B, 1) - metric scale
                - camera_poses: (B, N, 4, 4) - camera poses
                - aspect_ratio: float - aspect ratio of input images
                - (training only) student_feat_for_distill: (B*N, 4*H_lr*W_lr, C)
                - (training only) patch_start_idx: int
        """
        # hr_x = x.clone()
        if lr_video is None:
            lr_video = hr_video.clone()

        batch_size, num_frames, _, original_height, original_width = hr_video.shape
        # device, dtype = x.device, X.dtype
        original_aspect_ratio = original_width / original_height
        original_shape = (original_height, original_width)

        lr_x = self.downsample_input(lr_video, lr_max_size, lr_resolution)
        lr_x = (lr_x - self.image_mean) / self.image_std

        lr_token_shape = (
            lr_x.shape[-2] // self.patch_size,
            lr_x.shape[-1] // self.patch_size,
        )

        # NOTE LR stream ##############################################
        lr_hidden, lr_pos, lr_student_feat_for_distill = self.lr_stream(lr_x)
        # lr_token_shape = tuple(lr_hidden.shape[-2:])
        ################################################################

        # NOTE HR stream ##############################################
        hr_hidden, hr_cls_token = self.hr_stream(
            hr_video, hr_max_size=hr_max_size, hr_num_tokens=hr_num_tokens
        )

        hr_token_shape = tuple(hr_hidden.shape[-2:])
        hr_hidden = rearrange(hr_hidden, "b c h w -> b (h w) c")

        # NOTE fusion here ##############################################
        fused_features = self.fuse(hr_hidden, hr_token_shape, lr_hidden, lr_token_shape)

        # NOTE Output head ##############################################
        camera_poses = self.lr_head(lr_hidden, lr_pos, lr_token_shape)
        local_points, conf = self.hr_head(
            fused_features,
            hr_token_shape,
            original_aspect_ratio,
            original_shape,
        )
        metric_scale = self.metric_head(
            fused_features, hr_cls_token, num_frames
        )  # (B, 1)

        ################################################################

        local_points = rearrange(
            local_points, "(b t) h w c -> b t h w c", b=batch_size, t=num_frames
        )
        conf = rearrange(conf, "(b t) c h w -> b t h w c", b=batch_size, t=num_frames)
        camera_poses = rearrange(
            camera_poses, "(b t) i j -> b t i j", b=batch_size, t=num_frames
        )

        output_dict = dict(
            local_points=local_points,
            conf=conf,
            metric_scale=metric_scale,
            camera_poses=camera_poses,
            aspect_ratio=original_aspect_ratio,
        )

        if self.training and torch.is_grad_enabled():
            output_dict["student_feat_for_distill"] = lr_student_feat_for_distill
            output_dict["patch_start_idx"] = self.patch_start_idx

        return output_dict

    def forward_chunk(
        self,
        hr_video: torch.Tensor,
        lr_video: Optional[torch.Tensor] = None,
        lr_max_size: Optional[int] = None,
        lr_resolution: Optional[Tuple[int, int]] = None,
        hr_max_size: Optional[int] = None,
        hr_num_tokens: Optional[int] = None,
        chunk_size: int = 8,
    ) -> Dict[str, Any]:
        """
        Forward pass with chunked processing for high-resolution inputs.
        Processes frames in chunks to reduce memory usage.

        Args:
            hr_video: (B, N, 3, H, W) - input high resolution images
            lr_video: (B, N, 3, H, W) - input low resolution images
            lr_max_size: int or None - maximum size for low resolution stream (default: 518)
            lr_resolution: (H_lr, W_lr) or None - explicit low resolution size
            hr_max_size: int or None - maximum size for high resolution stream
            hr_num_tokens: int or None - target number of tokens for high resolution stream
            chunk_size: int - number of frames to process together in each chunk (default: 8)
        Returns:
            output_dict: dict containing:
                - local_points: (B, N, H, W, 3) - 3D points in local camera space
                - conf: (B, N, H, W, 1) - confidence map
                - metric_scale: (B, 1) - metric scale
                - camera_poses: (B, N, 4, 4) - camera poses
                - aspect_ratio: float - aspect ratio of input images
                - (training only) student_feat_for_distill: (B*N, 4*H_lr*W_lr, C)
                - (training only) patch_start_idx: int
        """
        if lr_video is None:
            lr_video = hr_video.clone()

        batch_size, num_frames, _, original_height, original_width = hr_video.shape
        original_aspect_ratio = original_width / original_height
        original_shape = (original_height, original_width)

        # NOTE LR stream - process all frames at once (it's low resolution)
        lr_x = self.downsample_input(lr_video, lr_max_size, lr_resolution)
        lr_x = (lr_x - self.image_mean) / self.image_std

        lr_token_shape = (
            lr_x.shape[-2] // self.patch_size,
            lr_x.shape[-1] // self.patch_size,
        )
        lr_hidden, lr_pos, lr_student_feat_for_distill = self.lr_stream(lr_x)

        # Process camera poses from LR stream (relatively cheap)
        camera_poses = self.lr_head(lr_hidden, lr_pos, lr_token_shape)
        camera_poses = rearrange(
            camera_poses, "(b t) i j -> b t i j", b=batch_size, t=num_frames
        )

        # Reshape lr_hidden for chunked access
        lr_hidden_per_frame = lr_hidden.reshape(
            batch_size, num_frames, -1, lr_hidden.shape[-1]
        )

        # NOTE Chunked processing for HR stream and fusion
        num_chunks = (num_frames + chunk_size - 1) // chunk_size

        all_local_points = []
        all_conf = []
        all_fused_cls_tokens = []
        # all_fused_features = []
        # all_hr_cls_tokens = []

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, num_frames)
            chunk_frames = chunk_end - chunk_start

            # HR stream for this chunk
            chunk_hr_x = hr_video[:, chunk_start:chunk_end]
            chunk_hr_hidden, chunk_hr_cls_token = self.hr_stream(
                chunk_hr_x, hr_max_size=hr_max_size, hr_num_tokens=hr_num_tokens
            )

            hr_token_shape = tuple(chunk_hr_hidden.shape[-2:])
            chunk_hr_hidden = rearrange(chunk_hr_hidden, "b c h w -> b (h w) c")

            # Get corresponding LR features for this chunk
            chunk_lr_hidden = lr_hidden_per_frame[:, chunk_start:chunk_end]
            chunk_lr_hidden = chunk_lr_hidden.reshape(
                batch_size * chunk_frames, -1, lr_hidden.shape[-1]
            )

            # Fusion
            chunk_fused_features = self.fuse(
                chunk_hr_hidden, hr_token_shape, chunk_lr_hidden, lr_token_shape
            )

            # HR head
            chunk_local_points, chunk_conf = self.hr_head(
                chunk_fused_features,
                hr_token_shape,
                original_aspect_ratio,
                original_shape,
            )

            chunk_fused_cls_token = self.cls_decoder(
                chunk_hr_cls_token[:, None], chunk_fused_features, num_frames=chunk_frames
            )
            chunk_fused_cls_token = chunk_fused_cls_token + chunk_hr_cls_token[:, None]

            chunk_local_points = rearrange(chunk_local_points, "(b t) h w c -> b t h w c", b=batch_size, t=chunk_frames)
            chunk_conf = rearrange(chunk_conf, "(b t) c h w -> b t h w c", b=batch_size, t=chunk_frames)
            chunk_fused_cls_token = rearrange(chunk_fused_cls_token, "(b t) 1 c -> b c t", t=chunk_frames)

            all_local_points.append(chunk_local_points)
            all_conf.append(chunk_conf)
            all_fused_cls_tokens.append(chunk_fused_cls_token)

        # Concatenate all chunks
        local_points = torch.cat(all_local_points, dim=1)
        conf = torch.cat(all_conf, dim=1)

        fused_cls_token = torch.cat(all_fused_cls_tokens, dim=2)
        fused_cls_token = F.adaptive_avg_pool1d(fused_cls_token, 1).squeeze(
            -1
        )  # (B, C)
        metric_scale = self.hr_scale_head(fused_cls_token)
        metric_scale = metric_scale.exp()  # (B, 1)


        output_dict = dict(
            local_points=local_points,
            conf=conf,
            metric_scale=metric_scale,
            camera_poses=camera_poses,
            aspect_ratio=original_aspect_ratio,
        )

        if self.training and torch.is_grad_enabled():
            output_dict["student_feat_for_distill"] = lr_student_feat_for_distill
            output_dict["patch_start_idx"] = self.patch_start_idx

        return output_dict

    @torch.inference_mode()
    def infer(
        self,
        hr_video: torch.Tensor,
        lr_video: Optional[torch.Tensor] = None,
        lr_max_size: Optional[int] = None,
        lr_resolution: Optional[Tuple[int, int]] = None,
        hr_max_size: Optional[int] = None,
        hr_num_tokens: Optional[int] = None,
        chunk_size: Optional[int] = None,
        refine: bool = False,
        to_cpu: bool = True,
        enable_autocast: bool = True,
    ) -> Dict[str, Any]:
        """
        Inference function with post-processing to generate final outputs.

        Args:
            hr_video: (B, N, 3, H, W) - input high resolution images
            lr_video: (B, N, 3, H, W) - input low resolution images
            lr_max_size: int or None - maximum size for low resolution stream (default: 518)
            lr_resolution: (H_lr, W_lr) or None - explicit low resolution size
            hr_max_size: int or None - maximum size for high resolution stream
            hr_num_tokens: int or None - target number of tokens for high resolution stream
            chunk_size: int or None - if provided, use chunked forward pass with this chunk size
            refine: bool - whether to refine points using recovered intrinsics (default: False)
        Returns:
            output_dict: dict containing (all on CPU, batch dimension squeezed):
                - global_points: (N, H, W, 3) - 3D points in global/world space
                - local_points: (N, H, W, 3) - 3D points in local camera space (metric scale applied)
                - camera_poses: (N, 4, 4) - camera poses (metric scale applied)
                - intrinsics: (N, 3, 3) - camera intrinsics
                - metric_scale: (1,) or scalar - metric scale
                - mask: (N, H, W) - binary mask (from confidence > 0.2)
        """

        with torch.amp.autocast(enabled=enable_autocast, dtype=torch.bfloat16, device_type="cuda"):
            if chunk_size is None:
                output = self.forward(
                    hr_video=hr_video,
                    lr_video=lr_video,
                    lr_max_size=lr_max_size,
                    lr_resolution=lr_resolution,
                    hr_max_size=hr_max_size,
                    hr_num_tokens=hr_num_tokens,
                )
            else:
                output = self.forward_chunk(
                    hr_video=hr_video,
                    lr_video=lr_video,
                    lr_max_size=lr_max_size,
                    lr_resolution=lr_resolution,
                    hr_max_size=hr_max_size,
                    hr_num_tokens=hr_num_tokens,
                    chunk_size=chunk_size,
                )

        with torch.autocast(device_type=self.device.type, dtype=torch.float32):

            # NOTE sky masking
            if output.get("conf", None) is None:
                mask = torch.ones_like(output["local_points"][..., 0]).bool()
            else:
                mask = torch.sigmoid(output["conf"]).squeeze(-1) > 0.2

            camera_poses = output["camera_poses"]
            local_points = output["local_points"]
            metric_scale = output.get("metric_scale", None)

            if metric_scale is not None:
                if metric_scale.shape[1] > 1:
                    metric_scale = metric_scale.mean(dim=1, keepdim=True)

                local_points = local_points * metric_scale[..., None, None, None]
                camera_poses[:, :, :3, 3] = (
                    camera_poses[:, :, :3, 3] * metric_scale[..., None]
                )

            output["local_points"] = local_points
            output["mask"] = mask
            output["camera_poses"] = camera_poses
            output["metric_scale"] = metric_scale

            output = self.get_intrinsics(output)


            # NOTE the refine is based on MoGE (https://github.com/microsoft/MoGe/blob/07444410f1e33f402353b99d6ccd26bd31e469e8/moge/model/v2.py#L267) 
            # currently refining point with focal shift create layered artifacts, so we avoid it for now
            if refine:
                print("Refining points with focal shift")
                output = self.refine_points(output)

            global_points = torch.einsum(
                "bnij, bnhwj -> bnhwi",
                output["camera_poses"],
                homogenize_points(output["local_points"]),
            )[..., :3]
            output["global_points"] = global_points

            final_keys = [
                "global_points",
                "local_points",
                "camera_poses",
                "intrinsics",
                "metric_scale",
                "mask",
            ]
            final_output = dict()
            for k in output.keys():
                if isinstance(output[k], torch.Tensor) and k in final_keys:
                    final_output[k] = output[k].squeeze(0)
                    if to_cpu:
                        final_output[k] = final_output[k].cpu()

            return final_output

    def get_intrinsics(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recover camera intrinsics from predicted point map.

        Args:
            output: dict containing:
                - local_points: (B, N, H, W, 3) - 3D points in local camera space
                - mask: (B, N, H, W) - binary mask
                - aspect_ratio: float
                - fov_x: float or None - horizontal field of view (if known)
        Returns:
            output: dict with added:
                - intrinsics: (B, N, 3, 3) - camera intrinsics
                - shift: (B, N) - depth shift for refinement
        """
        points = output["local_points"]
        mask = output["mask"]
        aspect_ratio = output["aspect_ratio"]
        fov_x = output.get("fov_x", None)

        if mask is not None:
            mask_binary = mask > 0.5
        else:
            mask_binary = None

        # Convert affine point map to camera-space. Recover depth and intrinsics from point map.
        # NOTE: Focal here is the focal length relative to half the image diagonal
        if fov_x is None:
            # Recover focal and shift from predicted point map
            focal, shift = recover_focal_shift(points, mask_binary)
        else:
            # Focal is known, recover shift only
            focal = (
                aspect_ratio
                / (1 + aspect_ratio**2) ** 0.5
                / torch.tan(
                    torch.deg2rad(
                        torch.as_tensor(fov_x, device=points.device, dtype=points.dtype)
                        / 2
                    )
                )
            )
            if focal.ndim == 0:
                focal = focal[None].expand(points.shape[0])
            _, shift = recover_focal_shift(points, mask_binary, focal=focal)
        fx, fy = (
            focal / 2 * (1 + aspect_ratio**2) ** 0.5 / aspect_ratio,
            focal / 2 * (1 + aspect_ratio**2) ** 0.5,
        )
        intrinsics = utils3d.torch.intrinsics_from_focal_center(fx, fy, 0.5, 0.5)

        output["intrinsics"] = intrinsics
        output["shift"] = shift

        return output

    def refine_points(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine predicted 3D points using recovered intrinsics.

        Args:
            output: dict containing:
                - local_points: (B, N, H, W, 3) - 3D points in affine space
                - shift: (B, N) - depth shift
                - intrinsics: (B, N, 3, 3) - camera intrinsics
                - mask: (B, N, H, W) - binary mask
        Returns:
            output: dict with refined:
                - local_points: (B, N, H, W, 3) - refined 3D points in camera space
                - mask: (B, N, H, W) - updated mask
        """
        points = output["local_points"]
        shift = output["shift"]
        intrinsics = output["intrinsics"]
        mask = output["mask"]

        refined_points = points.clone()
        refined_points[..., 2] += shift[..., None, None]
        if mask is not None:
            mask &= (
                refined_points[..., 2] > 0
            )  # in case depth is contains negative values (which should never happen in practice)
        refined_depth = refined_points[..., 2].clone()

        refined_points = utils3d.torch.depth_to_points(
            refined_depth, intrinsics=intrinsics
        )

        output["local_points"] = refined_points
        output["mask"] = mask

        return output
