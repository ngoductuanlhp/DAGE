import torch
from einops import rearrange, einsum

def get_scale_multiplier(
    intrinsics: torch.Tensor,  # "*#batch 3 3"
    pixel_size: torch.Tensor,  # "*#batch 2"
    multiplier: float = 0.1,
) -> torch.Tensor:  # " *batch"
    xy_multipliers = multiplier * einsum(
        intrinsics[..., :2, :2].float().inverse().to(intrinsics),
        pixel_size,
        "... i j, j -> ... i",
    )
    return xy_multipliers.sum(dim=-1)

    
def reg_dense_offsets(xyz, shift=6.0):
    d = xyz.norm(dim=-1, keepdim=True)
    return xyz / d.clamp(min=1e-8) * (torch.exp(d - shift) - torch.exp(-shift))

def reg_dense_scales(scales):
    return scales.exp()

def reg_dense_scales_da3(scales, depths, scale_min, scale_max, intr_normed, pixel_size, W, H):
    dtype, device = scales.dtype, scales.device
    scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
    pixel_size = 1 / torch.tensor((W, H), dtype=dtype, device=device)
    multiplier = get_scale_multiplier(intr_normed, pixel_size)
    gs_scales = scales * depths[..., None] * multiplier[..., None, None, None]
    gs_scales = rearrange(gs_scales, "b v h w d -> b (v h w) d")
    return gs_scales

def reg_dense_rotation(rotations, eps=1e-8):
    return rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

def reg_dense_sh(sh):
    return rearrange(sh, '... (d_sh xyz) -> ... d_sh xyz', xyz=3)

def reg_dense_opacities(opacities):
    return opacities.sigmoid()

def reg_dense_weights(weights):
    return weights.sigmoid()
