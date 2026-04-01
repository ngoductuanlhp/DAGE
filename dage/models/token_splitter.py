import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class TokenSplitter(nn.Module):
    def __init__(self, in_dim, patch_size=(2,2)):
        super().__init__()
        self.dim_factor = patch_size[0] * patch_size[1]
        # Add LayerNorm before projection for training stability
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim*self.dim_factor),
            nn.SiLU(),
            nn.Linear(in_dim*self.dim_factor, in_dim*self.dim_factor),
        )

    def forward(self, x: torch.Tensor, patch_start_idx: int, patch_hw: Tuple[int, int], B: int, N: int) -> torch.Tensor:
        patch_h, patch_w = patch_hw

        register_tokens = x[:, :patch_start_idx, :]  # (B*N, num_register, D)
        spatial_tokens = x[:, patch_start_idx:, :]   # (B*N, H/p*W/p, D)
        
        # Normalize then project spatial tokens for better alignment with teacher
        spatial_tokens = self.norm(spatial_tokens)
        spatial_tokens = self.proj(spatial_tokens)  # (B*N, H/p/2*W/p/2, D*4)
        
        # Upsample spatial tokens using pixel_shuffle
        spatial_tokens = spatial_tokens.reshape(B*N, patch_h, patch_w, -1)  # (B*N, H/p/2, W/p/2, D*4)
        spatial_tokens = spatial_tokens.permute(0, 3, 1, 2)  # (B*N, D*4, H/p/2, W/p/2)
        spatial_tokens = F.pixel_shuffle(spatial_tokens, upscale_factor=2)  # (B*N, D, H/p, W/p)
        spatial_tokens = spatial_tokens.flatten(2).transpose(1, 2)  # (B*N, H/p*W/p, D)
        
        # Concatenate unchanged register tokens with upsampled spatial tokens
        output = torch.cat([register_tokens, spatial_tokens], dim=1)  # (B*N, num_register+H/p*W/p, D)
        n_tokens = output.shape[1]
        output = output.reshape(B, N*n_tokens, -1) # because teacher output is in shape of (B, N*n_tokens, D)
        return output