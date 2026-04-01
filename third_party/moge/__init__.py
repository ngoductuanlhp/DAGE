import torch
import torch.nn as nn
import sys
from einops import rearrange

# sys.path.append('third_party/moge')
from moge.model.moge_model import MoGeModel
from moge.model.moge_model_v2 import MoGeModelV2


class MoGe(nn.Module):
    
    def __init__(self, model_name='MoGeModel', cache_dir=None, pretrained_path=None):
        super().__init__()
        self.model = eval(model_name).from_pretrained(
        'Ruicheng/moge-vitl' if pretrained_path is None else pretrained_path, 
        cache_dir=cache_dir
        ).eval()


    @torch.no_grad()
    def forward_image(self, image: torch.Tensor, **kwargs):
        # image: b, 3, h, w 0,1
        output = self.model.infer(image, resolution_level=9, apply_mask=False, **kwargs)
        points = output['points'] # b,h,w,3
        masks = output['mask'] # b,h,w
        return points, masks
    
    @torch.no_grad()
    def forward_video(self, video: torch.Tensor, **kwargs):
        # video: b, t, 3, h, w 0,1
        B, T, C, H, W = video.shape
        pseudo_image = video.reshape(B*T, C, H, W)

        total_pseudo_frames = B*T
        chunk_size = 32
        if total_pseudo_frames > chunk_size:
            # chunk_size = 32
            num_chunks = total_pseudo_frames // chunk_size + 1 if total_pseudo_frames % chunk_size != 0 else total_pseudo_frames // chunk_size

            points_list = []
            masks_list = []
            for n in range(num_chunks):
                chunk_start = n * chunk_size
                chunk_end = min(chunk_start + chunk_size, total_pseudo_frames)
                chunk_video = pseudo_image[chunk_start:chunk_end]
                chunk_output = self.model.infer(chunk_video, resolution_level=9, apply_mask=False, **kwargs)
                points_list.append(chunk_output['points'])
                masks_list.append(chunk_output['mask'])
            points = torch.cat(points_list, dim=0)
            masks = torch.cat(masks_list, dim=0)
            points = rearrange(points, '(b t) h w c -> b t h w c', b=B, t=T)
            masks = rearrange(masks, '(b t) h w -> b t h w', b=B, t=T)
        else:
            output = self.model.infer(pseudo_image, resolution_level=9, apply_mask=False, **kwargs)
            points = rearrange(output['points'], '(b t) h w c -> b t h w c', b=B, t=T)
            masks = rearrange(output['mask'], '(b t) h w -> b t h w', b=B, t=T)
        return points, masks
    
class MoGeTemporal(nn.Module):
    
    def __init__(self, pretrained_model_name_or_path='Ruicheng/moge-vitl', model_name='MoGeModelTemporal', cache_dir=None):
        super().__init__()
        self.model = eval(model_name).from_pretrained(
            pretrained_model_name_or_path, 
            cache_dir=cache_dir,
            strict=True, # NOTE for loading image-base model pretrain
            ).eval()


    @torch.no_grad()
    def forward_video(self, video: torch.Tensor, infer_sliding_window: bool = False, resolution_level: int = 9, **kwargs):
        # video: b, t, 3, h, w 0,1
        if infer_sliding_window:
            assert hasattr(self.model, 'infer_sliding_window'), f"Model {type(self.model).__name__} does not have 'infer_sliding_window' method"
            output = self.model.infer_sliding_window(video, resolution_level=9, apply_mask=False, **kwargs)
        else:
            output = self.model.infer(video, resolution_level=resolution_level, apply_mask=False, **kwargs)
        points = output['points'] # b,t,h,w,3
        masks = output['mask'] # b,t,h,w

        if "pose" in output:
            poses = output['pose'] # b,t,7
            return points, masks, poses
        
        return points, masks