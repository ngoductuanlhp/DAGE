import torch
from matplotlib import cm

def robust_min_max(tensor, quantile=0.99):
    T, H, W = tensor.shape
    min_vals = []
    max_vals = []
    for i in range(T):
        min_vals.append(torch.quantile(tensor[i], q=1-quantile, interpolation='nearest').item())
        max_vals.append(torch.quantile(tensor[i], q=quantile, interpolation='nearest').item())
    return min(min_vals), max(max_vals) 


class ColorMapper:
    def __init__(self, colormap: str = "inferno"):
        cmap = cm.get_cmap(colormap)
        # Handle both ListedColormap and LinearSegmentedColormap
        if hasattr(cmap, 'colors'):
            self.colormap = torch.tensor(cmap.colors)
        else:
            # Sample the colormap at 256 points for LinearSegmentedColormap
            import numpy as np
            self.colormap = torch.tensor(cmap(np.linspace(0, 1, 256))[:, :3])

    def apply(self, image: torch.Tensor, v_min=None, v_max=None):
        # assert len(image.shape) == 2
        if v_min is None:
            v_min = image.min()
        if v_max is None:
            v_max = image.max()
        image = (image - v_min) / (v_max - v_min)
        image = (image * 255).long()
        colormap = self.colormap.to(image.device)
        image = colormap[image]
        return image

def color_video_disp(disp, colormap='inferno'):
    visualizer = ColorMapper(colormap=colormap)
    disp_img = visualizer.apply(disp, v_min=0, v_max=1)
    return disp_img     

def pmap_to_disp(point_maps, valid_masks, gamma=0.5, method='gamma'):
    """
    Convert point maps to colored disparity visualization.
    
    Args:
        point_maps: (T, H, W, 3) point maps
        valid_masks: (T, H, W) boolean mask
        gamma: Power for gamma correction. Values < 1 brighten dark regions (far objects).
               Recommended: 0.5 for balanced, 0.7 for subtle, 0.3 for aggressive brightening
        method: Brightness enhancement method
               - 'gamma': Power transform (default)
               - 'log': Logarithmic scaling (good for wide range)
               - 'sqrt': Square root (moderate brightening)
               - 'percentile': Clip extreme values (reduces dynamic range)
    """
    disp_map = 1.0 / (point_maps[..., 2] + 1e-4)
    min_disparity, max_disparity = robust_min_max(disp_map)
    disp_map = torch.clamp((disp_map - min_disparity) / (max_disparity - min_disparity+1e-4), 0, 1)
    
    # Apply brightness enhancement
    if method == 'gamma' and gamma != 1.0:
        # Gamma correction: brightens dark regions (low values)
        disp_map = torch.pow(disp_map, gamma)
    elif method == 'log':
        # Logarithmic: strongly compresses high values, expands low values
        disp_map = torch.log(1 + disp_map * 9) / torch.log(torch.tensor(10.0))
    elif method == 'sqrt':
        # Square root: moderate brightening of dark regions
        disp_map = torch.sqrt(disp_map)
    elif method == 'percentile':
        # Percentile-based: clip outliers to reduce dynamic range
        p_low, p_high = 0.02, 0.98
        v_low = torch.quantile(disp_map[valid_masks], p_low)
        v_high = torch.quantile(disp_map[valid_masks], p_high)
        disp_map = torch.clamp((disp_map - v_low) / (v_high - v_low + 1e-4), 0, 1)
    
    disp_map = color_video_disp(disp_map)
    disp_map[~valid_masks] = 0
    return disp_map
    # imageio.mimsave(os.path.join(args.save_dir, os.path.basename(args.data[:-4])+'_disp.mp4'), disp, fps=24, quality=9, macro_block_size=1)

def pmap_to_depth(point_maps, valid_masks, gamma=0.5, method='gamma'):
    """
    Convert point maps to colored disparity visualization.
    
    Args:
        point_maps: (T, H, W, 3) point maps
        valid_masks: (T, H, W) boolean mask
        gamma: Power for gamma correction. Values < 1 brighten dark regions (far objects).
               Recommended: 0.5 for balanced, 0.7 for subtle, 0.3 for aggressive brightening
        method: Brightness enhancement method
               - 'gamma': Power transform (default)
               - 'log': Logarithmic scaling (good for wide range)
               - 'sqrt': Square root (moderate brightening)
               - 'percentile': Clip extreme values (reduces dynamic range)
    """
    depth_map = point_maps[..., 2]
    min_depth, max_depth = robust_min_max(depth_map)
    depth_map = torch.clamp((depth_map - min_depth) / (max_depth - min_depth+1e-4), 0, 1)
    depth_map = color_video_disp(depth_map, colormap='Spectral_r')
    depth_map[~valid_masks] = 0
    return depth_map

def create_temporal_slice(video_frames, col_idx=None):
    """
    Create a temporal slice visualization by extracting one column from each frame
    and concatenating them horizontally.
    
    Args:
        video_frames: (T, H, W, 3) tensor of colored frames
        col_idx: Column index to extract, or None for W//2 (center)
        
    Returns:
        Tensor of shape (H, T, 3) showing temporal evolution
    """
    T, H, W, C = video_frames.shape
    
    # Default to center column
    if col_idx is None:
        col_idx = W // 2
    
    # Extract column col_idx from all frames: (T, H, 3)
    column_slice = video_frames[:, :, col_idx, :]  # (T, H, 3)
    
    # Transpose to (H, T, 3)
    temporal_slice = column_slice.transpose(0, 1)  # (H, T, 3)
    
    return temporal_slice
