"""
Edge-based loss functions for depth/disparity prediction.

Available Loss Classes:
-----------------------
1. MultiScaleCombinedGradientLoss: 
   - Computes both Scharr (1st order) and Laplacian (2nd order) gradients
   - Shares preprocessing (SSI, masking, downsampling) for efficiency
   - Best for comprehensive edge preservation
   
2. MultiScaleLaplaceLoss:
   - Computes only Laplacian (2nd order) gradients  
   - More efficient than Scharr for isotropic edge detection
   
3. MultiScaleDeriLoss:
   - Computes either Scharr or Laplace gradients separately
   - Flexible but less efficient than specialized classes

Efficiency Comparison (operations per scale):
--------------------------------------------
Using Scharr + Laplace separately: 4 convolutions + 2x preprocessing
MultiScaleCombinedGradientLoss:    3 convolutions + 1x preprocessing ⭐ BEST
MultiScaleLaplaceLoss:             1 convolution + 1x preprocessing  ⭐ FASTEST
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def ssi_normalize_depth(depth, mask=None):
    if mask is not None:
        # Only compute statistics over valid pixels
        valid_depth = depth[mask > 0.5]
        if valid_depth.numel() > 0:
            median = torch.median(valid_depth)
            abs_diff = torch.abs(depth - median)
            mean_abs_diff = torch.mean(abs_diff[mask > 0.5])
        else:
            # Fallback if no valid pixels
            median = torch.median(depth)
            abs_diff = torch.abs(depth - median)
            mean_abs_diff = torch.mean(abs_diff)
    else:
        median = torch.median(depth)
        abs_diff = torch.abs(depth - median)  
        mean_abs_diff = torch.mean(abs_diff)
    
    normalized_depth = (depth - median) / (mean_abs_diff + 1e-8)
    return normalized_depth

class TrimMAELoss:
    def __init__(self, trim=0.2):
        self.trim = trim

    def __call__(self, prediction, target):
        res = (prediction - target).abs()
        sorted_res, _ = torch.sort(res.view(-1), descending=False)
        trimmed = sorted_res[: int(len(res) * (1.0 - self.trim))]
        return trimmed.sum() / len(res)

class MultiScaleCombinedGradientLoss(nn.Module):
    """
    Multi-scale combined gradient loss using both Scharr and Laplacian operators.
    
    Efficiently computes both first-order (Scharr) and second-order (Laplacian) gradients
    while sharing preprocessing operations like SSI normalization, masking, and downsampling.
    This avoids redundant computations compared to using separate loss functions.
    
    Args:
        norm (int): Loss norm (1 for L1, 2 for L2). Default: 1
        scales (int): Number of pyramid scales. Default: 6
        trim (bool): Whether to use trimmed MAE loss. Default: False
        ssi (bool): Whether to use scale-shift invariant normalization. Default: False
        amp (bool): Whether to use automatic mixed precision (fp16). Default: False
        scharr_weight (float): Weight for Scharr gradient loss. Default: 1.0
        laplace_weight (float): Weight for Laplacian loss. Default: 1.0
    """
    def __init__(self, norm=1, scales=6, trim=False, ssi=False, amp=False, 
                 scharr_weight=1.0, laplace_weight=1.0):
        super().__init__()
        self.name = "MultiScaleCombinedGradientLoss"
        dtype = torch.float16 if amp else torch.float
        
        # Scharr operators for first-order gradients
        self.scharr_x = torch.tensor([[[[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]]], dtype=dtype).cuda()
        self.scharr_y = torch.tensor([[[[-3, -10, -3], [0, 0, 0], [3, 10, 3]]]], dtype=dtype).cuda()
        
        # Laplace operator for second-order gradients
        self.laplace_op = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=dtype).cuda()
        
        if norm == 1:
            self.loss_function = nn.L1Loss(reduction='mean')
        elif norm == 2:
            self.loss_function = nn.MSELoss(reduction='mean')
        if trim:
            self.loss_function = TrimMAELoss()
        
        self.ssi = ssi
        self.scales = scales
        self.scharr_weight = scharr_weight
        self.laplace_weight = laplace_weight
        
        # Create 3x3 kernel of ones for mask convolution
        self.mask_kernel = torch.ones((1, 1, 3, 3), dtype=dtype).cuda()

    def compute_gradients(self, input_tensor):
        """
        Compute both Scharr and Laplacian gradients efficiently.
        
        Args:
            input_tensor: Input tensor [B, C, H, W]
            
        Returns:
            tuple: (scharr_x, scharr_y, laplacian)
        """
        groups = input_tensor.shape[1]
        
        # Scharr gradients
        scharr_x_op = self.scharr_x.repeat(groups, 1, 1, 1)
        scharr_y_op = self.scharr_y.repeat(groups, 1, 1, 1)
        grad_x = F.conv2d(input_tensor, scharr_x_op, groups=groups)
        grad_y = F.conv2d(input_tensor, scharr_y_op, groups=groups)
        
        # Laplacian
        laplace_op = self.laplace_op.repeat(groups, 1, 1, 1)
        laplacian = F.conv2d(input_tensor, laplace_op, groups=groups)
        
        return grad_x, grad_y, laplacian

    def forward(self, prediction, target, mask=None):
        # Apply SSI normalization once for both operators
        if self.ssi:
            prediction_ = ssi_normalize_depth(prediction, mask)
            target_ = ssi_normalize_depth(target, mask)
        else:
            prediction_ = prediction
            target_ = target

        mask_ = mask
        
        total_scharr_loss = 0.0
        total_laplace_loss = 0.0
        
        for scale in range(self.scales):
            # Compute all gradients once
            pred_grad_x, pred_grad_y, pred_laplacian = self.compute_gradients(prediction_)
            tgt_grad_x, tgt_grad_y, tgt_laplacian = self.compute_gradients(target_)
            
            if mask_ is not None:
                # Detect and filter out unusual pixels once
                unusual_mask = torch.zeros_like(mask_, dtype=torch.bool)
                
                # Check for invalid values (NaN, Inf)
                unusual_mask = unusual_mask | ~torch.isfinite(prediction_)
                unusual_mask = unusual_mask | ~torch.isfinite(target_)
                
                # Check for near-zero or negative depths
                if not self.ssi:
                    unusual_mask = unusual_mask | (prediction_ <= 1e-3)
                    unusual_mask = unusual_mask | (target_ <= 1e-3)
                    unusual_mask = unusual_mask | (prediction_ > 1e4)
                    unusual_mask = unusual_mask | (target_ > 1e4)
                else:
                    unusual_mask = unusual_mask | (torch.abs(prediction_) > 50.0)
                    unusual_mask = unusual_mask | (torch.abs(target_) > 50.0)
                
                # Update mask to exclude unusual pixels
                mask_ = mask_ & ~unusual_mask
                
                # Apply 3x3 convolution to ensure all pixels in patch are valid
                mask_float = mask_.float()
                groups = mask_float.shape[1]
                mask_kernel_repeated = self.mask_kernel.repeat(groups, 1, 1, 1)
                grad_mask = F.conv2d(mask_float, mask_kernel_repeated, groups=groups)
                grad_mask = grad_mask >= 9.0
                
                # Compute Scharr loss
                loss_x = torch.abs(pred_grad_x - tgt_grad_x) * grad_mask
                loss_y = torch.abs(pred_grad_y - tgt_grad_y) * grad_mask
                
                # Compute Laplacian loss
                loss_laplacian = torch.abs(pred_laplacian - tgt_laplacian) * grad_mask
                
                # Compute mean over valid pixels
                valid_count = grad_mask.sum()
                if valid_count > 0:
                    scharr_loss = (loss_x.sum() + loss_y.sum()) / valid_count
                    laplace_loss = loss_laplacian.sum() / valid_count
                    
                    total_scharr_loss += scharr_loss.clamp_max(5.0)
                    total_laplace_loss += laplace_loss.clamp_max(5.0)
            else:
                # No mask - use loss function directly
                scharr_loss = self.loss_function(pred_grad_x, tgt_grad_x) + \
                             self.loss_function(pred_grad_y, tgt_grad_y)
                laplace_loss = self.loss_function(pred_laplacian, tgt_laplacian)
                
                total_scharr_loss += scharr_loss
                total_laplace_loss += laplace_loss
            
            # Downsample once for next scale
            prediction_ = F.interpolate(prediction_, scale_factor=0.5)
            target_ = F.interpolate(target_, scale_factor=0.5)
            if mask_ is not None:
                mask_ = F.interpolate(mask_.float(), scale_factor=0.5, mode='nearest')
                mask_ = mask_ > 0.5
        
        # Combine losses with weights
        avg_scharr_loss = total_scharr_loss / self.scales
        avg_laplace_loss = total_laplace_loss / self.scales
        
        combined_loss = (self.scharr_weight * avg_scharr_loss + 
                        self.laplace_weight * avg_laplace_loss)
        
        return combined_loss


class MultiScaleLaplaceLoss(nn.Module):
    """
    Multi-scale Laplacian loss for depth/disparity prediction.
    
    Uses the Laplace operator to compute second-order derivatives. Since the Laplace
    operator is isotropic (same kernel for x and y directions), we only need to compute
    one Laplacian instead of separate x and y gradients, making it more efficient than
    MultiScaleDeriLoss with Laplace operator.
    
    Args:
        norm (int): Loss norm (1 for L1, 2 for L2). Default: 1
        scales (int): Number of pyramid scales. Default: 6
        trim (bool): Whether to use trimmed MAE loss. Default: False
        ssi (bool): Whether to use scale-shift invariant normalization. Default: False
        amp (bool): Whether to use automatic mixed precision (fp16). Default: False
    """
    def __init__(self, norm=1, scales=6, trim=False, ssi=False, amp=False):
        super().__init__()
        self.name = "MultiScaleLaplaceLoss"
        dtype = torch.float16 if amp else torch.float
        # Laplace operator - same for x and y (isotropic)
        self.laplace_op = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=dtype).cuda()
        if norm == 1:
            self.loss_function = nn.L1Loss(reduction='mean')
        elif norm == 2:
            self.loss_function = nn.MSELoss(reduction='mean')
        if trim:
            self.loss_function = TrimMAELoss()
        self.ssi = ssi
        self.scales = scales
        # Create 3x3 kernel of ones for mask convolution
        self.mask_kernel = torch.ones((1, 1, 3, 3), dtype=dtype).cuda()

    def compute_laplacian(self, input_tensor):
        """
        Compute Laplacian (second-order derivative) using convolution.
        
        Args:
            input_tensor: Input tensor [B, C, H, W]
            
        Returns:
            Laplacian of input [B, C, H-2, W-2]
        """
        groups = input_tensor.shape[1]
        laplace_op = self.laplace_op.repeat(groups, 1, 1, 1)
        laplacian = F.conv2d(input_tensor, laplace_op, groups=groups)
        return laplacian

    def forward(self, prediction, target, mask=None):
        if self.ssi:
            prediction_ = ssi_normalize_depth(prediction, mask)
            target_ = ssi_normalize_depth(target, mask)
        else:
            prediction_ = prediction
            target_ = target

        mask_ = mask
            
        total_loss = 0.0
        for scale in range(self.scales):
            laplacian_prediction = self.compute_laplacian(prediction_)
            laplacian_target = self.compute_laplacian(target_)
            
            if mask_ is not None:
                # Detect and filter out unusual pixels
                unusual_mask = torch.zeros_like(mask_, dtype=torch.bool)
                
                # Check for invalid values (NaN, Inf)
                unusual_mask = unusual_mask | ~torch.isfinite(prediction_)
                unusual_mask = unusual_mask | ~torch.isfinite(target_)
                
                # Check for near-zero or negative depths
                if not self.ssi:
                    unusual_mask = unusual_mask | (prediction_ <= 1e-3)
                    unusual_mask = unusual_mask | (target_ <= 1e-3)
                    unusual_mask = unusual_mask | (prediction_ > 1e4)
                    unusual_mask = unusual_mask | (target_ > 1e4)
                else:
                    unusual_mask = unusual_mask | (torch.abs(prediction_) > 50.0)
                    unusual_mask = unusual_mask | (torch.abs(target_) > 50.0)
                
                # Update mask to exclude unusual pixels
                mask_ = mask_ & ~unusual_mask
                
                # Apply 3x3 convolution to ensure all pixels in patch are valid
                mask_float = mask_.float()
                groups = mask_float.shape[1]
                mask_kernel_repeated = self.mask_kernel.repeat(groups, 1, 1, 1)
                grad_mask = F.conv2d(mask_float, mask_kernel_repeated, groups=groups)
                grad_mask = grad_mask >= 9.0
                
                # Apply mask to Laplacian
                loss_laplacian = torch.abs(laplacian_prediction - laplacian_target) * grad_mask
                
                # Compute mean over valid pixels only
                valid_count = grad_mask.sum()
                if valid_count > 0:
                    loss_mean = loss_laplacian.sum() / valid_count
                    total_loss += loss_mean.clamp_max(5.0)
            else:
                loss_laplacian = self.loss_function(laplacian_prediction, laplacian_target)
                total_loss += loss_laplacian
            
            # Downsample for next scale
            prediction_ = F.interpolate(prediction_, scale_factor=0.5)
            target_ = F.interpolate(target_, scale_factor=0.5)
            if mask_ is not None:
                mask_ = F.interpolate(mask_.float(), scale_factor=0.5, mode='nearest')
                mask_ = mask_ > 0.5
                
        return total_loss / self.scales


class MultiScaleDeriLoss(nn.Module):
    def __init__(self, operator='Scharr', norm=1, scales=6, trim=False, ssi=False, amp=False):
        super().__init__()
        self.name = "MultiScaleDerivativeLoss"
        self.operator = operator
        dtype = torch.float16 if amp else torch.float
        self.operators = {
            "Scharr": {
                'x': torch.tensor([[[[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]]], dtype=dtype).cuda(),
                'y': torch.tensor([[[[-3, -10, -3], [0, 0, 0], [3, 10, 3]]]], dtype=dtype).cuda(),
            },
            "Laplace": {
                'x': torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=dtype).cuda(),
                'y': torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=dtype).cuda(),
            }
        }
        self.op_x = self.operators[operator]['x']
        self.op_y = self.operators[operator]['y']
        if norm == 1:
            self.loss_function = nn.L1Loss(reduction='mean')
        elif norm == 2:
            self.loss_function == nn.MSELoss(reduction='mean')
        if trim:
            self.loss_function = TrimMAELoss()
        self.ssi = ssi
        self.scales = scales
        # Create 3x3 kernel of ones for mask convolution (to check if all pixels in patch are valid)
        self.mask_kernel = torch.ones((1, 1, 3, 3), dtype=dtype).cuda()

    def gradients(self, input_tensor):
        op_x, op_y = self.op_x, self.op_y
        groups = input_tensor.shape[1]
        op_x = op_x.repeat(groups, 1, 1, 1)
        op_y = op_y.repeat(groups, 1, 1, 1)
        grad_x = F.conv2d(input_tensor, op_x, groups=groups)
        grad_y = F.conv2d(input_tensor, op_y, groups=groups)
        return grad_x, grad_y

    def forward(self, prediction, target, mask=None):
        if self.ssi:
            prediction_ = ssi_normalize_depth(prediction, mask)
            target_ = ssi_normalize_depth(target, mask)
        else:
            prediction_ = prediction
            target_ = target
        # prediction_ = prediction_.unsqueeze(0)
        # target_ = target_.unsqueeze(0)

        mask_ = mask
            
        total_loss = torch.tensor(0.0, device=prediction_.device, requires_grad=True)
        for scale in range(self.scales):
            grad_prediction_x, grad_prediction_y = self.gradients(prediction_)
            grad_target_x, grad_target_y = self.gradients(target_)
            
            if mask_ is not None:
                # Detect and filter out unusual pixels before gradient computation
                unusual_mask = torch.zeros_like(mask_, dtype=torch.bool)
                
                # Check for invalid values (NaN, Inf)
                unusual_mask = unusual_mask | ~torch.isfinite(prediction_)
                unusual_mask = unusual_mask | ~torch.isfinite(target_)
                
                # Check for near-zero or negative depths (numerical instability)
                if not self.ssi:  # Only check depth values if not using SSI normalization
                    unusual_mask = unusual_mask | (prediction_ <= 1e-3)
                    unusual_mask = unusual_mask | (target_ <= 1e-3)
                    unusual_mask = unusual_mask | (prediction_ > 1e4)  # Extremely large values
                    unusual_mask = unusual_mask | (target_ > 1e4)
                else:
                    # For SSI normalized values, check for extreme normalized values
                    unusual_mask = unusual_mask | (torch.abs(prediction_) > 50.0)
                    unusual_mask = unusual_mask | (torch.abs(target_) > 50.0)
                
                # Update mask to exclude unusual pixels
                mask_ = mask_ & ~unusual_mask
                
                # Apply 3x3 convolution to mask to check if all pixels in each patch are valid
                # This ensures gradient is only valid if ALL pixels in the 3x3 patch are valid
                mask_float = mask_.float()
                groups = mask_float.shape[1]
                mask_kernel_repeated = self.mask_kernel.repeat(groups, 1, 1, 1)
                grad_mask = F.conv2d(mask_float, mask_kernel_repeated, groups=groups)
                # grad_mask == 9 means all 9 pixels in the patch are valid (1)
                grad_mask = grad_mask >= 9.0
                
                # Apply mask to gradients
                loss_x = torch.abs(grad_prediction_x - grad_target_x) * grad_mask
                loss_y = torch.abs(grad_prediction_y - grad_target_y) * grad_mask
                
                # Compute mean over valid pixels only
                valid_count = grad_mask.sum()
                if valid_count > 0:
                    loss_x_mean = loss_x.sum() / valid_count
                    loss_y_mean = loss_y.sum() / valid_count
                    # print(f"scale {2**scale}: loss_x_mean {loss_x_mean}, loss_y_mean {loss_y_mean}")
                    total_loss = total_loss + (loss_x_mean.clamp_max(5.0) + loss_y_mean.clamp_max(5.0))
                    # total_loss += (loss_x.sum() + loss_y.sum()) / valid_count
            else:
                loss_x = self.loss_function(grad_prediction_x, grad_target_x)
                loss_y = self.loss_function(grad_prediction_y, grad_target_y)
                total_loss = total_loss + torch.mean(loss_x + loss_y)
            
            prediction_ = F.interpolate(prediction_, scale_factor=0.5)
            target_ = F.interpolate(target_, scale_factor=0.5)
            if mask_ is not None:
                mask_ = F.interpolate(mask_.float(), scale_factor=0.5, mode='nearest')
                mask_ = mask_ > 0.5

        total_loss = total_loss / self.scales
        return total_loss


