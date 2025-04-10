# yolo_clip_detector/loss/dfl_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class DistributedFocalLoss(nn.Module):
    """
    Distributed Focal Loss (DFL) for bounding box regression
    
    This loss function is used to learn a distribution over a set of possible values
    rather than directly regressing to a single value.
    """
    
    def __init__(self, 
                 reg_max: int = 16, 
                 reduction: str = 'mean'):
        """
        Initialize the DistributedFocalLoss.
        
        Args:
            reg_max: Maximum value for regression distribution
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super().__init__()
        self.reg_max = reg_max
        self.reduction = reduction
        
        # Project tensor used for converting predictions to values
        self.project = torch.linspace(0, self.reg_max, self.reg_max + 1)
        
        logger.info(f"DistributedFocalLoss initialized with reg_max={reg_max}")
    
    def to(self, device):
        """Move loss module to specified device"""
        super().to(device)
        self.project = self.project.to(device)
        return self
    
    def forward(self, 
                pred_dfl: torch.Tensor, 
                target_dfl: torch.Tensor, 
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the DFL loss.
        
        Args:
            pred_dfl: Predicted distribution logits [batch_size, reg_max+1, ...]
            target_dfl: Target distribution indices [batch_size, ...]
            weights: Optional weights for loss calculation
            
        Returns:
            Loss value
        """
        # Ensure project tensor is on the same device
        if self.project.device != pred_dfl.device:
            self.project = self.project.to(pred_dfl.device)
        
        # Prepare target one-hot encoding
        target_dfl = target_dfl.long().clamp(0, self.reg_max)
        target_dfl_one_hot = F.one_hot(target_dfl, self.reg_max + 1).float()
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(
            pred_dfl.reshape(-1, self.reg_max + 1),
            target_dfl.reshape(-1),
            reduction='none'
        ).reshape(target_dfl.shape)
        
        # Apply weights if provided
        if weights is not None:
            loss = loss * weights
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

    def get_target(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Generate DFL targets from ground truth boxes.
        
        Args:
            pred_boxes: Predicted boxes in normalized format
            target_boxes: Target boxes in normalized format
            
        Returns:
            DFL targets tensor
        """
        # Calculate differences between predicted and target boxes
        # This is a simplified version - a real implementation would have more
        # sophisticated target assignment
        
        # For centers (xy), we calculate offsets
        xy_offsets = target_boxes[..., :2] - pred_boxes[..., :2]
        
        # For sizes (wh), we calculate log ratios
        wh_offsets = torch.log(target_boxes[..., 2:] / pred_boxes[..., 2:] + 1e-6)
        
        # Combine offsets
        offsets = torch.cat([xy_offsets, wh_offsets], dim=-1)
        
        # Scale to [0, reg_max]
        scaled_offsets = (offsets * self.reg_max).clamp(0, self.reg_max)
        
        # Round to integers
        targets = scaled_offsets.long()
        
        return targets
    
    def predict_from_dfl(self, pred_dfl: torch.Tensor) -> torch.Tensor:
        """
        Convert DFL predictions to box coordinates.
        
        Args:
            pred_dfl: DFL predictions tensor [batch_size, 4*(reg_max+1), ...]
            
        Returns:
            Box coordinates tensor
        """
        # Apply softmax to get distributions
        pred_dfl = pred_dfl.reshape(-1, 4, self.reg_max + 1)
        pred_dfl = F.softmax(pred_dfl, dim=-1)
        
        # Calculate expected values
        pred_dfl_proj = pred_dfl @ self.project.type_as(pred_dfl)
        
        # Reshape to original dimensions
        original_shape = list(pred_dfl.shape[:-2]) + [4]
        return pred_dfl_proj.reshape(original_shape)