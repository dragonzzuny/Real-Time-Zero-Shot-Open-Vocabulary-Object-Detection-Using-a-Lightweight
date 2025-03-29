# yolo_clip_detector/loss/iou_loss.py 파일 내용
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging
import math

logger = logging.getLogger(__name__)

class IoULoss(nn.Module):
    """
    IoU (Intersection over Union) Loss for bounding box regression
    
    Supports multiple IoU variants:
    - 'iou': Standard IoU
    - 'giou': Generalized IoU
    - 'diou': Distance IoU
    - 'ciou': Complete IoU
    """
    
    def __init__(self, 
                 iou_type: str = 'ciou', 
                 reduction: str = 'mean',
                 eps: float = 1e-7):
        """
        Initialize the IoULoss.
        
        Args:
            iou_type: Type of IoU to use ('iou', 'giou', 'diou', 'ciou')
            reduction: Reduction method ('none', 'mean', 'sum')
            eps: Small epsilon value for numerical stability
        """
        super().__init__()
        self.iou_type = iou_type.lower()
        self.reduction = reduction
        self.eps = eps
        
        assert self.iou_type in ['iou', 'giou', 'diou', 'ciou'], f"Unknown IoU type: {self.iou_type}"
        assert self.reduction in ['none', 'mean', 'sum'], f"Unknown reduction method: {self.reduction}"
        
        logger.info(f"IoULoss initialized with iou_type={iou_type}, reduction={reduction}")
    
    def forward(self, 
                pred_boxes: torch.Tensor, 
                target_boxes: torch.Tensor, 
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the IoU loss between predicted and target boxes.
        
        Args:
            pred_boxes: Predicted boxes of shape (batch_size, num_boxes, 4) in (x1, y1, x2, y2) format
            target_boxes: Target boxes of shape (batch_size, num_boxes, 4) in (x1, y1, x2, y2) format
            weights: Optional weights for each box of shape (batch_size, num_boxes)
            
        Returns:
            Loss value
        """
        # Compute IoU and loss
        iou, loss = self._compute_iou_loss(pred_boxes, target_boxes)
        
        # Apply weights if provided
        if weights is not None:
            # 디버깅용 로깅 추가
            logger.info(f"Loss shape: {loss.shape}, weights shape: {weights.shape}")
            
            # 텐서 차원을 고려하여 올바른 dimension에서 크기를 확인
            if loss.dim() == 3 and weights.dim() == 2:
                if weights.shape[1] != loss.shape[1]:
                    logger.info(f"Reshaping weights from {weights.shape} to match loss shape {loss.shape}")
                    if weights.shape[1] > loss.shape[1]:
                        weights = weights[:, :loss.shape[1]]
                    else:
                        padding = torch.zeros(
                            weights.shape[0], 
                            loss.shape[1] - weights.shape[1],
                            device=weights.device,
                            dtype=weights.dtype
                        )
                        weights = torch.cat([weights, padding], dim=1)
                
                # 차원 맞추기
                weights = weights.unsqueeze(-1)
            
            # 만약 크기가 여전히 맞지 않는다면, 경고 출력 후 weights 없이 진행
            if weights.shape[1] != loss.shape[1]:
                logger.warning(f"Weights shape {weights.shape} still doesn't match loss shape {loss.shape}. Ignoring weights.")
            else:
                loss = loss * weights
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
    
    def _compute_iou_loss(self, 
                         pred_boxes: torch.Tensor, 
                         target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU and the corresponding loss.
        
        Args:
            pred_boxes: Predicted boxes in (x1, y1, x2, y2) format
            target_boxes: Target boxes in (x1, y1, x2, y2) format
            
        Returns:
            Tuple of (IoU, Loss)
        """
        # Extract coordinates
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes.unbind(-1)
        target_x1, target_y1, target_x2, target_y2 = target_boxes.unbind(-1)
        
        # Calculate area of predicted and target boxes
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        
        # Calculate intersection area
        x1_max = torch.maximum(pred_x1, target_x1)
        y1_max = torch.maximum(pred_y1, target_y1)
        x2_min = torch.minimum(pred_x2, target_x2)
        y2_min = torch.minimum(pred_y2, target_y2)
        
        width = (x2_min - x1_max).clamp(min=0)
        height = (y2_min - y1_max).clamp(min=0)
        intersection = width * height
        
        # Calculate union area
        union = pred_area + target_area - intersection
        
        # Calculate IoU
        iou = intersection / (union + self.eps)
        
        # Standard IoU loss
        loss = 1 - iou
        
        if self.iou_type == 'iou':
            return iou, loss
        
        # Calculate coordinates for enclosing box (for GIoU)
        x1_min = torch.minimum(pred_x1, target_x1)
        y1_min = torch.minimum(pred_y1, target_y1)
        x2_max = torch.maximum(pred_x2, target_x2)
        y2_max = torch.maximum(pred_y2, target_y2)
        
        if self.iou_type == 'giou':
            # Calculate enclosing area
            enclosing_area = (x2_max - x1_min) * (y2_max - y1_min)
            
            # Calculate GIoU
            giou = iou - (enclosing_area - union) / (enclosing_area + self.eps)
            
            # GIoU loss
            loss = 1 - giou
            return iou, loss
        
        # Calculate centers and diagonals (for DIoU/CIoU)
        pred_cx = (pred_x1 + pred_x2) / 2
        pred_cy = (pred_y1 + pred_y2) / 2
        target_cx = (target_x1 + target_x2) / 2
        target_cy = (target_y1 + target_y2) / 2
        
        # Calculate squared distance between centers
        center_dist_squared = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        
        # Calculate squared diagonal length of enclosing box
        diagonal_squared = (x2_max - x1_min) ** 2 + (y2_max - y1_min) ** 2
        
        if self.iou_type == 'diou':
            # Calculate DIoU
            diou = iou - center_dist_squared / (diagonal_squared + self.eps)
            
            # DIoU loss
            loss = 1 - diou
            return iou, loss
        
        if self.iou_type == 'ciou':
            # Calculate aspect ratio consistency term
            pred_w = pred_x2 - pred_x1
            pred_h = pred_y2 - pred_y1
            target_w = target_x2 - target_x1
            target_h = target_y2 - target_y1
            
            pred_aspect = torch.atan(pred_w / (pred_h + self.eps))
            target_aspect = torch.atan(target_w / (target_h + self.eps))
            
            v = (4 / (math.pi ** 2)) * (pred_aspect - target_aspect) ** 2
            alpha = v / (1 - iou + v + self.eps)
            
            # Calculate CIoU
            ciou = iou - (center_dist_squared / (diagonal_squared + self.eps) + alpha * v)
            
            # CIoU loss
            loss = 1 - ciou
            return iou, loss
        
        # Default fallback to IoU loss (should never reach here)
        return iou, loss