# yolo_clip_detector/model/heads/box_head.py 파일 내용
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ConvBlock(nn.Module):
    """Standard convolution block with BatchNorm and SiLU activation"""
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 3, 
                 stride: int = 1, 
                 padding: Optional[int] = None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class BoxHead(nn.Module):
    """
    Box Regression Head for YOLO-CLIP detector
    
    This head is responsible for generating bounding box predictions.
    """
    
    def __init__(self, 
                 in_channels: List[int], 
                 hidden_dim: int = 256, 
                 reg_max: int = 16,
                 strides: List[int] = [8, 16, 32]):
        """
        Initialize the BoxHead.
        
        Args:
            in_channels: List of input channels for each feature level
            hidden_dim: Hidden dimension in the head
            reg_max: Maximum value for DFL (Distributed Focal Loss)
            strides: Strides for each feature level
        """
        super().__init__()
        
        # Create bounding box regression heads for each feature level
        self.box_convs = nn.ModuleList([
            nn.Sequential(
                ConvBlock(in_channel, hidden_dim, kernel_size=3),
                ConvBlock(hidden_dim, hidden_dim, kernel_size=3),
                nn.Conv2d(hidden_dim, 4 * (reg_max + 1), kernel_size=1)  # 4 for xywh, (reg_max+1) for DFL
            ) for in_channel in in_channels
        ])
        
        self.reg_max = reg_max
        self.strides = strides
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"BoxHead initialized with reg_max={reg_max}, strides={strides}")
    
    def _init_weights(self):
        """Initialize weights for the network"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Initialize convolution weights with kaiming normal
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through the BoxHead.
        
        Args:
            features: List of feature tensors from the backbone/neck
            
        Returns:
            Tuple of (List of bounding box predictions, List of grid information)
        """
        assert len(features) == len(self.box_convs), \
            f"Expected {len(self.box_convs)} feature maps, got {len(features)}"
        
        box_preds = []
        grids = []
        
        # Process each feature level
        for feat, box_conv, stride in zip(features, self.box_convs, self.strides):
            # Get feature map size
            batch_size, _, height, width = feat.shape
            
            # Apply box convolutions
            pred = box_conv(feat)
            
            # Create grid information for this level
            grid = self._create_grid(batch_size, height, width, stride, feat.device)
            
            box_preds.append(pred)
            grids.append(grid)
        
        return box_preds, grids
    
    def _create_grid(self, 
                     batch_size: int, 
                     height: int, 
                     width: int, 
                     stride: int, 
                     device: torch.device) -> torch.Tensor:
        """
        Create grid information for a feature level.
        
        Args:
            batch_size: Batch size
            height: Feature map height
            width: Feature map width
            stride: Feature map stride
            device: Device to create the grid on
            
        Returns:
            Grid information of shape (batch_size, height, width, 3)
            where the last dimension is (grid_cell_x, grid_cell_y, stride)
        """
        # Create grid coordinates
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing='ij'
        )
        
        # Create grid information: (x, y, stride)
        grid = torch.stack([grid_x, grid_y, torch.ones_like(grid_x) * stride], dim=-1)
        
        # Expand for batch dimension
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        return grid
    
    def decode_boxes(self, 
                     box_preds: List[torch.Tensor], 
                     grids: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode bounding box predictions into coordinates.
        
        Args:
            box_preds: List of box predictions from forward pass
            grids: List of grid information from forward pass
            
        Returns:
            Decoded boxes of shape (batch_size, total_predictions, 4)
        """
        all_boxes = []
        
        for pred, grid in zip(box_preds, grids):
            batch_size, _, height, width = pred.shape
            
            # Reshape to (batch_size, 4, reg_max+1, height, width)
            pred = pred.reshape(batch_size, 4, self.reg_max + 1, height, width)
            
            # Apply softmax to get distribution over reg_max+1 bins
            pred = pred.softmax(dim=2)
            
            # Project distribution to value using DFL
            reg_vals = pred * torch.arange(0, self.reg_max + 1, device=pred.device).float()
            reg_vals = reg_vals.sum(dim=2)  # (batch_size, 4, height, width)
            
            # Get grid information
            grid_xy = grid[..., :2]  # (batch_size, height, width, 2)
            stride = grid[..., 2:3]  # (batch_size, height, width, 1)
            
            # Permute reg_vals to match grid shape
            reg_vals = reg_vals.permute(0, 2, 3, 1)  # (batch_size, height, width, 4)
            
            # Decode boxes
            # x_center = (grid_x + offset_x) * stride
            # y_center = (grid_y + offset_y) * stride
            # w = exp(reg_vals[2]) * stride
            # h = exp(reg_vals[3]) * stride
            xy_center = (grid_xy + reg_vals[..., :2]) * stride
            wh = torch.exp(reg_vals[..., 2:]) * stride
            
            # Convert to (x1, y1, x2, y2) format
            boxes = torch.cat([
                xy_center - wh / 2,  # top-left (x1, y1)
                xy_center + wh / 2   # bottom-right (x2, y2)
            ], dim=-1)
            
            # Reshape to (batch_size, height*width, 4)
            boxes = boxes.reshape(batch_size, -1, 4)
            
            all_boxes.append(boxes)
        
        # Concatenate all feature levels
        return torch.cat(all_boxes, dim=1)