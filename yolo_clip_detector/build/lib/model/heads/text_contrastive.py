# yolo_clip_detector/model/heads/text_contrastive.py 파일 내용
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import logging
import math

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


class TextContrastiveHead(nn.Module):
    """
    Text Contrastive Head for YOLO-CLIP detector as described in YOLO-World paper
    
    This head computes object embeddings and calculates similarity with text embeddings.
    """
    
    def __init__(self, 
                 in_channels: int, 
                 embed_dim: int = 512,
                 hidden_dim: int = 256, 
                 reg_max: int = 16,
                 cls_alpha: float = 1.0,
                 cls_beta: float = 0.0,
                 width_scale: float = 1.0,
                 height_scale: float = 1.0):
        """
        Initialize the TextContrastiveHead.
        
        Args:
            in_channels: Number of input channels from the backbone/neck
            embed_dim: Dimension of embeddings (should match text embedding dimension)
            hidden_dim: Hidden dimension in the head
            reg_max: Maximum value for DFL (Distributed Focal Loss)
            cls_alpha: Scaling factor for similarity score
            cls_beta: Shifting factor for similarity score
            width_scale: Width scale for bounding box regression
            height_scale: Height scale for bounding box regression
        """
        super().__init__()
        
        # Object embedding convolution (for region-text matching)
        self.obj_embed_conv = nn.Sequential(
            ConvBlock(in_channels, hidden_dim, kernel_size=3),
            ConvBlock(hidden_dim, hidden_dim, kernel_size=3),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=1)
        )
        
        # Bounding box regression convolution
        self.box_conv = nn.Sequential(
            ConvBlock(in_channels, hidden_dim, kernel_size=3),
            ConvBlock(hidden_dim, hidden_dim, kernel_size=3),
            nn.Conv2d(hidden_dim, 4 * (reg_max + 1), kernel_size=1)  # 4 for xywh, (reg_max+1) for DFL
        )
        
        self.embed_dim = embed_dim
        self.reg_max = reg_max
        self.cls_alpha = cls_alpha
        self.cls_beta = cls_beta
        self.width_scale = width_scale
        self.height_scale = height_scale
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"TextContrastiveHead initialized with embed_dim={embed_dim}, reg_max={reg_max}")
    
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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the TextContrastiveHead.
        
        Args:
            x: Input tensor from the backbone/neck of shape (batch_size, in_channels, H, W)
            
        Returns:
            Tuple of (object embeddings, bounding box regression parameters)
        """
        # Compute object embeddings
        obj_embed = self.obj_embed_conv(x)
        
        # Compute bounding box regression
        box_preds = self.box_conv(x)
        
        return obj_embed, box_preds
    
    def compute_similarity(self, 
                          obj_embed: torch.Tensor, 
                          text_embed: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between object embeddings and text embeddings.
        
        Args:
            obj_embed: Object embeddings of shape (batch_size, embed_dim, H, W)
            text_embed: Text embeddings of shape (batch_size, num_classes, embed_dim)
            
        Returns:
            Similarity scores of shape (batch_size, num_classes, H, W)
        """
        # Reshape object embeddings
        batch_size, embed_dim, height, width = obj_embed.shape
        obj_embed = obj_embed.permute(0, 2, 3, 1).reshape(batch_size, height * width, embed_dim)
        
        # L2 normalize embeddings
        obj_embed = F.normalize(obj_embed, p=2, dim=-1)
        text_embed = F.normalize(text_embed, p=2, dim=-1)
        
        # Compute cosine similarity
        # obj_embed: (batch_size, H*W, embed_dim)
        # text_embed: (batch_size, num_classes, embed_dim)
        # similarity: (batch_size, H*W, num_classes)
        similarity = torch.matmul(obj_embed, text_embed.transpose(1, 2))
        
        # Apply affine transformation (scaling and shifting)
        similarity = self.cls_alpha * similarity + self.cls_beta
        
        # Reshape back to spatial format
        num_classes = text_embed.shape[1]
        similarity = similarity.transpose(1, 2).reshape(batch_size, num_classes, height, width)
        
        return similarity
    
    def decode_boxes(self, 
                     box_preds: torch.Tensor, 
                     grid_sizes: List[Tuple[int, int]],
                     strides: List[int]) -> torch.Tensor:
        """
        Decode bounding box predictions into coordinates.
        
        Args:
            box_preds: Box predictions from forward pass (batch_size, 4*(reg_max+1), H, W)
            grid_sizes: List of (height, width) for each feature level
            strides: List of strides for each feature level
            
        Returns:
            Decoded boxes of shape (batch_size, H*W, 4)
        """
        batch_size = box_preds.shape[0]
        decoded_boxes = []
        
        # Process each feature level
        box_idx = 0
        for (height, width), stride in zip(grid_sizes, strides):
            # Extract predictions for this level
            level_preds = box_preds[:, :, :height, :width]
            
            # Reshape to (batch_size, 4, reg_max+1, height, width)
            level_preds = level_preds.reshape(batch_size, 4, self.reg_max + 1, height, width)
            
            # Apply softmax to get distribution over reg_max+1 bins
            level_preds = level_preds.softmax(dim=2)
            
            # Project distribution to value using DFL
            reg_vals = level_preds * torch.arange(0, self.reg_max + 1, device=level_preds.device).float()
            reg_vals = reg_vals.sum(dim=2)
            
            # Generate grid cells
            grid_x, grid_y = torch.meshgrid(torch.arange(width, device=level_preds.device),
                                           torch.arange(height, device=level_preds.device),
                                           indexing='xy')
            
            # Reshape grid
            grid_xy = torch.stack([grid_x, grid_y], dim=0).repeat(batch_size, 1, 1, 1)
            
            # Decode boxes
            # x = (grid_x + reg_vals[0]) * stride
            # y = (grid_y + reg_vals[1]) * stride
            # w = exp(reg_vals[2]) * stride * width_scale
            # h = exp(reg_vals[3]) * stride * height_scale
            x_center = (grid_xy[:, 0, :, :] + reg_vals[:, 0, :, :]) * stride
            y_center = (grid_xy[:, 1, :, :] + reg_vals[:, 1, :, :]) * stride
            w = torch.exp(reg_vals[:, 2, :, :]) * stride * self.width_scale
            h = torch.exp(reg_vals[:, 3, :, :]) * stride * self.height_scale
            
            # Convert to (x1, y1, x2, y2) format
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2
            
            # Stack and reshape
            level_boxes = torch.stack([x1, y1, x2, y2], dim=1)
            level_boxes = level_boxes.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
            
            decoded_boxes.append(level_boxes)
            
            box_idx += height * width
        
        # Concatenate all feature levels
        return torch.cat(decoded_boxes, dim=1)