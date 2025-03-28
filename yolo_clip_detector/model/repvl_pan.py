# yolo_clip_detector/model/repvl_pan.py 파일 내용
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ConvBlock(nn.Module):
    """
    Standard convolution block with BatchNorm and SiLU activation
    """
    
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


class TextGuidedCSPLayer(nn.Module):
    """
    Text-guided Cross Stage Partial Layer (T-CSPLayer) as described in YOLO-World paper
    
    This layer integrates text embeddings into the vision backbone via max-sigmoid attention.
    """
    
    def __init__(self, in_channels: int, out_channels: int, text_dim: int, n_bottlenecks: int = 1):
        super().__init__()
        
        c_ = out_channels // 2  # Hidden channels
        
        self.cv1 = ConvBlock(in_channels, c_, kernel_size=1)
        self.cv2 = ConvBlock(in_channels, c_, kernel_size=1)
        self.cv3 = ConvBlock(2 * c_, out_channels, kernel_size=1)
        
        # Create a sequence of bottleneck blocks
        self.bottlenecks = nn.ModuleList([
            DarkBottleneck(c_, c_, shortcut=True) for _ in range(n_bottlenecks)
        ])
        
        # Text projection to image feature dimension
        self.text_proj = nn.Linear(text_dim, c_)
    
    def forward(self, x: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the TextGuidedCSPLayer.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, H, W)
            text_embeddings: Text embeddings of shape (batch_size, num_classes, text_dim)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, H, W)
        """
        # Apply first convolution
        y1 = self.cv1(x)
        
        # Process through bottlenecks with text guidance
        for bottleneck in self.bottlenecks:
            # Regular bottleneck processing
            y1_temp = bottleneck(y1)
            
            # Project text embeddings to match feature dimension
            projected_text = self.text_proj(text_embeddings)  # (B, num_classes, c_)
            
            # Reshape y1_temp for matrix multiplication
            B, C, H, W = y1_temp.shape
            y1_reshaped = y1_temp.permute(0, 2, 3, 1).reshape(B, H*W, C)  # (B, H*W, C)
            
            # Calculate attention scores and apply max-sigmoid attention
            # Matrix multiplication: (B, H*W, C) @ (B, num_classes, C).transpose(-1, -2) -> (B, H*W, num_classes)
            attention_scores = torch.matmul(y1_reshaped, projected_text.transpose(-1, -2))
            
            # Max over classes dimension and apply sigmoid
            max_scores, _ = torch.max(attention_scores, dim=-1, keepdim=True)  # (B, H*W, 1)
            attention_weights = torch.sigmoid(max_scores)  # (B, H*W, 1)
            
            # Apply attention weights
            y1_attended = y1_reshaped * attention_weights  # (B, H*W, C)
            
            # Reshape back to original format
            y1 = y1_attended.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Apply second convolution path
        y2 = self.cv2(x)
        
        # Concatenate and apply final convolution
        return self.cv3(torch.cat((y1, y2), dim=1))


class DarkBottleneck(nn.Module):
    """
    Bottleneck block used in Darknet backbone
    """
    
    def __init__(self, in_channels: int, out_channels: int, shortcut: bool = True):
        super().__init__()
        self.cv1 = ConvBlock(in_channels, out_channels // 2, kernel_size=1)
        self.cv2 = ConvBlock(out_channels // 2, out_channels, kernel_size=3)
        self.shortcut = shortcut and in_channels == out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.shortcut else self.cv2(self.cv1(x))


class ImagePoolingAttention(nn.Module):
    """
    Image Pooling Attention (I-Pooling Attention) as described in YOLO-World paper
    
    This module enhances text embeddings with image-aware information by aggregating
    image features and applying multi-head attention.
    """
    
    def __init__(self, in_channels: List[int], embed_dim: int, num_heads: int = 8):
        """
        Initialize the ImagePoolingAttention.
        
        Args:
            in_channels: List of input channel dimensions for each feature map
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.max_pool = nn.AdaptiveMaxPool2d((3, 3))  # Pool to 3x3 regions
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        # Create projection layers for each feature map to standardize dimensions
        self.projections = nn.ModuleList([
            nn.Linear(in_channel, embed_dim) for in_channel in in_channels
        ])
    
    def forward(self, text_embeddings: torch.Tensor, feature_maps: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the ImagePoolingAttention.
        
        Args:
            text_embeddings: Text embeddings of shape (batch_size, num_classes, embed_dim)
            feature_maps: List of multi-scale feature maps [(B, C3, H3, W3), (B, C4, H4, W4), (B, C5, H5, W5)]
            
        Returns:
            Updated text embeddings of shape (batch_size, num_classes, embed_dim)
        """
        batch_size = text_embeddings.shape[0]
        patch_tokens_list = []
        
        # Process each feature map
        for i, feature_map in enumerate(feature_maps):
            # Apply max pooling to get 3x3 regions
            pooled = self.max_pool(feature_map)  # (B, C, 3, 3)
            
            # Reshape to patch tokens
            B, C, H, W = pooled.shape
            patch_tokens = pooled.permute(0, 2, 3, 1).reshape(B, H*W, C)  # (B, 9, C)
            
            # Project to common embedding dimension
            patch_tokens = self.projections[i](patch_tokens)  # (B, 9, embed_dim)
            
            # Add to the list of patch tokens
            patch_tokens_list.append(patch_tokens)
        
        # Concatenate all patch tokens (3 feature maps × 9 patches = 27 patches)
        all_patch_tokens = torch.cat(patch_tokens_list, dim=1)  # (B, 27, embed_dim)
        
        # Apply multi-head attention to update text embeddings
        updated_text_embeddings, _ = self.mha(
            query=text_embeddings,
            key=all_patch_tokens,
            value=all_patch_tokens
        )
        
        # Add residual connection
        text_embeddings = text_embeddings + updated_text_embeddings
        
        return text_embeddings

class ImagePoolingAttention(nn.Module):
    """
    Image Pooling Attention (I-Pooling Attention) as described in YOLO-World paper
    
    This module enhances text embeddings with image-aware information by aggregating
    image features and applying multi-head attention.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        
        self.max_pool = nn.AdaptiveMaxPool2d((3, 3))  # Pool to 3x3 regions
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        # Add projection layers for each feature level to handle different channel dimensions
        self.projections = nn.ModuleList([
            nn.Linear(256, embed_dim),   # C3 projection
            nn.Linear(512, embed_dim),   # C4 projection
            nn.Linear(1024, embed_dim)   # C5 projection
        ])
    
    def forward(self, text_embeddings: torch.Tensor, feature_maps: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the ImagePoolingAttention.
        
        Args:
            text_embeddings: Text embeddings of shape (batch_size, num_classes, embed_dim)
            feature_maps: List of multi-scale feature maps [(B, C3, H3, W3), (B, C4, H4, W4), (B, C5, H5, W5)]
            
        Returns:
            Updated text embeddings of shape (batch_size, num_classes, embed_dim)
        """
        batch_size = text_embeddings.shape[0]
        patch_tokens_list = []
        
        # Process each feature map
        for i, feature_map in enumerate(feature_maps):
            # Apply max pooling to get 3x3 regions
            pooled = self.max_pool(feature_map)  # (B, C, 3, 3)
            
            # Reshape to patch tokens
            B, C, H, W = pooled.shape
            patch_tokens = pooled.permute(0, 2, 3, 1).reshape(B, H*W, C)  # (B, 9, C)
            
            # Project features to common embedding dimension using the appropriate projection
            patch_tokens = self.projections[i](patch_tokens)  # (B, 9, embed_dim)
            
            # Add to the list of patch tokens
            patch_tokens_list.append(patch_tokens)
        
        # Concatenate all patch tokens (3 feature maps × 9 patches = 27 patches)
        all_patch_tokens = torch.cat(patch_tokens_list, dim=1)  # (B, 27, embed_dim)
        
        # Apply multi-head attention to update text embeddings
        updated_text_embeddings, _ = self.mha(
            query=text_embeddings,
            key=all_patch_tokens,
            value=all_patch_tokens
        )
        
        # Add residual connection
        text_embeddings = text_embeddings + updated_text_embeddings
        
        return text_embeddings