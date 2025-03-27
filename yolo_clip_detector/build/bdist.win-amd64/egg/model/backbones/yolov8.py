# yolo_clip_detector/model/backbones/yolov8.py 파일 내용
import torch
import torch.nn as nn
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


class CSPLayer(nn.Module):
    """
    Cross Stage Partial Layer (CSP) used in YOLOv8
    """
    
    def __init__(self, in_channels: int, out_channels: int, n_bottlenecks: int = 1):
        super().__init__()
        
        c_ = out_channels // 2  # Hidden channels
        
        self.cv1 = ConvBlock(in_channels, c_, kernel_size=1)
        self.cv2 = ConvBlock(in_channels, c_, kernel_size=1)
        self.cv3 = ConvBlock(2 * c_, out_channels, kernel_size=1)
        
        # Create a sequence of bottleneck blocks
        self.bottlenecks = nn.Sequential(
            *[DarkBottleneck(c_, c_, shortcut=True) for _ in range(n_bottlenecks)]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.bottlenecks(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) module
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        c_ = in_channels // 2  # Hidden channels
        
        self.cv1 = ConvBlock(in_channels, c_, kernel_size=1)
        self.cv2 = ConvBlock(c_ * 4, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), dim=1))


class YOLOv8Backbone(nn.Module):
    """
    YOLOv8n backbone implementation
    """
    
    def __init__(self, 
                 in_channels: int = 3, 
                 variant: str = 'n', 
                 width_multiplier: Optional[float] = None,
                 depth_multiplier: Optional[float] = None):
        super().__init__()
        
        # Define width and depth multipliers based on the variant
        variant_configs = {
            'n': {'width': 0.25, 'depth': 0.33},
            's': {'width': 0.50, 'depth': 0.33},
            'm': {'width': 0.75, 'depth': 0.67},
            'l': {'width': 1.00, 'depth': 1.00},
            'x': {'width': 1.25, 'depth': 1.33},
        }
        
        if variant not in variant_configs:
            logger.warning(f"Unknown variant '{variant}', defaulting to 'n'")
            variant = 'n'
        
        # Use custom multipliers if provided, otherwise use variant defaults
        wm = width_multiplier if width_multiplier is not None else variant_configs[variant]['width']
        dm = depth_multiplier if depth_multiplier is not None else variant_configs[variant]['depth']
        
        # Define channel dimensions
        # Base channels: 64, 128, 256, 512, 1024
        base_channels = [64, 128, 256, 512, 1024]
        channels = [max(int(c * wm), 16) for c in base_channels]
        
        # Define number of bottlenecks per stage
        base_depths = [1, 2, 4, 8]
        depths = [max(int(d * dm), 1) for d in base_depths]
        
        # Stem
        self.stem = ConvBlock(in_channels, channels[0], kernel_size=3, stride=2)
        
        # Stage 1
        self.stage1 = nn.Sequential(
            ConvBlock(channels[0], channels[1], kernel_size=3, stride=2),
            CSPLayer(channels[1], channels[1], n_bottlenecks=depths[0])
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            ConvBlock(channels[1], channels[2], kernel_size=3, stride=2),
            CSPLayer(channels[2], channels[2], n_bottlenecks=depths[1])
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            ConvBlock(channels[2], channels[3], kernel_size=3, stride=2),
            CSPLayer(channels[3], channels[3], n_bottlenecks=depths[2])
        )
        
        # Stage 4
        self.stage4 = nn.Sequential(
            ConvBlock(channels[3], channels[4], kernel_size=3, stride=2),
            CSPLayer(channels[4], channels[4], n_bottlenecks=depths[3]),
            SPPF(channels[4], channels[4], kernel_size=5)
        )
        
        # Store output channels for each stage
        self.out_channels = [channels[2], channels[3], channels[4]]
        logger.info(f"YOLOv8{variant} backbone initialized with output channels: {self.out_channels}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the YOLOv8 backbone.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Tuple of three feature maps from different stages of the backbone
        """
        x = self.stem(x)
        x = self.stage1(x)
        c3 = self.stage2(x)    # P3 / 8
        c4 = self.stage3(c3)   # P4 / 16
        c5 = self.stage4(c4)   # P5 / 32
        
        return c3, c4, c5