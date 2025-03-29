# yolo_clip_detector/loss/region_text_contrastive.py 파일 내용
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class RegionTextContrastiveLoss(nn.Module):
    """
    Region-Text Contrastive Loss for YOLO-CLIP detector as described in YOLO-World paper
    
    This loss function computes a contrastive loss between region features and text embeddings.
    """
    
    def __init__(self, 
                 temperature: float = 0.1, 
                 reduction: str = 'mean',
                 topk: int = 3,
                 label_smoothing: float = 0.0):
        """
        Initialize the RegionTextContrastiveLoss.
        
        Args:
            temperature: Temperature parameter for scaling logits
            reduction: Reduction method ('mean', 'sum', 'none')
            topk: Top-k positives to use for each region
            label_smoothing: Label smoothing parameter
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.topk = topk
        self.label_smoothing = label_smoothing
        
        logger.info(f"RegionTextContrastiveLoss initialized with temperature={temperature}, "
                   f"reduction={reduction}, topk={topk}, label_smoothing={label_smoothing}")
    
    def forward(self, 
            region_features: torch.Tensor, 
            text_embeddings: torch.Tensor, 
            region_labels: torch.Tensor,
            valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the region-text contrastive loss.
        
        Args:
            region_features: Region features of shape (batch_size, num_regions, embed_dim)
            text_embeddings: Text embeddings of shape (batch_size, num_classes, embed_dim)
            region_labels: Ground truth labels of shape (batch_size, num_regions) or (batch_size, num_regions, num_classes)
            valid_mask: Mask indicating valid regions of shape (batch_size, num_regions)
            
        Returns:
            Loss value
        """
        batch_size, num_regions, embed_dim = region_features.shape
        num_classes = text_embeddings.shape[1]
        max_objects = region_labels.shape[1]
        
        # 디버깅 출력
        logger.debug(f"region_features shape: {region_features.shape}")
        logger.debug(f"region_labels shape: {region_labels.shape}")
        logger.debug(f"num_classes: {num_classes}")
        
        # region_features 크기 조정 로직 추가
        if num_regions != max_objects:
            logger.info(f"Adjusting region_features from {num_regions} to {max_objects}")
            if num_regions > max_objects:
                # 크기가 더 큰 경우 잘라내기
                region_features = region_features[:, :max_objects, :]
            else:
                # 크기가 더 작은 경우 패딩 추가
                padding = torch.zeros(batch_size, max_objects - num_regions, embed_dim,
                                    device=region_features.device)
                region_features = torch.cat([region_features, padding], dim=1)
                
                # valid_mask도 같이 조정
                if valid_mask is not None:
                    padding_mask = torch.zeros(batch_size, max_objects - num_regions,
                                            dtype=torch.bool, device=valid_mask.device)
                    valid_mask = torch.cat([valid_mask, padding_mask], dim=1)
        
        # Normalize features and embeddings
        region_features = F.normalize(region_features, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        # Compute similarity scores
        # region_features: (batch_size, num_regions, embed_dim)
        # text_embeddings: (batch_size, num_classes, embed_dim)
        # similarity: (batch_size, num_regions, num_classes)
        similarity = torch.bmm(region_features, text_embeddings.transpose(1, 2))
        
        # Scale similarity scores by temperature
        logits = similarity / self.temperature
        
        # Process region labels
        if region_labels.dim() == 2:  # If labels are class indices
            # 클래스 인덱스가 num_classes보다 크거나 같은 경우 처리
            invalid_labels = region_labels >= num_classes
            if invalid_labels.any():
                # 잘못된 레이블을 0으로 설정하고 나중에 손실 계산에서 제외
                logger.warning(f"WARNING: {invalid_labels.sum().item()} labels exceed num_classes={num_classes}")
                region_labels = torch.where(invalid_labels, torch.zeros_like(region_labels), region_labels)
                
                # valid_mask가 없으면 생성하고, 있으면 업데이트
                if valid_mask is None:
                    valid_mask = ~invalid_labels
                else:
                    valid_mask = valid_mask & ~invalid_labels
            
            # Convert to one-hot encoding
            labels_oh = F.one_hot(region_labels, num_classes).float()
        else:  # If labels are already one-hot encoded
            labels_oh = region_labels
        
        # Apply label smoothing if needed
        if self.label_smoothing > 0:
            labels_oh = (1 - self.label_smoothing) * labels_oh + self.label_smoothing / num_classes
        
        # If no valid mask is provided, create one that marks all regions as valid
        if valid_mask is None:
            valid_mask = torch.ones(batch_size, max_objects, dtype=torch.bool, device=region_features.device)
        
        # Calculate top-k positives for each region - BOOLEAN 오류 수정
        if self.topk > 1:
            # Calculate similarity for positive classes
            pos_sim = similarity * labels_oh
            
            # Get top-k positive similarities for each region
            topk_values, _ = torch.topk(pos_sim, min(self.topk, num_classes), dim=-1)
            
            # 계산할 때 스칼라 값으로 변환하여 Boolean 텐서 오류 방지
            topk_min = min(self.topk, int(labels_oh.sum(dim=-1).clamp(min=1).min().item()))
            pos_weight = topk_values.sum(dim=-1) / topk_min
            pos_weight = pos_weight.unsqueeze(-1)
            
            # Apply weights to labels
            weighted_labels = labels_oh * pos_weight
        else:
            weighted_labels = labels_oh
        
        # Compute cross-entropy loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(weighted_labels * log_probs)
        
        # Apply valid mask
        valid_mask = valid_mask.unsqueeze(-1).expand_as(loss)
        loss = loss * valid_mask
        
        # Normalize by the number of positive labels per region
        pos_count = labels_oh.sum(dim=-1).clamp(min=1)
        loss = loss.sum(dim=-1) / pos_count
        
        # Apply reduction
        if self.reduction == 'mean':
            valid_mask_sum = valid_mask.sum()
            if valid_mask_sum > 0:
                return loss.sum() / valid_mask_sum
            else:
                return torch.tensor(0.0, device=loss.device)
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss