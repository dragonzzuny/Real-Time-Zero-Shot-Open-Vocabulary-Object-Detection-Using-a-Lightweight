# yolo_clip_detector/model/yolo_clip.py 파일 내용
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union
import logging
from .backbones.yolov8 import YOLOv8Backbone
from .repvl_pan import RepVLPAN
from .heads.text_contrastive import TextContrastiveHead
from .heads.box_head import BoxHead

from yolo_clip_detector.clip.text_encoder import CLIPTextEncoder
from yolo_clip_detector.clip.vocab_builder import VocabularyBuilder

logger = logging.getLogger(__name__)

class YOLOCLIP(nn.Module):
    """
    YOLO-CLIP: Open-Vocabulary Object Detection with Vision-Language Model
    
    This model combines YOLOv8 with CLIP text embeddings for open-vocabulary detection.
    
    Attributes:
        backbone (YOLOv8Backbone): The YOLOv8 backbone for feature extraction
        neck (RepVLPAN): The Reparam. Vision-Language PAN
        text_encoder (CLIPTextEncoder): The CLIP text encoder
        contrastive_heads (nn.ModuleList): Text contrastive heads for each feature level
        box_head (BoxHead): Box regression head
        strides (List[int]): Strides for each feature level
        vocab_builder (VocabularyBuilder): Vocabulary builder for text embeddings
        offline_mode (bool): Whether to use offline vocabulary (for inference)
        offline_vocabulary (torch.Tensor): Offline vocabulary tensor (for inference)
    """
    
    def __init__(self, 
                 backbone_variant: str = 'n',
                 clip_model: str = 'ViT-B/32',
                 embed_dim: int = 512,
                 num_classes: int = 80,
                 strides: List[int] = [8, 16, 32],
                 reg_max: int = 16,
                 offline_mode: bool = False):
        """
        Initialize the YOLO-CLIP model.
        
        Args:
            backbone_variant: YOLOv8 backbone variant ('n', 's', 'm', 'l')
            clip_model: CLIP model variant
            embed_dim: Embedding dimension
            num_classes: Number of classes
            strides: Strides for each feature level
            reg_max: Maximum value for DFL (Distributed Focal Loss)
            offline_mode: Whether to use offline vocabulary
        """
        super().__init__()
        
        # Initialize YOLOv8 backbone
        self.backbone = YOLOv8Backbone(variant=backbone_variant)
        in_channels = self.backbone.out_channels
        
        # Initialize CLIP text encoder
        self.text_encoder = CLIPTextEncoder(model_name=clip_model, embed_dim=embed_dim)
        
        # Initialize RepVL-PAN
        self.neck = RepVLPAN(in_channels=in_channels, 
                            out_channels=in_channels, 
                            text_dim=embed_dim,
                            n_bottlenecks=2)
        
        # Initialize text contrastive heads for each feature level
        self.contrastive_heads = nn.ModuleList([
            TextContrastiveHead(in_channels=in_channels[i], 
                               embed_dim=embed_dim, 
                               hidden_dim=256, 
                               reg_max=reg_max)
            for i in range(len(in_channels))
        ])
        
        # Initialize box head
        self.box_head = BoxHead(in_channels=in_channels, 
                               hidden_dim=256,
                               reg_max=reg_max,
                               strides=strides)
        
        # Store strides
        self.strides = strides
        
        # Initialize vocabulary builder
        self.vocab_builder = VocabularyBuilder(text_encoder=self.text_encoder)
        
        # Set offline mode and initialize offline vocabulary
        self.offline_mode = offline_mode
        self.offline_vocabulary = None
        
        # Initialize offline vocabulary if in offline mode
        if offline_mode:
            self.offline_vocabulary = torch.zeros((num_classes, embed_dim), device=next(self.parameters()).device)
            logger.info(f"YOLO-CLIP initialized in offline mode with empty vocabulary")
        
        logger.info(f"YOLO-CLIP initialized with backbone variant={backbone_variant}, "
                  f"embed_dim={embed_dim}, num_classes={num_classes}")
    
    def forward(self, 
                images: torch.Tensor, 
                text_prompts: Optional[Union[List[str], List[List[str]]]] = None,
                class_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the YOLO-CLIP model.
        
        Args:
            images: Input images of shape (batch_size, 3, height, width)
            text_prompts: Text prompts for online vocabulary (ignored in offline mode)
                Can be a list of strings or a batch of lists of strings
            class_names: Class names for offline vocabulary (used only if offline_vocabulary is None)
            
        Returns:
            Dictionary containing predictions (boxes, scores, class_ids, embeddings)
        """
        batch_size = images.shape[0]
        
        # 텍스트 임베딩 처리 부분 수정
        if self.offline_mode:
            if self.offline_vocabulary is not None:
                text_embeddings = self.offline_vocabulary.unsqueeze(0).expand(batch_size, -1, -1)
            elif class_names is not None:
                # Build offline vocabulary from class names and store it
                self.offline_vocabulary = self.vocab_builder.build_online_vocabulary(class_names)
                text_embeddings = self.offline_vocabulary.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                raise ValueError("In offline mode, either offline_vocabulary or class_names must be provided")
        else:
            if text_prompts is None:
                raise ValueError("In online mode, text_prompts must be provided")
            
            # 텍스트 프롬프트 처리 개선
            if isinstance(text_prompts, list):
                if len(text_prompts) > 0 and isinstance(text_prompts[0], list):
                    # 배치의 각 항목에 대한 텍스트 프롬프트가 개별적으로 제공된 경우
                    all_embeddings = []
                    for i in range(batch_size):
                        # 배치 크기보다 프롬프트가 적은 경우 마지막 프롬프트 재사용
                        sample_prompts = text_prompts[i] if i < len(text_prompts) else text_prompts[-1]
                        emb = self.text_encoder(sample_prompts)
                        all_embeddings.append(emb)
                    
                    # 배치의 모든 임베딩 스택
                    if len(all_embeddings) == 1:
                        text_embeddings = all_embeddings[0].unsqueeze(0).expand(batch_size, -1, -1)
                    else:
                        try:
                            text_embeddings = torch.stack(all_embeddings)
                        except RuntimeError:
                            # 임베딩 차원이 일치하지 않는 경우
                            max_classes = max([e.shape[0] for e in all_embeddings])
                            padded_embeddings = []
                            for emb in all_embeddings:
                                if emb.shape[0] < max_classes:
                                    padding = torch.zeros(max_classes - emb.shape[0], emb.shape[1], device=emb.device)
                                    padded_emb = torch.cat([emb, padding], dim=0)
                                    padded_embeddings.append(padded_emb)
                                else:
                                    padded_embeddings.append(emb)
                            text_embeddings = torch.stack(padded_embeddings)
                else:
                    # 모든 샘플에 동일한 프롬프트 집합 사용
                    text_embeddings = self.text_encoder(text_prompts).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Extract features from backbone
        features = self.backbone(images)
        
        # Process features through RepVL-PAN
        pan_features, updated_text_embeddings = self.neck(features, text_embeddings)
        
        # Compute object embeddings and similarities from each feature level
        all_obj_embeddings = []
        all_similarities = []
        
        for i, (feat, head) in enumerate(zip(pan_features, self.contrastive_heads)):
            # Extract object embeddings and box predictions
            obj_embed, _ = head(feat)
            
            # Compute similarity with text embeddings
            similarity = head.compute_similarity(obj_embed, updated_text_embeddings)
            
            # Append to lists
            all_obj_embeddings.append(obj_embed)
            all_similarities.append(similarity)
        
        # Compute box predictions using box head
        box_preds, grids = self.box_head(pan_features)
        
        # Decode boxes
        boxes = self.box_head.decode_boxes(box_preds, grids)
        
        # Get class scores and IDs from similarities
        scores_list = []
        class_ids_list = []
        
        for similarity in all_similarities:
            # Get max scores and corresponding class IDs
            scores, class_ids = similarity.max(dim=1)
            scores_list.append(scores)
            class_ids_list.append(class_ids)
        
        # Concatenate scores and class IDs from different feature levels
        scores = torch.cat([s.flatten(1) for s in scores_list], dim=1)
        class_ids = torch.cat([c.flatten(1) for c in class_ids_list], dim=1)
        
        # Collect object embeddings from different feature levels
        obj_embeddings = []
        for i, embed in enumerate(all_obj_embeddings):
            B, C, H, W = embed.shape
            obj_embeddings.append(embed.permute(0, 2, 3, 1).reshape(B, H*W, C))
        
        obj_embeddings = torch.cat(obj_embeddings, dim=1)
        
        return {
            'boxes': boxes,               # (batch_size, num_boxes, 4)
            'scores': scores,             # (batch_size, num_boxes)
            'class_ids': class_ids,       # (batch_size, num_boxes)
            'obj_embeddings': obj_embeddings,  # (batch_size, num_boxes, embed_dim)
            'text_embeddings': updated_text_embeddings,  # Include text embeddings in output
            'box_preds': box_preds  # Include box predictions for loss calculation
        }
    
    def set_offline_vocabulary(self, 
                              class_names: List[str], 
                              save_path: Optional[str] = None) -> None:
        """
        Set offline vocabulary for inference.
        
        Args:
            class_names: List of class names
            save_path: Path to save the vocabulary (optional)
        """
        self.offline_mode = True
        self.offline_vocabulary = self.vocab_builder.build_online_vocabulary(class_names)
        
        # Save vocabulary if path is provided
        if save_path is not None:
            self.vocab_builder.build_offline_vocabulary(class_names, save_path)
        
        logger.info(f"Offline vocabulary set with {len(class_names)} classes")
    
    def load_offline_vocabulary(self, path: str) -> None:
        """
        Load offline vocabulary from a file.
        
        Args:
            path: Path to the vocabulary file
        """
        self.offline_mode = True
        vocab_dict = self.vocab_builder.load_offline_vocabulary(path)
        
        # Convert dictionary to tensor
        class_names = list(vocab_dict.keys())
        embeddings = []
        
        for name in class_names:
            embeddings.append(vocab_dict[name])
        
        self.offline_vocabulary = torch.stack(embeddings)
        
        logger.info(f"Offline vocabulary loaded with {len(class_names)} classes")