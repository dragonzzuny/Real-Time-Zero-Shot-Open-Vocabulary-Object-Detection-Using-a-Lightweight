# yolo_clip_detector/clip/vocab_builder.py 파일 내용
import torch
import json
import os
import logging
from typing import List, Dict, Optional, Union
import numpy as np
from .text_encoder import CLIPTextEncoder

logger = logging.getLogger(__name__)

class VocabularyBuilder:
    """
    Build and manage online and offline vocabularies for YOLO-CLIP detector
    """
    
    def __init__(self, 
                 text_encoder: CLIPTextEncoder,
                 prompt_templates: Optional[List[str]] = None):
        """
        Initialize the VocabularyBuilder.
        
        Args:
            text_encoder: The CLIP text encoder module
            prompt_templates: List of prompt templates (default: None, uses default templates)
        """
        self.text_encoder = text_encoder
        self.offline_vocab = {}
        
        # Default prompt templates following CLIP paper
        if prompt_templates is None:
            self.prompt_templates = [
                "a photo of a {}",
                "a photograph of a {}",
                "an image of a {}",
                "a picture of a {}",
                "{}"
            ]
        else:
            self.prompt_templates = prompt_templates
            
        logger.info(f"VocabularyBuilder initialized with {len(self.prompt_templates)} prompt templates")
    
    def build_online_vocabulary(self, class_names: List[str]) -> torch.Tensor:
        """
        Build an online vocabulary for the given class names.
        
        Args:
            class_names: List of class names to encode
            
        Returns:
            Tensor of text embeddings of shape (num_classes, embed_dim)
        """
        # Format prompts with templates
        all_prompts = []
        for class_name in class_names:
            for template in self.prompt_templates:
                all_prompts.append(template.format(class_name))
        
        # Encode all prompts
        embeddings = self.text_encoder(all_prompts)
        
        # Reshape and average over templates
        num_classes = len(class_names)
        num_templates = len(self.prompt_templates)
        embeddings = embeddings.reshape(num_classes, num_templates, -1)
        embeddings = embeddings.mean(dim=1)
        
        # Renormalize after averaging
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings
    
    def build_offline_vocabulary(self, 
                                class_names: List[str], 
                                save_path: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Build and save an offline vocabulary for the given class names.
        
        Args:
            class_names: List of class names to encode
            save_path: Path to save the vocabulary (default: None, doesn't save)
            
        Returns:
            Dictionary mapping class names to embeddings
        """
        # Get embeddings for the vocabulary
        embeddings = self.build_online_vocabulary(class_names)
        
        # Create a dictionary mapping class names to embeddings
        vocab_dict = {}
        for i, class_name in enumerate(class_names):
            vocab_dict[class_name] = embeddings[i]
        
        # Save the vocabulary if a path is provided
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Convert tensors to numpy arrays for saving
            save_dict = {k: v.cpu().numpy().tolist() for k, v in vocab_dict.items()}
            
            with open(save_path, 'w') as f:
                json.dump(save_dict, f)
            
            logger.info(f"Offline vocabulary saved to {save_path}")
        
        self.offline_vocab = vocab_dict
        return vocab_dict
    
    def load_offline_vocabulary(self, path: str) -> Dict[str, torch.Tensor]:
        """
        Load an offline vocabulary from a file.
        
        Args:
            path: Path to the vocabulary file
            
        Returns:
            Dictionary mapping class names to embeddings
        """
        with open(path, 'r') as f:
            vocab_dict = json.load(f)
        
        # Convert lists back to tensors
        device = self.text_encoder.device
        for k, v in vocab_dict.items():
            vocab_dict[k] = torch.tensor(v, device=device)
        
        self.offline_vocab = vocab_dict
        logger.info(f"Loaded offline vocabulary with {len(vocab_dict)} classes from {path}")
        return vocab_dict
    
    def get_vocabulary_matrix(self, class_names: Optional[List[str]] = None) -> torch.Tensor:
        """
        Get a vocabulary matrix for the specified class names.
        
        Args:
            class_names: List of class names to include (default: None, uses all)
            
        Returns:
            Tensor of text embeddings of shape (num_classes, embed_dim)
        """
        if class_names is None:
            class_names = list(self.offline_vocab.keys())
        
        # If offline vocabulary is empty, build a new one
        if not self.offline_vocab:
            logger.warning("Offline vocabulary is empty, building a new one")
            self.build_offline_vocabulary(class_names)
        
        # Collect embeddings for the requested classes
        embeddings_list = []
        for class_name in class_names:
            if class_name in self.offline_vocab:
                embeddings_list.append(self.offline_vocab[class_name])
            else:
                logger.warning(f"Class {class_name} not found in offline vocabulary, computing on-the-fly")
                # Compute embedding for this class
                embedding = self.build_online_vocabulary([class_name])[0]
                embeddings_list.append(embedding)
        
        # Stack embeddings into a matrix
        embeddings_matrix = torch.stack(embeddings_list)
        return embeddings_matrix