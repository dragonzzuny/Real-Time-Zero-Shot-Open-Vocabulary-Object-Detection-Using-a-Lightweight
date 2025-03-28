import torch
import torch.nn as nn
import clip
from typing import List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

class CLIPTextEncoder(nn.Module):
    """
    CLIP Text Encoder module for encoding text prompts into embeddings
    """
    
    def __init__(self, 
                 model_name: str = "ViT-B/32", 
                 embed_dim: int = 512,
                 device: Optional[Union[str, torch.device]] = None):
        """
        Initialize the CLIPTextEncoder.
        
        Args:
            model_name: CLIP model variant (default: "ViT-B/32")
            embed_dim: Dimension of text embeddings (default: 512)
            device: Device to run the model on (default: None, uses CUDA if available)
        """
        super().__init__()
        self.model_name = model_name
        self.embed_dim = embed_dim
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        logger.info(f"Loading CLIP model: {model_name} on {self.device}")
        self.clip_model, self.preprocess = clip.load(model_name, device=self.device)
        self.text_encoder = self.clip_model.encode_text
        
        # Freeze the text encoder by default
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        logger.info(f"CLIP Text Encoder initialized with embedding dimension: {embed_dim}")
    
    def forward(self, text_prompts: Union[List[str], List[List[str]]]) -> torch.Tensor:
        """
        Encode a list of text prompts into embeddings.
        
        Args:
            text_prompts: List of text prompts to encode or a batch of lists of text prompts
        
        Returns:
            Normalized text embeddings tensor
        """
        # Check if input is a batch of lists or a single list
        if isinstance(text_prompts[0], list):
            # Handle batch of lists with different lengths
            batch_embeddings = []
            
            for prompts in text_prompts:
                # Tokenize and encode text for this instance
                tokens = clip.tokenize(prompts).to(self.device)
                embeddings = self.text_encoder(tokens)
                
                # Normalize embeddings
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                
                # Average the embeddings if there are multiple prompts
                if len(prompts) > 1:
                    embedding = embeddings.mean(dim=0, keepdim=True)
                else:
                    embedding = embeddings
                    
                batch_embeddings.append(embedding)
            
            # Stack the batch embeddings
            return torch.cat(batch_embeddings, dim=0)
        else:
            # Handle single list of prompts
            text_tokens = clip.tokenize(text_prompts).to(self.device)
            text_embeddings = self.text_encoder(text_tokens)
            
            # Normalize embeddings
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            
            return text_embeddings
    
    def encode_vocabulary(self, vocabulary: List[str]) -> torch.Tensor:
        """
        Encode a list of vocabulary items for offline use.
        
        Args:
            vocabulary: List of vocabulary items (class names)
        
        Returns:
            Vocabulary embeddings tensor of shape (num_vocab, embed_dim)
        """
        # Format prompts with "a photo of a [class]" template
        formatted_prompts = [f"a photo of a {item}" for item in vocabulary]
        return self.forward(formatted_prompts)
    
    def unfreeze(self):
        """Unfreeze the text encoder for fine-tuning"""
        for param in self.clip_model.parameters():
            param.requires_grad = True
        logger.info("CLIP Text Encoder unfrozen for fine-tuning")
    
    def freeze(self):
        """Freeze the text encoder"""
        for param in self.clip_model.parameters():
            param.requires_grad = False
        logger.info("CLIP Text Encoder frozen")