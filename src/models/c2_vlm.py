"""
C2-VLM main model implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .vision_encoder import VisionEncoder
from .text_encoder import TextEncoder
from .fusion_module import FusionModule


class C2VLM(nn.Module):
    """
    Contextual Contrastive Vision-Language Model (C2-VLM).
    
    This model integrates vision and text encoders with a fusion module
    to perform multimodal understanding tasks with contextual awareness.
    """
    
    def __init__(
        self,
        vision_config: Dict,
        text_config: Dict,
        fusion_config: Dict,
        temperature: float = 0.07,
    ):
        """
        Initialize C2-VLM model.
        
        Args:
            vision_config: Configuration for vision encoder
            text_config: Configuration for text encoder  
            fusion_config: Configuration for fusion module
            temperature: Temperature parameter for contrastive learning
        """
        super().__init__()
        
        self.vision_encoder = VisionEncoder(**vision_config)
        self.text_encoder = TextEncoder(**text_config)
        self.fusion_module = FusionModule(
            vision_dim=self.vision_encoder.output_dim,
            text_dim=self.text_encoder.output_dim,
            **fusion_config
        )
        
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / temperature)))
        
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to feature representations."""
        return self.vision_encoder(images)
    
    def encode_text(self, text_tokens: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode text to feature representations."""
        return self.text_encoder(text_tokens, attention_mask)
    
    def forward(
        self, 
        images: torch.Tensor, 
        text_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of C2-VLM model.
        
        Args:
            images: Batch of images [B, C, H, W]
            text_tokens: Batch of tokenized text [B, seq_len]
            attention_mask: Attention mask for text [B, seq_len]
            return_loss: Whether to compute and return contrastive loss
            
        Returns:
            Dictionary containing model outputs and optionally loss
        """
        # Encode inputs
        image_features = self.encode_image(images)
        text_features = self.encode_text(text_tokens, attention_mask)
        
        # Contextual fusion
        fused_outputs = self.fusion_module(image_features, text_features, attention_mask)
        
        outputs = {
            'image_features': image_features,
            'text_features': text_features,
            'fused_features': fused_outputs['fused_features'],
            'image_embeddings': fused_outputs['image_embeddings'],
            'text_embeddings': fused_outputs['text_embeddings'],
            'logits_per_image': fused_outputs['logits_per_image'],
            'logits_per_text': fused_outputs['logits_per_text'],
            'attention_weights': fused_outputs.get('attention_weights'),
        }
        
        if return_loss:
            loss = self.compute_contrastive_loss(
                fused_outputs['logits_per_image'], 
                fused_outputs['logits_per_text']
            )
            outputs['loss'] = loss
            
        return outputs
    
    def compute_contrastive_loss(
        self, 
        logits_per_image: torch.Tensor, 
        logits_per_text: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss for image-text pairs."""
        batch_size = logits_per_image.size(0)
        labels = torch.arange(batch_size, device=logits_per_image.device)
        
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        
        return (loss_i2t + loss_t2i) / 2