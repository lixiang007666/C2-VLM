"""
Vision encoder implementation for C2-VLM.
"""

import torch
import torch.nn as nn
from typing import Optional, Union
import timm
from transformers import AutoImageProcessor


class VisionEncoder(nn.Module):
    """
    Vision encoder using pre-trained vision transformer or CNN backbone.
    """
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        freeze_backbone: bool = False,
        use_cls_token: bool = True,
    ):
        """
        Initialize vision encoder.
        
        Args:
            model_name: Name of the backbone model
            pretrained: Whether to use pretrained weights
            img_size: Input image size
            patch_size: Patch size for vision transformer
            embed_dim: Embedding dimension
            freeze_backbone: Whether to freeze backbone parameters
            use_cls_token: Whether to use CLS token for global representation
        """
        super().__init__()
        
        self.model_name = model_name
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        
        # Load backbone model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='',  # Remove global pooling
        )
        
        # Get output dimension from backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            dummy_output = self.backbone(dummy_input)
            if isinstance(dummy_output, tuple):
                dummy_output = dummy_output[0]
            self.backbone_dim = dummy_output.size(-1)
        
        # Projection layer to desired embedding dimension
        if self.backbone_dim != embed_dim:
            self.projection = nn.Linear(self.backbone_dim, embed_dim)
        else:
            self.projection = nn.Identity()
        
        self.output_dim = embed_dim
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of vision encoder.
        
        Args:
            images: Batch of images [B, C, H, W]
            
        Returns:
            Image features [B, embed_dim]
        """
        batch_size = images.size(0)
        
        # Extract features using backbone
        features = self.backbone(images)  # [B, num_patches + 1, backbone_dim] for ViT
        
        if isinstance(features, tuple):
            features = features[0]
            
        # Handle different model outputs
        if len(features.shape) == 3:  # Vision Transformer output
            if self.use_cls_token:
                # Use CLS token (first token) as global representation
                features = features[:, 0]  # [B, backbone_dim]
            else:
                # Use mean pooling over all patches
                features = features[:, 1:].mean(dim=1)  # [B, backbone_dim]
        elif len(features.shape) == 4:  # CNN output
            # Global average pooling
            features = features.mean(dim=[2, 3])  # [B, backbone_dim]
        elif len(features.shape) == 2:  # Already pooled
            pass  # [B, backbone_dim]
        else:
            raise ValueError(f"Unexpected feature shape: {features.shape}")
        
        # Project to desired dimension
        features = self.projection(features)  # [B, embed_dim]
        
        # Apply layer normalization
        features = self.layer_norm(features)
        
        return features
    
    def get_patch_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get patch-level embeddings for detailed analysis.
        
        Args:
            images: Batch of images [B, C, H, W]
            
        Returns:
            Patch embeddings [B, num_patches, embed_dim]
        """
        features = self.backbone(images)
        
        if isinstance(features, tuple):
            features = features[0]
            
        if len(features.shape) == 3:  # Vision Transformer
            # Remove CLS token, keep only patch tokens
            patch_features = features[:, 1:]  # [B, num_patches, backbone_dim]
        else:
            raise NotImplementedError("Patch embeddings only supported for Vision Transformers")
        
        # Project patch features
        patch_features = self.projection(patch_features)  # [B, num_patches, embed_dim]
        patch_features = self.layer_norm(patch_features)
        
        return patch_features


class ResNetVisionEncoder(VisionEncoder):
    """ResNet-based vision encoder for comparison."""
    
    def __init__(self, model_name: str = "resnet50", **kwargs):
        kwargs['model_name'] = model_name
        super().__init__(**kwargs)


class EfficientNetVisionEncoder(VisionEncoder):
    """EfficientNet-based vision encoder."""
    
    def __init__(self, model_name: str = "efficientnet_b3", **kwargs):
        kwargs['model_name'] = model_name
        super().__init__(**kwargs)