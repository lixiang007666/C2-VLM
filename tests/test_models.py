"""
Basic tests for C2-VLM model components.
"""

import pytest
import torch
import tempfile
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.c2_vlm import C2VLM
from models.vision_encoder import VisionEncoder
from models.text_encoder import TextEncoder
from models.fusion_module import FusionModule


class TestVisionEncoder:
    """Test vision encoder functionality."""
    
    def test_vision_encoder_initialization(self):
        """Test vision encoder can be initialized."""
        config = {
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,  # Use False for testing to avoid downloading
            'img_size': 224,
            'patch_size': 16,
            'embed_dim': 768
        }
        encoder = VisionEncoder(**config)
        assert encoder is not None
        assert encoder.output_dim == 768
    
    def test_vision_encoder_forward(self):
        """Test vision encoder forward pass."""
        config = {
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'img_size': 224,
            'patch_size': 16,
            'embed_dim': 768
        }
        encoder = VisionEncoder(**config)
        
        # Test with dummy input
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        
        features = encoder(images)
        assert features.shape == (batch_size, 768)


class TestTextEncoder:
    """Test text encoder functionality."""
    
    def test_text_encoder_initialization(self):
        """Test text encoder can be initialized."""
        config = {
            'model_name': 'distilbert-base-uncased',  # Smaller model for testing
            'max_length': 77,
            'embed_dim': 768
        }
        encoder = TextEncoder(**config)
        assert encoder is not None
        assert encoder.output_dim == 768
    
    def test_text_encoder_forward(self):
        """Test text encoder forward pass."""
        config = {
            'model_name': 'distilbert-base-uncased',
            'max_length': 77,
            'embed_dim': 768
        }
        encoder = TextEncoder(**config)
        
        # Test with dummy input
        batch_size = 2
        seq_len = 77
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        features = encoder(input_ids, attention_mask)
        assert features.shape == (batch_size, 768)


class TestFusionModule:
    """Test fusion module functionality."""
    
    def test_fusion_module_initialization(self):
        """Test fusion module can be initialized."""
        config = {
            'vision_dim': 768,
            'text_dim': 768,
            'hidden_dim': 512,
            'projection_dim': 512
        }
        fusion = FusionModule(**config)
        assert fusion is not None
    
    def test_fusion_module_forward(self):
        """Test fusion module forward pass."""
        config = {
            'vision_dim': 768,
            'text_dim': 768,
            'hidden_dim': 512,
            'projection_dim': 512
        }
        fusion = FusionModule(**config)
        
        # Test with dummy inputs
        batch_size = 2
        vision_features = torch.randn(batch_size, 768)
        text_features = torch.randn(batch_size, 768)
        
        outputs = fusion(vision_features, text_features)
        
        assert 'fused_features' in outputs
        assert 'vision_embeddings' in outputs
        assert 'text_embeddings' in outputs
        assert 'logits_per_image' in outputs
        assert 'logits_per_text' in outputs
        
        assert outputs['vision_embeddings'].shape == (batch_size, 512)
        assert outputs['text_embeddings'].shape == (batch_size, 512)
        assert outputs['logits_per_image'].shape == (batch_size, batch_size)
        assert outputs['logits_per_text'].shape == (batch_size, batch_size)


class TestC2VLM:
    """Test complete C2-VLM model."""
    
    def test_c2vlm_initialization(self):
        """Test C2-VLM model can be initialized."""
        vision_config = {
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'img_size': 224,
            'embed_dim': 768
        }
        
        text_config = {
            'model_name': 'distilbert-base-uncased',
            'max_length': 77,
            'embed_dim': 768
        }
        
        fusion_config = {
            'hidden_dim': 512,
            'projection_dim': 512
        }
        
        model = C2VLM(
            vision_config=vision_config,
            text_config=text_config,
            fusion_config=fusion_config
        )
        
        assert model is not None
    
    def test_c2vlm_forward(self):
        """Test C2-VLM model forward pass."""
        vision_config = {
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'img_size': 224,
            'embed_dim': 768
        }
        
        text_config = {
            'model_name': 'distilbert-base-uncased',
            'max_length': 77,
            'embed_dim': 768
        }
        
        fusion_config = {
            'hidden_dim': 512,
            'projection_dim': 512
        }
        
        model = C2VLM(
            vision_config=vision_config,
            text_config=text_config,
            fusion_config=fusion_config
        )
        
        # Test with dummy inputs
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        text_tokens = torch.randint(0, 1000, (batch_size, 77))
        attention_mask = torch.ones(batch_size, 77)
        
        outputs = model(images, text_tokens, attention_mask, return_loss=True)
        
        assert 'image_features' in outputs
        assert 'text_features' in outputs
        assert 'loss' in outputs
        assert 'logits_per_image' in outputs
        assert 'logits_per_text' in outputs
        
        assert outputs['loss'].item() >= 0  # Loss should be non-negative
    
    def test_c2vlm_similarity_computation(self):
        """Test similarity computation."""
        vision_config = {
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'img_size': 224,
            'embed_dim': 768
        }
        
        text_config = {
            'model_name': 'distilbert-base-uncased',
            'max_length': 77,
            'embed_dim': 768
        }
        
        fusion_config = {
            'hidden_dim': 512,
            'projection_dim': 512
        }
        
        model = C2VLM(
            vision_config=vision_config,
            text_config=text_config,
            fusion_config=fusion_config
        )
        
        # Test similarity computation
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        text_tokens = torch.randint(0, 1000, (batch_size, 77))
        attention_mask = torch.ones(batch_size, 77)
        
        similarities = model.get_image_text_similarity(images, text_tokens, attention_mask)
        
        assert similarities.shape == (batch_size, batch_size)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])