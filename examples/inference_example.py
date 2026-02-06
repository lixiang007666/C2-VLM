#!/usr/bin/env python3
"""
Simple inference example for C2-VLM.
"""

import torch
import yaml
from PIL import Image
import argparse
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.c2_vlm import C2VLM
from utils.config import Config
from transformers import AutoTokenizer
import torchvision.transforms as transforms


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="C2-VLM Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Config file")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--text", type=str, required=True, help="Input text description")
    return parser.parse_args()


def load_model(model_path: str, config_path: str):
    """Load trained C2-VLM model."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = Config(config)
    
    # Initialize model
    model = C2VLM(
        vision_config=config.model.vision_encoder,
        text_config=config.model.text_encoder,
        fusion_config=config.model.fusion_module,
        temperature=config.model.fusion_module.temperature
    )
    
    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.text_encoder.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device, config


def preprocess_image(image_path: str, image_size: int = 224):
    """Preprocess input image."""
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)  # Add batch dimension


def preprocess_text(text: str, tokenizer, max_length: int = 77):
    """Preprocess input text."""
    tokens = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return tokens['input_ids'], tokens['attention_mask']


def compute_similarity(model, image, text_tokens, attention_mask, device):
    """Compute image-text similarity."""
    image = image.to(device)
    text_tokens = text_tokens.to(device)
    attention_mask = attention_mask.to(device)
    
    with torch.no_grad():
        outputs = model(
            images=image,
            text_tokens=text_tokens,
            attention_mask=attention_mask,
            return_loss=False
        )
    
    # Get similarity score
    similarity = outputs['logits_per_image'][0, 0].item()
    
    return similarity, outputs


def main():
    """Main inference function."""
    args = parse_args()
    
    print("Loading model...")
    model, tokenizer, device, config = load_model(args.model_path, args.config)
    print(f"Model loaded on device: {device}")
    
    print("Preprocessing inputs...")
    # Preprocess image
    image = preprocess_image(args.image, config.data.image_size)
    print(f"Image shape: {image.shape}")
    
    # Preprocess text
    text_tokens, attention_mask = preprocess_text(
        args.text, tokenizer, config.data.text_max_length
    )
    print(f"Text tokens shape: {text_tokens.shape}")
    
    print("Computing similarity...")
    similarity, outputs = compute_similarity(
        model, image, text_tokens, attention_mask, device
    )
    
    print(f"\nResults:")
    print(f"Image: {args.image}")
    print(f"Text: {args.text}")
    print(f"Similarity Score: {similarity:.4f}")
    
    # Additional analysis
    print(f"\nDetailed Analysis:")
    print(f"Image embedding norm: {outputs['image_embeddings'].norm().item():.4f}")
    print(f"Text embedding norm: {outputs['text_embeddings'].norm().item():.4f}")
    print(f"Logit scale: {outputs['logit_scale'].item():.4f}")
    
    # Interpretation
    if similarity > 0.5:
        print("✅ High similarity - Image and text are well matched!")
    elif similarity > 0.0:
        print("⚠️ Medium similarity - Image and text have some correspondence")
    else:
        print("❌ Low similarity - Image and text don't match well")


if __name__ == "__main__":
    main()