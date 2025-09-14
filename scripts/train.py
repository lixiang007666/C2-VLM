#!/usr/bin/env python3
"""
Training script for C2-VLM model.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.c2_vlm import C2VLM
from data.dataset import VisionLanguageDataset
from training.trainer import Trainer
from utils.config import Config
from utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train C2-VLM model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--gpu", 
        type=str, 
        default="0",
        help="GPU device(s) to use"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./outputs",
        help="Output directory for logs and checkpoints"
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set GPU devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = Config(config)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logger(output_dir / "train.log")
    logger.info(f"Starting training with config: {args.config}")
    logger.info(f"Output directory: {output_dir}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and config.hardware.device == "cuda" else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = C2VLM(
        vision_config=config.model.vision_encoder,
        text_config=config.model.text_encoder,
        fusion_config=config.model.fusion_module,
        temperature=config.model.fusion_module.temperature
    )
    
    # Move model to device
    model = model.to(device)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        output_dir=output_dir,
        logger=logger
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    try:
        trainer.train()
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint(output_dir / "interrupted_checkpoint.pth")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()