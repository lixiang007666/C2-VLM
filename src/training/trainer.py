"""
Trainer class for C2-VLM model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, Any, Optional

from ..data.dataset import create_dataset, collate_fn
from ..utils import AverageMeter, save_checkpoint, load_checkpoint


class Trainer:
    """Trainer class for C2-VLM model."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Any,
        device: torch.device,
        output_dir: Path,
        logger: logging.Logger
    ):
        """
        Initialize trainer.
        
        Args:
            model: C2-VLM model
            config: Training configuration
            device: Training device
            output_dir: Output directory
            logger: Logger instance
        """
        self.model = model
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.logger = logger
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.text_encoder.model_name
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup data loaders
        self._setup_data_loaders()
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # Setup tensorboard logging
        self.writer = SummaryWriter(output_dir / "tensorboard")
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_metric = 0.0
        
    def _setup_data_loaders(self):
        """Setup train and validation data loaders."""
        # Training dataset
        train_dataset = create_dataset(
            dataset_name=self.config.data.dataset_name,
            data_root=self.config.data.data_root,
            split=self.config.data.train_split,
            image_size=self.config.data.image_size,
            text_max_length=self.config.data.text_max_length,
            tokenizer=self.tokenizer
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            collate_fn=collate_fn,
            drop_last=True
        )
        
        # Validation dataset
        val_dataset = create_dataset(
            dataset_name=self.config.data.dataset_name,
            data_root=self.config.data.data_root,
            split=self.config.data.val_split,
            image_size=self.config.data.image_size,
            text_max_length=self.config.data.text_max_length,
            tokenizer=self.tokenizer
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            collate_fn=collate_fn
        )
        
        self.logger.info(f"Train dataset size: {len(train_dataset)}")
        self.logger.info(f"Validation dataset size: {len(val_dataset)}")
        
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        # Optimizer
        if self.config.training.optimizer.type.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=self.config.training.optimizer.betas,
                eps=self.config.training.optimizer.eps
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer.type}")
        
        # Learning rate scheduler
        if self.config.training.scheduler.type.lower() == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs * len(self.train_loader),
                eta_min=self.config.training.scheduler.eta_min
            )
        else:
            self.scheduler = None
            
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        loss_meter = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                images=images,
                text_tokens=input_ids,
                attention_mask=attention_mask,
                return_loss=True
            )
            
            loss = outputs['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip_value > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_value
                )
            
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            # Update metrics
            loss_meter.update(loss.item(), images.size(0))
            self.current_step += 1
            
            # Logging
            if self.current_step % self.config.logging.log_frequency == 0:
                lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/loss', loss.item(), self.current_step)
                self.writer.add_scalar('train/learning_rate', lr, self.current_step)
                
                self.logger.info(
                    f"Step {self.current_step}: loss={loss.item():.4f}, lr={lr:.6f}"
                )
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
        
        return {'loss': loss_meter.avg}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        loss_meter = AverageMeter()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    images=images,
                    text_tokens=input_ids,
                    attention_mask=attention_mask,
                    return_loss=True
                )
                
                loss = outputs['loss']
                loss_meter.update(loss.item(), images.size(0))
        
        metrics = {'val_loss': loss_meter.avg}
        
        # Log validation metrics
        for key, value in metrics.items():
            self.writer.add_scalar(f'val/{key}', value, self.current_epoch)
        
        return metrics
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            if epoch % self.config.evaluation.eval_frequency == 0:
                val_metrics = self.validate()
                
                # Check if this is the best model
                current_metric = -val_metrics['val_loss']  # Use negative loss as metric
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.save_checkpoint(self.output_dir / "best_model.pth")
                    self.logger.info(f"New best model saved at epoch {epoch}")
                
                self.logger.info(
                    f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
                    f"val_loss={val_metrics['val_loss']:.4f}"
                )
            
            # Save checkpoint
            if epoch % self.config.checkpointing.save_frequency == 0:
                self.save_checkpoint(self.output_dir / f"checkpoint_epoch_{epoch}.pth")
        
        self.logger.info("Training completed!")
        self.writer.close()
    
    def save_checkpoint(self, filepath: Path):
        """Save training checkpoint."""
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            step=self.current_step,
            best_metric=self.best_metric,
            filepath=filepath
        )
        
    def load_checkpoint(self, filepath: Path):
        """Load training checkpoint."""
        state = load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            filepath=filepath,
            device=self.device
        )
        
        self.current_epoch = state['epoch']
        self.current_step = state['step']
        self.best_metric = state['best_metric']