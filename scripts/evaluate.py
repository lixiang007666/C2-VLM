#!/usr/bin/env python3
"""
Evaluation script for C2-VLM model.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.c2_vlm import C2VLM
from data.dataset import create_dataset, collate_fn
from utils.config import Config
from utils.logger import setup_logger
from transformers import AutoTokenizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate C2-VLM model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="coco",
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="val",
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./eval_results",
        help="Output directory for evaluation results"
    )
    return parser.parse_args()


def compute_retrieval_metrics(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    k_values: list = [1, 5, 10]
) -> dict:
    """
    Compute image-text retrieval metrics.
    
    Args:
        image_embeddings: Image embeddings [N, D]
        text_embeddings: Text embeddings [N, D]
        k_values: List of k values for recall@k
        
    Returns:
        Dictionary of metrics
    """
    # Compute similarity matrix
    similarities = image_embeddings @ text_embeddings.t()  # [N, N]
    
    # Image-to-text retrieval
    i2t_ranks = []
    for i in range(len(similarities)):
        sim_i = similarities[i]
        sorted_indices = torch.argsort(sim_i, descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        i2t_ranks.append(rank)
    
    # Text-to-image retrieval
    t2i_ranks = []
    for i in range(len(similarities)):
        sim_i = similarities[:, i]
        sorted_indices = torch.argsort(sim_i, descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        t2i_ranks.append(rank)
    
    # Compute recall@k
    metrics = {}
    for k in k_values:
        i2t_recall_k = np.mean([1 if rank <= k else 0 for rank in i2t_ranks])
        t2i_recall_k = np.mean([1 if rank <= k else 0 for rank in t2i_ranks])
        
        metrics[f'i2t_recall@{k}'] = i2t_recall_k
        metrics[f't2i_recall@{k}'] = t2i_recall_k
        metrics[f'recall@{k}'] = (i2t_recall_k + t2i_recall_k) / 2
    
    # Mean rank
    metrics['i2t_mean_rank'] = np.mean(i2t_ranks)
    metrics['t2i_mean_rank'] = np.mean(t2i_ranks)
    metrics['mean_rank'] = (metrics['i2t_mean_rank'] + metrics['t2i_mean_rank']) / 2
    
    return metrics


def evaluate_model(model, data_loader, device, logger):
    """Evaluate model on dataset."""
    model.eval()
    
    all_image_embeddings = []
    all_text_embeddings = []
    
    logger.info("Computing embeddings...")
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluation"):
            # Move batch to device
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(
                images=images,
                text_tokens=input_ids,
                attention_mask=attention_mask,
                return_loss=False
            )
            
            # Collect embeddings
            all_image_embeddings.append(outputs['image_embeddings'].cpu())
            all_text_embeddings.append(outputs['text_embeddings'].cpu())
    
    # Concatenate all embeddings
    image_embeddings = torch.cat(all_image_embeddings, dim=0)
    text_embeddings = torch.cat(all_text_embeddings, dim=0)
    
    logger.info(f"Computed embeddings: {image_embeddings.shape}")
    
    # Compute metrics
    metrics = compute_retrieval_metrics(image_embeddings, text_embeddings)
    
    return metrics


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = Config(config)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logger(output_dir / "eval.log")
    logger.info(f"Starting evaluation with config: {args.config}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Dataset: {args.dataset}, Split: {args.split}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = C2VLM(
        vision_config=config.model.vision_encoder,
        text_config=config.model.text_encoder,
        fusion_config=config.model.fusion_module,
        temperature=config.model.fusion_module.temperature
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    logger.info("Model loaded successfully")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.text_encoder.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    dataset = create_dataset(
        dataset_name=args.dataset,
        data_root=config.data.data_root,
        split=args.split,
        image_size=config.data.image_size,
        text_max_length=config.data.text_max_length,
        tokenizer=tokenizer
    )
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn
    )
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Evaluate model
    metrics = evaluate_model(model, data_loader, device, logger)
    
    # Log results
    logger.info("Evaluation Results:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Save results
    results_file = output_dir / f"eval_results_{args.dataset}_{args.split}.json"
    import json
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()