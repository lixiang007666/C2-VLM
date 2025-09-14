"""
Dataset implementations for C2-VLM.
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, List, Any, Optional
from pathlib import Path


class VisionLanguageDataset(Dataset):
    """
    Base vision-language dataset for C2-VLM training.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: int = 224,
        text_max_length: int = 77,
        transform: Optional[transforms.Compose] = None,
        tokenizer = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_root: Root directory containing the data
            split: Data split ("train", "val", "test")
            image_size: Size to resize images to
            text_max_length: Maximum text sequence length
            transform: Image transforms
            tokenizer: Text tokenizer
        """
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.text_max_length = text_max_length
        self.tokenizer = tokenizer
        
        # Default image transforms
        if transform is None:
            if split == "train":
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
        
        # Load data
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load dataset annotations. Override in subclasses."""
        # This is a placeholder - actual implementation would load from specific dataset format
        return []
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample."""
        item = self.data[idx]
        
        # Load and process image
        image_path = self.data_root / item['image_path']
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Process text
        text = item['caption']
        if self.tokenizer:
            text_tokens = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.text_max_length,
                return_tensors='pt'
            )
            input_ids = text_tokens['input_ids'].squeeze(0)
            attention_mask = text_tokens['attention_mask'].squeeze(0)
        else:
            # Simple tokenization (words)
            words = text.lower().split()[:self.text_max_length]
            input_ids = torch.zeros(self.text_max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.text_max_length, dtype=torch.long)
            # This would need a proper vocabulary in practice
        
        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'text': text,
            'image_id': item.get('image_id', idx),
            'caption_id': item.get('caption_id', idx)
        }


class COCODataset(VisionLanguageDataset):
    """COCO Captions dataset implementation."""
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load COCO annotations."""
        annotation_file = self.data_root / f"annotations/captions_{self.split}2017.json"
        
        if not annotation_file.exists():
            # Return dummy data for demonstration
            return [
                {
                    'image_path': 'images/dummy_image.jpg',
                    'caption': 'A sample image caption for demonstration.',
                    'image_id': 0,
                    'caption_id': 0
                }
            ]
        
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        # Create image_id to filename mapping
        image_id_to_filename = {
            img['id']: img['file_name'] 
            for img in annotations['images']
        }
        
        # Process annotations
        data = []
        for ann in annotations['annotations']:
            image_id = ann['image_id']
            filename = image_id_to_filename[image_id]
            
            data.append({
                'image_path': f"images/{self.split}2017/{filename}",
                'caption': ann['caption'],
                'image_id': image_id,
                'caption_id': ann['id']
            })
        
        return data


class Flickr30kDataset(VisionLanguageDataset):
    """Flickr30k dataset implementation."""
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load Flickr30k annotations."""
        # Placeholder implementation
        return [
            {
                'image_path': 'images/dummy_flickr.jpg',
                'caption': 'A sample Flickr30k caption.',
                'image_id': 0,
                'caption_id': 0
            }
        ]


def create_dataset(
    dataset_name: str,
    data_root: str,
    split: str,
    image_size: int = 224,
    text_max_length: int = 77,
    tokenizer = None
) -> VisionLanguageDataset:
    """
    Factory function to create datasets.
    
    Args:
        dataset_name: Name of the dataset ("coco", "flickr30k")
        data_root: Root directory containing the data
        split: Data split
        image_size: Image size
        text_max_length: Maximum text length
        tokenizer: Text tokenizer
        
    Returns:
        Dataset instance
    """
    if dataset_name.lower() == "coco":
        return COCODataset(data_root, split, image_size, text_max_length, tokenizer=tokenizer)
    elif dataset_name.lower() == "flickr30k":
        return Flickr30kDataset(data_root, split, image_size, text_max_length, tokenizer=tokenizer)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching."""
    keys = batch[0].keys()
    batched = {}
    
    for key in keys:
        if key in ['image', 'input_ids', 'attention_mask']:
            batched[key] = torch.stack([item[key] for item in batch])
        elif key in ['text']:
            batched[key] = [item[key] for item in batch]
        else:
            batched[key] = torch.tensor([item[key] for item in batch])
    
    return batched