# Dataset Preparation Guide

This document provides instructions for preparing datasets for C2-VLM training and evaluation.

## Supported Datasets

### COCO Captions

1. **Download the COCO dataset:**
   ```bash
   mkdir -p data/coco
   cd data/coco
   
   # Download images
   wget http://images.cocodataset.org/zips/train2017.zip
   wget http://images.cocodataset.org/zips/val2017.zip
   wget http://images.cocodataset.org/zips/test2017.zip
   
   # Download annotations
   wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
   wget http://images.cocodataset.org/annotations/image_info_test2017.zip
   
   # Extract files
   unzip train2017.zip
   unzip val2017.zip
   unzip test2017.zip
   unzip annotations_trainval2017.zip
   unzip image_info_test2017.zip
   ```

2. **Directory structure:**
   ```
   data/coco/
   ├── images/
   │   ├── train2017/
   │   ├── val2017/
   │   └── test2017/
   └── annotations/
       ├── captions_train2017.json
       ├── captions_val2017.json
       └── image_info_test2017.json
   ```

### Flickr30k

1. **Download the Flickr30k dataset:**
   ```bash
   mkdir -p data/flickr30k
   cd data/flickr30k
   
   # Download from official source (requires registration)
   # Follow instructions at: https://shannon.cs.illinois.edu/DenotationGraph/
   ```

2. **Directory structure:**
   ```
   data/flickr30k/
   ├── images/
   │   └── flickr30k_images/
   └── annotations/
       └── results_20130124.token
   ```

### Visual Genome

1. **Download Visual Genome dataset:**
   ```bash
   mkdir -p data/visual_genome
   cd data/visual_genome
   
   # Download images and annotations
   wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
   wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
   
   # Download region descriptions
   wget http://visualgenome.org/static/data/dataset/region_descriptions.json.zip
   ```

## Custom Dataset

To use your own dataset, create a dataset class inheriting from `VisionLanguageDataset`:

```python
from src.data.dataset import VisionLanguageDataset

class CustomDataset(VisionLanguageDataset):
    def _load_data(self):
        # Implement your data loading logic
        # Return list of dictionaries with keys:
        # - 'image_path': relative path to image
        # - 'caption': text description
        # - 'image_id': unique image identifier
        # - 'caption_id': unique caption identifier
        data = []
        # ... your loading code ...
        return data
```

## Data Format

Each dataset should provide data in the following format:

```python
{
    'image_path': 'path/to/image.jpg',
    'caption': 'A description of the image content',
    'image_id': unique_image_id,
    'caption_id': unique_caption_id
}
```

## Configuration

Update your configuration file to specify the dataset:

```yaml
data:
  dataset_name: "coco"  # or "flickr30k", "visual_genome", "custom"
  data_root: "./data/coco"
  image_size: 224
  text_max_length: 77
  num_workers: 4
  pin_memory: true
```

## Preprocessing Tips

1. **Image Preprocessing:**
   - Images are automatically resized to the specified size
   - Default normalization uses ImageNet statistics
   - Data augmentation is applied during training

2. **Text Preprocessing:**
   - Text is tokenized using the specified tokenizer
   - Sequences are padded/truncated to max_length
   - Special tokens are handled automatically

3. **Memory Considerations:**
   - Use appropriate `num_workers` based on your system
   - Enable `pin_memory` for faster GPU transfer
   - Consider reducing `batch_size` if you encounter OOM errors