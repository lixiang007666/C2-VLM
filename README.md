# C2-VLM

C2-VLM is a vision-language framework for cardio-cerebrovascular segmentation in CTA and MRA images. It combines a SAM ViT-B encoder with LoRA adaptation, a CLIP-ResNet50 local branch, BiomedCLIP text features, cross-scale expert-choice routing, and a convolutional segmentation decoder.

## Requirements

- Python 3.10
- PyTorch 2.4.1
- CUDA-capable GPU

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Data

C2-SegDB is available from [Hugging Face](https://huggingface.co/datasets/lixiangcog/C2-SegDB). The expected layout is:

```text
data/C2-SegDB/
├── images/
│   ├── aneurysm/
│   └── control/
└── labels/
    ├── aneurysm/
    └── control/
```

The loader creates a case-level stratified training and validation split. Use `--include-empty` to include every axial slice.

## Model inputs

Training requires a SAM ViT-B checkpoint and a BiomedCLIP prompt embedding file. Model files are not stored in this repository.

Generate the prompt embedding file from local BiomedCLIP model files:

```bash
python prepare_prompt_cache.py \
  --model-dir /path/to/BiomedCLIP \
  --biomedbert-config-dir /path/to/BiomedBERT \
  --output weights/biomedclip_prompt_bank.pt
```

## Training

```bash
python train_c2vlm.py \
  --data-root data/C2-SegDB \
  --sam-checkpoint weights/sam_vit_b_01ec64.pth \
  --prompt-embeddings weights/biomedclip_prompt_bank.pt \
  --output-dir runs/c2segdb \
  --epochs 100 \
  --image-size 1024 \
  --lora-rank 4 \
  --lora-alpha 16 \
  --experts 3 \
  --top-k 2
```

The same defaults are available through `train.sh`. Paths can be changed with `DATA_ROOT`, `SAM_CHECKPOINT`, `PROMPT_EMBEDDINGS`, `OUTPUT_DIR`, and `PYTHON_BIN`.

## Acknowledgements

This project uses components from [Segment Anything](https://github.com/facebookresearch/segment-anything), [OpenCLIP](https://github.com/mlfoundations/open_clip), [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224), and [E-SAM](https://github.com/Asphyxiate-Rye/E-SAM).

## License

See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
