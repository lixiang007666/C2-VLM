# C2-VLM

> **Guiding by Semantics, Seeing from Coarse to Fine: An Adaptive Vision-Language Model for Cardio-Cerebrovascular Segmentation**

C2-VLM is a unified segmentation framework for cardiovascular CTA and cerebrovascular MRA. It couples domain-adaptive visual representation with morphology-aware language guidance, allowing a single model to resolve modality shift, cross-organ vascular variation, and the severe scale imbalance between major trunks and distal branches.

## Framework

- **Visual Copilot.** A SAM ViT-B encoder is adapted with low-rank residual updates, while a parallel CLIP-ResNet50 pathway restores local texture and boundary cues that are attenuated by global self-attention.
- **Cross-scale Expert Mixing.** Intermediate encoder states are aligned and routed through an expert-choice Mixture-of-Experts block, enabling dynamic reuse of shallow geometric detail and deep semantic context.
- **Language Copilot.** BiomedCLIP encodes a morphology-focused prompt bank; attention aggregation contracts this bank into an input-compatible semantic prior and injects it into multi-scale visual features.
- **Non-Promptable Decoding.** The prediction head produces vessel masks without interactive points, boxes, or masks at inference time.
- **Topology-Aware Optimization.** Binary cross-entropy is coupled with soft clDice to balance voxel accuracy and vascular continuity.

The paper-aligned configuration uses SAM ViT-B, LoRA rank 4 with alpha 16, three experts with a top-k capacity factor of 2, BiomedCLIP text features, 1024 x 1024 axial slices, and a topology-loss weight of 0.8.

## Environment

- Python 3.10
- PyTorch 2.4.1
- CUDA-capable GPU

```bash
conda create -n c2vlm python=3.10 -y
conda activate c2vlm
pip install -r requirements.txt
```

## Model Assets

Training and inference require a SAM ViT-B checkpoint and a cached BiomedCLIP prompt bank. Checkpoints and generated embeddings are intentionally not stored in this repository.

```bash
python prepare_prompt_cache.py \
  --model-dir /path/to/BiomedCLIP \
  --biomedbert-config-dir /path/to/BiomedBERT \
  --output weights/biomedclip_prompt_bank.pt
```

## Data Interface

The public cerebrovascular portion of [C2-SegDB](https://huggingface.co/datasets/lixiangcog/C2-SegDB) follows this layout:

```text
data/C2-SegDB/
├── images/
│   ├── aneurysm/
│   └── control/
└── labels/
    ├── aneurysm/
    └── control/
```

Volumes are split at case level with group stratification. By default, training indexes vessel-containing axial slices; pass `--include-empty` to traverse each complete volume.

## Training

```bash
python train_c2vlm.py \
  --data-root data/C2-SegDB \
  --sam-checkpoint weights/sam_vit_b_01ec64.pth \
  --prompt-embeddings weights/biomedclip_prompt_bank.pt \
  --output-dir runs/c2segdb \
  --epochs 100 \
  --image-size 1024 \
  --learning-rate 1e-3 \
  --warmup-start 1e-5 \
  --lora-rank 4 \
  --lora-alpha 16 \
  --experts 3 \
  --top-k 2 \
  --topology-weight 0.8 \
  --include-empty
```

`train.sh` exposes the same entry point. Runtime paths can be overridden with `DATA_ROOT`, `SAM_CHECKPOINT`, `PROMPT_EMBEDDINGS`, `OUTPUT_DIR`, and `PYTHON_BIN`.

## Inference

`infer_c2vlm.py` performs prompt-free, slice-wise inference on a NIfTI volume or recursively on a directory of NIfTI volumes. Predictions retain the source affine, header geometry, and volume dimensions.

```bash
python infer_c2vlm.py \
  --input /path/to/input_volume_or_directory \
  --output-dir predictions \
  --checkpoint /path/to/latest.pt \
  --sam-checkpoint /path/to/sam_vit_b_01ec64.pth \
  --prompt-embeddings /path/to/biomedclip_prompt_bank.pt \
  --threshold 0.5 \
  --image-size 1024 \
  --batch-size 1
```

The inference architecture is reconstructed from the training arguments stored in `latest.pt`. Explicit architecture flags may be supplied when loading a checkpoint that does not contain its training configuration.

## Datasets Used in the Paper

| Dataset | Split | Modality | Access |
|---|---:|---:|---|
| COSTA-IXI-Guys | Train | TOF-MRA | [COSTA portal](https://imed.nimte.ac.cn/costa.html) / [Zenodo](https://zenodo.org/records/10957925) |
| COSTA-IXI-HH | Train | TOF-MRA | [COSTA portal](https://imed.nimte.ac.cn/costa.html) / [Zenodo](https://zenodo.org/records/10957925) |
| COSTA-IXI-IOP | Train | TOF-MRA | [COSTA portal](https://imed.nimte.ac.cn/costa.html) / [Zenodo](https://zenodo.org/records/10957925) |
| COSTA-ADAM | Train | TOF-MRA | [COSTA portal](https://imed.nimte.ac.cn/costa.html) / [Zenodo](https://zenodo.org/records/10957925) |
| TopCoW-MRA | Train | MRA | [TopCoW 2024 data](https://topcow24.grand-challenge.org/data/) |
| TubeTK-T1-MRA | Train | T1 / MRA | [TubeTK data](https://public.kitware.com/Wiki/TubeTK/Data) |
| SMILE-UHURA | Train | 7T TOF-MRA | [Synapse](https://www.synapse.org/Synapse:syn47164761/wiki/620033) |
| CereVessMRA-CN | Train | TOF-MRA | [Science Data Bank](https://doi.org/10.57760/sciencedb.13880) |
| ImageCAS (train partition) | Train | Coronary CTA | [Official repository](https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT) |
| COSTA-ICBM | Test | TOF-MRA | [COSTA portal](https://imed.nimte.ac.cn/costa.html) / [Zenodo](https://zenodo.org/records/10957925) |
| COSTA-LocH1 | Test | TOF-MRA | [COSTA portal](https://imed.nimte.ac.cn/costa.html) / [Zenodo](https://zenodo.org/records/10957925) |
| ImageCAS (test partition) | Test | Coronary CTA | [Official repository](https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT) |
| ASOCA | Test | Coronary CTA | [Challenge portal](https://asoca.grand-challenge.org/) |
| C2-SegDB-CBV | External validation | TOF-MRA | [Hugging Face](https://huggingface.co/datasets/lixiangcog/C2-SegDB) |
| C2-SegDB-CV | External validation | Coronary CTA | Not publicly released because of data-sharing constraints |

Access to some challenge datasets requires registration and acceptance of the provider's terms.

## Compared Methods

| Method | Reference implementation or paper |
|---|---|
| U-Net | [Paper](https://arxiv.org/abs/1505.04597) |
| 3D U-Net | [Paper](https://arxiv.org/abs/1606.06650) |
| SegFormer-B5 | [Official code](https://github.com/NVlabs/SegFormer) |
| nnU-Net ResEnc-L | [Official code](https://github.com/MIC-DKFZ/nnUNet) |
| SwinUNETR | [Official code](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR) |
| U-Mamba | [Official code](https://github.com/bowang-lab/U-Mamba) |
| DSCNet | [Official code](https://github.com/yaoleiqi/DSCNet) |
| vesselFM FT | [Official code](https://github.com/bwittmann/vesselFM) |
| SyncSAM | [Official code](https://github.com/Hhankyangg/SyncSAM) |
| Dino U-Net | [Official code](https://github.com/yifangao112/DinoUNet) |
| CESAR | [Official code](https://github.com/iMED-Lab/COSTA) |
| EI-Seg | [Official code](https://github.com/USTB-MEDAI/EI-Seg) |
| GBCNN | [Paper](https://doi.org/10.1109/TMI.2024.3435714) |
| ACE-ProtoNet | [Official code](https://github.com/d1c2x3/ACE-ProtoNet) |
