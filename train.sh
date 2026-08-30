#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data/C2-SegDB}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-${PROJECT_DIR}/weights/sam_vit_b_01ec64.pth}"
PROMPT_EMBEDDINGS="${PROMPT_EMBEDDINGS:-${PROJECT_DIR}/weights/biomedclip_prompt_bank.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/runs/c2segdb}"

"${PYTHON_BIN}" "${PROJECT_DIR}/train_c2vlm.py" \
  --data-root "${DATA_ROOT}" \
  --sam-checkpoint "${SAM_CHECKPOINT}" \
  --prompt-embeddings "${PROMPT_EMBEDDINGS}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 100 \
  --image-size 1024 \
  --batch-size 1 \
  --learning-rate 1e-3 \
  --warmup-start 1e-5 \
  --warmup-epochs 5 \
  --topology-weight 0.8 \
  --lora-rank 4 \
  --lora-alpha 16 \
  --experts 3 \
  --top-k 2 \
  "$@"
