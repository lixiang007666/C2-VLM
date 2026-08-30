"""Encode the offline 10-prompt morphology bank with BiomedCLIP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import _MODEL_CONFIGS


PROMPTS = [
    "Bright tubular structures with smooth continuous trajectories and sparse branching.",
    "A dense branching tree containing thick central trunks and progressively finer distal branches.",
    "Curved tubular structures with gradual caliber tapering and clear vessel continuity.",
    "Paired approximately symmetric branching structures surrounding a central vascular region.",
    "Tortuous vessels with multiple bifurcations and variable local curvature.",
    "Low-contrast thin vascular branches extending from brighter large vessels.",
    "Interconnected tubular paths with sharp bifurcation angles and few isolated fragments.",
    "A coarse-to-fine vascular tree with smooth boundaries and narrow distal endings.",
    "Asymmetric vessel branching with locally enlarged and tightly curved tubular segments.",
    "Fine continuous vessels forming a complex network around dominant central branches.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--biomedbert-config-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    bert_dir = Path(args.biomedbert_config_dir).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (model_dir / "open_clip_config.json").open() as stream:
        config = json.load(stream)
    model_cfg = config["model_cfg"]
    model_cfg["text_cfg"]["hf_model_name"] = str(bert_dir)
    model_cfg["text_cfg"]["hf_tokenizer_name"] = str(model_dir)
    model_name = "biomedclip_local"
    _MODEL_CONFIGS[model_name] = model_cfg
    tokenizer = get_tokenizer(model_name)
    model, _, _ = create_model_and_transforms(
        model_name=model_name,
        pretrained=str(model_dir / "open_clip_pytorch_model.bin"),
        pretrained_hf=False,
        **{f"image_{key}": value for key, value in config["preprocess_cfg"].items()},
    )
    model.eval()
    with torch.inference_mode():
        embeddings = model.encode_text(tokenizer(PROMPTS))
        embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1).cpu()
    torch.save({"embeddings": embeddings, "prompts": PROMPTS}, output)
    print(f"saved {tuple(embeddings.shape)} prompt bank to {output}")


if __name__ == "__main__":
    main()
