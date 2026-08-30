"""Prompt-free NIfTI inference for C2-VLM."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import nibabel as nib
import numpy as np
import torch

from .model import C2VLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run C2-VLM inference on one NIfTI volume or a directory tree."
    )
    parser.add_argument("--input", required=True, help="NIfTI file or input directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="C2-VLM training checkpoint")
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--prompt-embeddings", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", help="For example: cuda, cuda:1, or cpu")
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-alpha", type=float)
    parser.add_argument("--experts", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def is_nifti(path: Path) -> bool:
    return path.name.endswith(".nii") or path.name.endswith(".nii.gz")


def discover_inputs(input_path: Path) -> tuple[list[Path], Path | None]:
    if input_path.is_file():
        if not is_nifti(input_path):
            raise ValueError(f"Expected a .nii or .nii.gz file: {input_path}")
        return [input_path], None
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    volumes = sorted(path for path in input_path.rglob("*") if path.is_file() and is_nifti(path))
    if not volumes:
        raise FileNotFoundError(f"No NIfTI volumes found below {input_path}")
    return volumes, input_path


def nifti_stem(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def standardize_slice(slice_2d: np.ndarray) -> np.ndarray:
    finite = np.isfinite(slice_2d)
    foreground = slice_2d[finite & (slice_2d > 0)]
    values = foreground if foreground.size else slice_2d[finite]
    if not values.size:
        return np.zeros_like(slice_2d, dtype=np.float32)
    lo, hi = np.percentile(values, (0.5, 99.5))
    if hi <= lo:
        return np.zeros_like(slice_2d, dtype=np.float32)
    clipped = np.clip(np.nan_to_num(slice_2d, nan=lo, posinf=hi, neginf=lo), lo, hi)
    scaled = ((clipped - lo) / (hi - lo) * 255.0).astype(np.float32)
    return np.clip(scaled, 0.0, 255.0)


def architecture_value(
    name: str, explicit: int | float | None, checkpoint_args: dict[str, Any], default: int | float
) -> int | float:
    if explicit is not None:
        return explicit
    return checkpoint_args.get(name, default)


def load_model(args: argparse.Namespace, device: torch.device) -> C2VLM:
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint must contain a state dictionary")
    checkpoint_args = payload.get("args", {})
    if not isinstance(checkpoint_args, dict):
        checkpoint_args = {}

    model = C2VLM(
        sam_checkpoint=args.sam_checkpoint,
        prompt_embeddings=args.prompt_embeddings,
        lora_rank=int(architecture_value("lora_rank", args.lora_rank, checkpoint_args, 4)),
        lora_alpha=float(architecture_value("lora_alpha", args.lora_alpha, checkpoint_args, 16.0)),
        experts=int(architecture_value("experts", args.experts, checkpoint_args, 3)),
        top_k=int(architecture_value("top_k", args.top_k, checkpoint_args, 2)),
        local_pretrained=False,
    )
    state = payload.get("model", payload)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint field 'model' must be a state dictionary")
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def segment_volume(
    model: C2VLM,
    volume: np.ndarray,
    image_size: int,
    batch_size: int,
    threshold: float,
    device: torch.device,
    amp_enabled: bool,
) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, received shape {volume.shape}")
    height, width, depth = volume.shape
    prediction = np.zeros((height, width, depth), dtype=np.uint8)

    for start in range(0, depth, batch_size):
        stop = min(start + batch_size, depth)
        slices = []
        for index in range(start, stop):
            image = standardize_slice(volume[..., index])
            image = cv2.resize(
                image, (image_size, image_size), interpolation=cv2.INTER_LINEAR
            )
            slices.append(np.repeat(image[None, ...], 3, axis=0))
        images = torch.from_numpy(np.stack(slices)).to(device=device, dtype=torch.float32)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            probabilities = torch.sigmoid(model(images)).float().cpu().numpy()[:, 0]
        for offset, probability in enumerate(probabilities):
            restored = cv2.resize(
                probability, (width, height), interpolation=cv2.INTER_LINEAR
            )
            prediction[..., start + offset] = (restored >= threshold).astype(np.uint8)
    return prediction


def output_path_for(source: Path, root: Path | None, output_dir: Path) -> Path:
    relative_parent = source.relative_to(root).parent if root is not None else Path()
    destination = output_dir / relative_parent / f"{nifti_stem(source)}_seg.nii.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be within [0, 1]")
    if args.image_size < 16 or args.batch_size < 1:
        raise ValueError("--image-size must be at least 16 and --batch-size must be positive")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    amp_enabled = device.type == "cuda" and not args.no_amp

    inputs, input_root = discover_inputs(Path(args.input))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args, device)

    print(
        f"volumes={len(inputs)} device={device} image_size={args.image_size} "
        f"batch_size={args.batch_size} amp={amp_enabled}"
    )
    for index, source in enumerate(inputs, start=1):
        image = nib.load(str(source))
        volume = np.asanyarray(image.dataobj, dtype=np.float32)
        mask = segment_volume(
            model,
            volume,
            args.image_size,
            args.batch_size,
            args.threshold,
            device,
            amp_enabled,
        )
        destination = output_path_for(source, input_root, output_dir)
        header = image.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(mask, image.affine, header), str(destination))
        print(f"[{index}/{len(inputs)}] {source} -> {destination}", flush=True)


if __name__ == "__main__":
    main()
