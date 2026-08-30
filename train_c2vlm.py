"""C2-VLM training entry point."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Sampler

from c2segdb_dataset import (
    C2SegDBSliceDataset,
    build_slice_records,
    discover_cases,
    split_case_ids,
)
from c2vlm_model import C2VLM
from topology_loss import BCESoftClDiceLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/C2-SegDB")
    parser.add_argument("--sam-checkpoint", default="weights/sam_vit_b_01ec64.pth")
    parser.add_argument("--prompt-embeddings", default="weights/biomedclip_prompt_bank.pt")
    parser.add_argument("--output-dir", default="runs/c2segdb")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-start", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--topology-weight", type=float, default=0.8)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--experts", type=int, default=3)
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="E-SAM expert-choice capacity factor (paper: 2)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-slices", type=int)
    parser.add_argument("--max-val-slices", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="visit every axial slice, including slices without vessel labels",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-local-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_score(logits: torch.Tensor, target: torch.Tensor) -> float:
    prediction = (torch.sigmoid(logits) > 0.5).float()
    intersection = (prediction * target).sum()
    return float(((2 * intersection + 1) / (prediction.sum() + target.sum() + 1)).item())


def dice_statistics(
    logits: torch.Tensor, target: torch.Tensor
) -> tuple[float, float, float]:
    prediction = (torch.sigmoid(logits) > 0.5).float()
    return (
        float((prediction * target).sum().item()),
        float(prediction.sum().item()),
        float(target.sum().item()),
    )


class CaseGroupedSampler(Sampler[int]):
    """Shuffle cases and slices while keeping each case contiguous for NIfTI caching."""

    def __init__(self, records, seed: int) -> None:
        self.seed = seed
        self.epoch = 0
        self.by_case: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            self.by_case.setdefault(record.case_id, []).append(index)

    def __len__(self) -> int:
        return sum(len(indices) for indices in self.by_case.values())

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        case_ids = sorted(self.by_case)
        rng.shuffle(case_ids)
        ordered: list[int] = []
        for case_id in case_ids:
            indices = self.by_case[case_id].copy()
            rng.shuffle(indices)
            ordered.extend(indices)
        self.epoch += 1
        return iter(ordered)


def main() -> None:
    args = parse_args()
    if args.log_every < 1:
        raise ValueError("--log-every must be positive")
    if args.smoke:
        args.epochs = 1
        args.image_size = min(args.image_size, 256)
        args.max_train_slices = args.max_train_slices or 2
        args.max_val_slices = args.max_val_slices or 1
        args.max_steps = args.max_steps or 1
        args.workers = 0
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this SAM-based training entry")
    device = torch.device("cuda:0")

    cases = discover_cases(args.data_root)
    train_ids, val_ids = split_case_ids(cases, seed=args.seed)
    train_records = build_slice_records(
        cases,
        train_ids,
        include_empty=args.include_empty,
        max_slices=args.max_train_slices,
    )
    val_records = build_slice_records(
        cases,
        val_ids,
        include_empty=args.include_empty,
        max_slices=args.max_val_slices,
    )
    train_data = C2SegDBSliceDataset(train_records, args.image_size, training=True)
    val_data = C2SegDBSliceDataset(val_records, args.image_size, training=False)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=CaseGroupedSampler(train_records, args.seed),
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=1,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = C2VLM(
        args.sam_checkpoint,
        args.prompt_embeddings,
        args.lora_rank,
        args.lora_alpha,
        args.experts,
        args.top_k,
        local_pretrained=not args.no_local_pretrained,
    ).to(device)
    criterion = BCESoftClDiceLoss(args.topology_weight)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    amp_enabled = not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    def lr_lambda(epoch: int) -> float:
        if epoch < args.warmup_epochs:
            start_ratio = args.warmup_start / args.learning_rate
            return start_ratio + (1.0 - start_ratio) * epoch / max(1, args.warmup_epochs)
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w") as stream:
        json.dump(vars(args), stream, indent=2)

    print(
        f"cases={len(cases)} train_cases={len(train_ids)} val_cases={len(val_ids)} "
        f"train_slices={len(train_data)} val_slices={len(val_data)}"
    )
    print(
        f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)} "
        f"total_params={model.total_parameter_count():,} "
        f"trainable_params={model.trainable_parameter_count():,} amp={amp_enabled}"
    )

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True).float()
            masks = batch["mask"].to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, masks)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {loss}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.item()))
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == len(train_loader):
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={step + 1}/{len(train_loader)} "
                    f"loss={loss.item():.6f} lr={optimizer.param_groups[0]['lr']:.3e} "
                    f"peak_gpu_gib={peak_memory:.2f} "
                    f"moe_capacity={model.moe.last_capacity}/{model.moe.last_token_count}",
                    flush=True,
                )
            if args.max_steps is not None and step + 1 >= args.max_steps:
                break
        scheduler.step()
        print(
            f"epoch={epoch + 1}/{args.epochs} train_loss={np.mean(train_losses):.6f}",
            flush=True,
        )

        model.eval()
        validation_loss_sum = 0.0
        validation_intersection = 0.0
        validation_prediction_sum = 0.0
        validation_target_sum = 0.0
        validation_samples = 0
        with torch.inference_mode():
            for validation_step, batch in enumerate(val_loader):
                images = batch["image"].to(device).float()
                masks = batch["mask"].to(device).float()
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    logits = model(images)
                    batch_loss = float(criterion(logits, masks).item())
                batch_size = images.shape[0]
                validation_loss_sum += batch_loss * batch_size
                intersection, prediction_sum, target_sum = dice_statistics(logits, masks)
                validation_intersection += intersection
                validation_prediction_sum += prediction_sum
                validation_target_sum += target_sum
                validation_samples += batch_size
                if (
                    validation_step == 0
                    or (validation_step + 1) % args.log_every == 0
                    or validation_step + 1 == len(val_loader)
                ):
                    print(
                        f"validation_step={validation_step + 1}/{len(val_loader)}",
                        flush=True,
                    )
        validation_loss = validation_loss_sum / validation_samples
        validation_dice = (2 * validation_intersection + 1) / (
            validation_prediction_sum + validation_target_sum + 1
        )
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "validation_loss": validation_loss,
            "validation_dice": validation_dice,
            "validation_samples": validation_samples,
            "args": vars(args),
        }
        torch.save(checkpoint, output_dir / "latest.pt")
        print(
            f"validation_loss={validation_loss:.6f} validation_dice={validation_dice:.6f} "
            f"validation_samples={validation_samples} checkpoint={output_dir / 'latest.pt'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
