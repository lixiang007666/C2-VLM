"""2-D slice loader for the public C2-SegDB cerebrovascular subset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SliceRecord:
    image_path: str
    label_path: str
    slice_index: int
    case_id: str
    group: str


def discover_cases(root: str | Path) -> list[tuple[Path, Path, str, str]]:
    """Pair every unique ``*_0000.nii.gz`` volume with its label.

    The public repository currently contains byte-identical ``_0000`` and
    ``_0001`` copies.  Indexing only ``_0000`` prevents duplicate samples.
    """
    root = Path(root)
    cases: list[tuple[Path, Path, str, str]] = []
    for group_dir in sorted((root / "images").glob("*")):
        if not group_dir.is_dir():
            continue
        for image_path in sorted(group_dir.glob("case_*_0000.nii.gz")):
            case_id = image_path.name.removesuffix("_0000.nii.gz")
            label_path = root / "labels" / group_dir.name / f"{case_id}.nii.gz"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
            cases.append((image_path, label_path, case_id, group_dir.name))
    if not cases:
        raise FileNotFoundError(f"No C2-SegDB cases found below {root}")
    return cases


def split_case_ids(
    cases: Sequence[tuple[Path, Path, str, str]],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[set[str], set[str]]:
    """Create a deterministic, group-stratified case-level split."""
    rng = np.random.default_rng(seed)
    by_group: dict[str, list[str]] = {}
    for _, _, case_id, group in cases:
        by_group.setdefault(group, []).append(case_id)
    train_ids: set[str] = set()
    val_ids: set[str] = set()
    for ids in by_group.values():
        ids = sorted(ids)
        rng.shuffle(ids)
        n_val = max(1, int(round(len(ids) * val_fraction)))
        val_ids.update(ids[:n_val])
        train_ids.update(ids[n_val:])
    return train_ids, val_ids


def build_slice_records(
    cases: Iterable[tuple[Path, Path, str, str]],
    case_ids: set[str] | None = None,
    include_empty: bool = False,
    max_slices: int | None = None,
) -> list[SliceRecord]:
    records: list[SliceRecord] = []
    for image_path, label_path, case_id, group in cases:
        if case_ids is not None and case_id not in case_ids:
            continue
        # Reading each plane separately from a compressed NIfTI repeatedly
        # decompresses the volume. Load once per case and vectorize the index.
        label = np.asanyarray(nib.load(str(label_path)).dataobj)
        if include_empty:
            slice_indices = range(label.shape[2])
        else:
            slice_indices = np.flatnonzero(np.any(label > 0, axis=(0, 1))).tolist()
        for z in slice_indices:
            records.append(
                SliceRecord(str(image_path), str(label_path), int(z), case_id, group)
            )
            if max_slices is not None and len(records) >= max_slices:
                return records
    if not records:
        raise RuntimeError("No slices matched the requested C2-SegDB split")
    return records


class C2SegDBSliceDataset(Dataset):
    """Loads axial MRA slices with robust histogram standardization."""

    def __init__(
        self,
        records: Sequence[SliceRecord],
        image_size: int = 1024,
        training: bool = False,
    ) -> None:
        self.records = list(records)
        self.image_size = int(image_size)
        self.training = training
        self._cache_key: tuple[str, str] | None = None
        self._cache_image: np.ndarray | None = None
        self._cache_label: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.records)

    def _get_volumes(self, record: SliceRecord) -> tuple[np.ndarray, np.ndarray]:
        key = (record.image_path, record.label_path)
        if key != self._cache_key:
            self._cache_image = np.asanyarray(
                nib.load(record.image_path).dataobj, dtype=np.float32
            )
            self._cache_label = np.asanyarray(nib.load(record.label_path).dataobj)
            self._cache_key = key
        assert self._cache_image is not None and self._cache_label is not None
        return self._cache_image, self._cache_label

    @staticmethod
    def _standardize(slice_2d: np.ndarray) -> np.ndarray:
        foreground = slice_2d[slice_2d > 0]
        if foreground.size:
            lo, hi = np.percentile(foreground, (0.5, 99.5))
        else:
            lo, hi = float(slice_2d.min()), float(slice_2d.max())
        if hi <= lo:
            return np.zeros_like(slice_2d, dtype=np.float32)
        result = np.clip(slice_2d, lo, hi)
        return ((result - lo) / (hi - lo) * 255.0).astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        record = self.records[index]
        image_volume, label_volume = self._get_volumes(record)
        image = self._standardize(image_volume[..., record.slice_index])
        mask = (label_volume[..., record.slice_index] > 0).astype(np.float32)

        image = cv2.resize(
            image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )
        mask = cv2.resize(
            mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST
        )
        if self.training:
            if np.random.random() < 0.5:
                image, mask = np.fliplr(image).copy(), np.fliplr(mask).copy()
            if np.random.random() < 0.5:
                image, mask = np.flipud(image).copy(), np.flipud(mask).copy()

        image_rgb = np.repeat(image[None, ...], 3, axis=0)
        return {
            "image": torch.from_numpy(image_rgb),
            "mask": torch.from_numpy(mask[None, ...]),
            "case_id": record.case_id,
            "slice_index": record.slice_index,
        }
