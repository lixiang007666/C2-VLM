"""Topology-aware segmentation loss."""

import torch
import torch.nn.functional as F
from torch import nn


def soft_skeletonize(x: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    skeleton = torch.zeros_like(x)
    image = x
    for _ in range(iterations):
        eroded = -F.max_pool2d(-image, 3, stride=1, padding=1)
        opened = F.max_pool2d(eroded, 3, stride=1, padding=1)
        delta = F.relu(image - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
        image = eroded
    return skeleton


class BCESoftClDiceLoss(nn.Module):
    """(1-lambda) BCE + lambda clDice; paper uses lambda=0.8."""

    def __init__(self, topology_weight: float = 0.8, smooth: float = 1.0) -> None:
        super().__init__()
        self.topology_weight = topology_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(logits)
        pred_skeleton = soft_skeletonize(probability)
        target_skeleton = soft_skeletonize(target)
        precision = (pred_skeleton * target).sum(dim=(-2, -1))
        precision = (precision + self.smooth) / (
            pred_skeleton.sum(dim=(-2, -1)) + self.smooth
        )
        sensitivity = (target_skeleton * probability).sum(dim=(-2, -1))
        sensitivity = (sensitivity + self.smooth) / (
            target_skeleton.sum(dim=(-2, -1)) + self.smooth
        )
        cldice = 1.0 - (
            2.0 * precision * sensitivity + self.smooth
        ) / (precision + sensitivity + self.smooth)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        return (1.0 - self.topology_weight) * bce + self.topology_weight * cldice.mean()
