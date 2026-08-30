"""C2-VLM model components."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from open_clip import create_model
from torch import nn
from torch.utils.checkpoint import checkpoint

from segment_anything import sam_model_registry


class LoRAQKV(nn.Module):
    def __init__(self, qkv: nn.Linear, rank: int = 4, alpha: float = 16.0) -> None:
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.scale = alpha / rank
        self.a_q = nn.Linear(self.dim, rank, bias=False)
        self.b_q = nn.Linear(rank, self.dim, bias=False)
        self.a_v = nn.Linear(self.dim, rank, bias=False)
        self.b_v = nn.Linear(rank, self.dim, bias=False)
        nn.init.kaiming_uniform_(self.a_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b_q.weight)
        nn.init.zeros_(self.b_v.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        qkv[..., : self.dim] = qkv[..., : self.dim] + self.b_q(self.a_q(x)) * self.scale
        qkv[..., -self.dim :] = qkv[..., -self.dim :] + self.b_v(self.a_v(x)) * self.scale
        return qkv


class LocalEncoder(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        clip = create_model(
            "RN50",
            pretrained="openai" if pretrained else None,
            cache_dir=cache_dir,
        )
        backbone = clip.visual
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.act1,
            backbone.conv2,
            backbone.bn2,
            backbone.act2,
            backbone.conv3,
            backbone.bn3,
            backbone.act3,
            backbone.avgpool,
        )
        self.stages = nn.ModuleList(
            [backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]
        )
        self.channels = (256, 512, 1024, 2048)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs


class PromptAggregator(nn.Module):
    def __init__(self, prompt_dim: int, heads: int = 8) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, prompt_dim))
        nn.init.normal_(self.query, std=0.02)
        self.attention = nn.MultiheadAttention(prompt_dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(prompt_dim)
        self.ffn = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim * 4),
            nn.GELU(),
            nn.Linear(prompt_dim * 4, prompt_dim),
        )
        self.norm2 = nn.LayerNorm(prompt_dim)

    def forward(self, prompts: torch.Tensor) -> torch.Tensor:
        prompts = prompts.unsqueeze(0) if prompts.ndim == 2 else prompts
        query = self.query.expand(prompts.shape[0], -1, -1)
        attended, _ = self.attention(query, prompts, prompts, need_weights=False)
        x = self.norm1(query + attended)
        return self.norm2(x + self.ffn(x)).squeeze(1)


class LanguageAdapter(nn.Module):
    def __init__(self, text_dim: int, channels: int) -> None:
        super().__init__()
        self.project = nn.Linear(text_dim, channels)
        groups = 32 if channels >= 32 else 1
        self.adapter = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
        )

    def forward(self, visual: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.project(text))[:, :, None, None]
        return self.adapter(visual * (1.0 + gate))


class ExpertChoiceRouter(nn.Module):
    """E-SAM expert-choice noisy router with a capacity factor.

    Adapted from ``model/MoE.py`` in Asphyxiate-Rye/E-SAM (MIT).  In
    expert-choice routing, each expert selects its highest-scoring tokens;
    this is the routing direction described by Eqs. (2)--(4) of C2-VLM.
    """

    def __init__(self, dim: int, experts: int, capacity_factor: float) -> None:
        super().__init__()
        if experts < 1 or capacity_factor <= 0:
            raise ValueError("experts and capacity_factor must be positive")
        self.experts = experts
        self.capacity_factor = capacity_factor
        self.route = nn.Linear(dim, experts)
        self.noise = nn.Linear(dim, experts)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        flat = x.reshape(-1, x.shape[-1])
        logits = self.route(flat).transpose(0, 1)
        if self.training:
            noise_scale = F.softplus(self.noise(flat)).transpose(0, 1)
            logits = logits + torch.randn_like(logits) * noise_scale

        token_count = flat.shape[0]
        capacity = min(
            token_count,
            max(1, math.ceil(token_count * self.capacity_factor / self.experts)),
        )
        selected_logits, selected_indices = logits.topk(capacity, dim=-1)
        selected_weights = selected_logits.softmax(dim=-1)
        return selected_weights, selected_indices, capacity


class ExpertMLP(nn.Module):
    """The four-times expansion expert used by E-SAM."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ExpertChoiceTokenMoE(nn.Module):
    """Device-independent form of E-SAM's ExpertChoiceTokenSparseMoE."""

    def __init__(self, dim: int, experts: int, capacity_factor: float) -> None:
        super().__init__()
        self.router = ExpertChoiceRouter(dim, experts, capacity_factor)
        self.experts = nn.ModuleList([ExpertMLP(dim) for _ in range(experts)])
        self.last_capacity = 0
        self.last_token_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights, indices, capacity = self.router(x)
        flat = x.reshape(-1, x.shape[-1])
        mixed = torch.zeros_like(flat)
        for expert_index, expert in enumerate(self.experts):
            selected = indices[expert_index]
            expert_output = expert(flat.index_select(0, selected))
            expert_weights = weights[expert_index, :, None].to(expert_output.dtype)
            expert_output = expert_output * expert_weights
            mixed.index_add_(0, selected, expert_output)
        self.last_capacity = capacity
        self.last_token_count = flat.shape[0]
        return x + mixed.reshape_as(x)


class CrossScaleExpertMixing(nn.Module):
    def __init__(
        self,
        in_dim: int = 768,
        out_dim: int = 256,
        experts: int = 3,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        if top_k <= 0:
            raise ValueError("top_k capacity factor must be positive")
        self.align = nn.ModuleList([nn.Conv2d(in_dim, out_dim, 1) for _ in range(3)])
        self.expert_moe = ExpertChoiceTokenMoE(out_dim, experts, float(top_k))
        self.attention_norm = nn.LayerNorm(out_dim)
        self.cross_stage_attention = nn.MultiheadAttention(
            out_dim, 8, batch_first=True
        )
        self.neck = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.GroupNorm(32, out_dim),
            nn.GELU(),
        )

    def forward(self, stages: list[torch.Tensor]) -> torch.Tensor:
        aligned = [layer(x.permute(0, 3, 1, 2)) for layer, x in zip(self.align, stages)]
        features = torch.stack(
            [feature.permute(0, 2, 3, 1) for feature in aligned], dim=1
        )
        batch, stage_count, height, width, channels = features.shape

        tokens = features.reshape(batch * stage_count, height * width, channels)
        tokens = self.expert_moe(tokens)
        tokens = tokens.reshape(batch, stage_count * height * width, channels)

        normalized = self.attention_norm(tokens)
        attended, _ = self.cross_stage_attention(
            normalized, normalized, normalized, need_weights=False
        )
        tokens = tokens + attended
        fused = tokens.reshape(batch, stage_count, height, width, channels).mean(dim=1)
        return self.neck(fused.permute(0, 3, 1, 2))

    @property
    def last_capacity(self) -> int:
        return self.expert_moe.last_capacity

    @property
    def last_token_count(self) -> int:
        return self.expert_moe.last_token_count


class NonPromptableDecoder(nn.Module):
    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        blocks = []
        current = channels
        for output in (128, 64, 32, 16):
            blocks.extend(
                [
                    nn.ConvTranspose2d(current, output, 2, stride=2),
                    nn.GroupNorm(8 if output >= 8 else 1, output),
                    nn.GELU(),
                    nn.Conv2d(output, output, 3, padding=1),
                    nn.GELU(),
                ]
            )
            current = output
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv2d(current, 1, 1)

    def forward(self, x: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        x = self.blocks(x)
        if x.shape[-2:] != output_size:
            x = F.interpolate(x, output_size, mode="bilinear", align_corners=False)
        return self.head(x)


class C2VLM(nn.Module):
    def __init__(
        self,
        sam_checkpoint: str,
        prompt_embeddings: str,
        lora_rank: int = 4,
        lora_alpha: float = 16.0,
        experts: int = 3,
        top_k: int = 2,
        local_pretrained: bool = True,
    ) -> None:
        super().__init__()
        if not Path(sam_checkpoint).is_file():
            raise FileNotFoundError(sam_checkpoint)
        sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
        self.image_encoder = sam.image_encoder
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad = False
        for block in self.image_encoder.blocks:
            block.attn.qkv = LoRAQKV(block.attn.qkv, lora_rank, lora_alpha)

        embeddings = torch.load(prompt_embeddings, map_location="cpu", weights_only=True)
        if isinstance(embeddings, dict):
            embeddings = embeddings["embeddings"]
        embeddings = F.normalize(embeddings.float(), dim=-1)
        self.register_buffer("prompt_embeddings", embeddings, persistent=True)
        text_dim = embeddings.shape[-1]

        self.local_encoder = LocalEncoder(pretrained=local_pretrained)
        self.prompt_aggregator = PromptAggregator(text_dim)
        self.language_adapters = nn.ModuleList(
            [LanguageAdapter(text_dim, channels) for channels in self.local_encoder.channels]
        )
        self.local_to_vit = nn.ModuleList(
            [nn.Conv2d(channels, 768, 1) for channels in self.local_encoder.channels]
        )
        self.moe = CrossScaleExpertMixing(768, 256, experts, top_k)
        self.decoder = NonPromptableDecoder(256)

        self.register_buffer(
            "sam_mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "sam_std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
        )

    def _position_embedding(self, height: int, width: int) -> torch.Tensor:
        position = self.image_encoder.pos_embed
        if position is None:
            return 0.0
        if position.shape[1:3] != (height, width):
            position = F.interpolate(
                position.permute(0, 3, 1, 2),
                (height, width),
                mode="bicubic",
                align_corners=False,
            ).permute(0, 2, 3, 1)
        return position

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output_size = images.shape[-2:]
        local_input = (images / 255.0 - self.clip_mean) / self.clip_std
        local_features = self.local_encoder(local_input)
        text = self.prompt_aggregator(self.prompt_embeddings).expand(images.shape[0], -1)
        local_features = [
            adapter(feature, text)
            for adapter, feature in zip(self.language_adapters, local_features)
        ]

        x = (images - self.sam_mean) / self.sam_std
        x = self.image_encoder.patch_embed(x)
        x = x + self._position_embedding(x.shape[1], x.shape[2])
        stages: list[torch.Tensor] = []
        stage_ends = (2, 5, 8, 11)
        stage_index = 0
        for block_index, block in enumerate(self.image_encoder.blocks):
            if self.training and x.requires_grad and x.shape[1] >= 32:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            if block_index == stage_ends[stage_index]:
                local = F.adaptive_avg_pool2d(
                    local_features[stage_index], (x.shape[1], x.shape[2])
                )
                x = x + self.local_to_vit[stage_index](local).permute(0, 2, 3, 1)
                stages.append(x)
                stage_index += 1

        main = self.image_encoder.neck(x.permute(0, 3, 1, 2))
        fused = main + self.moe(stages[:3])
        return self.decoder(fused, output_size)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
