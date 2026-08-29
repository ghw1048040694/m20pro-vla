"""Compact multimodal policy used by the MuJoCo VLA baseline.

The model lives in the installable package so training, replay, evaluation,
and future adapters share one checkpoint contract instead of importing one
CLI script from another.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from m20pro_vla.data.history import HISTORY_FEATURE_DIM, HISTORY_FEATURE_LABELS


TEXT_ENCODING = "utf8_byte_v1"
TEXT_TOKEN_LENGTH = 64
PHASE_LABELS = ("search", "approach", "stop")
VISUAL_GEOMETRY_LABELS = ("visible", "x_offset", "area", "rear_source")
VISUAL_GEOMETRY_DIM = len(VISUAL_GEOMETRY_LABELS)


class M20MuJoCoVLA(nn.Module):
    """Language-conditioned body-command predictor.

    The output is ``[forward, lateral, yaw, stop_logit]``. Joint-level
    execution remains exclusively owned by ``M20LowLevelController``.
    """

    def __init__(self, vision: str = "spatial_v2"):
        super().__init__()
        if vision not in {"global_v1", "spatial_v2"}:
            raise ValueError(f"unsupported M20 VLA vision contract: {vision}")
        self.vision = vision
        if vision == "global_v1":
            image_layers = [
                nn.Conv2d(6, 24, 5, stride=2, padding=2), nn.GroupNorm(6, 24), nn.GELU(),
                nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
                nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            ]
            image_dim = 64
        else:
            image_layers = [
                nn.Conv2d(6, 24, 5, stride=2, padding=2), nn.GroupNorm(6, 24), nn.GELU(),
                nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
                nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
                nn.AdaptiveAvgPool2d((3, 5)), nn.Flatten(),
                nn.Linear(64 * 3 * 5, 128), nn.LayerNorm(128), nn.GELU(),
            ]
            image_dim = 128
        self.image = nn.Sequential(*image_layers)
        self.lidar = nn.Sequential(
            nn.Linear(72, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 32), nn.GELU()
        )
        self.proprio = nn.Sequential(
            nn.Linear(45, 96), nn.LayerNorm(96), nn.GELU(), nn.Linear(96, 48), nn.GELU()
        )
        self.text_embedding = nn.Embedding(257, 32, padding_idx=0)
        self.text_encoder = nn.GRU(32, 64, batch_first=True)
        self.text_projection = nn.Sequential(nn.Linear(64, 32), nn.GELU())
        self.trunk = nn.Sequential(
            nn.Linear(image_dim + 32 + 48 + 32, 192), nn.LayerNorm(192), nn.GELU()
        )
        self.history_projection = nn.Sequential(
            nn.Linear(HISTORY_FEATURE_DIM, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 192),
        )
        self.target_context = nn.Embedding(3, 16)
        self.action_head = nn.Linear(192 + 16, 4)
        self.target_head = nn.Linear(192, 3)
        self.phase_head = nn.Linear(192, len(PHASE_LABELS))
        self.phase_action_bias = nn.Linear(len(PHASE_LABELS), 4, bias=False)
        self.visual_geometry_head = nn.Linear(192, VISUAL_GEOMETRY_DIM)
        self.visual_action_bias = nn.Sequential(
            nn.Linear(VISUAL_GEOMETRY_DIM + len(PHASE_LABELS), 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 4),
        )
        self.close_approach_bias = nn.Sequential(
            nn.Linear(192 + 16 + VISUAL_GEOMETRY_DIM + len(PHASE_LABELS) + HISTORY_FEATURE_DIM, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 4),
        )
        nn.init.zeros_(self.history_projection[-1].weight)
        nn.init.zeros_(self.history_projection[-1].bias)
        nn.init.zeros_(self.phase_action_bias.weight)
        nn.init.zeros_(self.visual_action_bias[-1].weight)
        nn.init.zeros_(self.visual_action_bias[-1].bias)
        nn.init.zeros_(self.close_approach_bias[-1].weight)
        nn.init.zeros_(self.close_approach_bias[-1].bias)

    @classmethod
    def for_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> "M20MuJoCoVLA":
        """Select the historical vision head from a checkpoint state dict."""
        vision = "spatial_v2" if "image.11.weight" in state_dict else "global_v1"
        return cls(vision=vision)

    @property
    def history_dim(self) -> int:
        return HISTORY_FEATURE_DIM

    def forward(self, rgb, lidar, proprio, language, history=None):
        latent = self.encode(rgb, lidar, proprio, language, history=history)
        target_logits = self.target_head(latent)
        phase_logits = self.phase_head(latent)
        target_context = self._target_context_from_logits(target_logits)
        return self.action_from_latent(latent, target_context, phase_logits=phase_logits, history=history)

    def encode(self, rgb, lidar, proprio, language, history=None):
        text_embedding = self.text_embedding(language)
        text_encoded, _ = self.text_encoder(text_embedding)
        lengths = language.ne(0).sum(dim=1).clamp_min(1)
        last_index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, text_encoded.size(-1))
        text_feature = text_encoded.gather(1, last_index).squeeze(1)
        fused = torch.cat((self.image(rgb), self.lidar(lidar), self.proprio(proprio), self.text_projection(text_feature)), dim=-1)
        latent = self.trunk(fused)
        if history is not None:
            latent = latent + self.history_projection(history.to(dtype=latent.dtype, device=latent.device))
        return latent

    def target_logits(self, rgb, lidar, proprio, language, history=None):
        return self.target_head(self.encode(rgb, lidar, proprio, language, history=history))

    def phase_logits(self, rgb, lidar, proprio, language, history=None):
        return self.phase_head(self.encode(rgb, lidar, proprio, language, history=history))

    def action_from_latent(self, latent, target_context, phase_logits=None, history=None):
        raw = self.action_head(torch.cat((latent, target_context), dim=-1))
        if phase_logits is not None:
            phase_prob = torch.softmax(phase_logits, dim=-1)
            raw = raw + self.phase_action_bias(phase_prob)
        else:
            phase_prob = torch.zeros((latent.size(0), len(PHASE_LABELS)), dtype=latent.dtype, device=latent.device)
        visual_raw = self.visual_geometry_head(latent)
        visual_features = torch.cat(
            (
                torch.sigmoid(visual_raw[:, 0:1]),
                torch.tanh(visual_raw[:, 1:2]),
                torch.sigmoid(visual_raw[:, 2:4]),
                phase_prob,
            ),
            dim=-1,
        )
        raw = raw + self.visual_action_bias(visual_features)
        if history is not None:
            history_features = history.to(dtype=latent.dtype, device=latent.device)
            raw = raw + self.close_approach_bias(
                torch.cat((latent, target_context, visual_features, history_features), dim=-1)
            )
        return torch.cat((torch.tanh(raw[:, :3]), raw[:, 3:4]), dim=-1)

    def _target_context_from_logits(self, target_logits):
        return torch.matmul(torch.softmax(target_logits, dim=-1), self.target_context.weight)


def encode_text(text: str, length: int = TEXT_TOKEN_LENGTH) -> np.ndarray:
    """Encode task text as UTF-8 bytes without tokenizer state.

    ASCII prompts keep their historical token IDs. Non-ASCII instructions,
    including Chinese ObjectNav templates, are now preserved instead of being
    dropped by ``errors="ignore"``.
    """
    tokens = np.zeros(length, dtype=np.int64)
    encoded = np.frombuffer(text.encode("utf-8")[:length], dtype=np.uint8).astype(np.int64) + 1
    tokens[: len(encoded)] = encoded
    return tokens


__all__ = [
    "M20MuJoCoVLA",
    "TEXT_ENCODING",
    "TEXT_TOKEN_LENGTH",
    "PHASE_LABELS",
    "HISTORY_FEATURE_DIM",
    "HISTORY_FEATURE_LABELS",
    "VISUAL_GEOMETRY_LABELS",
    "VISUAL_GEOMETRY_DIM",
    "encode_text",
]
