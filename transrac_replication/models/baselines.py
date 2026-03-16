from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BaselineConfig:
    num_frames: int = 64
    hidden_dim: int = 256


def _pairwise_l2_corr(x: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Build temporal self-similarity/correlation matrix from embeddings.

    x: [B, N, D]
    return: [B, N, N]
    """
    # torch.cdist gives [B,N,N]
    d = torch.cdist(x, x, p=2)
    c = torch.exp(-d / max(1e-6, temperature))
    return c


class _DensityFromCorr(nn.Module):
    def __init__(self, in_channels: int, num_frames: int, hidden_dim: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
        )
        self.fc = nn.Sequential(
            nn.Linear(num_frames, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_frames),
            nn.ReLU(inplace=True),
        )

    def forward(self, corr: torch.Tensor) -> torch.Tensor:
        # corr [B,C,N,N]
        x = self.fuse(corr).squeeze(1)  # [B,N,N]
        x = x.mean(dim=1)  # [B,N]
        return self.fc(x)


class _DensityFromFeatures(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B,N,D] -> [B,N]
        return self.net(x).squeeze(-1)


class RepNetLikeBaseline(nn.Module):
    """RepNet-like baseline using single-scale temporal self-similarity."""

    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.head = _DensityFromCorr(in_channels=1, num_frames=cfg.num_frames, hidden_dim=cfg.hidden_dim)

    def forward_from_embeddings(self, embeddings: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # use scale-1 embedding only
        x = embeddings["v1"]  # [B,N,D]
        c = _pairwise_l2_corr(x).unsqueeze(1)  # [B,1,N,N]
        density = self.head(c)
        # match TransRAC eval interface: corr [B,N,N,C]
        corr_out = c.permute(0, 2, 3, 1)
        return density, corr_out

    def forward(self, multi_scale: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("Use forward_from_embeddings with cached embedding dataset for baselines.")


class ZhangLikeBaseline(nn.Module):
    """Context-aware/scale-insensitive style baseline using multi-scale TSM fusion."""

    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.context = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = _DensityFromCorr(in_channels=16, num_frames=cfg.num_frames, hidden_dim=cfg.hidden_dim)

    def forward_from_embeddings(self, embeddings: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        c1 = _pairwise_l2_corr(embeddings["v1"])
        c4 = _pairwise_l2_corr(embeddings["v4"])
        c8 = _pairwise_l2_corr(embeddings["v8"])
        c = torch.stack([c1, c4, c8], dim=1)  # [B,3,N,N]
        feat = self.context(c)  # [B,16,N,N]
        density = self.head(feat)
        corr_out = feat.permute(0, 2, 3, 1)
        return density, corr_out

    def forward(self, multi_scale: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("Use forward_from_embeddings with cached embedding dataset for baselines.")


class X3DLikeBaseline(nn.Module):
    """X3D-inspired temporal conv baseline on cached embeddings."""

    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.LazyLinear(cfg.hidden_dim)
        self.temporal = nn.Sequential(
            nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = _DensityFromFeatures(cfg.hidden_dim, cfg.hidden_dim)

    def forward_from_embeddings(self, embeddings: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = embeddings["v1"]
        h = self.proj(x)
        h = h + self.temporal(h.transpose(1, 2)).transpose(1, 2)
        density = self.head(h)
        corr = _pairwise_l2_corr(h).unsqueeze(-1)
        return density, corr

    def forward(self, multi_scale: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("Use forward_from_embeddings with cached embedding dataset for baselines.")


class TANetLikeBaseline(nn.Module):
    """TANet-inspired baseline with temporal adaptive gating."""

    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.LazyLinear(cfg.hidden_dim)
        self.conv = nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=3, padding=1)
        self.gate = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.Sigmoid(),
        )
        self.head = _DensityFromFeatures(cfg.hidden_dim, cfg.hidden_dim)

    def forward_from_embeddings(self, embeddings: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = embeddings["v1"]
        h = self.proj(x)
        g = self.gate(h)
        t = self.conv(h.transpose(1, 2)).transpose(1, 2)
        h = h + g * t
        density = self.head(h)
        corr = _pairwise_l2_corr(h).unsqueeze(-1)
        return density, corr

    def forward(self, multi_scale: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("Use forward_from_embeddings with cached embedding dataset for baselines.")


class VideoSwinTLikeBaseline(nn.Module):
    """Video-SwinT-inspired transformer baseline on cached embeddings."""

    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.LazyLinear(cfg.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=8,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = _DensityFromFeatures(cfg.hidden_dim, cfg.hidden_dim)

    def forward_from_embeddings(self, embeddings: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = embeddings["v1"]
        h = self.proj(x)
        h = self.encoder(h)
        density = self.head(h)
        corr = _pairwise_l2_corr(h).unsqueeze(-1)
        return density, corr

    def forward(self, multi_scale: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("Use forward_from_embeddings with cached embedding dataset for baselines.")


class HuangSegLikeBaseline(nn.Module):
    """Action-segmentation-inspired temporal reasoning baseline."""

    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.LazyLinear(cfg.hidden_dim)
        self.temporal = nn.Sequential(
            nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.boundary_head = nn.Sequential(
            nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(cfg.hidden_dim // 2, 1, kernel_size=1),
        )

    def forward_from_embeddings(self, embeddings: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = embeddings["v1"]
        h = self.proj(x)
        t = self.temporal(h.transpose(1, 2))
        boundary = torch.sigmoid(self.boundary_head(t)).squeeze(1)
        smooth = F.avg_pool1d(boundary.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)
        density = F.relu(0.5 * boundary + 0.5 * smooth)
        corr = _pairwise_l2_corr(h).unsqueeze(-1)
        return density, corr

    def forward(self, multi_scale: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("Use forward_from_embeddings with cached embedding dataset for baselines.")


def build_baseline(model_name: str, cfg: BaselineConfig) -> nn.Module:
    name = model_name.lower()
    if name == "repnet":
        return RepNetLikeBaseline(cfg)
    if name == "zhang":
        return ZhangLikeBaseline(cfg)
    if name == "x3d":
        return X3DLikeBaseline(cfg)
    if name == "tanet":
        return TANetLikeBaseline(cfg)
    if name in {"swin", "videoswint", "video_swin"}:
        return VideoSwinTLikeBaseline(cfg)
    if name in {"huang", "huang_seg", "huangseg"}:
        return HuangSegLikeBaseline(cfg)
    raise ValueError(f"Unsupported baseline model: {model_name}")
