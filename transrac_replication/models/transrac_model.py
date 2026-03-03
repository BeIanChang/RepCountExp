from __future__ import annotations

import math
from dataclasses import dataclass
import importlib
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class TransRACConfig:
    num_frames: int = 64
    embedding_dim: int = 512
    num_heads: int = 4
    num_scales: int = 3
    hidden_dim: int = 512
    use_torchvision_backbone: bool = False


class FrozenScaleEncoder(nn.Module):
    """Backbone interface placeholder for frozen Video Swin features.

    Input scale sequence shape: [B, N, W, C, H, W]
    Output embedding shape: [B, N, D]
    """

    def __init__(self, in_channels: int = 3, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.proj = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, embedding_dim),
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, scale_seq: torch.Tensor) -> torch.Tensor:
        # scale_seq [B, N, W, C, H, W]
        if scale_seq.ndim != 6:
            raise ValueError(f"Expected 6D input, got {scale_seq.shape}")
        # temporal clip pooling over local window and spatial dims
        x = scale_seq.mean(dim=(2, 4, 5))  # [B, N, C]
        x = self.proj(x)  # [B, N, D]
        return x


class FrozenTorchvisionSwin3DEncoder(nn.Module):
    """Frozen Video Swin-T encoder wrapper.

    Input:  [B, N, W, C, H, W]
    Output: [B, N, D]
    """

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        try:
            video_mod = importlib.import_module("torchvision.models.video")
            swin3d_t = getattr(video_mod, "swin3d_t")
            weights = getattr(video_mod, "Swin3D_T_Weights").DEFAULT
        except Exception as exc:
            raise RuntimeError("torchvision video models are unavailable") from exc

        self.backbone = swin3d_t(weights=weights)
        self.backbone.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1, 1))
        self.proj = nn.Sequential(
            nn.Linear(768, embedding_dim),
            nn.ReLU(inplace=True),
        )

        for p in self.backbone.parameters():
            p.requires_grad = False

    def _extract_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        # x [B*N, C, T, H, W]
        with torch.no_grad():
            y = self.backbone.patch_embed(x)
            y = self.backbone.pos_drop(y)
            y = self.backbone.features(y)
            y = self.backbone.norm(y)
            y = y.permute(0, 4, 1, 2, 3)
            y = self.backbone.avgpool(y)
            y = torch.flatten(y, 1)
        return y

    def forward(self, scale_seq: torch.Tensor) -> torch.Tensor:
        if scale_seq.ndim != 6:
            raise ValueError(f"Expected 6D input, got {scale_seq.shape}")

        b, n, tw, c, h, w = scale_seq.shape
        x = scale_seq.reshape(b * n, tw, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # [B*N,C,T,H,W]
        if x.shape[2] < 2:
            x = x.repeat(1, 1, 2, 1, 1)
        x = (x - self.mean) / self.std

        feat = self._extract_backbone_features(x)  # [B*N,768]
        out = self.proj(feat)  # [B*N,D]
        out = out.view(b, n, -1)
        return out


class CorrelationHead(nn.Module):
    def __init__(self, dim: int = 512, heads: int = 4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        if self.head_dim * heads != dim:
            raise ValueError("embedding dim must be divisible by heads")
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B, N, D]
        b, n, d = x.shape
        q = self.q_proj(x).reshape(b, n, self.heads, self.head_dim).permute(0, 2, 1, 3)  # [B,H,N,Dh]
        k = self.k_proj(x).reshape(b, n, self.heads, self.head_dim).permute(0, 2, 1, 3)
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)  # [B,H,N,N]
        attn = F.softmax(attn_logits, dim=-1)
        return attn


class DensityPredictor(nn.Module):
    def __init__(self, in_channels: int = 12, hidden_dim: int = 512, num_frames: int = 64):
        super().__init__()
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.fc1 = nn.Linear(num_frames, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_frames)

    def forward(self, corr: torch.Tensor) -> torch.Tensor:
        # corr [B, N, N, C]
        b, n, _, c = corr.shape
        x = corr.permute(0, 3, 1, 2)  # [B,C,N,N]
        x = self.conv_fuse(x).squeeze(1)  # [B,N,N]
        x = x.mean(dim=1)  # [B,N]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class TransRACModel(nn.Module):
    def __init__(self, cfg: TransRACConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.use_torchvision_backbone:
            self.encoder = FrozenTorchvisionSwin3DEncoder(embedding_dim=cfg.embedding_dim)
        else:
            self.encoder = FrozenScaleEncoder(embedding_dim=cfg.embedding_dim)
        self.corr_heads = nn.ModuleDict(
            {
                "v1": CorrelationHead(dim=cfg.embedding_dim, heads=cfg.num_heads),
                "v4": CorrelationHead(dim=cfg.embedding_dim, heads=cfg.num_heads),
                "v8": CorrelationHead(dim=cfg.embedding_dim, heads=cfg.num_heads),
            }
        )
        self.predictor = DensityPredictor(
            in_channels=cfg.num_scales * cfg.num_heads,
            hidden_dim=cfg.hidden_dim,
            num_frames=cfg.num_frames,
        )

    def forward_from_embeddings(self, embeddings: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # embeddings keys: v1,v4,v8 each [B,N,D]
        corrs = []
        for key in ("v1", "v4", "v8"):
            emb = embeddings[key]
            attn = self.corr_heads[key](emb)  # [B,H,N,N]
            attn = attn.permute(0, 2, 3, 1)  # [B,N,N,H]
            corrs.append(attn)

        corr = torch.cat(corrs, dim=-1)  # [B,N,N,12]
        density = self.predictor(corr)  # [B,N]
        return density, corr

    def forward(self, multi_scale: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # multi_scale scale tensors should be [B, N, W, C, H, W]
        embeddings = {k: self.encoder(multi_scale[k]) for k in ("v1", "v4", "v8")}
        return self.forward_from_embeddings(embeddings)
