from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class ScaleSpec:
    window: int
    stride: int


def build_scale_sequence(
    frames: np.ndarray,
    window: int,
    stride: int,
    target_length: int,
) -> np.ndarray:
    """Build sliding-window subsequences and align temporal length.

    Args:
        frames: [N, C, H, W]
        window: Window size (1, 4, 8).
        stride: Sliding stride.
        target_length: Aligned output temporal length.

    Returns:
        Scale tensor [target_length, window, C, H, W]
    """
    if frames.ndim != 4:
        raise ValueError(f"Expected frames ndim=4, got {frames.ndim}")
    if window <= 0 or stride <= 0 or target_length <= 0:
        raise ValueError("window, stride, and target_length must be positive")

    n, c, h, w = frames.shape
    starts = list(range(0, max(1, n - window + 1), stride))
    if not starts:
        starts = [0]

    windows = []
    for s in starts:
        e = s + window
        clip = frames[s:e]
        if clip.shape[0] < window:
            pad_count = window - clip.shape[0]
            pad = np.repeat(clip[-1:], pad_count, axis=0)
            clip = np.concatenate([clip, pad], axis=0)
        windows.append(clip)

    seq = np.stack(windows, axis=0)  # [T', window, C, H, W]
    if seq.shape[0] < target_length:
        pad_count = target_length - seq.shape[0]
        tail = np.repeat(seq[-1:], pad_count, axis=0)
        seq = np.concatenate([seq, tail], axis=0)
    elif seq.shape[0] > target_length:
        seq = seq[:target_length]

    assert seq.shape == (target_length, window, c, h, w)
    return seq


def build_multiscale_sequences(
    frames: np.ndarray,
    scales: Dict[str, ScaleSpec],
    target_length: int,
) -> Dict[str, np.ndarray]:
    """Build aligned multi-scale subsequences.

    Returns dict with per-scale tensors [N, window, C, H, W].
    """
    out: Dict[str, np.ndarray] = {}
    for name, spec in scales.items():
        out[name] = build_scale_sequence(
            frames=frames,
            window=spec.window,
            stride=spec.stride,
            target_length=target_length,
        )
    return out
