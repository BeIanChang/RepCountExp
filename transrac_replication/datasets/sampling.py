from __future__ import annotations

from typing import Tuple

import numpy as np


def uniform_sample_indices(total_frames: int, num_samples: int = 64) -> np.ndarray:
    """Uniformly sample frame indices with replication padding.

    Args:
        total_frames: Number of frames in the source video.
        num_samples: Number of sampled frames.

    Returns:
        int64 array of shape [num_samples] with values in [0, total_frames-1].
    """
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    if total_frames == 1:
        return np.zeros((num_samples,), dtype=np.int64)

    indices = np.linspace(0, total_frames - 1, num=num_samples, dtype=np.float32)
    indices = np.round(indices).astype(np.int64)
    indices = np.clip(indices, 0, total_frames - 1)
    return indices


def sample_video_frames(frames: np.ndarray, num_samples: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a fixed number of frames from a frame tensor.

    Args:
        frames: Frame tensor [T, C, H, W].
        num_samples: Output temporal length.

    Returns:
        sampled_frames: [num_samples, C, H, W]
        indices: sampled index array [num_samples]
    """
    if frames.ndim != 4:
        raise ValueError(f"Expected frames ndim=4, got {frames.ndim}")
    total = frames.shape[0]
    idx = uniform_sample_indices(total, num_samples=num_samples)
    return frames[idx], idx
