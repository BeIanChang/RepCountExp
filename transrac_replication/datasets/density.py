from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def _normal_cdf(x: np.ndarray | float, mu: float, sigma: float) -> np.ndarray | float:
    z = (x - mu) / (sigma * math.sqrt(2.0))
    if isinstance(z, np.ndarray):
        return 0.5 * (1.0 + np.vectorize(math.erf)(z))
    return 0.5 * (1.0 + math.erf(z))


def cycle_to_gaussian(start: float, end: float, min_sigma: float = 1e-3) -> Tuple[float, float]:
    if end <= start:
        raise ValueError(f"Invalid cycle boundary: start={start}, end={end}")
    mu = 0.5 * (start + end)
    sigma = max(min_sigma, (end - start) / 6.0)
    return mu, sigma


def gaussian_bin_integral(k: int, mu: float, sigma: float) -> float:
    left = k - 0.5
    right = k + 0.5
    cdf_r = _normal_cdf(right, mu=mu, sigma=sigma)
    cdf_l = _normal_cdf(left, mu=mu, sigma=sigma)
    return float(cdf_r - cdf_l)


def map_cycles_to_sample_bins(
    cycles: Sequence[Tuple[float, float]],
    total_frames: int,
    num_bins: int,
) -> List[Tuple[float, float]]:
    """Map original frame boundary pairs to sampled-bin coordinates."""
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")

    if total_frames == 1:
        return [(0.0, 0.0) for _ in cycles]

    scale = (num_bins - 1) / float(total_frames - 1)
    mapped: List[Tuple[float, float]] = []
    for s, e in cycles:
        sm = float(np.clip(s * scale, 0.0, num_bins - 1))
        em = float(np.clip(e * scale, 0.0, num_bins - 1))
        if em < sm:
            sm, em = em, sm
        mapped.append((sm, em))
    return mapped


def build_density_map(
    cycles: Sequence[Tuple[float, float]],
    num_bins: int,
) -> np.ndarray:
    """Build 1D density map from cycle boundary pairs on sampled-bin axis."""
    density = np.zeros((num_bins,), dtype=np.float32)
    if not cycles:
        return density

    for start, end in cycles:
        if end <= start:
            continue
        mu, sigma = cycle_to_gaussian(start, end)
        for k in range(num_bins):
            density[k] += gaussian_bin_integral(k, mu=mu, sigma=sigma)
    return density


def build_density_from_original_cycles(
    cycles_original: Iterable[Tuple[float, float]],
    total_frames: int,
    num_bins: int = 64,
) -> np.ndarray:
    cycles = list(cycles_original)
    mapped = map_cycles_to_sample_bins(cycles, total_frames=total_frames, num_bins=num_bins)
    return build_density_map(mapped, num_bins=num_bins)
