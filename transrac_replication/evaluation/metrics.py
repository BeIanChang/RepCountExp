from __future__ import annotations

import numpy as np


def mae(pred_counts: np.ndarray, gt_counts: np.ndarray) -> float:
    pred = np.asarray(pred_counts, dtype=np.float32)
    gt = np.asarray(gt_counts, dtype=np.float32)
    return float(np.mean(np.abs(pred - gt)))


def mae_normalized(pred_counts: np.ndarray, gt_counts: np.ndarray, eps: float = 1e-6) -> float:
    """Paper-style normalized MAE: mean(|pred-gt| / gt)."""
    pred = np.asarray(pred_counts, dtype=np.float32)
    gt = np.asarray(gt_counts, dtype=np.float32)
    denom = np.maximum(gt, eps)
    return float(np.mean(np.abs(pred - gt) / denom))


def obo(pred_counts: np.ndarray, gt_counts: np.ndarray) -> float:
    pred = np.asarray(pred_counts, dtype=np.float32)
    gt = np.asarray(gt_counts, dtype=np.float32)
    return float(np.mean(np.abs(pred - gt) <= 1.0))
