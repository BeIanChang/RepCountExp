from __future__ import annotations

import torch
import torch.nn.functional as F


def density_mse_loss(pred_density: torch.Tensor, gt_density: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_density, gt_density)
