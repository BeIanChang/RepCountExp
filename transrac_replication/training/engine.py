from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from transrac_replication.evaluation.metrics import mae, mae_normalized, obo
from transrac_replication.training.losses import density_mse_loss


@dataclass
class TrainConfig:
    batch_size: int = 2
    max_steps: int = 100
    lr: float = 8e-6
    device: str = "cpu"
    eval_interval: int = 0
    out_dir: str = "transrac_replication/experiments/runs"


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def run_train_loop(model: torch.nn.Module, train_loader: DataLoader, cfg: TrainConfig) -> Dict[str, Union[float, str]]:
    device = torch.device(cfg.device)
    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.max_steps), eta_min=1e-7)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    if log_path.exists():
        log_path.unlink()
    log_path.write_text("step,loss,lr\n", encoding="utf-8")

    model.train()
    step = 0
    last_loss = 0.0
    while step < cfg.max_steps:
        for batch in train_loader:
            batch = _to_device(batch, device)
            if "v1" in batch:
                density_pred, _ = model({"v1": batch["v1"], "v4": batch["v4"], "v8": batch["v8"]})
            else:
                density_pred, _ = model.forward_from_embeddings(
                    {"v1": batch["x_v1"], "v4": batch["x_v4"], "v8": batch["x_v8"]}
                )
            loss = density_mse_loss(density_pred, batch["density_gt"])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            last_loss = float(loss.detach().cpu().item())
            step += 1
            lr = float(optimizer.param_groups[0]["lr"])
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"{step},{last_loss:.8f},{lr:.10f}\n")

            if step >= cfg.max_steps:
                break

    ckpt_path = out_dir / "model_last.pt"
    torch.save(model.state_dict(), ckpt_path)

    return {"last_loss": last_loss, "steps": float(step), "checkpoint": str(ckpt_path), "log_csv": str(log_path)}


@torch.no_grad()
def run_eval(model: torch.nn.Module, loader: DataLoader, device: str = "cpu") -> Dict[str, float]:
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    pred_counts = []
    gt_counts = []
    for batch in loader:
        batch = _to_device(batch, dev)
        if "v1" in batch:
            density_pred, _ = model({"v1": batch["v1"], "v4": batch["v4"], "v8": batch["v8"]})
        else:
            density_pred, _ = model.forward_from_embeddings(
                {"v1": batch["x_v1"], "v4": batch["x_v4"], "v8": batch["x_v8"]}
            )
        pred_count = density_pred.sum(dim=1).detach().cpu().numpy()
        gt_count = batch["gt_count"].detach().cpu().numpy()
        pred_counts.extend(pred_count.tolist())
        gt_counts.extend(gt_count.tolist())

    return {
        "mae": mae(pred_counts=np.array(pred_counts), gt_counts=np.array(gt_counts)),
        "mae_normalized": mae_normalized(pred_counts=np.array(pred_counts), gt_counts=np.array(gt_counts)),
        "obo": obo(pred_counts=np.array(pred_counts), gt_counts=np.array(gt_counts)),
    }
