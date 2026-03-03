from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transrac_replication.datasets.cached_dataset import CachedEmbeddingDataset
from transrac_replication.models.baselines import BaselineConfig, build_baseline
from transrac_replication.models.transrac_model import TransRACConfig, TransRACModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cached-embedding checkpoint with detailed metrics.")
    parser.add_argument("--model", type=str, choices=["transrac", "repnet", "zhang"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("transrac_replication/experiments/cache_embeddings_full"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def build_model(name: str) -> torch.nn.Module:
    if name == "transrac":
        return TransRACModel(TransRACConfig())
    return build_baseline(name, BaselineConfig())


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model = build_model(args.model)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    ds = CachedEmbeddingDataset(cache_dir=args.cache_dir, split=args.split)
    rows: list[dict[str, float | str]] = []
    for i, p in enumerate(ds.files, start=1):
        sample = ds[i - 1]
        emb = {
            "v1": sample["x_v1"].unsqueeze(0).to(device),
            "v4": sample["x_v4"].unsqueeze(0).to(device),
            "v8": sample["x_v8"].unsqueeze(0).to(device),
        }
        density, _ = model.forward_from_embeddings(emb)
        pred = float(density.sum(dim=1).item())
        gt = float(sample["gt_count"].item())
        rows.append({"video": p.stem, "gt_count": gt, "pred_count": pred})
        if i % 50 == 0 or i == len(ds.files):
            print(f"processed {i}/{len(ds.files)}")

    out = pd.DataFrame(rows)
    out["abs_err"] = (out["pred_count"] - out["gt_count"]).abs()
    out["pred_round"] = out["pred_count"].round()
    out["abs_err_round"] = (out["pred_round"] - out["gt_count"]).abs()
    out["is_obo"] = out["abs_err"] <= 1.0
    out["is_obo_round"] = out["abs_err_round"] <= 1.0
    out["is_exact_round"] = out["abs_err_round"] <= 0.0
    out["norm_err"] = out["abs_err"] / out["gt_count"].clip(lower=1.0)
    out["norm_err_p1"] = out["abs_err"] / (out["gt_count"] + 1e-1)
    out["norm_err_round"] = out["abs_err_round"] / out["gt_count"].clip(lower=1.0)
    out["norm_err_round_p1"] = out["abs_err_round"] / (out["gt_count"] + 1e-1)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    summary = {
        "model": args.model,
        "split": args.split,
        "n_videos": int(len(out)),
        "mae_raw": float(out["abs_err"].mean()),
        "mae_norm": float(out["norm_err"].mean()),
        "mae_norm_p1": float(out["norm_err_p1"].mean()),
        "obo": float(out["is_obo"].mean()),
        "mae_round": float(out["abs_err_round"].mean()),
        "mae_round_norm": float(out["norm_err_round"].mean()),
        "mae_round_norm_p1": float(out["norm_err_round_p1"].mean()),
        "obo_round": float(out["is_obo_round"].mean()),
        "exact_round": float(out["is_exact_round"].mean()),
        "checkpoint": str(args.checkpoint),
        "cache_dir": str(args.cache_dir),
        "out_csv": str(args.out_csv),
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
