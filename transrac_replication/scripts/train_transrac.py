from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from transrac_replication.datasets.cached_dataset import CachedEmbeddingDataset
from transrac_replication.datasets.repcount_dataset import RepCountTransRACDataset
from transrac_replication.models.transrac_model import TransRACConfig, TransRACModel
from transrac_replication.training.engine import TrainConfig, run_eval, run_train_loop
from transrac_replication.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TransRAC replication model (initial skeleton).")
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("transrac_replication/experiments/repcount_manifest.csv"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=8e-6)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--eval-split",
        type=str,
        default="valid",
        choices=["valid", "test"],
        help="Evaluation split to report after training.",
    )
    parser.add_argument(
        "--use-torchvision-backbone",
        action="store_true",
        help="Use frozen torchvision swin3d_t encoder instead of lightweight placeholder.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache dir with precomputed embeddings; if set, training uses cached dataset.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("transrac_replication/experiments/transrac"),
        help="Output directory for logs/checkpoints/summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.cache_dir is not None:
        train_ds = CachedEmbeddingDataset(cache_dir=args.cache_dir, split="train")
        val_ds = CachedEmbeddingDataset(cache_dir=args.cache_dir, split=args.eval_split)
    else:
        train_ds = RepCountTransRACDataset(
            manifest_csv=args.manifest_csv,
            split="train",
            num_frames=args.num_frames,
            frame_size=args.frame_size,
        )
        val_ds = RepCountTransRACDataset(
            manifest_csv=args.manifest_csv,
            split=args.eval_split,
            num_frames=args.num_frames,
            frame_size=args.frame_size,
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TransRACModel(
        TransRACConfig(num_frames=args.num_frames, use_torchvision_backbone=args.use_torchvision_backbone)
    )
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        device=args.device,
        out_dir=str(args.out_dir),
    )

    train_stats = run_train_loop(model, train_loader, cfg=train_cfg)
    eval_stats = run_eval(model, val_loader, device=args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "model": "transrac",
                "train_stats": train_stats,
                "eval_stats": eval_stats,
                "config": {
                    "batch_size": args.batch_size,
                    "max_steps": args.max_steps,
                    "lr": args.lr,
                    "device": args.device,
                    "eval_split": args.eval_split,
                    "cache_dir": None if args.cache_dir is None else str(args.cache_dir),
                    "use_torchvision_backbone": bool(args.use_torchvision_backbone),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("train_stats", train_stats)
    print("eval_stats", eval_stats)
    print("summary", summary_path)


if __name__ == "__main__":
    main()
