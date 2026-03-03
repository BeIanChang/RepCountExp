from __future__ import annotations

import argparse
from pathlib import Path
import json

from torch.utils.data import DataLoader

from transrac_replication.datasets.cached_dataset import CachedEmbeddingDataset
from transrac_replication.models.baselines import BaselineConfig, build_baseline
from transrac_replication.training.engine import TrainConfig, run_eval, run_train_loop
from transrac_replication.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train cached-embedding baselines (RepNet-like / Zhang-like).")
    parser.add_argument("--model", type=str, choices=["repnet", "zhang"], required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("transrac_replication/experiments/cache_embeddings_smoke"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
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
        "--out-dir",
        type=Path,
        default=Path("transrac_replication/experiments/baselines"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_ds = CachedEmbeddingDataset(cache_dir=args.cache_dir, split="train")
    val_ds = CachedEmbeddingDataset(cache_dir=args.cache_dir, split=args.eval_split)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_baseline(args.model, BaselineConfig())
    out_dir = args.out_dir / args.model
    cfg = TrainConfig(
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        device=args.device,
        out_dir=str(out_dir),
    )
    train_stats = run_train_loop(model, train_loader, cfg)
    eval_stats = run_eval(model, val_loader, device=args.device)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "train_stats": train_stats,
                "eval_stats": eval_stats,
                "config": {
                    "batch_size": args.batch_size,
                    "max_steps": args.max_steps,
                    "lr": args.lr,
                    "device": args.device,
                    "eval_split": args.eval_split,
                    "cache_dir": str(args.cache_dir),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("model", args.model)
    print("train_stats", train_stats)
    print("eval_stats", eval_stats)
    print("summary", summary_path)


if __name__ == "__main__":
    main()
