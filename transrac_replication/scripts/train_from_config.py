from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from transrac_replication.scripts.train_transrac import main as train_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch TransRAC training from YAML config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("transrac_replication/configs/transrac_repcount_a.yaml"),
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("transrac_replication/experiments/repcount_manifest.csv"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use-torchvision-backbone", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    seed = cfg.get("experiment", {}).get("seed", 42)

    cmd = [
        "--manifest-csv",
        str(args.manifest_csv),
        "--batch-size",
        str(train_cfg.get("batch_size", 16)),
        "--max-steps",
        str(train_cfg.get("max_steps", 16000)),
        "--lr",
        str(train_cfg.get("learning_rate", 8e-6)),
        "--num-frames",
        str(data_cfg.get("num_frames", 64)),
        "--frame-size",
        str(data_cfg.get("frame_size", 224)),
        "--seed",
        str(seed),
        "--device",
        str(args.device),
    ]

    if args.use_torchvision_backbone:
        cmd.append("--use-torchvision-backbone")
    if args.cache_dir is not None:
        cmd.extend(["--cache-dir", str(args.cache_dir)])

    import sys

    sys.argv = ["train_transrac.py", *cmd]
    train_main()


if __name__ == "__main__":
    main()
