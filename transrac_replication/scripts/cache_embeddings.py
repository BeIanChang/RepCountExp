from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transrac_replication.datasets.repcount_dataset import RepCountTransRACDataset
from transrac_replication.models.transrac_model import TransRACConfig, TransRACModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen encoder embeddings for TransRAC scales.")
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("transrac_replication/experiments/repcount_manifest.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("transrac_replication/experiments/cache_embeddings"),
    )
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--use-torchvision-backbone", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0, help="If >0, cache at most this many samples per split.")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    cfg = TransRACConfig(num_frames=args.num_frames, use_torchvision_backbone=args.use_torchvision_backbone)
    model = TransRACModel(cfg).to(device)
    model.eval()

    manifest = pd.read_csv(args.manifest_csv)
    for split in args.splits:
        split_df = manifest[manifest["split"] == split].reset_index(drop=True)
        if len(split_df) == 0:
            continue

        dataset = RepCountTransRACDataset(
            manifest_csv=args.manifest_csv,
            split=split,
            num_frames=args.num_frames,
            frame_size=args.frame_size,
        )
        out_split_dir = args.out_dir / split
        out_split_dir.mkdir(parents=True, exist_ok=True)

        limit = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)
        for i in range(limit):
            sample = dataset[i]
            video_id = str(split_df.iloc[i]["video_id"])
            out_path = out_split_dir / f"{video_id}.npz"

            v1 = sample["v1"].unsqueeze(0).to(device)
            v4 = sample["v4"].unsqueeze(0).to(device)
            v8 = sample["v8"].unsqueeze(0).to(device)

            x1 = model.encoder(v1).squeeze(0).cpu().numpy()
            x4 = model.encoder(v4).squeeze(0).cpu().numpy()
            x8 = model.encoder(v8).squeeze(0).cpu().numpy()

            np.savez_compressed(
                out_path,
                x_v1=x1,
                x_v4=x4,
                x_v8=x8,
                density_gt=sample["density_gt"].cpu().numpy(),
                gt_count=float(sample["gt_count"].item()),
            )

            if (i + 1) % 10 == 0 or (i + 1) == limit:
                print(f"[{split}] cached {i + 1}/{limit}")


if __name__ == "__main__":
    main()
