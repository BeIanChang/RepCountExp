from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TransRAC replication manifest from parsed index.")
    parser.add_argument(
        "--parsed-index-csv",
        type=Path,
        default=Path("outputs/00_index/master_index.parsed.csv"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("transrac_replication/experiments/repcount_manifest.csv"),
    )
    parser.add_argument(
        "--actions",
        nargs="*",
        default=None,
        help="Optional action filter list.",
    )
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="Drop rows with empty periods or missing count.",
    )
    return parser.parse_args()


def has_periods(s: str) -> bool:
    if not isinstance(s, str) or not s:
        return False
    try:
        v = json.loads(s)
    except json.JSONDecodeError:
        return False
    return isinstance(v, list) and len(v) > 0


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.parsed_index_csv)

    if args.actions:
        df = df[df["canonical_action"].isin(args.actions)].copy()

    if args.drop_empty:
        df = df[df["label_count"].notna()].copy()
        df = df[df["periods_json"].map(has_periods)].copy()

    keep_cols = [
        "video_id",
        "split",
        "video_path",
        "canonical_action",
        "label_count",
        "n_frames",
        "fps",
        "periods_json",
    ]
    out = df[keep_cols].reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote manifest: {args.out_csv} ({len(out)} rows)")


if __name__ == "__main__":
    main()
