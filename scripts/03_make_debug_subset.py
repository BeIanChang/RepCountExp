from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from common_llsp import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create balanced LLSP debug subset.")
    parser.add_argument(
        "--parsed-index-csv",
        type=Path,
        default=Path("outputs/00_index/master_index.parsed.csv"),
        help="Parsed index file from 02_parse_periods.py.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/00_index/subset_debug.csv"),
        help="Output CSV path for debug subset.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("outputs/00_index/subset_debug_report.json"),
        help="Output report path.",
    )
    parser.add_argument("--actions", nargs="+", default=["squat", "push_up", "pull_up"])
    parser.add_argument("--train", type=int, default=20)
    parser.add_argument("--valid", type=int, default=10)
    parser.add_argument("--test", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow rows with empty periods or missing count.",
    )
    return parser.parse_args()


def balanced_sample(df: pd.DataFrame, actions: List[str], target_n: int, rng: np.random.Generator) -> pd.DataFrame:
    if target_n <= 0 or df.empty:
        return df.iloc[0:0].copy()

    pools = {action: df[df["canonical_action"] == action].copy() for action in actions}
    selected_chunks: List[pd.DataFrame] = []

    base = target_n // len(actions)
    remainder = target_n % len(actions)

    for idx, action in enumerate(actions):
        take = base + (1 if idx < remainder else 0)
        pool = pools[action]
        if pool.empty or take <= 0:
            continue
        n_take = min(take, len(pool))
        chosen_idx = rng.choice(pool.index.to_numpy(), size=n_take, replace=False)
        selected = pool.loc[chosen_idx]
        selected_chunks.append(selected)
        pools[action] = pool.drop(index=chosen_idx)

    selected = pd.concat(selected_chunks, ignore_index=False) if selected_chunks else df.iloc[0:0].copy()

    still_need = max(0, target_n - len(selected))
    if still_need > 0:
        remaining_pool = df.drop(index=selected.index, errors="ignore")
        if not remaining_pool.empty:
            n_take = min(still_need, len(remaining_pool))
            chosen_idx = rng.choice(remaining_pool.index.to_numpy(), size=n_take, replace=False)
            selected = pd.concat([selected, remaining_pool.loc[chosen_idx]], ignore_index=False)

    return selected.sort_values(["split", "canonical_action", "video_id"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.parsed_index_csv)

    keep_mask = df["canonical_action"].isin(args.actions)
    if not args.allow_empty:
        keep_mask &= df["label_count"].notna()
        keep_mask &= df["n_periods"] > 0
    filtered = df[keep_mask].copy()

    split_targets = {"train": args.train, "valid": args.valid, "test": args.test}
    chunks: List[pd.DataFrame] = []
    deficits: Dict[str, int] = {}

    for split, target in split_targets.items():
        split_df = filtered[filtered["split"] == split].copy()
        chosen = balanced_sample(split_df, args.actions, target, rng)
        chunks.append(chosen)
        deficits[split] = int(max(0, target - len(chosen)))

    subset = pd.concat(chunks, ignore_index=True)
    ensure_dir(args.out_csv.parent)
    subset.to_csv(args.out_csv, index=False)

    report = {
        "config": {
            "actions": args.actions,
            "targets": split_targets,
            "seed": args.seed,
            "allow_empty": bool(args.allow_empty),
        },
        "result": {
            "subset_rows": int(len(subset)),
            "rows_per_split": {k: int(v) for k, v in subset.groupby("split").size().to_dict().items()},
            "rows_per_action": {k: int(v) for k, v in subset.groupby("canonical_action").size().to_dict().items()},
            "rows_per_split_action": {
                f"{s}:{a}": int(v)
                for (s, a), v in subset.groupby(["split", "canonical_action"]).size().to_dict().items()
            },
            "deficits": deficits,
        },
    }
    write_json(args.report_json, report)

    print(f"Wrote subset ({len(subset)} rows): {args.out_csv}")
    print(f"Wrote subset report: {args.report_json}")


if __name__ == "__main__":
    main()
