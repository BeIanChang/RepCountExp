from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from common_llsp import l_columns, read_annotations, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse LLSP L-columns into periods.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/LLSP"),
        help="Dataset root containing annotation/ and video/.",
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=Path("outputs/00_index/master_index.csv"),
        help="Master index from 01_build_index.py.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/00_index/master_index.parsed.csv"),
        help="Output parsed index CSV.",
    )
    parser.add_argument(
        "--quality-json",
        type=Path,
        default=Path("outputs/00_index/data_quality_report.json"),
        help="Data quality report JSON to update.",
    )
    return parser.parse_args()


def parse_row_periods(l_values: List[float], n_frames: float | int | None) -> Tuple[List[List[int]], Dict[str, Any]]:
    clean_vals = [int(round(v)) for v in l_values if pd.notna(v)]
    info: Dict[str, Any] = {
        "odd_location_values": False,
        "clipped_pairs": 0,
        "invalid_pairs": 0,
    }

    if len(clean_vals) % 2 == 1:
        info["odd_location_values"] = True
        clean_vals = clean_vals[:-1]

    periods: List[List[int]] = []
    max_frame = None
    if n_frames is not None and pd.notna(n_frames):
        max_frame = int(n_frames)

    for i in range(0, len(clean_vals), 2):
        start, end = clean_vals[i], clean_vals[i + 1]
        original = (start, end)
        if max_frame is not None:
            start = int(np.clip(start, 0, max_frame))
            end = int(np.clip(end, 0, max_frame))
            if (start, end) != original:
                info["clipped_pairs"] += 1
        if start < end:
            periods.append([start, end])
        else:
            info["invalid_pairs"] += 1

    return periods, info


def main() -> None:
    args = parse_args()

    index_df = pd.read_csv(args.index_csv)
    ann = read_annotations(args.data_root)
    l_cols = l_columns(ann)

    ann = ann.rename(columns={"name": "video_name"})
    merge_cols = ["split", "video_name"] + l_cols
    ann_small = ann[merge_cols].copy()
    merged = index_df.merge(ann_small, on=["split", "video_name"], how="left", validate="one_to_one")

    periods_json: List[str] = []
    n_periods: List[int] = []
    period_count_diff: List[float] = []
    parse_status: List[str] = []

    odd_rows = 0
    clipped_rows = 0
    invalid_rows = 0

    for _, row in merged.iterrows():
        values = [row[c] for c in l_cols]
        periods, info = parse_row_periods(values, row.get("n_frames"))
        periods_json.append(json.dumps(periods, separators=(",", ":")))
        n_periods.append(len(periods))

        label_count = pd.to_numeric(row.get("label_count"), errors="coerce")
        if pd.notna(label_count):
            period_count_diff.append(float(len(periods) - label_count))
        else:
            period_count_diff.append(np.nan)

        if info["invalid_pairs"] > 0:
            status = "invalid_pairs_removed"
        elif info["odd_location_values"]:
            status = "odd_values_trimmed"
        elif len(periods) == 0:
            status = "no_periods"
        else:
            status = "ok"
        parse_status.append(status)

        odd_rows += int(info["odd_location_values"])
        clipped_rows += int(info["clipped_pairs"] > 0)
        invalid_rows += int(info["invalid_pairs"] > 0)

    merged["periods_json"] = periods_json
    merged["n_periods"] = n_periods
    merged["period_count_diff"] = period_count_diff
    merged["parse_status"] = parse_status

    drop_cols = l_cols
    out_df = merged.drop(columns=drop_cols)
    out_df.to_csv(args.out_csv, index=False)

    mismatch = out_df[pd.to_numeric(out_df["period_count_diff"], errors="coerce") != 0]
    mismatch = mismatch[pd.to_numeric(mismatch["period_count_diff"], errors="coerce").notna()]

    parse_payload = {
        "parse_summary": {
            "rows_total": int(len(out_df)),
            "rows_ok": int((out_df["parse_status"] == "ok").sum()),
            "rows_no_periods": int((out_df["parse_status"] == "no_periods").sum()),
            "rows_odd_values_trimmed": int((out_df["parse_status"] == "odd_values_trimmed").sum()),
            "rows_invalid_pairs_removed": int((out_df["parse_status"] == "invalid_pairs_removed").sum()),
            "rows_with_any_odd_values": int(odd_rows),
            "rows_with_clipped_pairs": int(clipped_rows),
            "rows_with_invalid_pairs": int(invalid_rows),
            "rows_count_mismatch": int(len(mismatch)),
        },
        "examples": {
            "count_mismatch_rows": mismatch[
                ["split", "video_name", "label_count", "n_periods", "period_count_diff", "parse_status"]
            ]
            .head(20)
            .to_dict(orient="records")
        },
    }

    if args.quality_json.exists():
        base = json.loads(args.quality_json.read_text(encoding="utf-8"))
    else:
        base = {}

    merged_examples = {}
    merged_examples.update(base.get("examples", {}))
    merged_examples.update(parse_payload.get("examples", {}))

    base.update({k: v for k, v in parse_payload.items() if k != "examples"})
    base["examples"] = merged_examples
    write_json(args.quality_json, base)

    print(f"Wrote parsed index: {args.out_csv}")
    print(f"Updated quality report: {args.quality_json}")


if __name__ == "__main__":
    main()
