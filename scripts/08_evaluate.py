from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from common_llsp import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rep-count predictions.")
    parser.add_argument(
        "--pred-csvs",
        nargs="+",
        default=[
            "outputs/04_results/baseline_peak.csv",
            "outputs/04_results/proposed_phase_vote.csv",
        ],
        help="Prediction CSV paths to evaluate.",
    )
    parser.add_argument(
        "--out-table-csv",
        type=Path,
        default=Path("outputs/04_results/metrics_table.csv"),
        help="Output metrics table CSV.",
    )
    parser.add_argument(
        "--out-summary-md",
        type=Path,
        default=Path("outputs/04_results/metrics_summary.md"),
        help="Output markdown summary.",
    )
    parser.add_argument("--event-k", type=int, default=10, help="Tolerance (frames) for event match.")
    parser.add_argument(
        "--failure-top-n",
        type=int,
        default=3,
        help="Top-N failure videos per method/split to include in report.",
    )
    return parser.parse_args()


def parse_periods(value: str) -> List[Tuple[int, int]]:
    if not isinstance(value, str) or not value:
        return []
    try:
        arr = json.loads(value)
    except json.JSONDecodeError:
        return []
    out: List[Tuple[int, int]] = []
    for p in arr:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            s = int(p[0])
            e = int(p[1])
            if s < e:
                out.append((s, e))
    return out


def count_metrics(df: pd.DataFrame) -> Dict[str, float]:
    y_true = df["true_count"].to_numpy(dtype=float)
    y_pred = df["pred_count"].to_numpy(dtype=float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    oboa = float(np.mean(np.abs(err) <= 1.0))
    return {"mae": mae, "rmse": rmse, "oboa": oboa}


def event_match_counts(pred: List[Tuple[int, int]], gt: List[Tuple[int, int]], k: int) -> Tuple[int, int, int]:
    matched_gt = set()
    tp = 0
    for ps, pe in pred:
        best_idx = None
        best_cost = None
        for i, (gs, ge) in enumerate(gt):
            if i in matched_gt:
                continue
            if abs(ps - gs) <= k and abs(pe - ge) <= k:
                cost = abs(ps - gs) + abs(pe - ge)
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_idx = i
        if best_idx is not None:
            matched_gt.add(best_idx)
            tp += 1
    fp = max(0, len(pred) - tp)
    fn = max(0, len(gt) - tp)
    return tp, fp, fn


def event_metrics(df: pd.DataFrame, k: int) -> Dict[str, float]:
    tp = fp = fn = 0
    for _, row in df.iterrows():
        pred = parse_periods(row.get("pred_periods_json", ""))
        gt = parse_periods(row.get("true_periods_json", ""))
        tpi, fpi, fni = event_match_counts(pred, gt, k)
        tp += tpi
        fp += fpi
        fn += fni
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "event_precision": float(prec),
        "event_recall": float(rec),
        "event_f1": float(f1),
        "event_tp": int(tp),
        "event_fp": int(fp),
        "event_fn": int(fn),
    }


def collect_failure_cases(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    out = []
    for split, g in df.groupby("split"):
        worst = g.sort_values(["abs_err", "video_id"], ascending=[False, True]).head(top_n)
        out.append(worst)
    if not out:
        return df.iloc[0:0].copy()
    return pd.concat(out, ignore_index=True)


def main() -> None:
    args = parse_args()
    pred_paths = [Path(p) for p in args.pred_csvs]

    metrics_rows: List[Dict] = []
    summary_lines: List[str] = []
    failure_dir = Path("outputs/04_results/failure_cases")
    ensure_dir(failure_dir)

    for pred_path in pred_paths:
        if not pred_path.exists():
            print(f"Skipping missing prediction file: {pred_path}")
            continue

        df = pd.read_csv(pred_path)
        if "method" in df.columns:
            method = str(df["method"].iloc[0])
        else:
            method = pred_path.stem

        summary_lines.append(f"## {method}")

        for split_name, split_df in [("overall", df)] + [(s, g) for s, g in df.groupby("split")]:
            cm = count_metrics(split_df)
            em = event_metrics(split_df, args.event_k)
            metrics_rows.append(
                {
                    "method": method,
                    "split": split_name,
                    "n_videos": int(len(split_df)),
                    **cm,
                    **em,
                }
            )
            summary_lines.append(
                f"- `{split_name}` n={len(split_df)} | MAE={cm['mae']:.3f} RMSE={cm['rmse']:.3f} OBOA={cm['oboa']:.3f} "
                f"| P/R/F1@K={args.event_k}: {em['event_precision']:.3f}/{em['event_recall']:.3f}/{em['event_f1']:.3f}"
            )

        failures = collect_failure_cases(df, top_n=args.failure_top_n)
        failure_csv = failure_dir / f"{method}_top_failures.csv"
        failures.to_csv(failure_csv, index=False)
        summary_lines.append(f"- failure cases: `{failure_csv}`")
        summary_lines.append("")

    metrics_df = pd.DataFrame(metrics_rows)
    ensure_dir(args.out_table_csv.parent)
    metrics_df.to_csv(args.out_table_csv, index=False)

    ensure_dir(args.out_summary_md.parent)
    args.out_summary_md.write_text("\n".join(summary_lines).strip() + "\n", encoding="utf-8")

    print(f"Wrote metrics table: {args.out_table_csv}")
    print(f"Wrote metrics summary: {args.out_summary_md}")


if __name__ == "__main__":
    main()
