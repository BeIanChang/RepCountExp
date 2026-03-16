from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from common_llsp import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate phase-native and baseline-peak-online with confidence gating.")
    parser.add_argument(
        "--phase-csv",
        type=Path,
        default=Path("outputs/04_results/phase_native_online_partA_test_all_softpenalty.csv"),
    )
    parser.add_argument(
        "--peak-online-csv",
        type=Path,
        default=Path("outputs/04_results/baseline_peak_online_partA_test_all.csv"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/04_results/phase_native_peak_online_hybrid_partA_test_all.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("outputs/04_results/phase_native_peak_online_hybrid_summary.json"),
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=Path("outputs/04_results/phase_native_peak_online_hybrid_summary.md"),
    )
    return parser.parse_args()


def fold_id(video_id: str) -> int:
    h = hashlib.md5(video_id.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 5


def metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    ae = np.abs(pred - gt)
    return {
        "mae": float(np.mean(ae)),
        "mae_norm": float(np.mean(ae / np.clip(gt, 1.0, None))),
        "mae_norm_p1": float(np.mean(ae / (gt + 1e-1))),
        "obo": float(np.mean(ae <= 1.0)),
    }


def candidate_conf_columns(df: pd.DataFrame) -> List[str]:
    preferred = [
        "mean_confidence",
        "conf_phase",
        "conf_r",
        "conf_var",
        "conf_flip",
        "conf_pause",
    ]
    return [c for c in preferred if c in df.columns]


def main() -> None:
    args = parse_args()
    phase = pd.read_csv(args.phase_csv)
    peak = pd.read_csv(args.peak_online_csv)

    merged = phase.merge(
        peak[["video_id", "pred_count"]].rename(columns={"pred_count": "peak_pred_count"}),
        on="video_id",
        how="inner",
    ).copy()

    merged["gt"] = merged["true_count"].astype(float)
    merged["phase_pred"] = merged["pred_count"].astype(float)
    merged["peak_pred_count"] = merged["peak_pred_count"].astype(float)
    merged["fold"] = merged["video_id"].astype(str).map(fold_id)

    calib = merged[merged["fold"] <= 2].copy()
    holdout = merged[merged["fold"] >= 3].copy()

    conf_cols = candidate_conf_columns(merged)
    if not conf_cols:
        raise ValueError("No confidence columns found in phase csv")

    best = None
    search_rows: List[Dict[str, float | str]] = []

    for conf_col in conf_cols:
        cvals = calib[conf_col].to_numpy(dtype=float)
        ths = np.unique(np.quantile(cvals, np.linspace(0.05, 0.90, 18)))
        for th in ths:
            pred = np.where(calib[conf_col].to_numpy(dtype=float) < th, calib["peak_pred_count"], calib["phase_pred"])
            m = metrics(pred, calib["gt"].to_numpy(dtype=float))
            row = {
                "conf_col": conf_col,
                "threshold": float(th),
                "calib_mae_norm_p1": m["mae_norm_p1"],
                "calib_mae": m["mae"],
                "calib_obo": m["obo"],
            }
            search_rows.append(row)
            if best is None or row["calib_mae_norm_p1"] < best["calib_mae_norm_p1"]:
                best = row

    assert best is not None
    best_col = str(best["conf_col"])
    best_th = float(best["threshold"])

    for split_df in (calib, holdout, merged):
        split_df["hybrid_pred"] = np.where(
            split_df[best_col].to_numpy(dtype=float) < best_th,
            split_df["peak_pred_count"],
            split_df["phase_pred"],
        )
        split_df["fallback_used"] = split_df[best_col].to_numpy(dtype=float) < best_th

    out = merged[
        [
            "video_id",
            "split",
            "action",
            "true_count",
            "phase_pred",
            "peak_pred_count",
            best_col,
            "hybrid_pred",
            "fallback_used",
        ]
    ].copy()
    out = out.rename(columns={best_col: "selected_confidence"})
    out["abs_err_phase"] = (out["phase_pred"] - out["true_count"]).abs()
    out["abs_err_peak_online"] = (out["peak_pred_count"] - out["true_count"]).abs()
    out["abs_err_hybrid"] = (out["hybrid_pred"] - out["true_count"]).abs()

    ensure_dir(args.out_csv.parent)
    out.to_csv(args.out_csv, index=False)

    def pack(df: pd.DataFrame) -> Dict[str, float]:
        gt = df["gt"].to_numpy(dtype=float)
        return {
            "n": float(len(df)),
            "fallback_fraction": float(np.mean(df["fallback_used"].to_numpy(dtype=bool))),
            "phase_mae_norm_p1": metrics(df["phase_pred"].to_numpy(dtype=float), gt)["mae_norm_p1"],
            "peak_online_mae_norm_p1": metrics(df["peak_pred_count"].to_numpy(dtype=float), gt)["mae_norm_p1"],
            "hybrid_mae_norm_p1": metrics(df["hybrid_pred"].to_numpy(dtype=float), gt)["mae_norm_p1"],
            "phase_obo": metrics(df["phase_pred"].to_numpy(dtype=float), gt)["obo"],
            "peak_online_obo": metrics(df["peak_pred_count"].to_numpy(dtype=float), gt)["obo"],
            "hybrid_obo": metrics(df["hybrid_pred"].to_numpy(dtype=float), gt)["obo"],
            "phase_mae": metrics(df["phase_pred"].to_numpy(dtype=float), gt)["mae"],
            "peak_online_mae": metrics(df["peak_pred_count"].to_numpy(dtype=float), gt)["mae"],
            "hybrid_mae": metrics(df["hybrid_pred"].to_numpy(dtype=float), gt)["mae"],
        }

    summary = {
        "phase_csv": str(args.phase_csv),
        "peak_online_csv": str(args.peak_online_csv),
        "selected_confidence_col": best_col,
        "selected_threshold": best_th,
        "calib": pack(calib),
        "holdout": pack(holdout),
        "full": pack(merged),
        "search_rows": search_rows,
        "out_csv": str(args.out_csv),
    }

    ensure_dir(args.summary_json.parent)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Phase-native + Peak-online Hybrid", "",
        f"Selected confidence: `{best_col}`", f"Threshold: `{best_th:.6f}`", "",
        "| Split | n | Fallback% | Phase MAE | Peak MAE | Hybrid MAE | Phase MAE_norm_p1 | Peak MAE_norm_p1 | Hybrid MAE_norm_p1 | Phase OBO | Peak OBO | Hybrid OBO |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in ["calib", "holdout", "full"]:
        s = summary[k]
        lines.append(
            f"| {k} | {int(s['n'])} | {s['fallback_fraction']:.3f} | {s['phase_mae']:.4f} | {s['peak_online_mae']:.4f} | {s['hybrid_mae']:.4f} | "
            f"{s['phase_mae_norm_p1']:.4f} | {s['peak_online_mae_norm_p1']:.4f} | {s['hybrid_mae_norm_p1']:.4f} | {s['phase_obo']:.4f} | {s['peak_online_obo']:.4f} | {s['hybrid_obo']:.4f} |"
        )

    ensure_dir(args.summary_md.parent)
    args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.summary_json}")
    print(f"Wrote {args.summary_md}")


if __name__ == "__main__":
    main()
