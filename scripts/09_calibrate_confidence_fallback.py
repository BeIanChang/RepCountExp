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
    parser = argparse.ArgumentParser(description="Calibrate confidence-gated fallback from phase method to RepNet.")
    parser.add_argument("--phase-csv", type=Path, required=True, help="Phase-method prediction CSV.")
    parser.add_argument("--repnet-csv", type=Path, default=Path("transrac_replication/experiments/repnet_external_test_predictions.csv"))
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=Path("outputs/00_index/subset_partA_test_all.csv"),
        help="CSV with video_id/video_name mapping.",
    )
    parser.add_argument("--out-csv", type=Path, required=True, help="Output hybrid prediction CSV.")
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    return parser.parse_args()


def metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    abs_err = np.abs(pred - gt)
    return {
        "mae": float(np.mean(abs_err)),
        "mae_norm": float(np.mean(abs_err / np.clip(gt, 1.0, None))),
        "mae_norm_p1": float(np.mean(abs_err / (gt + 1e-1))),
        "obo": float(np.mean(abs_err <= 1.0)),
    }


def fold_id(video_id: str) -> int:
    h = hashlib.md5(video_id.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 5


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
    rep = pd.read_csv(args.repnet_csv)
    idx = pd.read_csv(args.index_csv)[["video_id", "video_name"]]

    phase = phase.merge(idx, on="video_id", how="left")
    rep = rep.rename(columns={"video": "video_name", "pred_count": "rep_pred_count"})
    merged = phase.merge(rep[["video_name", "rep_pred_count"]], on="video_name", how="left")
    merged = merged.dropna(subset=["rep_pred_count"]).copy()

    merged["gt"] = merged["true_count"].astype(float)
    merged["phase_pred"] = merged["pred_count"].astype(float)
    merged["rep_pred_count"] = merged["rep_pred_count"].astype(float)
    merged["fold"] = merged["video_id"].astype(str).map(fold_id)
    calib = merged[merged["fold"] <= 2].copy()
    holdout = merged[merged["fold"] >= 3].copy()

    conf_cols = candidate_conf_columns(merged)
    if not conf_cols:
        raise ValueError("No confidence columns found in phase CSV")

    best = None
    search_rows: List[Dict[str, float | str]] = []

    for conf_col in conf_cols:
        cvals = calib[conf_col].to_numpy(dtype=float)
        qs = np.linspace(0.05, 0.90, 18)
        ths = np.unique(np.quantile(cvals, qs))
        for th in ths:
            hybrid = np.where(calib[conf_col].to_numpy(dtype=float) < th, calib["rep_pred_count"], calib["phase_pred"])
            m = metrics(hybrid, calib["gt"].to_numpy(dtype=float))
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

    for split_name, split_df in [("calib", calib), ("holdout", holdout), ("full", merged)]:
        split_df["hybrid_pred"] = np.where(
            split_df[best_col].to_numpy(dtype=float) < best_th,
            split_df["rep_pred_count"],
            split_df["phase_pred"],
        )
        split_df["fallback_used"] = split_df[best_col].to_numpy(dtype=float) < best_th

    out_df = merged[[
        "video_id",
        "video_name",
        "action",
        "gt",
        "phase_pred",
        "rep_pred_count",
        best_col,
        "hybrid_pred",
        "fallback_used",
    ]].copy()
    out_df = out_df.rename(columns={"gt": "true_count", best_col: "selected_confidence", "rep_pred_count": "repnet_pred_count"})
    out_df["abs_err_phase"] = (out_df["phase_pred"] - out_df["true_count"]).abs()
    out_df["abs_err_repnet"] = (out_df["repnet_pred_count"] - out_df["true_count"]).abs()
    out_df["abs_err_hybrid"] = (out_df["hybrid_pred"] - out_df["true_count"]).abs()

    ensure_dir(args.out_csv.parent)
    out_df.to_csv(args.out_csv, index=False)

    def pack(split_df: pd.DataFrame) -> Dict[str, float]:
        gt = split_df["gt"].to_numpy(dtype=float)
        phase_pred = split_df["phase_pred"].to_numpy(dtype=float)
        rep_pred = split_df["rep_pred_count"].to_numpy(dtype=float)
        hybrid_pred = split_df["hybrid_pred"].to_numpy(dtype=float)
        out = {
            "n": float(len(split_df)),
            "fallback_fraction": float(np.mean(split_df["fallback_used"].to_numpy(dtype=bool))),
            "phase_mae_norm_p1": metrics(phase_pred, gt)["mae_norm_p1"],
            "repnet_mae_norm_p1": metrics(rep_pred, gt)["mae_norm_p1"],
            "hybrid_mae_norm_p1": metrics(hybrid_pred, gt)["mae_norm_p1"],
            "phase_obo": metrics(phase_pred, gt)["obo"],
            "repnet_obo": metrics(rep_pred, gt)["obo"],
            "hybrid_obo": metrics(hybrid_pred, gt)["obo"],
        }
        return out

    conf_corr = {
        col: float(merged[[col, "gt", "phase_pred"]].assign(abs_err=np.abs(merged["phase_pred"] - merged["gt"]))[[col, "abs_err"]].corr(method="spearman").iloc[0, 1])
        for col in conf_cols
    }

    summary = {
        "phase_csv": str(args.phase_csv),
        "repnet_csv": str(args.repnet_csv),
        "selected_confidence_col": best_col,
        "selected_threshold": best_th,
        "confidence_spearman_abs_err": conf_corr,
        "calib": pack(calib),
        "holdout": pack(holdout),
        "full": pack(merged),
        "search_rows": search_rows,
        "out_csv": str(args.out_csv),
    }

    ensure_dir(args.summary_json.parent)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        f"# Confidence Calibration: {Path(args.phase_csv).name}",
        "",
        f"Selected confidence feature: `{best_col}`",
        f"Selected threshold: `{best_th:.6f}`",
        "",
        "## MAE_norm_p1 / OBO",
        "",
        "| Split | n | Fallback% | Phase | RepNet | Hybrid | Phase OBO | RepNet OBO | Hybrid OBO |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name in ["calib", "holdout", "full"]:
        s = summary[split_name]
        md_lines.append(
            f"| {split_name} | {int(s['n'])} | {s['fallback_fraction']:.3f} | {s['phase_mae_norm_p1']:.4f} | "
            f"{s['repnet_mae_norm_p1']:.4f} | {s['hybrid_mae_norm_p1']:.4f} | {s['phase_obo']:.4f} | {s['repnet_obo']:.4f} | {s['hybrid_obo']:.4f} |"
        )

    md_lines.append("")
    md_lines.append("## Confidence Correlation (Spearman with abs error)")
    md_lines.append("")
    for col, val in conf_corr.items():
        md_lines.append(f"- `{col}`: {val:.4f}")

    ensure_dir(args.summary_md.parent)
    args.summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote hybrid predictions: {args.out_csv}")
    print(f"Wrote summary json: {args.summary_json}")
    print(f"Wrote summary md: {args.summary_md}")


if __name__ == "__main__":
    main()
