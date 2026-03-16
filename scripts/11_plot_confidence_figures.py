from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_llsp import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot detailed confidence figures for phase methods.")
    parser.add_argument("--phase-csv", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--tag", type=str, required=True, help="Short tag for output file names.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/04_results/confidence_figures"))
    return parser.parse_args()


def _safe_qcut(x: pd.Series, q: int) -> pd.Series:
    try:
        return pd.qcut(x, q=q, duplicates="drop")
    except ValueError:
        return pd.cut(x, bins=min(q, max(2, x.nunique())))


def plot_scatter(df: pd.DataFrame, conf_col: str, threshold: float, out_path: Path, title: str) -> None:
    x = df[conf_col].to_numpy(dtype=float)
    y = df["abs_err"].to_numpy(dtype=float)
    fallback = x < threshold

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.scatter(x[~fallback], y[~fallback], s=16, alpha=0.55, label="phase selected")
    ax.scatter(x[fallback], y[fallback], s=16, alpha=0.55, label="fallback region")
    ax.axvline(threshold, color="tab:red", linestyle="--", linewidth=1.2, label=f"threshold={threshold:.4f}")
    ax.set_xlabel(conf_col)
    ax.set_ylabel("abs error")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_hist(df: pd.DataFrame, conf_col: str, threshold: float, out_path: Path, title: str) -> None:
    x = df[conf_col].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.hist(x, bins=24, alpha=0.8, color="tab:blue")
    ax.axvline(threshold, color="tab:red", linestyle="--", linewidth=1.2, label=f"threshold={threshold:.4f}")
    ax.set_xlabel(conf_col)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_binned_table(df: pd.DataFrame, conf_col: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["conf_bin"] = _safe_qcut(tmp[conf_col], q=10)
    out = (
        tmp.groupby("conf_bin", observed=True)
        .agg(
            n=("video_id", "count"),
            conf_min=(conf_col, "min"),
            conf_max=(conf_col, "max"),
            mean_conf=(conf_col, "mean"),
            mae=("abs_err", "mean"),
            mae_norm_p1=("norm_err_p1", "mean"),
            obo=("is_obo", "mean"),
        )
        .reset_index(drop=True)
    )
    return out


def plot_binned_curve(binned: pd.DataFrame, out_path: Path, title: str) -> None:
    x = binned["mean_conf"].to_numpy(dtype=float)
    y1 = binned["mae_norm_p1"].to_numpy(dtype=float)
    y2 = binned["obo"].to_numpy(dtype=float)

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=120)
    ax1.plot(x, y1, marker="o", label="MAE_norm_p1")
    ax1.set_xlabel("mean confidence (bin)")
    ax1.set_ylabel("MAE_norm_p1")
    ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.plot(x, y2, marker="s", linestyle="--", color="tab:orange", label="OBO")
    ax2.set_ylabel("OBO")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_threshold_search(search_df: pd.DataFrame, selected_col: str, selected_th: float, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    for conf_col, g in search_df.groupby("conf_col"):
        g = g.sort_values("threshold")
        lw = 2.0 if conf_col == selected_col else 1.0
        alpha = 0.95 if conf_col == selected_col else 0.45
        ax.plot(g["threshold"], g["calib_mae_norm_p1"], linewidth=lw, alpha=alpha, label=conf_col)

    ax.axvline(selected_th, color="tab:red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("threshold")
    ax.set_ylabel("calib MAE_norm_p1")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    phase = pd.read_csv(args.phase_csv)
    summary = json.loads(args.calibration_json.read_text(encoding="utf-8"))
    selected_col = str(summary["selected_confidence_col"])
    selected_th = float(summary["selected_threshold"])

    phase["abs_err"] = (phase["pred_count"] - phase["true_count"]).abs()
    phase["is_obo"] = phase["abs_err"] <= 1.0
    phase["norm_err_p1"] = phase["abs_err"] / (phase["true_count"] + 1e-1)

    ensure_dir(args.out_dir)

    scatter_png = args.out_dir / f"{args.tag}_confidence_vs_abs_err.png"
    hist_png = args.out_dir / f"{args.tag}_confidence_hist.png"
    binned_csv = args.out_dir / f"{args.tag}_confidence_binned_metrics.csv"
    binned_png = args.out_dir / f"{args.tag}_confidence_binned_curve.png"
    search_png = args.out_dir / f"{args.tag}_threshold_search.png"
    report_md = args.out_dir / f"{args.tag}_confidence_report.md"

    plot_scatter(
        df=phase,
        conf_col=selected_col,
        threshold=selected_th,
        out_path=scatter_png,
        title=f"{args.tag}: confidence vs abs error",
    )
    plot_hist(
        df=phase,
        conf_col=selected_col,
        threshold=selected_th,
        out_path=hist_png,
        title=f"{args.tag}: confidence histogram",
    )

    binned = build_binned_table(phase, selected_col)
    binned.to_csv(binned_csv, index=False)
    plot_binned_curve(binned, binned_png, title=f"{args.tag}: binned confidence behavior")

    search_df = pd.DataFrame(summary.get("search_rows", []))
    if not search_df.empty:
        plot_threshold_search(
            search_df,
            selected_col=selected_col,
            selected_th=selected_th,
            out_path=search_png,
            title=f"{args.tag}: threshold search",
        )

    corr = summary.get("confidence_spearman_abs_err", {})
    md_lines = [
        f"# Confidence Report: {args.tag}",
        "",
        f"- phase csv: `{args.phase_csv}`",
        f"- calibration json: `{args.calibration_json}`",
        f"- selected confidence: `{selected_col}`",
        f"- selected threshold: `{selected_th:.6f}`",
        "",
        "## Figure files",
        f"- scatter: `{scatter_png}`",
        f"- histogram: `{hist_png}`",
        f"- binned curve: `{binned_png}`",
        f"- threshold search: `{search_png}`",
        f"- binned table: `{binned_csv}`",
        "",
        "## Spearman correlation with abs error",
    ]
    for k, v in corr.items():
        md_lines.append(f"- `{k}`: {float(v):.4f}")

    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote confidence report: {report_md}")


if __name__ == "__main__":
    main()
