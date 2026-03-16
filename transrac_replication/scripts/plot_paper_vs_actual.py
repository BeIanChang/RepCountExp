from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot paper-reported vs reproduced comparison figure.")
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=Path("transrac_replication/experiments/expanded_comparison_partA_test152.csv"),
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("transrac_replication/experiments/paper_vs_actual_comparison.png"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("transrac_replication/experiments/paper_vs_actual_comparison.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.comparison_csv)

    paper_rows = {
        "X3D": "X3D",
        "TANet": "TANet",
        "Video SwinT": "Video SwinT",
        "Huang et al.": "Huang et al.",
        "RepNet": "RepNet",
        "Zhang et al.": "Zhang et al.",
        "TransRAC": "TransRAC (paper)",
    }

    actual_rows = {
        "X3D": "X3D_cached16k_proxy",
        "TANet": "TANet_cached16k_proxy",
        "Video SwinT": "VideoSwinT_cached16k_proxy",
        "Huang et al.": "Huang_cached16k_proxy",
        "RepNet": "RepNet_external_paper64",
        "Zhang et al.": "Zhang_external_resnext101",
        "TransRAC": "TransRAC_official_ckpt",
    }

    proxy_set = {"X3D", "TANet", "Video SwinT", "Huang et al."}

    rows = []
    for family in ["X3D", "TANet", "Video SwinT", "Huang et al.", "RepNet", "Zhang et al.", "TransRAC"]:
        p = df[df["method"] == paper_rows[family]].iloc[0]
        a = df[df["method"] == actual_rows[family]].iloc[0]
        rows.append(
            {
                "family": family,
                "paper_method": str(p["method"]),
                "actual_method": str(a["method"]),
                "paper_mae_norm_p1": float(p["mae_norm_p1"]),
                "actual_mae_norm_p1": float(a["mae_norm_p1"]),
                "delta_mae_norm_p1": float(a["mae_norm_p1"] - p["mae_norm_p1"]),
                "paper_obo": float(p["obo"]),
                "actual_obo": float(a["obo"]),
                "delta_obo": float(a["obo"] - p["obo"]),
                "actual_n": int(a["n_videos"]),
                "is_proxy_cached151": family in proxy_set,
            }
        )

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    x = np.arange(len(out))
    w = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), dpi=120, sharex=True)

    proxy_mask = out["is_proxy_cached151"].to_numpy(dtype=bool)
    colors_actual = np.where(proxy_mask, "#6a3d9a", "#1f78b4")

    axes[0].bar(x - w / 2, out["paper_mae_norm_p1"].to_numpy(dtype=float), w, label="paper-reported", color="#9e9e9e")
    for i in range(len(out)):
        axes[0].bar(
            x[i] + w / 2,
            float(out.loc[i, "actual_mae_norm_p1"]),
            w,
            color=colors_actual[i],
            hatch="//" if proxy_mask[i] else None,
            label="reproduced actual" if i == 0 else None,
        )
    axes[0].set_ylabel("MAE_norm_p1 (lower better)")
    axes[0].set_title("Paper-reported vs reproduced actual")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(x - w / 2, out["paper_obo"].to_numpy(dtype=float), w, label="paper-reported", color="#9e9e9e")
    for i in range(len(out)):
        axes[1].bar(
            x[i] + w / 2,
            float(out.loc[i, "actual_obo"]),
            w,
            color=colors_actual[i],
            hatch="//" if proxy_mask[i] else None,
            label="reproduced actual" if i == 0 else None,
        )
    axes[1].set_ylabel("OBO (higher better)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(out["family"].tolist(), rotation=20, ha="right")
    axes[1].grid(axis="y", alpha=0.2)

    handles, labels = axes[1].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    axes[0].legend(uniq.values(), uniq.keys(), loc="upper left", fontsize=9)

    fig.text(
        0.01,
        0.01,
        "Blue=direct reproduced (n=152). Purple-hatched=proxy cached runs (n=151).",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(args.out_png)
    plt.close(fig)

    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()
