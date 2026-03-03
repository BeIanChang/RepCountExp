from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    root = Path("F:/Projects/FItCoach")
    exp_dir = root / "transrac_replication" / "experiments"
    out_csv = exp_dir / "final_benchmark_combined.csv"
    out_md = exp_dir / "final_benchmark_combined.md"
    out_simple_csv = exp_dir / "final_benchmark_simplified_partA_test152.csv"
    out_simple_md = exp_dir / "final_benchmark_simplified_partA_test152.md"

    repnet_summary = json.loads((exp_dir / "repnet_external_test_summary.json").read_text(encoding="utf-8"))
    zhang_summary = json.loads((exp_dir / "zhang_external_test_summary.json").read_text(encoding="utf-8"))

    repnet_pred = pd.read_csv(exp_dir / "repnet_external_test_predictions.csv")
    zhang_pred = pd.read_csv(exp_dir / "zhang_external_test_predictions.csv")

    subset = pd.read_csv(root / "outputs" / "00_index" / "subset_partA_test_3acts.csv")
    keep_names = (subset["video_id"].astype(str) + ".mp4").tolist()

    repnet_3acts = repnet_pred[repnet_pred["video"].isin(keep_names)].copy()
    zhang_3acts = zhang_pred[zhang_pred["video"].isin(keep_names)].copy()

    for df in (repnet_3acts, zhang_3acts):
        df["abs_err"] = np.abs(df["pred_count"] - df["gt_count"])
        df["is_obo"] = df["abs_err"] <= 1.0
        df["norm_err_p1"] = df["abs_err"] / (df["gt_count"] + 1e-1)

    phase_metrics_all = pd.read_csv(root / "outputs" / "04_results" / "metrics_table_partA_test_all.csv")
    phase_metrics_all = phase_metrics_all[phase_metrics_all["split"] == "overall"].set_index("method")

    phase_metrics_3acts = pd.read_csv(root / "outputs" / "04_results" / "metrics_table_partA_test_3acts.csv")
    phase_metrics_3acts = phase_metrics_3acts[phase_metrics_3acts["split"] == "overall"].set_index("method")

    rows = [
        {
            "method": "TransRAC_official_ckpt",
            "scope": "PartA_test_152",
            "n_videos": 152,
            "mae_raw": np.nan,
            "mae_norm_p1": 0.5826176742925063,
            "obo": 0.28289473684210525,
            "event_f1": np.nan,
            "notes": "official checkpoint inference",
        },
        {
            "method": "RepNet_external",
            "scope": "PartA_test_152",
            "n_videos": int(repnet_summary["n_videos"]),
            "mae_raw": float(repnet_summary["mae_raw"]),
            "mae_norm_p1": float(repnet_summary["mae_normalized_p1"]),
            "obo": float(repnet_summary["obo"]),
            "event_f1": np.nan,
            "notes": "external checkpoint",
        },
        {
            "method": "Zhang_external_resnext101",
            "scope": "PartA_test_152",
            "n_videos": int(zhang_summary["n_videos"]),
            "mae_raw": float(zhang_summary["mae_raw"]),
            "mae_norm_p1": float(zhang_summary["mae_normalized_p1"]),
            "obo": float(zhang_summary["obo"]),
            "event_f1": np.nan,
            "notes": "external checkpoint, GPU inference",
        },
    ]

    for method_key, alias in (
        ("baseline_fsm", "FSM_baseline"),
        ("phase_native_online_phase_crossing", "Phase_native_online"),
        ("proposed_phase_vote", "Phase_vote"),
    ):
        row = phase_metrics_all.loc[method_key]
        rows.append(
            {
                "method": alias,
                "scope": "PartA_test_152",
                "n_videos": int(row["n_videos"]),
                "mae_raw": float(row["mae"]),
                "mae_norm_p1": float(row["mae_norm_p1"]),
                "obo": float(row["oboa"]),
                "event_f1": float(row["event_f1"]),
                "notes": "pose-signal pipeline",
            }
        )

    rows.extend(
        [
            {
                "method": "RepNet_external",
                "scope": "PartA_test_3acts_53",
                "n_videos": int(len(repnet_3acts)),
                "mae_raw": float(repnet_3acts["abs_err"].mean()),
                "mae_norm_p1": float(repnet_3acts["norm_err_p1"].mean()),
                "obo": float(repnet_3acts["is_obo"].mean()),
                "event_f1": np.nan,
                "notes": "filtered to squat/push_up/pull_up",
            },
            {
                "method": "Zhang_external_resnext101",
                "scope": "PartA_test_3acts_53",
                "n_videos": int(len(zhang_3acts)),
                "mae_raw": float(zhang_3acts["abs_err"].mean()),
                "mae_norm_p1": float(zhang_3acts["norm_err_p1"].mean()),
                "obo": float(zhang_3acts["is_obo"].mean()),
                "event_f1": np.nan,
                "notes": "filtered to squat/push_up/pull_up",
            },
        ]
    )

    for method_key, alias in (
        ("baseline_fsm", "FSM_baseline"),
        ("phase_native_online_phase_crossing", "Phase_native_online"),
        ("proposed_phase_vote", "Phase_vote"),
    ):
        row = phase_metrics_3acts.loc[method_key]
        rows.append(
            {
                "method": alias,
                "scope": "PartA_test_3acts_53",
                "n_videos": int(row["n_videos"]),
                "mae_raw": float(row["mae"]),
                "mae_norm_p1": float(row["mae_norm_p1"]),
                "obo": float(row["oboa"]),
                "event_f1": float(row["event_f1"]),
                "notes": "pose-signal pipeline",
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)

    lines: list[str] = ["# Final Benchmark Combined", "", "## PartA test (152 videos)"]
    for _, row in out[out["scope"] == "PartA_test_152"].iterrows():
        mae_raw_val = float(row["mae_raw"])
        mae_raw = "NA" if np.isnan(mae_raw_val) else f"{mae_raw_val:.4f}"
        mae_norm_val = float(row["mae_norm_p1"])
        obo_val = float(row["obo"])
        lines.append(
            f"- {row['method']}: n={int(row['n_videos'])}, MAE_raw={mae_raw}, "
            f"MAE_norm_p1={mae_norm_val:.4f}, OBO={obo_val:.4f}"
        )

    lines.append("")
    lines.append("## PartA test 3 actions (53 videos: squat/push_up/pull_up)")
    for _, row in out[out["scope"] == "PartA_test_3acts_53"].iterrows():
        mae_norm_val = float(row["mae_norm_p1"])
        mae_norm = "NA" if np.isnan(mae_norm_val) else f"{mae_norm_val:.4f}"
        event_f1_val = float(row["event_f1"])
        event_f1 = "" if np.isnan(event_f1_val) else f", EventF1={event_f1_val:.4f}"
        mae_raw = float(row["mae_raw"])
        obo_val = float(row["obo"])
        lines.append(
            f"- {row['method']}: n={int(row['n_videos'])}, MAE_raw={mae_raw:.4f}, "
            f"MAE_norm_p1={mae_norm}, OBO={obo_val:.4f}{event_f1}"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    simple = out[out["scope"] == "PartA_test_152"].copy()
    simple = simple[["method", "n_videos", "mae_norm_p1", "obo"]].sort_values("mae_norm_p1").reset_index(drop=True)
    simple.to_csv(out_simple_csv, index=False)

    simple_lines = [
        "# PartA Test (152) Simplified Table",
        "",
        "| Method | N | MAE_norm_p1 | OBO |",
        "|---|---:|---:|---:|",
    ]
    for _, row in simple.iterrows():
        simple_lines.append(
            f"| {row['method']} | {int(row['n_videos'])} | {float(row['mae_norm_p1']):.4f} | {float(row['obo']):.4f} |"
        )
    out_simple_md.write_text("\n".join(simple_lines) + "\n", encoding="utf-8")

    print(f"Wrote combined benchmark CSV: {out_csv}")
    print(f"Wrote combined benchmark MD: {out_md}")
    print(f"Wrote simplified benchmark CSV: {out_simple_csv}")
    print(f"Wrote simplified benchmark MD: {out_simple_md}")


if __name__ == "__main__":
    main()
