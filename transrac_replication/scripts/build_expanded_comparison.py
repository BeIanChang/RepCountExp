from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    root = Path("F:/Projects/FItCoach")
    exp = root / "transrac_replication" / "experiments"
    out_csv = exp / "expanded_comparison_partA_test152.csv"
    out_md = exp / "expanded_comparison_partA_test152.md"
    out_cross_csv = exp / "expanded_comparison_crosspaper.csv"
    out_cross_md = exp / "expanded_comparison_crosspaper.md"

    rep_multi = json.loads((exp / "repnet_external_test_summary.json").read_text(encoding="utf-8"))
    rep_p64 = json.loads((exp / "repnet_external_test_paper64_summary.json").read_text(encoding="utf-8"))
    zhang_ext = json.loads((exp / "zhang_external_test_summary.json").read_text(encoding="utf-8"))
    view_fft = json.loads((root / "outputs" / "04_results" / "viewpoint_fft_partA_test_all_summary.json").read_text(encoding="utf-8"))
    x3d_cached = json.loads((exp / "x3d_cached16k_test_summary.json").read_text(encoding="utf-8"))
    tanet_cached = json.loads((exp / "tanet_cached16k_test_summary.json").read_text(encoding="utf-8"))
    videoswint_cached = json.loads((exp / "videoswint_cached16k_test_summary.json").read_text(encoding="utf-8"))
    huang_cached = json.loads((exp / "huang_cached16k_test_summary.json").read_text(encoding="utf-8"))
    phase_soft = pd.read_csv(root / "outputs" / "04_results" / "metrics_table_partA_test_all_softpenalty.csv")
    phase_soft = phase_soft[phase_soft["split"] == "overall"].set_index("method")
    peak_online = pd.read_csv(root / "outputs" / "04_results" / "metrics_table_baseline_peak_online_partA_test_all.csv")
    peak_online = peak_online[peak_online["split"] == "overall"].iloc[0]
    native_peak_hybrid = json.loads(
        (root / "outputs" / "04_results" / "phase_native_peak_online_hybrid_summary.json").read_text(encoding="utf-8")
    )

    rows = [
        {
            "method": "TransRAC_official_ckpt",
            "source": "ours_reproduced",
            "dataset": "RepCount-A test",
            "n_videos": 152,
            "mae_norm_p1": 0.5826176742925063,
            "obo": 0.28289473684210525,
            "notes": "official checkpoint inference",
        },
        {
            "method": "RepNet_external_multi_stride_full",
            "source": "ours_reproduced",
            "dataset": "RepCount-A test",
            "n_videos": int(rep_multi["n_videos"]),
            "mae_norm_p1": float(rep_multi["mae_normalized_p1"]),
            "obo": float(rep_multi["obo"]),
            "notes": "external weights + multi-stride search",
        },
        {
            "method": "RepNet_external_paper64",
            "source": "ours_reproduced",
            "dataset": "RepCount-A test",
            "n_videos": int(rep_p64["n_videos"]),
            "mae_norm_p1": float(rep_p64["mae_normalized_p1"]),
            "obo": float(rep_p64["obo"]),
            "notes": "single 64-frame protocol",
        },
        {
            "method": "Zhang_external_resnext101",
            "source": "ours_reproduced",
            "dataset": "RepCount-A test",
            "n_videos": int(zhang_ext["n_videos"]),
            "mae_norm_p1": float(zhang_ext["mae_normalized_p1"]),
            "obo": float(zhang_ext["obo"]),
            "notes": "external checkpoint",
        },
        {
            "method": "FSM_baseline",
            "source": "ours_reproduced",
            "dataset": "RepCount-A test",
            "n_videos": int(phase_soft.loc["baseline_fsm", "n_videos"]),
            "mae_norm_p1": float(phase_soft.loc["baseline_fsm", "mae_norm_p1"]),
            "obo": float(phase_soft.loc["baseline_fsm", "oboa"]),
            "notes": "pose-signal method",
        },
        {
            "method": "Phase_native_softpenalty",
            "source": "ours_reproduced",
            "dataset": "RepCount-A test",
            "n_videos": int(phase_soft.loc["phase_native_online_phase_crossing", "n_videos"]),
            "mae_norm_p1": float(phase_soft.loc["phase_native_online_phase_crossing", "mae_norm_p1"]),
            "obo": float(phase_soft.loc["phase_native_online_phase_crossing", "oboa"]),
            "notes": "added var/flip/pause soft penalties",
        },
        {
            "method": "Baseline_peak_online",
            "source": "ours_reproduced",
            "dataset": "RepCount-A test",
            "n_videos": int(peak_online["n_videos"]),
            "mae_norm_p1": float(peak_online["mae_norm_p1"]),
            "obo": float(peak_online["oboa"]),
            "notes": "causal trough-peak-trough with low lookahead",
        },
        {
            "method": "Phase_native_peak_online_hybrid",
            "source": "ours_reproduced",
            "dataset": "RepCount-A test",
            "n_videos": int(native_peak_hybrid["full"]["n"]),
            "mae_norm_p1": float(native_peak_hybrid["full"]["hybrid_mae_norm_p1"]),
            "obo": float(native_peak_hybrid["full"]["hybrid_obo"]),
            "notes": (
                "confidence gate: "
                f"{native_peak_hybrid['selected_confidence_col']}<{native_peak_hybrid['selected_threshold']:.4f}, "
                f"fallback={native_peak_hybrid['full']['fallback_fraction']:.3f}"
            ),
        },
        {
            "method": "ViewpointFFT_similarity",
            "source": "ours_reproduced",
            "dataset": "RepCount-A test",
            "n_videos": int(view_fft["n_videos"]),
            "mae_norm_p1": float(view_fft["mae_norm_p1"]),
            "obo": float(view_fft["obo"]),
            "notes": "skeleton cosine similarity + sliding FFT integration",
        },
        {
            "method": "X3D_cached16k_proxy",
            "source": "ours_reproduced_cached151",
            "dataset": "RepCount-A cached test",
            "n_videos": int(x3d_cached["n_videos"]),
            "mae_norm_p1": float(x3d_cached["mae_norm_p1"]),
            "obo": float(x3d_cached["obo"]),
            "notes": "method-inspired proxy on cached embeddings",
        },
        {
            "method": "TANet_cached16k_proxy",
            "source": "ours_reproduced_cached151",
            "dataset": "RepCount-A cached test",
            "n_videos": int(tanet_cached["n_videos"]),
            "mae_norm_p1": float(tanet_cached["mae_norm_p1"]),
            "obo": float(tanet_cached["obo"]),
            "notes": "method-inspired proxy on cached embeddings",
        },
        {
            "method": "VideoSwinT_cached16k_proxy",
            "source": "ours_reproduced_cached151",
            "dataset": "RepCount-A cached test",
            "n_videos": int(videoswint_cached["n_videos"]),
            "mae_norm_p1": float(videoswint_cached["mae_norm_p1"]),
            "obo": float(videoswint_cached["obo"]),
            "notes": "method-inspired proxy on cached embeddings",
        },
        {
            "method": "Huang_cached16k_proxy",
            "source": "ours_reproduced_cached151",
            "dataset": "RepCount-A cached test",
            "n_videos": int(huang_cached["n_videos"]),
            "mae_norm_p1": float(huang_cached["mae_norm_p1"]),
            "obo": float(huang_cached["obo"]),
            "notes": "action-seg inspired proxy on cached embeddings",
        },
    ]

    paper_rows = [
        ("X3D", 0.9105, 0.1059),
        ("TANet", 0.6624, 0.0993),
        ("Video SwinT", 0.5756, 0.1324),
        ("Huang et al.", 0.5267, 0.1589),
        ("RepNet", 0.9950, 0.0134),
        ("Zhang et al.", 0.8786, 0.1554),
        ("TransRAC (paper)", 0.4431, 0.2913),
    ]
    for method, mae, obo in paper_rows:
        rows.append(
            {
                "method": method,
                "source": "paper_reported",
                "dataset": "RepCount-A test",
                "n_videos": 152,
                "mae_norm_p1": mae,
                "obo": obo,
                "notes": "CVPR22 Table-2 reported number",
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)

    ours = out[out["source"] == "ours_reproduced"].copy()
    ours = ours.iloc[np.argsort(np.asarray(ours["mae_norm_p1"], dtype=float))]

    ours_cached = out[out["source"] == "ours_reproduced_cached151"].copy()
    ours_cached = ours_cached.iloc[np.argsort(np.asarray(ours_cached["mae_norm_p1"], dtype=float))]

    paper = out[out["source"] == "paper_reported"].copy()
    paper = paper.iloc[np.argsort(np.asarray(paper["mae_norm_p1"], dtype=float))]

    md = [
        "# Expanded Comparison on RepCount-A Test (152)",
        "",
        "## Ours Reproduced Runs",
        "",
        "| Method | MAE_norm_p1 | OBO | Notes |",
        "|---|---:|---:|---|",
    ]
    for _, r in ours.iterrows():
        md.append(f"| {r['method']} | {float(r['mae_norm_p1']):.4f} | {float(r['obo']):.4f} | {r['notes']} |")

    md += [
        "",
        "## Ours Reproduced Cached-Embedding Proxies (n=151)",
        "",
        "| Method | MAE_norm_p1 | OBO | Notes |",
        "|---|---:|---:|---|",
    ]
    for _, r in ours_cached.iterrows():
        md.append(f"| {r['method']} | {float(r['mae_norm_p1']):.4f} | {float(r['obo']):.4f} | {r['notes']} |")

    md += [
        "",
        "## Paper-Reported Methods (TransRAC CVPR22 Table-2)",
        "",
        "| Method | MAE_norm_p1 | OBO |",
        "|---|---:|---:|",
    ]
    for _, r in paper.iterrows():
        md.append(f"| {r['method']} | {float(r['mae_norm_p1']):.4f} | {float(r['obo']):.4f} |")

    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    cross_rows = [
        {
            "paper": "Viewpoint-Invariant Exercise Repetition Counting (arXiv:2107.13760)",
            "dataset": "MM-fit",
            "metric": "MAE/OBOA",
            "value": "0.06 / 0.94",
            "source": "paper_abstract/table",
            "notes": "Not directly comparable to RepCount-A",
        },
        {
            "paper": "Viewpoint-Invariant Exercise Repetition Counting (arXiv:2107.13760)",
            "dataset": "UI-PRMD",
            "metric": "MAE/OBOA",
            "value": "0.06 / 0.95",
            "source": "paper_abstract/table",
            "notes": "Not directly comparable to RepCount-A",
        },
        {
            "paper": "Our adaptation of 2107.13760 method",
            "dataset": "RepCount-A test",
            "metric": "MAE_norm_p1/OBO",
            "value": f"{view_fft['mae_norm_p1']:.4f} / {view_fft['obo']:.4f}",
            "source": "reproduced_here",
            "notes": "Skeleton cosine similarity + FFT integration baseline",
        },
    ]
    cross = pd.DataFrame(cross_rows)
    cross.to_csv(out_cross_csv, index=False)

    cross_md = [
        "# Cross-Paper Comparison Note",
        "",
        "| Paper/Run | Dataset | Metric | Value | Notes |",
        "|---|---|---|---|---|",
    ]
    for _, r in cross.iterrows():
        cross_md.append(f"| {r['paper']} | {r['dataset']} | {r['metric']} | {r['value']} | {r['notes']} |")
    out_cross_md.write_text("\n".join(cross_md) + "\n", encoding="utf-8")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_cross_csv}")
    print(f"Wrote {out_cross_md}")


if __name__ == "__main__":
    main()
