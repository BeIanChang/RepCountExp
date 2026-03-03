from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path("F:/Projects/FItCoach/transrac_replication/experiments")

    paper = {
        "RepNet": {"mae_norm_p1": 0.9950, "obo": 0.0134},
        "Zhang": {"mae_norm_p1": 0.8786, "obo": 0.1554},
        "TransRAC": {"mae_norm_p1": 0.4431, "obo": 0.2913},
    }

    rows = []

    def add_row(label: str, family: str, summary_path: Path, mae_key: str = "mae_normalized_p1", obo_key: str = "obo") -> None:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        mae = float(payload[mae_key])
        obo = float(payload[obo_key])
        rows.append(
            {
                "run": label,
                "family": family,
                "mae_norm_p1": mae,
                "obo": obo,
                "delta_mae": mae - paper[family]["mae_norm_p1"],
                "delta_obo": obo - paper[family]["obo"],
                "summary": str(summary_path),
            }
        )

    add_row(
        "repnet_external_multi_stride_full",
        "RepNet",
        root / "repnet_external_test_summary.json",
    )
    add_row(
        "repnet_external_paper64",
        "RepNet",
        root / "repnet_external_test_paper64_summary.json",
    )
    add_row(
        "repnet_like_cached16k",
        "RepNet",
        root / "repnet_like_16k_cached_test_summary.json",
        mae_key="mae_norm_p1",
    )

    add_row(
        "zhang_external_resnext101",
        "Zhang",
        root / "zhang_external_test_summary.json",
    )
    add_row(
        "zhang_like_cached16k",
        "Zhang",
        root / "zhang_like_16k_cached_test_summary.json",
        mae_key="mae_norm_p1",
    )

    rows.append(
        {
            "run": "transrac_official_ckpt",
            "family": "TransRAC",
            "mae_norm_p1": 0.5826176742925063,
            "obo": 0.28289473684210525,
            "delta_mae": 0.5826176742925063 - paper["TransRAC"]["mae_norm_p1"],
            "delta_obo": 0.28289473684210525 - paper["TransRAC"]["obo"],
            "summary": "official log",
        }
    )
    add_row(
        "transrac_cached16k",
        "TransRAC",
        root / "transrac_16k_cached_test_summary.json",
        mae_key="mae_norm_p1",
    )

    out = pd.DataFrame(rows).sort_values(["family", "mae_norm_p1"]).reset_index(drop=True)
    out_csv = root / "paper_alignment_attempt_v2.csv"
    out_md = root / "paper_alignment_attempt_v2.md"
    out.to_csv(out_csv, index=False)

    lines = [
        "# Paper Alignment Attempt V2",
        "",
        "Columns: MAE_norm_p1, OBO, and deltas vs paper table values.",
        "",
        "| Run | Family | MAE_norm_p1 | OBO | dMAE | dOBO |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in out.iterrows():
        lines.append(
            f"| {row['run']} | {row['family']} | {row['mae_norm_p1']:.4f} | {row['obo']:.4f} | {row['delta_mae']:+.4f} | {row['delta_obo']:+.4f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
