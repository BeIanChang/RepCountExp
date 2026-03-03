from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize TransRAC/baseline run summaries into one table.")
    parser.add_argument(
        "--summary-files",
        nargs="+",
        required=True,
        help="List of summary.json paths.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("transrac_replication/experiments/benchmark_table.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for s in args.summary_files:
        p = Path(s)
        payload = json.loads(p.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": payload.get("model", p.parent.name),
                "mae": payload.get("eval_stats", {}).get("mae"),
                "mae_normalized": payload.get("eval_stats", {}).get("mae_normalized"),
                "obo": payload.get("eval_stats", {}).get("obo"),
                "last_loss": payload.get("train_stats", {}).get("last_loss"),
                "steps": payload.get("train_stats", {}).get("steps"),
                "summary_path": str(p),
            }
        )

    out = pd.DataFrame(rows).sort_values(["mae_normalized", "obo"], ascending=[True, False]).reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(out.to_string(index=False))
    print(f"Wrote benchmark table: {args.out_csv}")


if __name__ == "__main__":
    main()
