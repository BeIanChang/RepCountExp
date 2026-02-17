from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.signal import find_peaks

from common_llsp import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline peak/trough rep counter.")
    parser.add_argument(
        "--subset-csv",
        type=Path,
        default=Path("outputs/00_index/subset_debug.csv"),
        help="Subset CSV with split/video/action info.",
    )
    parser.add_argument(
        "--signals-dir",
        type=Path,
        default=Path("outputs/03_signals"),
        help="Directory containing computed signal npz files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/exercises.yaml"),
        help="Exercise configuration yaml.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/04_results/baseline_peak.csv"),
        help="Output prediction CSV.",
    )
    parser.add_argument("--min-distance-sec", type=float, default=0.35)
    parser.add_argument("--min-prominence-z", type=float, default=0.25)
    parser.add_argument("--min-cycle-sec", type=float, default=0.3)
    parser.add_argument("--max-cycle-sec", type=float, default=8.0)
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
            out.append((int(p[0]), int(p[1])))
    return out


def zscore(x: np.ndarray) -> np.ndarray:
    m = float(np.nanmean(x))
    s = float(np.nanstd(x))
    if not np.isfinite(s) or s < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - m) / s).astype(np.float32)


def count_cycles(
    theta: np.ndarray,
    fps: float,
    min_distance_sec: float,
    min_prominence_z: float,
    min_cycle_sec: float,
    max_cycle_sec: float,
) -> Tuple[List[Tuple[int, int]], Dict[str, int]]:
    if len(theta) == 0:
        return [], {"n_peaks": 0, "n_troughs": 0}

    dist = max(1, int(round(min_distance_sec * max(fps, 1.0))))
    min_cycle_frames = max(1, int(round(min_cycle_sec * max(fps, 1.0))))
    max_cycle_frames = max(min_cycle_frames + 1, int(round(max_cycle_sec * max(fps, 1.0))))

    theta_z = zscore(theta)
    prom = max(0.05, float(min_prominence_z))
    peaks, _ = find_peaks(theta_z, distance=dist, prominence=prom)
    troughs, _ = find_peaks(-theta_z, distance=dist, prominence=prom)

    troughs = np.sort(troughs)
    peaks = np.sort(peaks)

    cycles: List[Tuple[int, int]] = []
    for i in range(len(troughs) - 1):
        t0 = int(troughs[i])
        t1 = int(troughs[i + 1])
        span = t1 - t0
        if span < min_cycle_frames or span > max_cycle_frames:
            continue
        in_between = peaks[(peaks > t0) & (peaks < t1)]
        if len(in_between) == 0:
            continue
        cycles.append((t0, t1))

    return cycles, {"n_peaks": int(len(peaks)), "n_troughs": int(len(troughs))}


def main() -> None:
    args = parse_args()
    subset = pd.read_csv(args.subset_csv)
    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rows: List[Dict] = []
    for _, row in subset.iterrows():
        split = row["split"]
        video_id = row["video_id"]
        action = row["canonical_action"]
        true_periods = parse_periods(row.get("periods_json", ""))
        true_count = int(row["n_periods"]) if pd.notna(row.get("n_periods")) else len(true_periods)

        signal_file = args.signals_dir / split / f"{video_id}.npz"
        status = "ok"
        primary_signal = ""
        pred_periods: List[Tuple[int, int]] = []
        debug_peaks = 0
        debug_troughs = 0

        if not signal_file.exists():
            status = "missing_signal"
            pred_count = 0
        elif action not in cfg:
            status = "missing_action_config"
            pred_count = 0
        else:
            data = np.load(signal_file)
            primary_signal = str(cfg[action]["signals"][0]["name"])
            theta_key = f"theta_{primary_signal}"
            if theta_key not in data.files:
                status = "missing_primary_theta"
                pred_count = 0
            else:
                theta = data[theta_key].astype(np.float32)
                fps = float(data["fps"])
                pred_periods, dbg = count_cycles(
                    theta=theta,
                    fps=fps,
                    min_distance_sec=args.min_distance_sec,
                    min_prominence_z=args.min_prominence_z,
                    min_cycle_sec=args.min_cycle_sec,
                    max_cycle_sec=args.max_cycle_sec,
                )
                debug_peaks = dbg["n_peaks"]
                debug_troughs = dbg["n_troughs"]
                pred_count = len(pred_periods)

        rows.append(
            {
                "video_id": video_id,
                "split": split,
                "action": action,
                "true_count": true_count,
                "pred_count": int(pred_count),
                "abs_err": int(abs(int(pred_count) - int(true_count))),
                "pred_periods_json": json.dumps(pred_periods, separators=(",", ":")),
                "true_periods_json": json.dumps(true_periods, separators=(",", ":")),
                "primary_signal": primary_signal,
                "debug_n_peaks": debug_peaks,
                "debug_n_troughs": debug_troughs,
                "status": status,
                "method": "baseline_peak",
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["split", "video_id"]).reset_index(drop=True)
    ensure_dir(args.out_csv.parent)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote baseline predictions: {args.out_csv}")


if __name__ == "__main__":
    main()
