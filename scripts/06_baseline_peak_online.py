from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from common_llsp import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online-compatible peak/trough baseline rep counter.")
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
        default=Path("outputs/04_results/baseline_peak_online.csv"),
        help="Output prediction CSV.",
    )
    parser.add_argument("--min-distance-sec", type=float, default=0.35)
    parser.add_argument("--min-prominence-z", type=float, default=0.25)
    parser.add_argument("--min-cycle-sec", type=float, default=0.3)
    parser.add_argument("--max-cycle-sec", type=float, default=8.0)
    parser.add_argument(
        "--lookahead-sec",
        type=float,
        default=0.35,
        help="Low-latency lookahead for online extrema confirmation.",
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
            out.append((int(p[0]), int(p[1])))
    return out


def zscore(x: np.ndarray) -> np.ndarray:
    m = float(np.nanmean(x))
    s = float(np.nanstd(x))
    if not np.isfinite(s) or s < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - m) / s).astype(np.float32)


def _confirm_extreme(
    theta_z: np.ndarray,
    idx: int,
    kind: str,
    lookahead_frames: int,
    prominence_z: float,
    upto: int,
) -> bool:
    left = max(0, idx - lookahead_frames)
    right = min(len(theta_z) - 1, idx + lookahead_frames, upto)
    if right - left < 2:
        return False

    seg = theta_z[left : right + 1]
    val = float(theta_z[idx])
    eps = 1e-6

    if kind == "peak":
        is_local = val >= float(np.max(seg)) - eps
        left_min = float(np.min(theta_z[left : idx + 1]))
        right_min = float(np.min(theta_z[idx : right + 1]))
        prom = val - max(left_min, right_min)
        return bool(is_local and prom >= prominence_z)

    is_local = val <= float(np.min(seg)) + eps
    left_max = float(np.max(theta_z[left : idx + 1]))
    right_max = float(np.max(theta_z[idx : right + 1]))
    prom = min(left_max, right_max) - val
    return bool(is_local and prom >= prominence_z)


def detect_extrema_online(
    theta_z: np.ndarray,
    min_distance_frames: int,
    prominence_z: float,
    lookahead_frames: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(theta_z)
    if n < 3:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    pending_peaks: List[int] = []
    pending_troughs: List[int] = []
    peaks: List[int] = []
    troughs: List[int] = []
    last_peak = -10**9
    last_trough = -10**9

    def process_pending(upto: int) -> None:
        nonlocal last_peak, last_trough

        keep_peaks: List[int] = []
        for idx in pending_peaks:
            if upto - idx < lookahead_frames and upto < n - 1:
                keep_peaks.append(idx)
                continue
            if idx - last_peak < min_distance_frames:
                continue
            if _confirm_extreme(theta_z, idx, "peak", lookahead_frames, prominence_z, upto):
                peaks.append(idx)
                last_peak = idx
        pending_peaks[:] = keep_peaks

        keep_troughs: List[int] = []
        for idx in pending_troughs:
            if upto - idx < lookahead_frames and upto < n - 1:
                keep_troughs.append(idx)
                continue
            if idx - last_trough < min_distance_frames:
                continue
            if _confirm_extreme(theta_z, idx, "trough", lookahead_frames, prominence_z, upto):
                troughs.append(idx)
                last_trough = idx
        pending_troughs[:] = keep_troughs

    d_prev = float(theta_z[1] - theta_z[0])
    for t in range(2, n):
        d_curr = float(theta_z[t] - theta_z[t - 1])
        cand = t - 1
        if d_prev > 0.0 and d_curr <= 0.0:
            pending_peaks.append(cand)
        if d_prev < 0.0 and d_curr >= 0.0:
            pending_troughs.append(cand)

        process_pending(t)
        d_prev = d_curr

    process_pending(n - 1)
    return np.asarray(peaks, dtype=np.int32), np.asarray(troughs, dtype=np.int32)


def count_cycles_online(
    theta: np.ndarray,
    fps: float,
    min_distance_sec: float,
    min_prominence_z: float,
    min_cycle_sec: float,
    max_cycle_sec: float,
    lookahead_sec: float,
) -> Tuple[List[Tuple[int, int]], Dict[str, int]]:
    if len(theta) == 0:
        return [], {"n_peaks": 0, "n_troughs": 0}

    dist = max(1, int(round(min_distance_sec * max(fps, 1.0))))
    lookahead = max(1, int(round(lookahead_sec * max(fps, 1.0))))
    min_cycle_frames = max(1, int(round(min_cycle_sec * max(fps, 1.0))))
    max_cycle_frames = max(min_cycle_frames + 1, int(round(max_cycle_sec * max(fps, 1.0))))

    theta_z = zscore(theta)
    prom = max(0.05, float(min_prominence_z))
    peaks, troughs = detect_extrema_online(theta_z, min_distance_frames=dist, prominence_z=prom, lookahead_frames=lookahead)

    peaks = np.sort(peaks)
    troughs = np.sort(troughs)

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
                pred_periods, dbg = count_cycles_online(
                    theta=theta,
                    fps=fps,
                    min_distance_sec=args.min_distance_sec,
                    min_prominence_z=args.min_prominence_z,
                    min_cycle_sec=args.min_cycle_sec,
                    max_cycle_sec=args.max_cycle_sec,
                    lookahead_sec=args.lookahead_sec,
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
                "candidate_source": "baseline_peak_online",
                "status": status,
                "method": "baseline_peak_online",
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["split", "video_id"]).reset_index(drop=True)
    ensure_dir(args.out_csv.parent)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote online baseline predictions: {args.out_csv}")


if __name__ == "__main__":
    main()
