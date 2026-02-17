from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from common_llsp import ensure_dir


def _sign(x: float, eps: float = 1e-3) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


@dataclass
class FSMState:
    rep_count: int = 0
    phase: str = "BOTTOM"
    reached_top: bool = False
    t_start: Optional[float] = None
    last_metric: Optional[float] = None
    progress: float = 0.0
    frame_start: Optional[int] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FSM baseline counter (FitCoach-style, no calibration).")
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
        help="Directory containing signal npz files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/exercises.yaml"),
        help="Exercise config yaml.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/04_results/baseline_fsm.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--start-th", type=float, default=0.15, help="Progress threshold to start up phase.")
    parser.add_argument("--top-th", type=float, default=0.85, help="Progress threshold for top detection.")
    parser.add_argument("--bottom-th", type=float, default=0.15, help="Progress threshold for rep completion.")
    parser.add_argument("--early-fail-th", type=float, default=0.10, help="Early-fail progress threshold.")
    parser.add_argument("--t-min", type=float, default=0.30, help="Minimum rep duration in seconds.")
    parser.add_argument("--t-max", type=float, default=8.00, help="Maximum rep duration in seconds.")
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
            s, e = int(p[0]), int(p[1])
            if s < e:
                out.append((s, e))
    return out


def normalize_progress(theta: np.ndarray) -> Tuple[np.ndarray, float, float]:
    finite = theta[np.isfinite(theta)]
    if len(finite) == 0:
        return np.zeros_like(theta, dtype=np.float32), 0.0, 1.0
    theta_low = float(np.nanpercentile(finite, 5))
    theta_high = float(np.nanpercentile(finite, 95))
    if theta_high <= theta_low + 1e-6:
        theta_low = float(np.nanmin(finite))
        theta_high = float(np.nanmax(finite))
    denom = max(1e-6, theta_high - theta_low)
    p = (theta - theta_low) / denom
    p = np.clip(p, -0.2, 1.2).astype(np.float32)
    return p, theta_low, theta_high


def run_fsm(
    theta: np.ndarray,
    fps: float,
    start_th: float,
    top_th: float,
    bottom_th: float,
    early_fail_th: float,
    t_min: float,
    t_max: float,
) -> Tuple[List[Tuple[int, int]], Dict[str, float]]:
    p, theta_low, theta_high = normalize_progress(theta)
    state = FSMState()
    periods: List[Tuple[int, int]] = []
    dt = 1.0 / max(fps, 1e-6)

    for i, metric in enumerate(theta):
        if not np.isfinite(metric):
            continue
        t = i * dt
        prog = float(p[i])
        state.progress = prog

        last = metric if state.last_metric is None else state.last_metric
        direction = _sign(float(metric - last))

        if state.t_start is not None:
            elapsed = t - state.t_start
            if elapsed > t_max and state.phase != "BOTTOM":
                state.phase = "BOTTOM"
                state.t_start = None
                state.reached_top = False
                state.frame_start = None

        phase = state.phase
        if phase == "BOTTOM":
            if prog > start_th and direction > 0:
                state.phase = "UP_PHASE"
                state.t_start = t
                state.reached_top = False
                state.frame_start = i

        elif phase == "UP_PHASE":
            if prog >= top_th:
                state.phase = "TOP"
                state.reached_top = True
            elif prog <= early_fail_th and direction < 0:
                state.phase = "BOTTOM"
                state.t_start = None
                state.reached_top = False
                state.frame_start = None

        elif phase == "TOP":
            if prog < top_th and direction < 0:
                state.phase = "DOWN_PHASE"

        elif phase == "DOWN_PHASE":
            if prog <= bottom_th:
                if state.reached_top and state.t_start is not None:
                    duration = t - state.t_start
                    if duration >= t_min:
                        state.rep_count += 1
                        start_frame = i if state.frame_start is None else state.frame_start
                        periods.append((int(start_frame), int(i)))
                state.phase = "BOTTOM"
                state.t_start = None
                state.reached_top = False
                state.frame_start = None

        state.last_metric = float(metric)

    meta = {
        "theta_low": theta_low,
        "theta_high": theta_high,
        "final_rep_count": float(state.rep_count),
    }
    return periods, meta


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
        pred_periods: List[Tuple[int, int]] = []
        primary_signal = ""
        theta_low = np.nan
        theta_high = np.nan

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
                pred_periods, meta = run_fsm(
                    theta=theta,
                    fps=fps,
                    start_th=args.start_th,
                    top_th=args.top_th,
                    bottom_th=args.bottom_th,
                    early_fail_th=args.early_fail_th,
                    t_min=args.t_min,
                    t_max=args.t_max,
                )
                theta_low = meta["theta_low"]
                theta_high = meta["theta_high"]
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
                "theta_low_est": theta_low,
                "theta_high_est": theta_high,
                "status": status,
                "method": "baseline_fsm",
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["split", "video_id"]).reset_index(drop=True)
    ensure_dir(args.out_csv.parent)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote FSM baseline predictions: {args.out_csv}")


if __name__ == "__main__":
    main()
