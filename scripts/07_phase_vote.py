from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from common_llsp import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-cycle validation with redundancy voting.")
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path("outputs/04_results/baseline_peak.csv"),
        help="Baseline output CSV that provides candidate windows.",
    )
    parser.add_argument(
        "--signals-dir",
        type=Path,
        default=Path("outputs/03_signals"),
        help="Directory with signal npz files.",
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
        default=Path("outputs/04_results/proposed_phase_vote.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--vote-k", type=int, default=2)
    parser.add_argument("--sigma-pi", type=float, default=0.35)
    parser.add_argument("--radius-target", type=float, default=1.0)
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
            s = int(p[0])
            e = int(p[1])
            if s < e:
                out.append((s, e))
    return out


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def score_window(
    phi: np.ndarray,
    rad: np.ndarray,
    t0: int,
    t1: int,
    delta_min: float,
    delta_max: float,
    radius_min: float,
    sigma: float,
    radius_target: float,
) -> Tuple[bool, float]:
    if t0 < 0 or t1 >= len(phi) or t1 <= t0:
        return False, 0.0
    p0 = float(phi[t0])
    p1 = float(phi[t1])
    if not np.isfinite(p0) or not np.isfinite(p1):
        return False, 0.0
    delta = p1 - p0
    delta_abs = abs(delta)

    segment = rad[t0 : t1 + 1]
    med_r = float(np.nanmedian(segment)) if len(segment) > 0 else 0.0
    if not np.isfinite(med_r):
        med_r = 0.0

    phase_ok = delta_min <= delta_abs <= delta_max
    radius_ok = med_r > radius_min
    passed = phase_ok and radius_ok

    score_phase = math.exp(-abs(delta_abs - (2.0 * math.pi)) / max(sigma, 1e-6))
    score_radius = clamp(med_r / max(radius_target, 1e-6), 0.0, 1.0)
    score = float((score_phase + score_radius) * 0.5)
    return passed, score


def main() -> None:
    args = parse_args()
    base = pd.read_csv(args.baseline_csv)
    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rows: List[Dict] = []
    for _, row in base.iterrows():
        split = row["split"]
        video_id = row["video_id"]
        action = row["action"]
        true_count = int(row["true_count"])
        true_periods = parse_periods(row.get("true_periods_json", ""))
        candidates = parse_periods(row.get("pred_periods_json", ""))

        signal_file = args.signals_dir / split / f"{video_id}.npz"
        accepted_periods: List[Tuple[int, int]] = []
        confidence_list: List[float] = []
        status = "ok"

        if not signal_file.exists():
            status = "missing_signal"
        elif action not in cfg:
            status = "missing_action_config"
        else:
            conf_action = cfg[action]
            signal_names = [s["name"] for s in conf_action["signals"]]
            phase_cfg = conf_action.get("phase", {})
            delta_min = float(phase_cfg.get("delta_phi_min_pi", 1.7)) * math.pi
            delta_max = float(phase_cfg.get("delta_phi_max_pi", 2.5)) * math.pi
            radius_min = float(phase_cfg.get("radius_min", 0.3))
            sigma = float(args.sigma_pi) * math.pi

            data = np.load(signal_file)

            for t0, t1 in candidates:
                pass_count = 0
                scores: List[float] = []
                for sig in signal_names:
                    phi_key = f"phi_{sig}"
                    r_key = f"r_{sig}"
                    if phi_key not in data.files or r_key not in data.files:
                        continue
                    passed, score = score_window(
                        phi=data[phi_key],
                        rad=data[r_key],
                        t0=t0,
                        t1=t1,
                        delta_min=delta_min,
                        delta_max=delta_max,
                        radius_min=radius_min,
                        sigma=sigma,
                        radius_target=args.radius_target,
                    )
                    if passed:
                        pass_count += 1
                    scores.append(score)

                if pass_count >= args.vote_k:
                    accepted_periods.append((t0, t1))
                    confidence_list.append(float(np.mean(scores)) if len(scores) else 0.0)

        pred_count = len(accepted_periods)
        mean_conf = float(np.mean(confidence_list)) if len(confidence_list) else 0.0

        rows.append(
            {
                "video_id": video_id,
                "split": split,
                "action": action,
                "true_count": true_count,
                "pred_count": int(pred_count),
                "abs_err": int(abs(pred_count - true_count)),
                "pred_periods_json": json.dumps(accepted_periods, separators=(",", ":")),
                "true_periods_json": json.dumps(true_periods, separators=(",", ":")),
                "candidate_count": int(len(candidates)),
                "accepted_count": int(pred_count),
                "mean_confidence": mean_conf,
                "vote_k": int(args.vote_k),
                "status": status,
                "method": "proposed_phase_vote",
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["split", "video_id"]).reset_index(drop=True)
    ensure_dir(args.out_csv.parent)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote phase-vote predictions: {args.out_csv}")


if __name__ == "__main__":
    main()
