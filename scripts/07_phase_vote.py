from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from common_llsp import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-cycle validation with redundancy voting.")
    parser.add_argument(
        "--candidate-source",
        type=str,
        default="baseline_peak",
        choices=["baseline_peak", "online_crossing", "native_online_csv"],
        help="Where candidate windows come from.",
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path("outputs/04_results/baseline_peak.csv"),
        help="Baseline output CSV that provides candidate windows.",
    )
    parser.add_argument(
        "--subset-csv",
        type=Path,
        default=Path("outputs/00_index/subset_debug.csv"),
        help="Subset CSV (used when --candidate-source=online_crossing).",
    )
    parser.add_argument(
        "--native-csv",
        type=Path,
        default=Path("outputs/04_results/phase_native_online_partA_test_all_softpenalty.csv"),
        help="Phase-native output CSV that provides online candidate windows.",
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
    parser.add_argument("--pause-delta-pi", type=float, default=0.06, help="Per-step |dphi| below this (in pi units) is pause-like.")
    parser.add_argument("--phase-var-sigma", type=float, default=1.0, help="Soft penalty scale for phase-step variance.")
    parser.add_argument("--flip-sigma", type=float, default=0.35, help="Soft penalty scale for phase-step sign flips.")
    parser.add_argument("--pause-sigma", type=float, default=0.45, help="Soft penalty scale for pause-rate in window.")
    parser.add_argument("--cross-hyst-pi", type=float, default=0.08, help="Crossing hysteresis in pi units for online candidates.")
    parser.add_argument("--min-cycle-sec", type=float, default=0.3)
    parser.add_argument("--max-cycle-sec", type=float, default=8.0)
    parser.add_argument("--cooldown-sec", type=float, default=0.2)
    parser.add_argument("--warmup-sec", type=float, default=2.5)
    parser.add_argument("--moving-window-sec", type=float, default=0.8)
    parser.add_argument("--moving-quantile", type=float, default=0.35)
    parser.add_argument("--moving-floor", type=float, default=0.02)
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


def wrap_pi_arr(x: np.ndarray) -> np.ndarray:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def wrap_pi(x: float) -> float:
    return float((x + math.pi) % (2.0 * math.pi) - math.pi)


def circular_mean(vals: np.ndarray) -> float:
    if vals.size == 0:
        return 0.0
    s = np.sin(vals).mean()
    c = np.cos(vals).mean()
    return float(np.arctan2(s, c))


def extract_online_crossing_candidates(
    phi: np.ndarray,
    r: np.ndarray,
    fps: float,
    cross_hyst_pi: float,
    min_cycle_sec: float,
    max_cycle_sec: float,
    cooldown_sec: float,
    warmup_sec: float,
    moving_window_sec: float,
    moving_quantile: float,
    moving_floor: float,
) -> List[Tuple[int, int]]:
    n = len(phi)
    if n < 3:
        return []

    wrapped = np.asarray([wrap_pi(float(v)) for v in phi], dtype=np.float32)
    warmup_frames = max(5, int(round(warmup_sec * max(fps, 1.0))))
    psi0 = circular_mean(wrapped[: min(n, warmup_frames)])
    rel = np.asarray([wrap_pi(float(v - psi0)) for v in wrapped], dtype=np.float32)

    hyst = float(cross_hyst_pi) * math.pi
    min_frames = max(1, int(round(min_cycle_sec * max(fps, 1.0))))
    max_frames = max(min_frames + 1, int(round(max_cycle_sec * max(fps, 1.0))))
    cooldown_frames = max(0, int(round(cooldown_sec * max(fps, 1.0))))
    moving_win = max(3, int(round(moving_window_sec * max(fps, 1.0))))

    state = "IDLE"
    start = None
    prev_rel = None
    cooldown = 0
    out: List[Tuple[int, int]] = []

    for t in range(n):
        if cooldown > 0:
            cooldown -= 1

        lo = max(0, t - moving_win + 1)
        r_recent = np.asarray(r[lo : t + 1], dtype=np.float32)
        if r_recent.size == 0:
            moving = False
        else:
            thr = max(float(moving_floor), float(np.quantile(r_recent, moving_quantile)))
            moving = float(r[t]) > thr

        cur = float(rel[t])
        if state == "IDLE":
            if moving:
                state = "ACTIVE"
                start = None
                prev_rel = cur
            continue

        if not moving:
            state = "IDLE"
            start = None
            prev_rel = cur
            continue

        prev = cur if prev_rel is None else float(prev_rel)
        crossing = prev <= -hyst and cur >= hyst
        if crossing and cooldown == 0:
            if start is None:
                start = t
            else:
                span = int(t - start)
                if min_frames <= span <= max_frames:
                    out.append((int(start), int(t)))
                    cooldown = cooldown_frames
                start = t
        prev_rel = cur

    return out


def build_records_from_subset(subset: pd.DataFrame) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for _, row in subset.iterrows():
        split = str(row["split"])
        video_id = str(row["video_id"])
        action = str(row["canonical_action"])
        true_val = row.get("periods_json", "")
        true_periods = parse_periods(str(true_val) if true_val is not None else "")
        n_periods = row.get("n_periods")
        if n_periods is not None and bool(pd.notna(n_periods)):
            true_count = int(n_periods)
        else:
            true_count = len(true_periods)
        records.append(
            {
                "split": split,
                "video_id": video_id,
                "action": action,
                "true_count": true_count,
                "true_periods": true_periods,
                "candidates": [],
                "needs_online_candidates": True,
            }
        )
    return records


def build_records_from_baseline(base: pd.DataFrame) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for _, row in base.iterrows():
        true_val = row.get("true_periods_json", "")
        pred_val = row.get("pred_periods_json", "")
        records.append(
            {
                "split": str(row["split"]),
                "video_id": str(row["video_id"]),
                "action": str(row["action"]),
                "true_count": int(row["true_count"]),
                "true_periods": parse_periods(str(true_val) if true_val is not None else ""),
                "candidates": parse_periods(str(pred_val) if pred_val is not None else ""),
                "needs_online_candidates": False,
            }
        )
    return records


def build_records_from_native(native: pd.DataFrame) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for _, row in native.iterrows():
        true_val = row.get("true_periods_json", "")
        pred_val = row.get("pred_periods_json", "")
        records.append(
            {
                "split": str(row["split"]),
                "video_id": str(row["video_id"]),
                "action": str(row["action"]),
                "true_count": int(row["true_count"]),
                "true_periods": parse_periods(str(true_val) if true_val is not None else ""),
                "candidates": parse_periods(str(pred_val) if pred_val is not None else ""),
                "needs_online_candidates": False,
            }
        )
    return records


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
    phase_var_sigma: float,
    flip_sigma: float,
    pause_sigma: float,
    pause_delta: float,
) -> Tuple[bool, float, float, float, float, float, float, float]:
    if t0 < 0 or t1 >= len(phi) or t1 <= t0:
        return False, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0

    seg_phi = np.asarray(phi[t0 : t1 + 1], dtype=np.float32)
    if len(seg_phi) < 2:
        return False, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
    d = np.diff(seg_phi)
    d = wrap_pi_arr(d)
    delta_abs = float(abs(np.sum(d)))

    mean_abs = float(np.mean(np.abs(d)))
    std_d = float(np.std(d))
    var_norm = float(std_d / max(mean_abs, 1e-3))
    sign = np.sign(d)
    sign = sign[sign != 0]
    if len(sign) < 2:
        flip_rate = 0.0
    else:
        flip_rate = float(np.mean(sign[1:] != sign[:-1]))
    pause_rate = float(np.mean(np.abs(d) < pause_delta))

    segment = rad[t0 : t1 + 1]
    med_r = float(np.nanmedian(segment)) if len(segment) > 0 else 0.0
    if not np.isfinite(med_r):
        med_r = 0.0

    phase_ok = delta_min <= delta_abs <= delta_max
    radius_ok = med_r > radius_min
    passed = phase_ok and radius_ok

    score_phase = math.exp(-abs(delta_abs - (2.0 * math.pi)) / max(sigma, 1e-6))
    score_radius = clamp(med_r / max(radius_target, 1e-6), 0.0, 1.0)
    score_var = float(np.exp(-var_norm / max(phase_var_sigma, 1e-6)))
    score_flip = float(np.exp(-flip_rate / max(flip_sigma, 1e-6)))
    score_pause = float(np.exp(-pause_rate / max(pause_sigma, 1e-6)))
    score = float(0.40 * score_phase + 0.20 * score_radius + 0.15 * score_var + 0.15 * score_flip + 0.10 * score_pause)
    return passed, score, var_norm, flip_rate, pause_rate, score_phase, score_radius, score_var


def main() -> None:
    args = parse_args()
    if args.candidate_source == "baseline_peak":
        base = pd.read_csv(args.baseline_csv)
        records = build_records_from_baseline(base)
    elif args.candidate_source == "native_online_csv":
        native = pd.read_csv(args.native_csv)
        records = build_records_from_native(native)
    else:
        subset = pd.read_csv(args.subset_csv)
        records = build_records_from_subset(subset)

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rows: List[Dict] = []
    for rec in records:
        split = str(rec["split"])
        video_id = str(rec["video_id"])
        action = str(rec["action"])
        true_count = int(rec["true_count"])
        true_periods = list(rec["true_periods"])
        candidates: List[Tuple[int, int]] = list(rec["candidates"])
        needs_online_candidates = bool(rec.get("needs_online_candidates", False))

        signal_file = args.signals_dir / split / f"{video_id}.npz"
        accepted_periods: List[Tuple[int, int]] = []
        confidence_list: List[float] = []
        var_norm_all: List[float] = []
        flip_rate_all: List[float] = []
        pause_rate_all: List[float] = []
        conf_phase_all: List[float] = []
        conf_r_all: List[float] = []
        conf_var_all: List[float] = []
        conf_flip_all: List[float] = []
        conf_pause_all: List[float] = []
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
            if needs_online_candidates:
                primary_name = str(conf_action["signals"][0]["name"])
                phi_key = f"phi_{primary_name}"
                r_key = f"r_{primary_name}"
                if phi_key not in data.files or r_key not in data.files:
                    status = "missing_primary_phi_or_r"
                    candidates = []
                else:
                    fps = float(data["fps"]) if "fps" in data.files else 30.0
                    candidates = extract_online_crossing_candidates(
                        phi=np.asarray(data[phi_key], dtype=np.float32),
                        r=np.asarray(data[r_key], dtype=np.float32),
                        fps=fps,
                        cross_hyst_pi=float(args.cross_hyst_pi),
                        min_cycle_sec=float(args.min_cycle_sec),
                        max_cycle_sec=float(args.max_cycle_sec),
                        cooldown_sec=float(args.cooldown_sec),
                        warmup_sec=float(args.warmup_sec),
                        moving_window_sec=float(args.moving_window_sec),
                        moving_quantile=float(args.moving_quantile),
                        moving_floor=float(args.moving_floor),
                    )

            for t0, t1 in candidates:
                pass_count = 0
                scores: List[float] = []
                for sig in signal_names:
                    phi_key = f"phi_{sig}"
                    r_key = f"r_{sig}"
                    if phi_key not in data.files or r_key not in data.files:
                        continue
                    passed, score, var_norm, flip_rate, pause_rate, c_phase, c_r, c_var = score_window(
                        phi=data[phi_key],
                        rad=data[r_key],
                        t0=t0,
                        t1=t1,
                        delta_min=delta_min,
                        delta_max=delta_max,
                        radius_min=radius_min,
                        sigma=sigma,
                        radius_target=args.radius_target,
                        phase_var_sigma=args.phase_var_sigma,
                        flip_sigma=args.flip_sigma,
                        pause_sigma=args.pause_sigma,
                        pause_delta=args.pause_delta_pi * math.pi,
                    )
                    if passed:
                        pass_count += 1
                    scores.append(score)
                    var_norm_all.append(var_norm)
                    flip_rate_all.append(flip_rate)
                    pause_rate_all.append(pause_rate)
                    conf_phase_all.append(c_phase)
                    conf_r_all.append(c_r)
                    conf_var_all.append(c_var)
                    conf_flip_all.append(float(np.exp(-flip_rate / max(args.flip_sigma, 1e-6))))
                    conf_pause_all.append(float(np.exp(-pause_rate / max(args.pause_sigma, 1e-6))))

                if pass_count >= args.vote_k:
                    accepted_periods.append((t0, t1))
                    confidence_list.append(float(np.mean(scores)) if len(scores) else 0.0)

        pred_count = len(accepted_periods)
        mean_conf = float(np.mean(confidence_list)) if len(confidence_list) else 0.0
        mean_var_norm = float(np.mean(var_norm_all)) if var_norm_all else 0.0
        mean_flip_rate = float(np.mean(flip_rate_all)) if flip_rate_all else 0.0
        mean_pause_rate = float(np.mean(pause_rate_all)) if pause_rate_all else 0.0
        conf_phase = float(np.mean(conf_phase_all)) if conf_phase_all else 0.0
        conf_r = float(np.mean(conf_r_all)) if conf_r_all else 0.0
        conf_var = float(np.mean(conf_var_all)) if conf_var_all else 0.0
        conf_flip = float(np.mean(conf_flip_all)) if conf_flip_all else 0.0
        conf_pause = float(np.mean(conf_pause_all)) if conf_pause_all else 0.0

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
                "mean_var_norm": mean_var_norm,
                "mean_flip_rate": mean_flip_rate,
                "mean_pause_rate": mean_pause_rate,
                "conf_phase": conf_phase,
                "conf_r": conf_r,
                "conf_var": conf_var,
                "conf_flip": conf_flip,
                "conf_pause": conf_pause,
                "vote_k": int(args.vote_k),
                "candidate_source": str(args.candidate_source),
                "status": status,
                "method": (
                    "proposed_phase_vote"
                    if args.candidate_source == "baseline_peak"
                    else "proposed_phase_vote_native_candidates"
                    if args.candidate_source == "native_online_csv"
                    else "proposed_phase_vote_online"
                ),
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["split", "video_id"]).reset_index(drop=True)
    ensure_dir(args.out_csv.parent)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote phase-vote predictions: {args.out_csv}")


if __name__ == "__main__":
    main()
