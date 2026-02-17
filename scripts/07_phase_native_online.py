from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from common_llsp import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-native online counter (phase-crossing segmentation, streaming replay).")
    parser.add_argument(
        "--subset-csv",
        type=Path,
        default=Path("outputs/00_index/subset_debug.csv"),
        help="Subset CSV with split/video/action and GT periods.",
    )
    parser.add_argument(
        "--signals-dir",
        type=Path,
        default=Path("outputs/03_signals"),
        help="Directory with per-video signal npz files.",
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
        default=Path("outputs/04_results/phase_native_online.csv"),
        help="Output prediction CSV.",
    )
    parser.add_argument("--vote-k", type=int, default=2)
    parser.add_argument("--tol-pi", type=float, default=0.4, help="Completion threshold: 2pi - tol.")
    parser.add_argument("--eps-phi", type=float, default=0.35, help="Relative phase error tolerance.")
    parser.add_argument("--t-min", type=float, default=0.25)
    parser.add_argument("--t-max", type=float, default=3.0)
    parser.add_argument("--cooldown-sec", type=float, default=0.2)
    parser.add_argument("--moving-window-sec", type=float, default=0.3)
    parser.add_argument("--norm-window-sec", type=float, default=3.0)
    parser.add_argument("--r-quantile", type=float, default=0.1)
    parser.add_argument("--r-floor", type=float, default=0.05)
    parser.add_argument("--ema-alpha", type=float, default=0.25, help="EMA for theta.")
    parser.add_argument("--omega-alpha", type=float, default=0.35, help="EMA for omega.")
    parser.add_argument("--reject-consume-max-pi", type=float, default=1.0, help="Max pi consumed on reject.")
    parser.add_argument("--warmup-sec", type=float, default=2.5, help="Warmup window for adaptive crossing landmark.")
    parser.add_argument("--cross-hyst-pi", type=float, default=0.08, help="Phase crossing hysteresis in pi units.")
    parser.add_argument(
        "--diagnostic-dir",
        type=Path,
        default=Path("outputs/01_preview/crossing_diagnostics"),
        help="Directory for crossing diagnostics plots.",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Disable diagnostic plots.",
    )
    return parser.parse_args()


def wrap_pi(x: float) -> float:
    return float((x + math.pi) % (2.0 * math.pi) - math.pi)


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


def median_iqr(values: Deque[float], floor: float = 1e-3) -> Tuple[float, float]:
    if not values:
        return 0.0, 1.0
    arr = np.asarray(values, dtype=np.float32)
    med = float(np.median(arr))
    q1 = float(np.quantile(arr, 0.25))
    q3 = float(np.quantile(arr, 0.75))
    iqr = max(floor, q3 - q1)
    return med, iqr


@dataclass
class SignalState:
    theta_ema: Optional[float] = None
    omega_ema: float = 0.0
    prev_angle_raw: Optional[float] = None
    phi: float = 0.0


class OnlineSignalTracker:
    def __init__(self, fps: float, norm_window_frames: int, ema_alpha: float, omega_alpha: float):
        self.fps = max(1e-6, fps)
        self.norm_window_frames = max(10, norm_window_frames)
        self.ema_alpha = float(np.clip(ema_alpha, 1e-3, 1.0))
        self.omega_alpha = float(np.clip(omega_alpha, 1e-3, 1.0))
        self.state = SignalState()
        self.theta_hist: Deque[float] = deque(maxlen=self.norm_window_frames)
        self.omega_hist: Deque[float] = deque(maxlen=self.norm_window_frames)

    def update(self, theta_raw: float) -> Tuple[float, float, float]:
        s = self.state

        if not np.isfinite(theta_raw):
            theta_raw = s.theta_ema if s.theta_ema is not None else 0.0

        if s.theta_ema is None:
            theta_f = float(theta_raw)
        else:
            theta_f = self.ema_alpha * float(theta_raw) + (1.0 - self.ema_alpha) * s.theta_ema

        prev_theta = theta_f if s.theta_ema is None else s.theta_ema
        omega_raw = (theta_f - prev_theta) * self.fps
        omega_f = self.omega_alpha * omega_raw + (1.0 - self.omega_alpha) * s.omega_ema

        self.theta_hist.append(theta_f)
        self.omega_hist.append(omega_f)
        med_t, iqr_t = median_iqr(self.theta_hist)
        med_o, iqr_o = median_iqr(self.omega_hist)

        theta_p = (theta_f - med_t) / iqr_t
        omega_p = (omega_f - med_o) / iqr_o

        angle = float(np.arctan2(omega_p, theta_p))
        if s.prev_angle_raw is None:
            phi = angle
        else:
            delta = angle - s.prev_angle_raw
            delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
            phi = s.phi + delta

        r = float(np.sqrt(theta_p * theta_p + omega_p * omega_p))

        s.theta_ema = theta_f
        s.omega_ema = omega_f
        s.prev_angle_raw = angle
        s.phi = phi
        return phi, r, angle


class PhaseNativeCounter:
    def __init__(
        self,
        signal_names: List[str],
        primary_signal: str,
        fps: float,
        vote_k: int,
        tol_pi: float,
        eps_phi: float,
        t_min: float,
        t_max: float,
        cooldown_sec: float,
        moving_window_sec: float,
        norm_window_sec: float,
        r_quantile: float,
        r_floor: float,
        ema_alpha: float,
        omega_alpha: float,
        reject_consume_max_pi: float,
        warmup_sec: float,
        cross_hyst_pi: float,
    ):
        self.signal_names = signal_names
        self.primary_signal = primary_signal
        self.fps = max(1e-6, fps)
        self.vote_k = vote_k
        self.tol = tol_pi * math.pi
        self.eps_phi = eps_phi
        self.t_min = t_min
        self.t_max = t_max
        self.cooldown_frames = max(0, int(round(cooldown_sec * self.fps)))
        self.moving_window = max(3, int(round(moving_window_sec * self.fps)))
        self.norm_window = max(20, int(round(norm_window_sec * self.fps)))
        self.r_quantile = float(np.clip(r_quantile, 0.01, 0.5))
        self.r_floor = r_floor
        self.reject_consume = reject_consume_max_pi * math.pi
        self.warmup_frames = max(5, int(round(warmup_sec * self.fps)))
        self.cross_hyst = cross_hyst_pi * math.pi

        self.trackers: Dict[str, OnlineSignalTracker] = {
            n: OnlineSignalTracker(
                fps=self.fps,
                norm_window_frames=self.norm_window,
                ema_alpha=ema_alpha,
                omega_alpha=omega_alpha,
            )
            for n in self.signal_names
        }

        self.phi_hist: Dict[str, List[float]] = {n: [] for n in self.signal_names}
        self.psi_hist: Dict[str, List[float]] = {n: [] for n in self.signal_names}
        self.r_hist: Dict[str, List[float]] = {n: [] for n in self.signal_names}

        self.state = "IDLE"
        self.cooldown = 0
        self.cross_start: Optional[int] = None
        self.prev_rel_primary: Optional[float] = None
        self.psi0: Optional[float] = None

        self.primary_psi_hist: List[float] = []
        self.primary_r_hist: List[float] = []

        self.count = 0
        self.periods: List[Tuple[int, int]] = []
        self.confidences: List[float] = []
        self.rejects = 0
        self.crossings: List[int] = []
        self.accepted_frames: List[int] = []
        self.rejected_frames: List[int] = []
        self.moving_trace: List[int] = []
        self.n_candidates = 0
        self.reject_fail_vote = 0
        self.reject_fail_phi = 0
        self.reject_fail_r = 0

    def _fit_psi0(self) -> None:
        if self.psi0 is not None:
            return
        if len(self.primary_psi_hist) < self.warmup_frames:
            return

        psi = np.asarray(self.primary_psi_hist[: self.warmup_frames], dtype=np.float32)
        r = np.asarray(self.primary_r_hist[: self.warmup_frames], dtype=np.float32)
        if len(psi) == 0:
            self.psi0 = 0.0
            return

        r_med = float(np.median(r))
        mask = r >= r_med
        psi_sel = psi[mask] if np.any(mask) else psi

        if len(psi_sel) == 0:
            self.psi0 = 0.0
            return

        sin_m = float(np.mean(np.sin(psi_sel)))
        cos_m = float(np.mean(np.cos(psi_sel)))
        if abs(sin_m) < 1e-6 and abs(cos_m) < 1e-6:
            self.psi0 = float(np.median(psi_sel))
        else:
            self.psi0 = float(np.arctan2(sin_m, cos_m))

    def _recent_r(self, name: str, t: int) -> np.ndarray:
        arr = self.r_hist[name]
        if not arr:
            return np.asarray([], dtype=np.float32)
        start = max(0, t - self.norm_window + 1)
        return np.asarray(arr[start : t + 1], dtype=np.float32)

    def _moving_gate(self, t: int) -> bool:
        r_arr = self.r_hist[self.primary_signal]
        if len(r_arr) < self.moving_window:
            return False
        recent = np.asarray(r_arr[-self.moving_window :], dtype=np.float32)
        r_recent_all = self._recent_r(self.primary_signal, t)
        if len(r_recent_all) < self.moving_window:
            return False
        r_min = max(self.r_floor, float(np.quantile(r_recent_all, self.r_quantile)))
        return float(np.mean(recent)) > r_min

    def _window_delta_local(self, name: str, start: int, end: int) -> float:
        psi = self.psi_hist[name]
        if start < 0 or end >= len(psi) or end <= start:
            return 0.0
        seg = psi[start : end + 1]
        if len(seg) < 2:
            return 0.0
        acc = 0.0
        prev = float(seg[0])
        for cur in seg[1:]:
            d = wrap_pi(float(cur) - prev)
            acc += d
            prev = float(cur)
        return abs(acc)

    def _validate_window(self, start: int, end: int) -> Tuple[bool, float, int, int, int]:
        secondary_names = [n for n in self.signal_names if n != self.primary_signal]
        required_secondary = max(1, min(len(secondary_names), self.vote_k - 1))

        delta_primary = self._window_delta_local(self.primary_signal, start, end)
        primary_ok_phi = abs(delta_primary - 2.0 * math.pi) / (2.0 * math.pi) < self.eps_phi

        scores: List[float] = []
        fail_phi = 0
        fail_r = 0
        sec_pass = 0

        if not primary_ok_phi:
            fail_phi += 1

        primary_rad = np.asarray(self.r_hist[self.primary_signal][start : end + 1], dtype=np.float32)
        if len(primary_rad) > 0:
            r_med = float(np.median(primary_rad))
            r_recent = self._recent_r(self.primary_signal, end)
            r_min = self.r_floor if len(r_recent) == 0 else max(self.r_floor, float(np.quantile(r_recent, self.r_quantile)))
            score_phi = float(np.exp(-abs(delta_primary - 2.0 * math.pi) / (0.5 * math.pi)))
            score_r = float(np.clip(r_med / max(r_min, 1e-3), 0.0, 2.0) / 2.0)
            scores.append(0.5 * (score_phi + score_r))

        for n in secondary_names:
            rad = np.asarray(self.r_hist[n][start : end + 1], dtype=np.float32)
            if len(rad) == 0:
                continue

            delta = self._window_delta_local(n, start, end)
            ok_phi = delta > math.pi

            r_med = float(np.median(rad))
            r_recent_all = self._recent_r(n, end)
            r_min = self.r_floor if len(r_recent_all) == 0 else max(self.r_floor, float(np.quantile(r_recent_all, self.r_quantile)))
            ok_r = r_med > r_min

            if not ok_phi:
                fail_phi += 1
            if not ok_r:
                fail_r += 1
            if ok_phi and ok_r:
                sec_pass += 1

            score_phi = float(np.exp(-max(0.0, math.pi - delta) / (0.5 * math.pi)))
            score_r = float(np.clip(r_med / max(r_min, 1e-3), 0.0, 2.0) / 2.0)
            scores.append(0.5 * (score_phi + score_r))

        pass_count = int(primary_ok_phi) + sec_pass
        ok = primary_ok_phi and (sec_pass >= required_secondary)
        conf = float(np.mean(scores)) if scores else 0.0
        return ok, conf, pass_count, fail_phi, fail_r

    def _phase_direction(self, t: int) -> int:
        phi_arr = self.phi_hist[self.primary_signal]
        if len(phi_arr) < 3:
            return 1
        start = max(0, t - self.moving_window + 1)
        recent = np.asarray(phi_arr[start : t + 1], dtype=np.float32)
        if len(recent) < 3:
            return 1
        dphi = np.diff(recent)
        med = float(np.median(dphi)) if len(dphi) > 0 else 0.0
        return 1 if med >= 0.0 else -1

    def update(self, t: int, theta_values: Dict[str, float]) -> None:
        for n in self.signal_names:
            phi, r, psi = self.trackers[n].update(theta_values[n])
            self.phi_hist[n].append(phi)
            self.psi_hist[n].append(psi)
            self.r_hist[n].append(r)

        if self.cooldown > 0:
            self.cooldown -= 1
            return

        moving = self._moving_gate(t)
        self.moving_trace.append(1 if moving else 0)
        phi_p = self.phi_hist[self.primary_signal][t]
        wrapped = wrap_pi(phi_p)

        self.primary_psi_hist.append(wrapped)
        self.primary_r_hist.append(self.r_hist[self.primary_signal][t])
        self._fit_psi0()

        if self.psi0 is None:
            return

        rel = wrap_pi(wrapped - self.psi0)

        if self.state == "IDLE":
            if moving:
                self.state = "ACTIVE"
                self.cross_start = None
                self.prev_rel_primary = rel
            return

        if self.state != "ACTIVE":
            self.state = "IDLE"
            self.cross_start = None
            self.prev_rel_primary = rel
            return

        if not moving:
            self.state = "IDLE"
            self.cross_start = None
            self.prev_rel_primary = rel
            return

        prev_rel = rel if self.prev_rel_primary is None else self.prev_rel_primary
        crossing = prev_rel <= -self.cross_hyst and rel >= self.cross_hyst

        if crossing:
            self.crossings.append(t)
            if self.cross_start is None:
                self.cross_start = t
            else:
                start = self.cross_start
                end = t
                dur = (end - start) / self.fps

                if self.t_min <= dur <= self.t_max:
                    self.n_candidates += 1
                    ok, conf, _, fail_phi, fail_r = self._validate_window(start, end)
                    if ok:
                        self.count += 1
                        self.periods.append((start, end))
                        self.confidences.append(conf)
                        self.accepted_frames.append(end)
                    else:
                        self.rejects += 1
                        self.reject_fail_vote += 1
                        self.reject_fail_phi += fail_phi
                        self.reject_fail_r += fail_r
                        self.rejected_frames.append(end)
                    self.cooldown = self.cooldown_frames

                self.cross_start = t

        self.prev_rel_primary = rel


def run_on_video(
    signal_path: Path,
    signal_names: List[str],
    primary_signal: str,
    args: argparse.Namespace,
) -> Dict:
    data = np.load(signal_path)
    fps = float(data["fps"])

    theta_series: Dict[str, np.ndarray] = {}
    for n in signal_names:
        k = f"theta_raw_{n}"
        if k not in data.files:
            k = f"theta_{n}"
        if k not in data.files:
            raise KeyError(f"Missing theta signal for {n} in {signal_path}")
        theta_series[n] = data[k].astype(np.float32)

    n_frames = len(theta_series[primary_signal])
    counter = PhaseNativeCounter(
        signal_names=signal_names,
        primary_signal=primary_signal,
        fps=fps,
        vote_k=args.vote_k,
        tol_pi=args.tol_pi,
        eps_phi=args.eps_phi,
        t_min=args.t_min,
        t_max=args.t_max,
        cooldown_sec=args.cooldown_sec,
        moving_window_sec=args.moving_window_sec,
        norm_window_sec=args.norm_window_sec,
        r_quantile=args.r_quantile,
        r_floor=args.r_floor,
        ema_alpha=args.ema_alpha,
        omega_alpha=args.omega_alpha,
        reject_consume_max_pi=args.reject_consume_max_pi,
        warmup_sec=args.warmup_sec,
        cross_hyst_pi=args.cross_hyst_pi,
    )

    for t in range(n_frames):
        frame_values = {n: float(theta_series[n][t]) for n in signal_names}
        counter.update(t, frame_values)

    return {
        "pred_count": counter.count,
        "pred_periods": counter.periods,
        "mean_confidence": float(np.mean(counter.confidences)) if counter.confidences else 0.0,
        "n_rejects": int(counter.rejects),
        "fps": fps,
        "n_frames": n_frames,
        "psi": counter.primary_psi_hist,
        "r": counter.primary_r_hist,
        "moving": counter.moving_trace,
        "crossings": counter.crossings,
        "accepted": counter.accepted_frames,
        "rejected": counter.rejected_frames,
        "psi0": counter.psi0,
        "n_crossings": len(counter.crossings),
        "n_candidates": int(counter.n_candidates),
        "n_accepted": len(counter.periods),
        "reject_fail_vote": int(counter.reject_fail_vote),
        "reject_fail_phi": int(counter.reject_fail_phi),
        "reject_fail_r": int(counter.reject_fail_r),
        "moving_fraction": float(np.mean(counter.moving_trace)) if counter.moving_trace else 0.0,
    }


def plot_diagnostics(
    out_path: Path,
    split: str,
    video_id: str,
    action: str,
    out: Dict,
    gt_periods: List[Tuple[int, int]],
) -> None:
    psi = np.asarray(out["psi"], dtype=np.float32)
    rad = np.asarray(out["r"], dtype=np.float32)
    moving = np.asarray(out["moving"], dtype=np.int32)
    x = np.arange(len(psi), dtype=np.int32)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), dpi=110, sharex=True)
    fig.suptitle(f"{split}/{video_id} | {action} | phase-crossing diagnostics")

    axes[0].plot(x, psi, linewidth=1.0, label="psi (wrapped)")
    if out.get("psi0") is not None:
        axes[0].axhline(float(out["psi0"]), color="tab:purple", linestyle="--", linewidth=1.0, label="psi0")
    axes[0].set_ylabel("psi")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(x, rad, linewidth=1.0, color="tab:orange", label="r")
    axes[1].set_ylabel("r")
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(x, moving, linewidth=1.0, color="tab:green", label="moving")
    axes[2].set_ylabel("moving")
    axes[2].set_xlabel("frame")
    axes[2].legend(loc="upper right", fontsize=8)

    for ax in axes:
        for c in out["crossings"]:
            ax.axvline(int(c), color="tab:blue", alpha=0.25, linewidth=0.8)
        for c in out["accepted"]:
            ax.axvline(int(c), color="tab:green", alpha=0.35, linewidth=1.0)
        for c in out["rejected"]:
            ax.axvline(int(c), color="tab:red", alpha=0.35, linewidth=1.0)
        for s, e in gt_periods:
            ax.axvline(int(s), color="black", alpha=0.15, linewidth=0.8)
            ax.axvline(int(e), color="black", alpha=0.15, linewidth=0.8)

    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    subset = pd.read_csv(args.subset_csv)
    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rows: List[Dict] = []
    diag_targets: Dict[str, str] = {}
    if not args.no_diagnostics:
        for action in ["push_up", "pull_up", "squat"]:
            action_rows = subset[subset["canonical_action"] == action]
            if len(action_rows) > 0:
                diag_targets[action] = str(action_rows.iloc[0]["video_id"])
    for _, row in subset.iterrows():
        split = str(row["split"])
        video_id = str(row["video_id"])
        action = str(row["canonical_action"])
        periods_value = row.get("periods_json", "")
        true_periods = parse_periods(str(periods_value) if periods_value is not None else "")
        n_periods_value = row.get("n_periods")
        n_periods_text = "" if n_periods_value is None else str(n_periods_value)
        try:
            true_count = int(float(n_periods_text))
        except (TypeError, ValueError):
            true_count = len(true_periods)

        signal_file = args.signals_dir / split / f"{video_id}.npz"
        status = "ok"
        pred_count = 0
        pred_periods: List[Tuple[int, int]] = []
        mean_conf = 0.0
        n_rejects = 0
        primary_signal = ""
        n_crossings = 0
        n_candidates = 0
        n_accepted = 0
        reject_fail_vote = 0
        reject_fail_phi = 0
        reject_fail_r = 0
        moving_fraction = 0.0

        if not signal_file.exists():
            status = "missing_signal"
        elif action not in cfg:
            status = "missing_action_config"
        else:
            action_cfg = cfg[action]
            signal_names = [s["name"] for s in action_cfg["signals"]]
            primary_signal = signal_names[0]
            try:
                out = run_on_video(signal_file, signal_names, primary_signal, args)
                pred_count = int(out["pred_count"])
                pred_periods = out["pred_periods"]
                mean_conf = float(out["mean_confidence"])
                n_rejects = int(out["n_rejects"])
                n_crossings = int(out["n_crossings"])
                n_candidates = int(out["n_candidates"])
                n_accepted = int(out["n_accepted"])
                reject_fail_vote = int(out["reject_fail_vote"])
                reject_fail_phi = int(out["reject_fail_phi"])
                reject_fail_r = int(out["reject_fail_r"])
                moving_fraction = float(out["moving_fraction"])

                if not args.no_diagnostics and diag_targets.get(action) == video_id:
                    diag_path = args.diagnostic_dir / f"{action}_{split}_{video_id}.png"
                    plot_diagnostics(diag_path, split, video_id, action, out, true_periods)
            except Exception:
                status = "run_failed"

        rows.append(
            {
                "video_id": video_id,
                "split": split,
                "action": action,
                "true_count": true_count,
                "pred_count": pred_count,
                "abs_err": int(abs(pred_count - true_count)),
                "pred_periods_json": json.dumps(pred_periods, separators=(",", ":")),
                "true_periods_json": json.dumps(true_periods, separators=(",", ":")),
                "primary_signal": primary_signal,
                "mean_confidence": mean_conf,
                "n_rejects": n_rejects,
                "n_crossings": n_crossings,
                "n_candidates": n_candidates,
                "n_accepted": n_accepted,
                "reject_fail_vote": reject_fail_vote,
                "reject_fail_phi": reject_fail_phi,
                "reject_fail_r": reject_fail_r,
                "moving_fraction": moving_fraction,
                "status": status,
                "method": "phase_native_online_phase_crossing",
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["split", "video_id"]).reset_index(drop=True)
    ensure_dir(args.out_csv.parent)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote phase-native online predictions: {args.out_csv}")


if __name__ == "__main__":
    main()
