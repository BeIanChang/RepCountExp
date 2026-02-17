from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.signal import savgol_filter

from common_llsp import ensure_dir


LANDMARK_INDEX = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute theta/omega/phi/r signals from pose landmarks.")
    parser.add_argument(
        "--subset-csv",
        type=Path,
        default=Path("outputs/00_index/subset_debug.csv"),
        help="Input subset CSV with split/video ids and canonical_action.",
    )
    parser.add_argument(
        "--pose-dir",
        type=Path,
        default=Path("outputs/02_pose"),
        help="Directory containing pose npz files by split.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/exercises.yaml"),
        help="Exercise signal config yaml.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/03_signals"),
        help="Output directory for signal npz files.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=Path("outputs/01_preview"),
        help="Output directory for preview plots.",
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=Path("outputs/03_signals/signal_run_report.csv"),
        help="Output run report CSV.",
    )
    return parser.parse_args()


def angle_abc(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    ba_norm = np.linalg.norm(ba, axis=1)
    bc_norm = np.linalg.norm(bc, axis=1)
    denom = ba_norm * bc_norm
    valid = denom > 1e-8
    out = np.full((a.shape[0],), np.nan, dtype=np.float32)
    if np.any(valid):
        cosang = np.sum(ba[valid] * bc[valid], axis=1) / denom[valid]
        cosang = np.clip(cosang, -1.0, 1.0)
        out[valid] = np.degrees(np.arccos(cosang)).astype(np.float32)
    return out


def angle_to_vertical(v: np.ndarray) -> np.ndarray:
    v_norm = np.linalg.norm(v, axis=1)
    valid = v_norm > 1e-8
    out = np.full((v.shape[0],), np.nan, dtype=np.float32)
    if np.any(valid):
        vertical = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        dot = np.sum(v[valid] * vertical, axis=1)
        cosang = np.clip(dot / v_norm[valid], -1.0, 1.0)
        out[valid] = np.degrees(np.arccos(cosang)).astype(np.float32)
    return out


def interpolate_nan(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float32).copy()
    n = len(y)
    idx = np.arange(n)
    good = np.isfinite(y)
    if good.sum() == 0:
        return y
    if good.sum() == 1:
        y[:] = y[good][0]
        return y
    y[~good] = np.interp(idx[~good], idx[good], y[good])
    return y


def smooth_signal(x: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    y = interpolate_nan(x)
    if len(y) < 5 or np.all(~np.isfinite(y)):
        return y
    win = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if win < 5:
        return y
    po = min(polyorder, win - 1)
    try:
        return savgol_filter(y, window_length=win, polyorder=po, mode="interp").astype(np.float32)
    except ValueError:
        return y


def robust_scale(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    q1 = np.nanpercentile(x, 25)
    q3 = np.nanpercentile(x, 75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr < 1e-6:
        std = np.nanstd(x)
        if not np.isfinite(std) or std < 1e-6:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - med) / std).astype(np.float32)
    return ((x - med) / iqr).astype(np.float32)


def resolve_triplet(landmarks: np.ndarray, names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = landmarks[:, LANDMARK_INDEX[names[0]], :]
    b = landmarks[:, LANDMARK_INDEX[names[1]], :]
    c = landmarks[:, LANDMARK_INDEX[names[2]], :]
    return a, b, c


def resolve_pair(landmarks: np.ndarray, names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    a = landmarks[:, LANDMARK_INDEX[names[0]], :]
    b = landmarks[:, LANDMARK_INDEX[names[1]], :]
    return a, b


def average_pair(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = np.full_like(left, np.nan, dtype=np.float32)
    left_ok = np.isfinite(left)
    right_ok = np.isfinite(right)
    both = left_ok & right_ok
    out[both] = ((left[both] + right[both]) * 0.5).astype(np.float32)
    left_only = left_ok & ~right_ok
    out[left_only] = left[left_only].astype(np.float32)
    right_only = right_ok & ~left_ok
    out[right_only] = right[right_only].astype(np.float32)
    return out


def compute_signals_for_video(
    landmarks: np.ndarray,
    valid_mask: np.ndarray,
    fps: float,
    action_cfg: Dict,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    win = int(action_cfg.get("smoothing", {}).get("window", 15))
    poly = int(action_cfg.get("smoothing", {}).get("polyorder", 3))

    for signal_cfg in action_cfg["signals"]:
        name = signal_cfg["name"]
        kind = signal_cfg["kind"]

        if kind == "bilateral_angle":
            l_a, l_b, l_c = resolve_triplet(landmarks, signal_cfg["left"])
            r_a, r_b, r_c = resolve_triplet(landmarks, signal_cfg["right"])
            theta_left = angle_abc(l_a, l_b, l_c)
            theta_right = angle_abc(r_a, r_b, r_c)
            theta_raw = average_pair(theta_left, theta_right)
        elif kind == "bilateral_vertical":
            l_hip, l_sh = resolve_pair(landmarks, signal_cfg["left"])
            r_hip, r_sh = resolve_pair(landmarks, signal_cfg["right"])
            left_vec = l_sh - l_hip
            right_vec = r_sh - r_hip
            theta_left = angle_to_vertical(left_vec)
            theta_right = angle_to_vertical(right_vec)
            theta_raw = average_pair(theta_left, theta_right)
        else:
            raise ValueError(f"Unsupported signal kind: {kind}")

        theta_raw[~valid_mask] = np.nan
        theta = smooth_signal(theta_raw, window=win, polyorder=poly)
        omega = np.gradient(theta).astype(np.float32) * np.float32(fps if np.isfinite(fps) and fps > 0 else 1.0)

        theta_n = robust_scale(theta)
        omega_n = robust_scale(omega)
        phi = np.unwrap(np.arctan2(omega_n, theta_n)).astype(np.float32)
        r = np.sqrt(theta_n**2 + omega_n**2).astype(np.float32)

        out[f"theta_raw_{name}"] = theta_raw.astype(np.float32)
        out[f"theta_{name}"] = theta.astype(np.float32)
        out[f"omega_{name}"] = omega.astype(np.float32)
        out[f"phi_{name}"] = phi.astype(np.float32)
        out[f"r_{name}"] = r.astype(np.float32)

    out["valid_mask"] = valid_mask.astype(bool)
    out["time_sec"] = (np.arange(landmarks.shape[0], dtype=np.float32) / np.float32(max(fps, 1e-6))).astype(np.float32)
    return out


def make_preview_plot(
    out_path: Path,
    signal_names: List[str],
    result: Dict[str, np.ndarray],
    split: str,
    video_id: str,
    action: str,
) -> None:
    primary = signal_names[0]
    t = result["time_sec"]
    theta = result[f"theta_{primary}"]
    omega = result[f"omega_{primary}"]
    phi = result[f"phi_{primary}"]
    rad = result[f"r_{primary}"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=110)
    fig.suptitle(f"{split}/{video_id} | action={action} | primary={primary}")

    ax = axes[0, 0]
    for name in signal_names:
        ax.plot(t, result[f"theta_{name}"], linewidth=1.0, label=f"theta:{name}")
    ax.set_title("theta(t)")
    ax.set_xlabel("sec")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[0, 1]
    for name in signal_names:
        ax.plot(t, result[f"omega_{name}"], linewidth=1.0, label=f"omega:{name}")
    ax.set_title("omega(t)")
    ax.set_xlabel("sec")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1, 0]
    ax.plot(theta, omega, linewidth=1.0)
    ax.set_title(f"phase portrait ({primary})")
    ax.set_xlabel("theta")
    ax.set_ylabel("omega")

    ax = axes[1, 1]
    ax.plot(t, phi, linewidth=1.0, label="phi")
    ax.plot(t, rad, linewidth=1.0, label="r")
    ax.set_title(f"phi(t), r(t) ({primary})")
    ax.set_xlabel("sec")
    ax.legend(fontsize=8)

    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    subset = pd.read_csv(args.subset_csv)
    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    report_rows: List[Dict] = []
    total = len(subset)

    for idx, row in subset.iterrows():
        split = row["split"]
        video_id = row["video_id"]
        action = row["canonical_action"]
        pose_path = args.pose_dir / split / f"{video_id}.npz"
        signal_out = args.out_dir / split / f"{video_id}.npz"
        preview_out = args.preview_dir / split / f"{video_id}_plots.png"

        status = "ok"
        message = ""

        if action not in cfg:
            status = "unsupported_action"
            message = f"Action not in config: {action}"
            report_rows.append(
                {
                    "split": split,
                    "video_id": video_id,
                    "action": action,
                    "status": status,
                    "message": message,
                    "signal_count": 0,
                    "output_npz": "",
                    "preview_png": "",
                }
            )
            print(f"[{idx + 1}/{total}] {split}/{video_id}: {status}")
            continue

        if not pose_path.exists():
            status = "missing_pose"
            message = f"Pose file not found: {pose_path}"
            report_rows.append(
                {
                    "split": split,
                    "video_id": video_id,
                    "action": action,
                    "status": status,
                    "message": message,
                    "signal_count": 0,
                    "output_npz": "",
                    "preview_png": "",
                }
            )
            print(f"[{idx + 1}/{total}] {split}/{video_id}: {status}")
            continue

        pose = np.load(pose_path)
        landmarks = pose["landmarks"].astype(np.float32)
        valid_mask = pose["valid_mask"].astype(bool)
        fps = float(pose["fps"])
        n_frames = int(pose["n_frames"])

        action_cfg = cfg[action]
        signal_names = [s["name"] for s in action_cfg["signals"]]
        result = compute_signals_for_video(
            landmarks=landmarks,
            valid_mask=valid_mask,
            fps=fps,
            action_cfg=action_cfg,
        )

        save_dict: Dict[str, np.ndarray] = {
            "fps": np.float32(fps),
            "n_frames": np.int32(n_frames),
            "valid_mask": result["valid_mask"],
            "time_sec": result["time_sec"],
            "signal_names": np.array(signal_names, dtype="U64"),
        }
        for k, v in result.items():
            if k in {"valid_mask", "time_sec"}:
                continue
            save_dict[k] = v

        ensure_dir(signal_out.parent)
        np.savez_compressed(signal_out, **save_dict)
        make_preview_plot(preview_out, signal_names, result, split, video_id, action)

        report_rows.append(
            {
                "split": split,
                "video_id": video_id,
                "action": action,
                "status": status,
                "message": message,
                "signal_count": len(signal_names),
                "n_frames": n_frames,
                "valid_ratio": float(valid_mask.mean()) if len(valid_mask) > 0 else 0.0,
                "output_npz": str(signal_out),
                "preview_png": str(preview_out),
            }
        )
        print(f"[{idx + 1}/{total}] {split}/{video_id}: {status} signals={len(signal_names)}")

    report_df = pd.DataFrame(report_rows).sort_values(["split", "video_id"]).reset_index(drop=True)
    ensure_dir(args.report_csv.parent)
    report_df.to_csv(args.report_csv, index=False)
    print(f"Wrote signal report: {args.report_csv}")


if __name__ == "__main__":
    main()
