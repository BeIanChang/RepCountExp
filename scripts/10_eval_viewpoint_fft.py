from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from common_llsp import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate viewpoint-invariant FFT counting baseline from skeleton similarity.")
    parser.add_argument(
        "--subset-csv",
        type=Path,
        default=Path("outputs/00_index/subset_partA_test_all.csv"),
        help="Subset CSV with split/video_id and GT count.",
    )
    parser.add_argument(
        "--pose-dir",
        type=Path,
        default=Path("outputs/02_pose"),
        help="Directory containing pose npz by split.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/04_results/viewpoint_fft_partA_test_all.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("outputs/04_results/viewpoint_fft_partA_test_all_summary.json"),
    )
    parser.add_argument("--window", type=int, default=256, help="Sliding FFT window size (frames).")
    parser.add_argument("--step", type=int, default=1, help="Sliding step size (frames).")
    parser.add_argument("--anchor-frames", type=int, default=30, help="Number of initial frames to average as anchor.")
    parser.add_argument("--min-hz", type=float, default=0.1)
    parser.add_argument("--max-hz", type=float, default=4.0)
    parser.add_argument("--max-videos", type=int, default=0)
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


def fill_nan_2d(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    t, d = y.shape
    idx = np.arange(t)
    for j in range(d):
        col = y[:, j]
        mask = np.isfinite(col)
        if mask.all():
            continue
        if not np.any(mask):
            y[:, j] = 0.0
            continue
        y[:, j] = np.interp(idx, idx[mask], col[mask])
    return y


def skeleton_similarity_series(landmarks: np.ndarray, valid_mask: np.ndarray, anchor_frames: int) -> np.ndarray:
    # landmarks: [T, J, C]
    t = landmarks.shape[0]
    x = landmarks.astype(np.float32)
    center = np.nanmean(x, axis=1, keepdims=True)
    x = x - center
    flat = x.reshape(t, -1)

    if valid_mask is not None and len(valid_mask) == t:
        valid_mask = valid_mask.astype(bool)
        flat[~valid_mask] = np.nan

    flat = fill_nan_2d(flat)
    a_len = min(max(1, anchor_frames), t)
    anchor = np.mean(flat[:a_len], axis=0)
    a_norm = float(np.linalg.norm(anchor))
    if a_norm < 1e-6:
        return np.zeros((t,), dtype=np.float32)

    norms = np.linalg.norm(flat, axis=1)
    denom = np.maximum(norms * a_norm, 1e-6)
    sim = (flat @ anchor) / denom
    return np.clip(sim, -1.0, 1.0).astype(np.float32)


def count_from_similarity_fft(sim: np.ndarray, fps: float, window: int, step: int, min_hz: float, max_hz: float) -> float:
    n = len(sim)
    if n < 4:
        return 0.0

    if n < window:
        window = n
    window = max(8, int(window))
    step = max(1, int(step))

    total = 0.0
    n_used = 0
    for s in range(0, n - window + 1, step):
        seg = sim[s : s + window].astype(np.float32)
        seg = seg - float(np.mean(seg))
        spec = np.abs(np.fft.rfft(seg))
        if spec.shape[0] <= 1:
            continue
        hz = np.fft.rfftfreq(window, d=1.0 / max(fps, 1e-6))
        valid = (hz >= min_hz) & (hz <= max_hz)
        if valid.sum() == 0:
            continue
        spec_valid = spec.copy()
        spec_valid[~valid] = 0.0
        spec_valid[0] = 0.0
        k = int(np.argmax(spec_valid))
        if k <= 0:
            continue
        f_hz = float(hz[k])
        total += f_hz * (step / max(fps, 1e-6))
        n_used += 1

    if n_used == 0:
        return 0.0
    return float(total)


def main() -> None:
    args = parse_args()
    subset = pd.read_csv(args.subset_csv)
    if args.max_videos > 0:
        subset = subset.head(args.max_videos).copy()

    rows: List[Dict] = []
    for i, row in subset.iterrows():
        split = str(row["split"])
        video_id = str(row["video_id"])
        pose_file = args.pose_dir / split / f"{video_id}.npz"

        periods_val = row.get("periods_json", "")
        periods = parse_periods(str(periods_val) if periods_val is not None else "")
        n_periods_val = row.get("n_periods")
        try:
            gt_count = int(float(n_periods_val)) if pd.notna(n_periods_val) else len(periods)
        except (TypeError, ValueError):
            gt_count = len(periods)

        status = "ok"
        pred = 0.0
        if not pose_file.exists():
            status = "missing_pose"
        else:
            try:
                d = np.load(pose_file)
                landmarks = d["landmarks"]
                valid_mask = d["valid_mask"] if "valid_mask" in d.files else np.ones((landmarks.shape[0],), dtype=bool)
                fps = float(d["fps"]) if "fps" in d.files else 30.0

                sim = skeleton_similarity_series(landmarks, valid_mask, anchor_frames=args.anchor_frames)
                pred = count_from_similarity_fft(
                    sim=sim,
                    fps=fps,
                    window=args.window,
                    step=args.step,
                    min_hz=args.min_hz,
                    max_hz=args.max_hz,
                )
            except Exception:
                status = "run_failed"
                pred = 0.0

        rows.append(
            {
                "video_id": video_id,
                "split": split,
                "action": str(row.get("canonical_action", "")),
                "true_count": int(gt_count),
                "pred_count": float(pred),
                "abs_err": float(abs(float(pred) - float(gt_count))),
                "status": status,
                "method": "viewpoint_fft_similarity",
            }
        )
        if (i + 1) % 20 == 0 or (i + 1) == len(subset):
            print(f"processed {i + 1}/{len(subset)}")

    out = pd.DataFrame(rows)
    out["is_obo"] = out["abs_err"] <= 1.0
    out["norm_err"] = out["abs_err"] / out["true_count"].clip(lower=1.0)
    out["norm_err_p1"] = out["abs_err"] / (out["true_count"] + 1e-1)

    ensure_dir(args.out_csv.parent)
    out.to_csv(args.out_csv, index=False)

    summary = {
        "n_videos": int(len(out)),
        "n_failed": int((out["status"] != "ok").sum()),
        "mae_raw": float(out["abs_err"].mean()),
        "mae_norm": float(out["norm_err"].mean()),
        "mae_norm_p1": float(out["norm_err_p1"].mean()),
        "obo": float(out["is_obo"].mean()),
        "window": int(args.window),
        "step": int(args.step),
        "anchor_frames": int(args.anchor_frames),
        "min_hz": float(args.min_hz),
        "max_hz": float(args.max_hz),
        "out_csv": str(args.out_csv),
    }

    ensure_dir(args.summary_json.parent)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
