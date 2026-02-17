from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from common_llsp import ensure_dir


MP_POSE = mp.solutions.pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract pose landmarks with MediaPipe.")
    parser.add_argument(
        "--subset-csv",
        type=Path,
        default=Path("outputs/00_index/subset_debug.csv"),
        help="CSV listing videos to process.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/02_pose"),
        help="Output base directory for pose npz files.",
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=Path("outputs/02_pose/pose_run_report.csv"),
        help="Output report CSV path.",
    )
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--min-det", type=float, default=0.5)
    parser.add_argument("--min-track", type=float, default=0.5)
    parser.add_argument("--static-image-mode", action="store_true")
    return parser.parse_args()


def extract_video_pose(
    video_path: Path,
    model_complexity: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    static_image_mode: bool,
) -> Dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "status": "video_open_failed",
            "fps": np.nan,
            "n_frames": 0,
            "landmarks": np.zeros((0, 33, 3), dtype=np.float32),
            "visibility": np.zeros((0, 33), dtype=np.float32),
            "valid_mask": np.zeros((0,), dtype=bool),
            "img_wh": np.array([np.nan, np.nan], dtype=np.float32),
        }

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    landmarks: List[np.ndarray] = []
    visibility: List[np.ndarray] = []
    valid_mask: List[bool] = []
    img_wh: np.ndarray | None = None

    with MP_POSE.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if img_wh is None:
                h, w = frame.shape[:2]
                img_wh = np.array([float(w), float(h)], dtype=np.float32)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)

            if result.pose_landmarks is None:
                landmarks.append(np.full((33, 3), np.nan, dtype=np.float32))
                visibility.append(np.zeros((33,), dtype=np.float32))
                valid_mask.append(False)
                continue

            frame_lm = np.array(
                [[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark],
                dtype=np.float32,
            )
            frame_vis = np.array([lm.visibility for lm in result.pose_landmarks.landmark], dtype=np.float32)

            landmarks.append(frame_lm)
            visibility.append(frame_vis)
            valid_mask.append(True)

    cap.release()

    n_out = len(landmarks)
    if n_out == 0:
        return {
            "status": "no_frames",
            "fps": fps if fps > 0 else np.nan,
            "n_frames": 0,
            "landmarks": np.zeros((0, 33, 3), dtype=np.float32),
            "visibility": np.zeros((0, 33), dtype=np.float32),
            "valid_mask": np.zeros((0,), dtype=bool),
            "img_wh": np.array([np.nan, np.nan], dtype=np.float32) if img_wh is None else img_wh,
            "expected_frames": expected_frames,
        }

    return {
        "status": "ok",
        "fps": fps if fps > 0 else np.nan,
        "n_frames": n_out,
        "landmarks": np.stack(landmarks, axis=0),
        "visibility": np.stack(visibility, axis=0),
        "valid_mask": np.array(valid_mask, dtype=bool),
        "img_wh": np.array([np.nan, np.nan], dtype=np.float32) if img_wh is None else img_wh,
        "expected_frames": expected_frames,
    }


def main() -> None:
    args = parse_args()
    subset = pd.read_csv(args.subset_csv)

    required_cols = {"split", "video_id", "video_path"}
    missing_cols = sorted(required_cols - set(subset.columns))
    if missing_cols:
        raise ValueError(f"subset CSV missing required columns: {missing_cols}")

    ensure_dir(args.out_dir)
    ensure_dir(args.report_csv.parent)

    report_rows: List[Dict] = []

    total = len(subset)
    for idx, row in subset.iterrows():
        split = row["split"]
        video_id = row["video_id"]
        video_path = Path(str(row["video_path"]))

        output_path = args.out_dir / split / f"{video_id}.npz"
        ensure_dir(output_path.parent)

        result = extract_video_pose(
            video_path=video_path,
            model_complexity=args.model_complexity,
            min_detection_confidence=args.min_det,
            min_tracking_confidence=args.min_track,
            static_image_mode=args.static_image_mode,
        )

        np.savez_compressed(
            output_path,
            landmarks=result["landmarks"],
            visibility=result["visibility"],
            fps=np.float32(result["fps"]),
            n_frames=np.int32(result["n_frames"]),
            valid_mask=result["valid_mask"],
            image_wh=result["img_wh"],
        )

        n_frames = int(result["n_frames"])
        valid_frames = int(result["valid_mask"].sum()) if n_frames > 0 else 0
        valid_ratio = (valid_frames / n_frames) if n_frames > 0 else 0.0

        report_rows.append(
            {
                "split": split,
                "video_id": video_id,
                "video_path": str(video_path),
                "status": result["status"],
                "fps": result["fps"],
                "expected_n_frames": int(result.get("expected_frames", n_frames)),
                "n_frames": n_frames,
                "valid_frames": valid_frames,
                "valid_ratio": valid_ratio,
                "dropped_frames": int(max(0, int(result.get("expected_frames", n_frames)) - n_frames)),
                "output_npz": str(output_path),
            }
        )

        print(f"[{idx + 1}/{total}] {split}/{video_id}: status={result['status']} valid_ratio={valid_ratio:.3f}")

    report_df = pd.DataFrame(report_rows).sort_values(["split", "video_id"]).reset_index(drop=True)
    report_df.to_csv(args.report_csv, index=False)

    print(f"Wrote pose report: {args.report_csv}")


if __name__ == "__main__":
    main()
