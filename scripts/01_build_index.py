from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd

from common_llsp import ensure_dir, l_columns, normalize_action, read_annotations, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LLSP master index.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/LLSP"),
        help="Dataset root containing annotation/ and video/.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/00_index/master_index.csv"),
        help="Output CSV path for master index.",
    )
    parser.add_argument(
        "--quality-json",
        type=Path,
        default=Path("outputs/00_index/data_quality_report.json"),
        help="Output JSON path for data quality report.",
    )
    return parser.parse_args()


def get_video_meta(video_path: Path) -> Tuple[float, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.nan, -1, np.nan
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration = (n_frames / fps) if fps > 0 else np.nan
    return fps, n_frames, duration


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    df = read_annotations(data_root)
    l_cols = l_columns(df)

    if "name" not in df.columns or "type" not in df.columns or "count" not in df.columns:
        raise ValueError("Expected columns: name, type, count")

    df["video_name"] = df["name"].astype(str)
    df["video_id"] = df["video_name"].str.replace(".mp4", "", regex=False)
    df["video_path"] = df.apply(
        lambda row: (data_root / "video" / row["split"] / row["video_name"]).as_posix(), axis=1
    )
    df["video_exists"] = df["video_path"].map(lambda p: Path(p).exists())
    df["raw_type"] = df["type"].astype(str)
    df["canonical_action"] = df["raw_type"].map(normalize_action)
    df["label_count"] = pd.to_numeric(df["count"], errors="coerce")
    df["l_non_nan"] = df[l_cols].notna().sum(axis=1)

    fps_list = []
    frames_list = []
    duration_list = []
    for video_path, exists in zip(df["video_path"], df["video_exists"]):
        if not exists:
            fps_list.append(np.nan)
            frames_list.append(np.nan)
            duration_list.append(np.nan)
            continue
        fps, n_frames, duration = get_video_meta(Path(video_path))
        fps_list.append(fps)
        frames_list.append(n_frames)
        duration_list.append(duration)

    df["fps"] = fps_list
    df["n_frames"] = frames_list
    df["duration_sec"] = duration_list

    out_columns = [
        "video_id",
        "video_name",
        "split",
        "video_path",
        "raw_type",
        "canonical_action",
        "label_count",
        "l_non_nan",
        "fps",
        "n_frames",
        "duration_sec",
    ]

    out_df = df[out_columns].copy()
    ensure_dir(args.out_csv.parent)
    out_df.to_csv(args.out_csv, index=False)

    missing_video_rows = df.loc[~df["video_exists"], ["split", "video_name"]]
    bad_meta_rows = df.loc[(df["video_exists"]) & ((df["fps"] <= 0) | (df["n_frames"] <= 0)), ["split", "video_name", "fps", "n_frames"]]
    rows_missing_count = df.loc[df["label_count"].isna(), ["split", "video_name", "raw_type"]]
    rows_no_locations = df.loc[df["l_non_nan"] == 0, ["split", "video_name", "raw_type", "label_count"]]

    quality_payload: Dict = {
        "summary": {
            "rows_total": int(len(df)),
            "rows_per_split": {k: int(v) for k, v in df.groupby("split").size().to_dict().items()},
            "unique_raw_actions": int(df["raw_type"].nunique()),
            "unique_canonical_actions": int(df["canonical_action"].nunique()),
            "missing_videos": int((~df["video_exists"]).sum()),
            "missing_count_rows": int(df["label_count"].isna().sum()),
            "rows_without_locations": int((df["l_non_nan"] == 0).sum()),
            "bad_video_metadata_rows": int(len(bad_meta_rows)),
        },
        "raw_action_distribution": {k: int(v) for k, v in df["raw_type"].value_counts().to_dict().items()},
        "canonical_action_distribution": {
            k: int(v) for k, v in df["canonical_action"].value_counts().to_dict().items()
        },
        "examples": {
            "missing_video_rows": missing_video_rows.head(20).to_dict(orient="records"),
            "bad_video_metadata_rows": bad_meta_rows.head(20).to_dict(orient="records"),
            "rows_missing_count": rows_missing_count.head(20).to_dict(orient="records"),
            "rows_no_locations": rows_no_locations.head(20).to_dict(orient="records"),
        },
    }
    write_json(args.quality_json, quality_payload)

    print(f"Wrote {len(out_df)} rows to {args.out_csv}")
    print(f"Wrote quality report to {args.quality_json}")


if __name__ == "__main__":
    main()
