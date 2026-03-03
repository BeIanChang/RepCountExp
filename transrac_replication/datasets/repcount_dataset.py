from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from transrac_replication.datasets.density import build_density_from_original_cycles
from transrac_replication.datasets.multiscale import ScaleSpec, build_multiscale_sequences
from transrac_replication.datasets.sampling import uniform_sample_indices


def parse_periods_json(s: str) -> List[Tuple[float, float]]:
    if not isinstance(s, str) or not s:
        return []
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        return []
    out: List[Tuple[float, float]] = []
    for p in data:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            a, b = float(p[0]), float(p[1])
            if b > a:
                out.append((a, b))
    return out


def load_video_frames(video_path: str, frame_size: int = 224) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_LINEAR)
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))  # [C,H,W]
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return np.stack(frames, axis=0)


class RepCountTransRACDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        split: str,
        num_frames: int = 64,
        frame_size: int = 224,
    ):
        super().__init__()
        df = pd.read_csv(manifest_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows for split={split} in {manifest_csv}")

        self.num_frames = num_frames
        self.frame_size = frame_size
        self.scales: Dict[str, ScaleSpec] = {
            "v1": ScaleSpec(window=1, stride=1),
            "v4": ScaleSpec(window=4, stride=2),
            "v8": ScaleSpec(window=8, stride=4),
        }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        video_path = str(row["video_path"])
        frames = load_video_frames(video_path, frame_size=self.frame_size)  # [T,C,H,W]

        total_frames = frames.shape[0]
        sample_idx = uniform_sample_indices(total_frames=total_frames, num_samples=self.num_frames)
        sampled = frames[sample_idx]

        multi = build_multiscale_sequences(
            sampled,
            scales=self.scales,
            target_length=self.num_frames,
        )

        periods = parse_periods_json(str(row.get("periods_json", "")))
        density = build_density_from_original_cycles(
            periods,
            total_frames=int(total_frames),
            num_bins=self.num_frames,
        )

        out = {
            "v1": torch.from_numpy(multi["v1"]),
            "v4": torch.from_numpy(multi["v4"]),
            "v8": torch.from_numpy(multi["v8"]),
            "density_gt": torch.from_numpy(density),
            "gt_count": torch.tensor(float(len(periods)), dtype=torch.float32),
        }
        return out
