from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedEmbeddingDataset(Dataset):
    def __init__(self, cache_dir: str | Path, split: str):
        self.cache_dir = Path(cache_dir)
        self.files: List[Path] = sorted((self.cache_dir / split).glob("*.npz"))
        if not self.files:
            raise ValueError(f"No cached npz files found in {self.cache_dir / split}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        p = self.files[index]
        d = np.load(p)
        return {
            "x_v1": torch.from_numpy(d["x_v1"].astype(np.float32)),
            "x_v4": torch.from_numpy(d["x_v4"].astype(np.float32)),
            "x_v8": torch.from_numpy(d["x_v8"].astype(np.float32)),
            "density_gt": torch.from_numpy(d["density_gt"].astype(np.float32)),
            "gt_count": torch.tensor(float(d["gt_count"]), dtype=torch.float32),
        }
