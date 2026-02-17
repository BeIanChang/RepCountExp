from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import pandas as pd


SPLITS = ("train", "valid", "test")


CANONICAL_ACTION_MAP = {
    "frontraise": "front_raise",
    "front_raise": "front_raise",
    "pullups": "pull_up",
    "pull_up": "pull_up",
    "pushups": "push_up",
    "push_up": "push_up",
    "benchpressing": "bench_pressing",
    "bench_pressing": "bench_pressing",
    "jumpjacks": "jump_jack",
    "jump_jack": "jump_jack",
    "squant": "squat",
    "squat": "squat",
    "pommelhorse": "pommel",
    "pommel": "pommel",
    "situp": "situp",
    "battle_rope": "battle_rope",
    "others": "others",
    "soccer_jungle": "soccer_jungle",
}


def normalize_action(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return CANONICAL_ACTION_MAP.get(key, key)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_annotations(data_root: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for split in SPLITS:
        csv_path = data_root / "annotation" / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {csv_path}")
        df = pd.read_csv(csv_path)
        df["split"] = split
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def l_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("L") and c[1:].isdigit()]
    cols.sort(key=lambda c: int(c[1:]))
    return cols


def write_json(path: Path, payload: Dict) -> None:
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, float) and math.isnan(obj):
            return None
        return obj

    ensure_dir(path.parent)
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")
