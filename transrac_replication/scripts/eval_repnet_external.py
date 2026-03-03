from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate external RepNet checkpoint on RepCount-A split.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("F:/Projects/RepNet_pytorch_ext"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("F:/Projects/RepNet_pytorch_ext/checkpoints/pytorch_weights.pth"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("F:/Projects/FItCoach/data/LLSP"),
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--strides", nargs="+", type=int, default=[1, 2, 3, 4, 8])
    parser.add_argument(
        "--protocol",
        type=str,
        default="multi_stride_full",
        choices=["multi_stride_full", "paper64"],
        help="multi_stride_full: current full-video multi-stride inference; paper64: single 64-frame clip, stride=1.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="If >0, evaluate only first N videos for quick check.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("transrac_replication/experiments/repnet_external_test_predictions.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("transrac_replication/experiments/repnet_external_test_summary.json"),
    )
    return parser.parse_args()


def read_video_frames(video_path: Path, transform: T.Compose) -> Tuple[List[np.ndarray], List[torch.Tensor], float]:
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    raw_frames: List[np.ndarray] = []
    frames: List[torch.Tensor] = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        raw_frames.append(frame)
        tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(tensor)
    cap.release()
    return raw_frames, frames, fps


def sample_fixed_length(frames: List[torch.Tensor], n: int = 64) -> List[torch.Tensor]:
    if not frames:
        return []
    if len(frames) == 1:
        return [frames[0] for _ in range(n)]
    idx = np.linspace(0, len(frames) - 1, n)
    idx = np.clip(np.round(idx).astype(np.int64), 0, len(frames) - 1)
    return [frames[int(i)] for i in idx]


def main() -> None:
    args = parse_args()

    import sys

    sys.path.insert(0, str(args.repo_root.resolve()))
    from repnet.model import RepNet

    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5),
        ]
    )

    model = RepNet()
    state_dict = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval().to(args.device)

    ann_csv = args.data_root / "annotation" / f"{args.split}.csv"
    video_dir = args.data_root / "video" / args.split
    ann = pd.read_csv(ann_csv)
    if args.max_videos > 0:
        ann = ann.head(args.max_videos).copy()

    rows = []
    failed = 0
    for i, row in enumerate(ann.itertuples(index=False), start=1):
        name = str(getattr(row, "name"))
        gt_raw = getattr(row, "count")
        gt = float(gt_raw) if pd.notna(gt_raw) else np.nan
        video_path = video_dir / name
        try:
            _, frames, _ = read_video_frames(video_path, transform)
            if args.protocol == "paper64":
                stride = 1
                clip = sample_fixed_length(frames, n=64)
                if len(clip) < 64:
                    pred = 0.0
                    conf = 0.0
                else:
                    chunk = torch.stack(clip, dim=0).unsqueeze(0).movedim(1, 2).to(args.device)
                    with torch.no_grad():
                        raw_period_length, raw_periodicity, _ = model(chunk)
                    confidence, _, period_count, _ = model.get_counts(
                        raw_period_length[0].cpu(),
                        raw_periodicity[0].cpu(),
                        stride,
                    )
                    pred = float(period_count[-1].item()) if period_count.numel() > 0 else 0.0
                    conf = float(confidence.item())
            else:
                best = None
                for stride in args.strides:
                    stride_frames = frames[::stride]
                    usable = (len(stride_frames) // 64) * 64
                    stride_frames = stride_frames[:usable]
                    if len(stride_frames) < 64:
                        continue
                    chunk = torch.stack(stride_frames, dim=0).unflatten(0, (-1, 64)).movedim(1, 2).to(args.device)

                    raw_period_length, raw_periodicity = [], []
                    with torch.no_grad():
                        for b in range(chunk.shape[0]):
                            batch_period_length, batch_periodicity, _ = model(chunk[b].unsqueeze(0))
                            raw_period_length.append(batch_period_length[0].cpu())
                            raw_periodicity.append(batch_periodicity[0].cpu())
                    raw_period_length = torch.cat(raw_period_length)
                    raw_periodicity = torch.cat(raw_periodicity)
                    confidence, _, period_count, _ = model.get_counts(raw_period_length, raw_periodicity, stride)
                    pred = float(period_count[-1].item()) if period_count.numel() > 0 else 0.0
                    cand = (float(confidence.item()), pred, stride)
                    if best is None or cand[0] > best[0]:
                        best = cand

                if best is None:
                    pred = 0.0
                    stride = -1
                    conf = 0.0
                else:
                    conf, pred, stride = best
            error = ""
        except Exception as exc:  # pragma: no cover - defensive runtime path
            failed += 1
            pred = 0.0
            stride = -1
            conf = 0.0
            error = str(exc)

        rows.append(
            {
                "video": name,
                "gt_count": gt,
                "pred_count": pred,
                "best_stride": stride,
                "best_conf": conf,
                "error": error,
            }
        )
        print(f"processed {i}/{len(ann)} (failed={failed})")

    out = pd.DataFrame(rows)
    out["abs_err"] = (out["pred_count"] - out["gt_count"]).abs()
    out["is_obo"] = out["abs_err"] <= 1.0
    out["pred_count_round"] = out["pred_count"].round()
    out["abs_err_round"] = (out["pred_count_round"] - out["gt_count"]).abs()
    out["is_obo_round"] = out["abs_err_round"] <= 1.0
    out["is_exact_round"] = out["abs_err_round"] <= 0.0
    # Two MAE styles: paper-normalized and official test_loop style with +0.1 stabilizer
    out["norm_err"] = out["abs_err"] / out["gt_count"].clip(lower=1e-6)
    out["norm_err_p1"] = out["abs_err"] / (out["gt_count"] + 1e-1)
    out["norm_err_round"] = out["abs_err_round"] / out["gt_count"].clip(lower=1.0)
    out["norm_err_round_p1"] = out["abs_err_round"] / (out["gt_count"] + 1e-1)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    summary = {
        "n_videos": int(len(out)),
        "n_failed": int(failed),
        "mae_raw": float(out["abs_err"].mean()),
        "mae_normalized": float(out["norm_err"].mean()),
        "mae_normalized_p1": float(out["norm_err_p1"].mean()),
        "obo": float(out["is_obo"].mean()),
        "mae_round": float(out["abs_err_round"].mean()),
        "mae_round_normalized": float(out["norm_err_round"].mean()),
        "mae_round_normalized_p1": float(out["norm_err_round_p1"].mean()),
        "obo_round": float(out["is_obo_round"].mean()),
        "exact_round": float(out["is_exact_round"].mean()),
        "protocol": args.protocol,
        "out_csv": str(args.out_csv),
    }

    import json

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
