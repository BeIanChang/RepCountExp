from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Zhang et al. external checkpoint on RepCount-A split.")
    parser.add_argument("--repo-root", type=Path, default=Path("F:/Projects/DeepTemporalRepCounting_ext"))
    parser.add_argument("--weights", type=Path, default=Path("F:/Projects/DeepTemporalRepCounting_ext/resnext101.pth"))
    parser.add_argument("--data-root", type=Path, default=Path("F:/Projects/FItCoach/data/LLSP"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--val-dataset-mode", type=str, default="quva", choices=["quva", "yt_seg", "ucf_aug"])
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0, help="Start row index in split annotations (0-based, inclusive).")
    parser.add_argument("--end-index", type=int, default=0, help="End row index in split annotations (0-based, exclusive). 0 means until end.")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("transrac_replication/experiments/zhang_external_test_predictions.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("transrac_replication/experiments/zhang_external_test_summary.json"),
    )
    return parser.parse_args()


def scale_longerside_and_pad(img_rgb: np.ndarray, size: int) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    if w >= h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)
    resized = cv2.resize(img_rgb, (ow, oh), interpolation=cv2.INTER_LINEAR)

    pad_h = max(size - oh, 0)
    pad_w = max(size - ow, 0)
    top = int(pad_h / 2)
    bottom = int(pad_h - top)
    left = int(pad_w / 2)
    right = int(pad_w - left)
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def preprocess_video(video_path: Path, sample_size: int, norm_value: float, mean: list[float], std: list[float]) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    frames: list[torch.Tensor] = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = scale_longerside_and_pad(rgb, sample_size)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float() / float(norm_value)
        for c in range(3):
            x[c] = (x[c] - mean[c]) / std[c]
        frames.append(x)
    cap.release()

    if not frames:
        return torch.zeros((3, 0, sample_size, sample_size), dtype=torch.float32)
    return torch.stack(frames, dim=1)


def build_inputs(sample_inputs: torch.Tensor, state: tuple[int, int, int], sample_len: int, opt, device: torch.device) -> torch.Tensor:
    inputs = torch.zeros((3, int(opt.basic_duration), int(opt.sample_size), int(opt.sample_size)), dtype=torch.float32, device=device)
    mean_t = torch.tensor(opt.mean, dtype=torch.float32, device=device).view(3, 1, 1)
    padding = mean_t.repeat(1, int(opt.sample_size), int(opt.sample_size))

    sduration = int(opt.basic_duration // 2)
    lp, mp, rp = state

    pos = int(lp - (mp - lp + 1) * opt.l_context_ratio)
    pos2 = int(mp)
    steps = (pos2 - pos + 1) * 1.0 / max(sduration - 1, 1)
    for j in range(sduration):
        p = int(max(pos, min(pos2, round(pos - 0.5 + j * steps))))
        if p < 0 or p >= sample_len:
            inputs[:, j, :, :] = padding
        else:
            inputs[:, j, :, :] = sample_inputs[:, p, :, :].to(device)

    pos = int(mp + 1)
    pos2 = int(rp + (rp - pos + 1) * (opt.r_context_ratio - 1))
    steps = (pos2 - pos + 1) * 1.0 / max(sduration - 1, 1)
    for j in range(sduration):
        p = int(max(pos, min(pos2, round(pos - 0.5 + j * steps))))
        idx = sduration + j
        if p < 0 or p >= sample_len:
            inputs[:, idx, :, :] = padding
        else:
            inputs[:, idx, :, :] = sample_inputs[:, p, :, :].to(device)

    return inputs


def count_video(sample_inputs: torch.Tensor, model: torch.nn.Module, opt, dataset_mode: str, device: torch.device) -> float:
    sample_len = int(sample_inputs.shape[1])
    if sample_len < 8:
        return 0.0

    val_opt = {
        "merge_level": 5,
        "merge_w": 0.5,
        "min_scale": 0.03,
        "max_scale": 0.35,
        "init_scale_num": 30,
        "abandon_second_box": False,
    }
    if dataset_mode == "ucf_aug":
        val_opt["min_cycles"] = 2
    else:
        val_opt["min_cycles"] = 4
    if dataset_mode == "yt_seg":
        val_opt["merge_w"] = 0.1

    level_pow = int(2 ** val_opt["merge_level"])
    mp = np.zeros((val_opt["merge_level"], level_pow), dtype=np.int64)
    lp_l = np.zeros((val_opt["merge_level"], level_pow), dtype=np.int64)
    lp_r = np.zeros((val_opt["merge_level"], level_pow), dtype=np.int64)
    rp_l = np.zeros((val_opt["merge_level"], level_pow), dtype=np.int64)
    rp_r = np.zeros((val_opt["merge_level"], level_pow), dtype=np.int64)

    max_mp = 4
    max_score = -1e6

    with torch.no_grad():
        for k in range(val_opt["init_scale_num"]):
            p = float(k) / float(max(val_opt["init_scale_num"] - 1, 1))
            powers_level = (val_opt["max_scale"] / val_opt["min_scale"]) ** p
            mp_k = sample_len * val_opt["min_scale"] * powers_level
            mid_pt = sample_len / 2.0
            inputs = build_inputs(sample_inputs, (int(mid_pt - mp_k), int(mid_pt), int(mid_pt + mp_k + 1)), sample_len, opt, device)
            pred_cls, pred_box, _, _ = model(inputs.unsqueeze(0))
            pred_box = torch.clamp(pred_box, min=-0.5, max=0.5)

            cls_prob = torch.softmax(pred_cls[0], dim=0)
            for a in range(3, 4):
                box_exp = math.exp(float(pred_box[0][a].item()))
                pred_seg = box_exp * float(opt.anchors[a])
                score = float(cls_prob[1][a].item())
                cand_mp = mp_k * pred_seg
                if score > max_score and cand_mp >= 4 and cand_mp < sample_len / float(val_opt["min_cycles"]):
                    max_score = score
                    max_mp = int(round(cand_mp))

        for _ in range(4):
            mid_pt = sample_len / 2.0
            inputs = build_inputs(sample_inputs, (int(mid_pt - max_mp), int(mid_pt), int(mid_pt + max_mp + 1)), sample_len, opt, device)
            pred_cls, pred_box, _, _ = model(inputs.unsqueeze(0))
            pred_box = torch.clamp(pred_box, min=-0.5, max=0.5)
            cls_prob = torch.softmax(pred_cls[0], dim=0)

            best_local_score = -1e6
            tmp = max_mp
            for a in range(3, 4):
                box_exp = math.exp(float(pred_box[0][a].item()))
                pred_seg = box_exp * float(opt.anchors[a])
                score = float(cls_prob[1][a].item())
                mp_k = tmp * pred_seg
                if score > best_local_score and mp_k >= 4 and mp_k < sample_len / float(val_opt["min_cycles"]):
                    best_local_score = score
                    max_mp = int(round(float(max_mp * (1 - val_opt["merge_w"]) + mp_k * val_opt["merge_w"])))

        for l2 in range(level_pow):
            mp[0, l2] = int(float(sample_len) / float(level_pow + 1) * (l2 + 0.5))
            lp_l[0, l2] = mp[0, l2] - max_mp
            rp_l[0, l2] = mp[0, l2] + max_mp + 1
            lp_r[0, l2] = lp_l[0, l2]
            rp_r[0, l2] = rp_l[0, l2]

        save_lp = save_mp = save_rp = 0
        load_lp = load_mp = load_rp = 0
        for l1 in range(1, val_opt["merge_level"]):
            steps = int(2 ** (val_opt["merge_level"] - l1 - 1))
            pos = -steps
            for l2 in range(int(2 ** l1)):
                pos += 2 * steps
                if l1 == 1:
                    iters = 4
                elif l1 == 2:
                    iters = 2
                else:
                    iters = 1

                for it in range(iters):
                    if it == 0:
                        load_mp = int(mp[l1 - 1, pos])
                        load_lp = int(round(float(lp_l[l1 - 1, pos] + lp_r[l1 - 1, pos]) / 2.0))
                        load_rp = int(round(float(rp_l[l1 - 1, pos] + rp_r[l1 - 1, pos]) / 2.0))
                    else:
                        load_mp = int(save_mp)
                        load_lp = int(round(float(save_lp) * val_opt["merge_w"] + float(load_lp) * (1.0 - val_opt["merge_w"])))
                        load_rp = int(round(float(save_rp) * val_opt["merge_w"] + float(load_rp) * (1.0 - val_opt["merge_w"])))

                    inputs = build_inputs(sample_inputs, (load_lp, load_mp, load_rp), sample_len, opt, device)
                    pred_cls_1, pred_box_1, pred_cls_2, pred_box_2 = model(inputs.unsqueeze(0))
                    pred_box_1 = torch.clamp(pred_box_1, min=-0.5, max=0.5)
                    pred_box_2 = torch.clamp(pred_box_2, min=-0.5, max=0.5)
                    cls_prob_1 = torch.softmax(pred_cls_1[0], dim=0)
                    cls_prob_2 = torch.softmax(pred_cls_2[0], dim=0)

                    best_s1, action_1 = -1e6, 1.0
                    best_s2, action_2 = -1e6, 1.0
                    for a in range(int(opt.n_classes)):
                        act1 = math.exp(float(pred_box_1[0][a].item())) * float(opt.anchors[a])
                        sc1 = float(cls_prob_1[1][a].item())
                        if sc1 > best_s1:
                            best_s1, action_1 = sc1, act1

                        act2 = math.exp(float(pred_box_2[0][a].item())) * float(opt.anchors[a])
                        sc2 = float(cls_prob_2[1][a].item())
                        if sc2 > best_s2:
                            best_s2, action_2 = sc2, act2

                    if val_opt["abandon_second_box"]:
                        action_2 = action_1

                    seg_len_1 = (load_mp - load_lp + 1) * action_1
                    seg_len_2 = (load_rp - load_mp) * action_2
                    seg_len_1 = min(max(4, seg_len_1), sample_len / float(val_opt["min_cycles"]))
                    seg_len_2 = min(max(4, seg_len_2), sample_len / float(val_opt["min_cycles"]))
                    new_mp = int(load_mp)
                    new_lp = int(new_mp - seg_len_1 + 1)
                    new_rp = int(new_mp + seg_len_2)

                    fail_flag = (new_mp - new_lp + 1) < 4 or (new_rp - new_mp) < 4
                    if fail_flag:
                        save_lp, save_mp, save_rp = load_lp, load_mp, load_rp
                    else:
                        save_lp, save_mp, save_rp = new_lp, new_mp, new_rp

                l_segments = float(save_lp) * val_opt["merge_w"] + float(load_lp) * (1.0 - val_opt["merge_w"])
                r_segments = float(save_rp) * val_opt["merge_w"] + float(load_rp) * (1.0 - val_opt["merge_w"])
                for s in range(-steps, 0):
                    idx = pos + s
                    mp[l1, idx] = mp[l1 - 1, idx]
                    lp_r[l1, idx] = int(round(mp[l1 - 1, idx] + (l_segments - mp[l1 - 1, pos])))
                    rp_r[l1, idx] = int(round(mp[l1 - 1, idx] + (r_segments - mp[l1 - 1, pos])))
                    if l1 <= 2 or l1 == val_opt["merge_level"] - 1 or l2 == 0:
                        lp_l[l1, idx] = lp_r[l1, idx]
                        rp_l[l1, idx] = rp_r[l1, idx]
                    else:
                        lp_l[l1, idx] = lp_l[l1 - 1, idx]
                        rp_l[l1, idx] = rp_l[l1 - 1, idx]

                for s in range(0, steps):
                    idx = pos + s
                    mp[l1, idx] = mp[l1 - 1, idx]
                    lp_l[l1, idx] = int(round(mp[l1 - 1, idx] + (l_segments - mp[l1 - 1, pos])))
                    rp_l[l1, idx] = int(round(mp[l1 - 1, idx] + (r_segments - mp[l1 - 1, pos])))
                    if l1 <= 2 or l1 == val_opt["merge_level"] - 1 or l2 == int(2 ** l1) - 1:
                        lp_r[l1, idx] = lp_l[l1, idx]
                        rp_r[l1, idx] = rp_l[l1, idx]
                    else:
                        lp_r[l1, idx] = lp_r[l1 - 1, idx]
                        rp_r[l1, idx] = rp_r[l1 - 1, idx]

    left_vals: list[float] = []
    right_vals: list[float] = []
    last = val_opt["merge_level"] - 1
    for k in range(level_pow):
        lp_avg = int(round(float(lp_l[last, k] + lp_r[last, k]) / 2.0))
        rp_avg = int(round(float(rp_l[last, k] + rp_r[last, k]) / 2.0))
        pos1 = int(lp_avg - (mp[last, k] - lp_avg + 1) * opt.l_context_ratio)
        pos2 = int(rp_avg + (rp_avg - mp[last, k] + 0) * (opt.r_context_ratio - 1))
        if pos1 >= 0 and pos2 < sample_len:
            left_vals.append(1.0 / float(max(mp[last, k] - lp_avg + 1, 1)))
            right_vals.append(1.0 / float(max(rp_avg - mp[last, k], 1)))

    if not left_vals or not right_vals:
        count = float(sample_len) / float(max_mp + 1)
    else:
        count = float(sample_len) * float((sum(left_vals) + sum(right_vals)) * 0.5) / float(len(left_vals))

    return float(round(count))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    sys.path.insert(0, str(args.repo_root.resolve()))
    from models.model import generate_model
    from utils.mean import get_mean, get_std

    checkpoint = torch.load(args.weights, map_location="cpu")
    opt = checkpoint["opt"]
    opt.learning_policy = "2stream"
    opt.no_cuda = device.type != "cuda"
    opt.mean = get_mean(getattr(opt, "norm_value", 255), dataset=getattr(opt, "mean_std_dataset", "quva"))
    opt.std = get_std(getattr(opt, "norm_value", 255), dataset=getattr(opt, "mean_std_dataset", "quva"))

    model, _ = generate_model(opt)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to(device).eval()

    ann_csv = args.data_root / "annotation" / f"{args.split}.csv"
    video_dir = args.data_root / "video" / args.split
    ann = pd.read_csv(ann_csv)
    if args.start_index > 0 or args.end_index > 0:
        end_idx = None if args.end_index <= 0 else int(args.end_index)
        ann = ann.iloc[int(args.start_index) : end_idx].copy()
    if args.max_videos > 0:
        ann = ann.head(args.max_videos).copy()

    rows: list[dict[str, float | str]] = []
    failed = 0
    for i, row in ann.iterrows():
        name = str(row["name"])
        gt = float(row["count"]) if pd.notna(row["count"]) else float("nan")
        video_path = video_dir / name
        try:
            sample_inputs = preprocess_video(video_path, int(opt.sample_size), float(opt.norm_value), opt.mean, opt.std)
            pred = count_video(sample_inputs, model, opt, args.val_dataset_mode, device)
            err = ""
        except Exception as exc:  # pragma: no cover - defensive runtime path
            pred = 0.0
            err = str(exc)
            failed += 1

        rows.append({"video": name, "gt_count": gt, "pred_count": pred, "error": err})
        print(f"processed {len(rows)}/{len(ann)} (failed={failed})")

    out = pd.DataFrame(rows)
    out["abs_err"] = (out["pred_count"] - out["gt_count"]).abs()
    out["is_obo"] = out["abs_err"] <= 1.0
    out["norm_err"] = out["abs_err"] / out["gt_count"].clip(lower=1e-6)
    out["norm_err_p1"] = out["abs_err"] / (out["gt_count"] + 1e-1)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    summary = {
        "n_videos": int(len(out)),
        "n_failed": int(failed),
        "mae_raw": float(out["abs_err"].mean()),
        "mae_normalized": float(out["norm_err"].mean()),
        "mae_normalized_p1": float(out["norm_err_p1"].mean()),
        "obo": float(out["is_obo"].mean()),
        "dataset_mode": args.val_dataset_mode,
        "out_csv": str(args.out_csv),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
