# Alignment Results (Current)

This note records the latest aligned experiments for RepCount-A and our in-house methods.

## 1) Official TransRAC checkpoint inference (Part-A test, 152 videos)

- Checkpoint: `F:\Projects\TransRAC_official\pretrained\transrac_ckpt_pytorch_171.pt`
- Backbone pretrained file: `F:\Projects\TransRAC_official\pretrained\swin_tiny_patch244_window877_kinetics400_1k.pth`
- Data path: `F:\Projects\TransRAC_official\repcountA_npz64\test`

Observed result:

- MAE (normalized-p1 style): `0.5826`
- OBO: `0.2829`

## 2) External RepNet checkpoint inference (Part-A test, 152 videos)

- Summary file: `transrac_replication/experiments/repnet_external_test_summary.json`
- Prediction file: `transrac_replication/experiments/repnet_external_test_predictions.csv`

Observed result:

- MAE (raw): `2.4429`
- MAE (normalized-p1): `0.4885`
- OBO: `0.3289`

## 3) External Zhang checkpoint inference (ResNeXt-101, Part-A test, 152 videos)

- Checkpoint: `F:\Projects\DeepTemporalRepCounting_ext\resnext101.pth`
- GPU env: `F:\Projects\venvs\fitcoach_cuda`
- Summary file: `transrac_replication/experiments/zhang_external_test_summary.json`
- Prediction file: `transrac_replication/experiments/zhang_external_test_predictions.csv`

Observed result:

- MAE (raw): `5.3421`
- MAE (normalized-p1): `0.8705`
- OBO: `0.3355`

## 4) FSM / Phase-based methods (Part-A full test, 152 videos)

Subset used: `outputs/00_index/subset_partA_test_all.csv`.

- Pose report: `outputs/02_pose/pose_run_report_partA_test_all.csv`
- Signal report: `outputs/03_signals/signal_run_report_partA_test_all.csv`
- Metrics table: `outputs/04_results/metrics_table_partA_test_all.csv`

Observed result:

- `baseline_fsm`: MAE `7.0592`, MAE_norm_p1 `0.4453`, OBOA `0.4934`, Event-F1 `0.3339`
- `phase_native_online_phase_crossing`: MAE `6.6382`, MAE_norm_p1 `0.6686`, OBOA `0.1579`, Event-F1 `0.4291`
- `proposed_phase_vote`: MAE `6.4868`, MAE_norm_p1 `0.4356`, OBOA `0.2697`, Event-F1 `0.4046`

Reference 3-action subset results (53 videos) are still available at `outputs/04_results/metrics_table_partA_test_3acts.csv`.

## 5) Combined table

- CSV: `transrac_replication/experiments/final_benchmark_combined.csv`
- Markdown: `transrac_replication/experiments/final_benchmark_combined.md`

Metric note: `MAE_norm_p1` is computed as `mean(abs(pred-gt)/(gt+0.1))`, aligned with official TransRAC testing script style.

## 6) Caveats

- The official TransRAC repo does not provide a direct, runnable official RepNet/Zhang baseline path for RepCount-A, so external checkpoints/wrappers were used.
- TransRAC is reported on full test-152, while FSM/phase currently cover the 3-action subset (53 videos).
