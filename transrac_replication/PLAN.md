# TransRAC Replication Execution Plan

## Completed (Build Stage)

- Project scaffold created (`transrac_replication/*`)
- Data pipeline primitives implemented:
  - 64-frame uniform sampling with short-video padding
  - multi-scale subsequence generation (`V1`, `V4`, `V8`) with temporal alignment to 64
- Density map target generation implemented from cycle boundaries
- Core model skeleton implemented:
  - frozen encoder interface
  - per-scale temporal correlation heads
  - multi-scale correlation concat and density predictor
- Optional real frozen encoder path implemented:
  - torchvision `swin3d_t` integration
  - single-step train/eval smoke test passed on CPU
- Feature cache pipeline implemented:
  - cache script saves per-video `x_v1/x_v4/x_v8` + GT density/count
  - cached training path smoke tested
- Baseline training scripts implemented (cached embeddings):
  - RepNet-like baseline
  - Zhang-like context/scale baseline
  - smoke runs completed with MAE/OBO output
- Preliminary comparable run completed on a cached 50/50 subset (train/valid):
  - table generated at `transrac_replication/experiments/benchmark_table_50.csv`
- Next milestone run completed on expanded cached subset (train=200, test=151):
  - table generated at `transrac_replication/experiments/benchmark_table_200_test.csv`
  - metrics include both raw MAE and paper-style normalized MAE
- Full-cache milestone completed (train=757, test=151):
  - table generated at `transrac_replication/experiments/benchmark_table_full_test.csv`
  - TransRAC currently leads normalized MAE in this setting
- Smoke/unit tests passing:
  - `python -m transrac_replication.tests.test_data_pipeline`
  - `python -m transrac_replication.tests.test_density`
  - `python -m transrac_replication.tests.test_model`
- Manifest generated for available RepCount parsed index:
  - `transrac_replication/experiments/repcount_manifest.csv`

## Next Actions (Immediate)

1. Add paper-aligned training schedule (16k steps, LR decay, logging/checkpoints).
2. Add train dataloader that reads videos and returns:
   - sampled frames [64, 3, 224, 224]
   - `V1/V4/V8`
   - density GT [64]
3. Implement evaluation pipeline for MAE/OBO in paper-compatible report format.
4. Upgrade baselines from "replication-like" stubs to closer paper implementations.
5. Launch longer runs on full split cache and collect comparable metrics.

## Validation Gates

- One-batch overfit test reaches very low loss.
- Frozen backbone has zero gradient updates.
- Density sum tracks count on sample set.
- End-to-end run produces MAE/OBO report on validation split.
