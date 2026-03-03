# TransRAC Replication Workspace

This workspace implements a faithful, executable replication pipeline for TransRAC.

Implemented components:

- 64-frame sampler (uniform, causal-compatible)
- Multi-scale subsequence builder (`V1`, `V4`, `V8`)
- Density map ground-truth generator from cycle boundaries
- Initial model skeleton (frozen encoder interface + temporal correlation + density head)
- Optional frozen Video Swin-T encoder integration via torchvision (`--use-torchvision-backbone`)
- Feature caching pipeline for faster training iterations

## Folder Layout

```text
transrac_replication/
├── ASSUMPTIONS.md
├── configs/
├── datasets/
├── models/
├── training/
├── evaluation/
├── utils/
├── scripts/
├── experiments/
└── tests/
```

## Smoke Tests

Run from repository root:

```bash
python -m transrac_replication.tests.test_data_pipeline
python -m transrac_replication.tests.test_density
python -m transrac_replication.tests.test_model
python -m transrac_replication.scripts.train_transrac --manifest-csv transrac_replication/experiments/repcount_manifest.csv --batch-size 1 --max-steps 1 --frame-size 64 --device cpu --use-torchvision-backbone
python -m transrac_replication.scripts.cache_embeddings --manifest-csv transrac_replication/experiments/repcount_manifest.csv --out-dir transrac_replication/experiments/cache_embeddings_smoke --frame-size 64 --splits train valid --max-samples 1
python -m transrac_replication.scripts.train_transrac --cache-dir transrac_replication/experiments/cache_embeddings_smoke --batch-size 1 --max-steps 1 --device cpu
python -m transrac_replication.scripts.train_baseline --model repnet --cache-dir transrac_replication/experiments/cache_embeddings_smoke --batch-size 1 --max-steps 5 --device cpu --out-dir transrac_replication/experiments/baselines_smoke
python -m transrac_replication.scripts.train_baseline --model zhang --cache-dir transrac_replication/experiments/cache_embeddings_smoke --batch-size 1 --max-steps 5 --device cpu --out-dir transrac_replication/experiments/baselines_smoke
python -m transrac_replication.scripts.summarize_runs --summary-files transrac_replication/experiments/transrac_50/summary.json transrac_replication/experiments/baselines_50/repnet/summary.json transrac_replication/experiments/baselines_50/zhang/summary.json --out-csv transrac_replication/experiments/benchmark_table_50.csv
```

## Next Milestones

1. Integrate real RepCount metadata loader.
2. Build training loop with frozen Swin-T backbone features.
3. Reproduce TransRAC core metrics on RepCount Part-A.
4. Add baseline replication scripts (RepNet, Zhang et al.).

## Baselines (Current)

- `repnet` baseline: `transrac_replication.models.baselines.RepNetLikeBaseline`
- `zhang` baseline: `transrac_replication.models.baselines.ZhangLikeBaseline`

Both baselines currently train from cached embeddings and report MAE/OBO on validation split.

## Latest Milestone Results

- Expanded cached run table: `transrac_replication/experiments/benchmark_table_200_test.csv`
- Setup: cached embeddings (`train=200`, `test=151`), `max_steps=2000`
- Reported metrics include:
  - `mae` (raw absolute count error)
  - `mae_normalized` (paper-style `|pred-gt|/gt` mean)
  - `obo`

- Full-cache run table: `transrac_replication/experiments/benchmark_table_full_test.csv`
- Setup: cached embeddings (`train=757`, `test=151`), `max_steps=2000`
