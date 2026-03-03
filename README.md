# RepCountExp

This repo tracks RepCount Part-A experiments across official/external deep models and in-house FSM/phase methods.

## 1) Main benchmark (Part-A test, n=152)

Metric protocol used in this table:
- `MAE_norm_p1 = mean(|pred - gt| / (gt + 0.1))`
- `OBO = mean(|pred - gt| <= 1)`

| Method | N | MAE_norm_p1 | OBO |
|---|---:|---:|---:|
| Phase_vote | 152 | 0.4356 | 0.2697 |
| FSM_baseline | 152 | 0.4453 | 0.4934 |
| RepNet_external (multi-stride full) | 152 | 0.4885 | 0.3289 |
| TransRAC_official_ckpt | 152 | 0.5826 | 0.2829 |
| Phase_native_online | 152 | 0.6686 | 0.1579 |
| Zhang_external_resnext101 | 152 | 0.8705 | 0.3355 |

## 2) Paper alignment snapshot (TransRAC Table-2 style)

| Family | Paper (MAE/OBO) | Closest run here | Current (MAE/OBO) |
|---|---|---|---|
| RepNet | 0.9950 / 0.0134 | RepNet external `paper64` | 0.7994 / 0.2500 |
| Zhang et al. | 0.8786 / 0.1554 | Zhang external `resnext101` | 0.8705 / 0.3355 |
| Ours (TransRAC) | 0.4431 / 0.2913 | official ckpt inference | 0.5826 / 0.2829 |

## 3) RepNet protocol sensitivity (same checkpoint)

| Protocol | N | MAE_norm_p1 | OBO |
|---|---:|---:|---:|
| multi-stride full | 152 | 0.4885 | 0.3289 |
| paper64 (single 64-frame clip, GPU) | 152 | 0.7994 | 0.2500 |

## 4) Historical self-trained replication runs (cached embeddings, 16k)

These are in-house replication runs (not official released training pipeline).

| Model | N | MAE_norm_p1 | OBO |
|---|---:|---:|---:|
| transrac_cached16k | 151 | 0.9614 | 0.0795 |
| repnet_like_cached16k | 151 | 1.3461 | 0.0464 |
| zhang_like_cached16k | 151 | 1.3282 | 0.0530 |

## 5) Result files

- Combined benchmark: `transrac_replication/experiments/final_benchmark_combined.csv`
- Simplified benchmark: `transrac_replication/experiments/final_benchmark_simplified_partA_test152.md`
- Paper alignment comparison: `transrac_replication/experiments/paper_alignment_attempt_v2.md`
- Full FSM/phase metrics (test-152): `outputs/04_results/metrics_table_partA_test_all.csv`
