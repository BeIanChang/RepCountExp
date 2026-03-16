# Confidence Calibration: phase_native_online_partA_test_all_softpenalty.csv

Selected confidence feature: `mean_confidence`
Selected threshold: `0.670717`

## MAE_norm_p1 / OBO

| Split | n | Fallback% | Phase | RepNet | Hybrid | Phase OBO | RepNet OBO | Hybrid OBO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| calib | 84 | 0.298 | 0.7057 | 0.7105 | 0.5263 | 0.2976 | 0.2738 | 0.3333 |
| holdout | 68 | 0.441 | 0.4729 | 0.2139 | 0.3376 | 0.1912 | 0.3971 | 0.2794 |
| full | 152 | 0.362 | 0.6016 | 0.4883 | 0.4419 | 0.2500 | 0.3289 | 0.3092 |

## Confidence Correlation (Spearman with abs error)

- `mean_confidence`: -0.0008
- `conf_phase`: 0.0082
- `conf_r`: 0.0713
- `conf_var`: -0.1146
- `conf_flip`: -0.1329
- `conf_pause`: 0.2550
