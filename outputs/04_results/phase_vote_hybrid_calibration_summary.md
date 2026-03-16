# Confidence Calibration: proposed_phase_vote_partA_test_all_softpenalty.csv

Selected confidence feature: `conf_r`
Selected threshold: `0.836298`

## MAE_norm_p1 / OBO

| Split | n | Fallback% | Phase | RepNet | Hybrid | Phase OBO | RepNet OBO | Hybrid OBO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| calib | 84 | 0.750 | 0.4100 | 0.7105 | 0.2908 | 0.2619 | 0.2738 | 0.2619 |
| holdout | 68 | 0.706 | 0.4673 | 0.2139 | 0.2480 | 0.2794 | 0.3971 | 0.3971 |
| full | 152 | 0.730 | 0.4356 | 0.4883 | 0.2716 | 0.2697 | 0.3289 | 0.3224 |

## Confidence Correlation (Spearman with abs error)

- `mean_confidence`: 0.1249
- `conf_phase`: -0.1195
- `conf_r`: -0.1830
- `conf_var`: 0.2935
- `conf_flip`: 0.0505
- `conf_pause`: 0.4825
