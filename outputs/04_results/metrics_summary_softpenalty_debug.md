## baseline_fsm
- `overall` n=40 | MAE=5.825 MAE_norm=0.344 MAE_norm_p1=0.337 RMSE=11.882 OBOA=0.525 | P/R/F1@K=10: 0.025/0.015/0.019
- `test` n=10 | MAE=4.800 MAE_norm=0.314 MAE_norm_p1=0.303 RMSE=9.022 OBOA=0.700 | P/R/F1@K=10: 0.000/0.000/0.000
- `train` n=20 | MAE=7.950 MAE_norm=0.398 MAE_norm_p1=0.392 RMSE=15.361 OBOA=0.450 | P/R/F1@K=10: 0.066/0.029/0.040
- `valid` n=10 | MAE=2.600 MAE_norm=0.266 MAE_norm_p1=0.263 RMSE=3.376 OBOA=0.500 | P/R/F1@K=10: 0.000/0.000/0.000
- failure cases: `outputs\04_results\failure_cases\baseline_fsm_top_failures.csv`

## phase_native_online_phase_crossing
- `overall` n=40 | MAE=4.150 MAE_norm=0.454 MAE_norm_p1=0.442 RMSE=5.441 OBOA=0.225 | P/R/F1@K=10: 0.331/0.245/0.282
- `test` n=10 | MAE=4.400 MAE_norm=0.469 MAE_norm_p1=0.455 RMSE=6.148 OBOA=0.300 | P/R/F1@K=10: 0.264/0.176/0.211
- `train` n=20 | MAE=3.700 MAE_norm=0.428 MAE_norm_p1=0.415 RMSE=5.000 OBOA=0.250 | P/R/F1@K=10: 0.350/0.256/0.296
- `valid` n=10 | MAE=4.800 MAE_norm=0.490 MAE_norm_p1=0.484 RMSE=5.532 OBOA=0.100 | P/R/F1@K=10: 0.349/0.294/0.319
- failure cases: `outputs\04_results\failure_cases\phase_native_online_phase_crossing_top_failures.csv`

## proposed_phase_vote
- `overall` n=40 | MAE=3.425 MAE_norm=0.359 MAE_norm_p1=0.350 RMSE=4.937 OBOA=0.325 | P/R/F1@K=10: 0.249/0.185/0.213
- `test` n=10 | MAE=3.400 MAE_norm=0.269 MAE_norm_p1=0.265 RMSE=5.550 OBOA=0.300 | P/R/F1@K=10: 0.052/0.038/0.044
- `train` n=20 | MAE=3.750 MAE_norm=0.457 MAE_norm_p1=0.442 RMSE=5.045 OBOA=0.350 | P/R/F1@K=10: 0.465/0.339/0.392
- `valid` n=10 | MAE=2.800 MAE_norm=0.254 MAE_norm_p1=0.251 RMSE=3.975 OBOA=0.300 | P/R/F1@K=10: 0.000/0.000/0.000
- failure cases: `outputs\04_results\failure_cases\proposed_phase_vote_top_failures.csv`
