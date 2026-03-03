## baseline_fsm
- `overall` n=40 | MAE=5.825 RMSE=11.882 OBOA=0.525 | P/R/F1@K=10: 0.025/0.015/0.019
- `test` n=10 | MAE=4.800 RMSE=9.022 OBOA=0.700 | P/R/F1@K=10: 0.000/0.000/0.000
- `train` n=20 | MAE=7.950 RMSE=15.361 OBOA=0.450 | P/R/F1@K=10: 0.066/0.029/0.040
- `valid` n=10 | MAE=2.600 RMSE=3.376 OBOA=0.500 | P/R/F1@K=10: 0.000/0.000/0.000
- failure cases: `outputs\04_results\failure_cases\baseline_fsm_top_failures.csv`

## phase_native_online_phase_crossing
- `overall` n=40 | MAE=6.475 RMSE=7.933 OBOA=0.100 | P/R/F1@K=10: 0.332/0.180/0.233
- `test` n=10 | MAE=6.700 RMSE=8.313 OBOA=0.200 | P/R/F1@K=10: 0.250/0.122/0.164
- `train` n=20 | MAE=5.750 RMSE=7.277 OBOA=0.100 | P/R/F1@K=10: 0.364/0.213/0.269
- `valid` n=10 | MAE=7.700 RMSE=8.758 OBOA=0.000 | P/R/F1@K=10: 0.333/0.167/0.222
- failure cases: `outputs\04_results\failure_cases\phase_native_online_phase_crossing_top_failures.csv`

## proposed_phase_vote
- `overall` n=40 | MAE=3.425 RMSE=4.937 OBOA=0.325 | P/R/F1@K=10: 0.249/0.185/0.213
- `test` n=10 | MAE=3.400 RMSE=5.550 OBOA=0.300 | P/R/F1@K=10: 0.052/0.038/0.044
- `train` n=20 | MAE=3.750 RMSE=5.045 OBOA=0.350 | P/R/F1@K=10: 0.465/0.339/0.392
- `valid` n=10 | MAE=2.800 RMSE=3.975 OBOA=0.300 | P/R/F1@K=10: 0.000/0.000/0.000
- failure cases: `outputs\04_results\failure_cases\proposed_phase_vote_top_failures.csv`
