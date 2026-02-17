## baseline_fsm
- `overall` n=40 | MAE=5.825 RMSE=11.882 OBOA=0.525 | P/R/F1@K=10: 0.025/0.015/0.019
- `test` n=10 | MAE=4.800 RMSE=9.022 OBOA=0.700 | P/R/F1@K=10: 0.000/0.000/0.000
- `train` n=20 | MAE=7.950 RMSE=15.361 OBOA=0.450 | P/R/F1@K=10: 0.066/0.029/0.040
- `valid` n=10 | MAE=2.600 RMSE=3.376 OBOA=0.500 | P/R/F1@K=10: 0.000/0.000/0.000
- failure cases: `outputs\04_results\failure_cases\baseline_fsm_top_failures.csv`

## phase_native_online
- `overall` n=40 | MAE=7.000 RMSE=9.042 OBOA=0.100 | P/R/F1@K=10: 0.364/0.176/0.237
- `test` n=10 | MAE=6.700 RMSE=8.654 OBOA=0.100 | P/R/F1@K=10: 0.391/0.191/0.256
- `train` n=20 | MAE=6.950 RMSE=9.206 OBOA=0.150 | P/R/F1@K=10: 0.449/0.224/0.299
- `valid` n=10 | MAE=7.400 RMSE=9.088 OBOA=0.000 | P/R/F1@K=10: 0.125/0.056/0.077
- failure cases: `outputs\04_results\failure_cases\phase_native_online_top_failures.csv`

## phase_native_online_phase_crossing
- `overall` n=40 | MAE=4.025 RMSE=5.189 OBOA=0.200 | P/R/F1@K=10: 0.323/0.247/0.280
- `test` n=10 | MAE=4.700 RMSE=6.301 OBOA=0.200 | P/R/F1@K=10: 0.238/0.153/0.186
- `train` n=20 | MAE=3.200 RMSE=4.290 OBOA=0.250 | P/R/F1@K=10: 0.359/0.282/0.316
- `valid` n=10 | MAE=5.000 RMSE=5.586 OBOA=0.100 | P/R/F1@K=10: 0.315/0.270/0.291
- failure cases: `outputs\04_results\failure_cases\phase_native_online_phase_crossing_top_failures.csv`
