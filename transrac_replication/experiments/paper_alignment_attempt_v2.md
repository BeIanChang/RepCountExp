# Paper Alignment Attempt V2

Columns: MAE_norm_p1, OBO, and deltas vs paper table values.

| Run | Family | MAE_norm_p1 | OBO | dMAE | dOBO |
|---|---|---:|---:|---:|---:|
| repnet_external_multi_stride_full | RepNet | 0.4885 | 0.3289 | -0.5065 | +0.3155 |
| repnet_external_paper64 | RepNet | 0.7994 | 0.2500 | -0.1956 | +0.2366 |
| repnet_like_cached16k | RepNet | 1.3461 | 0.0464 | +0.3511 | +0.0330 |
| transrac_official_ckpt | TransRAC | 0.5826 | 0.2829 | +0.1395 | -0.0084 |
| transrac_cached16k | TransRAC | 0.9614 | 0.0795 | +0.5183 | -0.2118 |
| zhang_external_resnext101 | Zhang | 0.8705 | 0.3355 | -0.0081 | +0.1801 |
| zhang_like_cached16k | Zhang | 1.3282 | 0.0530 | +0.4496 | -0.1024 |
