# Final Benchmark Combined

## PartA test (152 videos)
- TransRAC_official_ckpt: n=152, MAE_raw=NA, MAE_norm_p1=0.5826, OBO=0.2829
- RepNet_external: n=152, MAE_raw=2.4429, MAE_norm_p1=0.4885, OBO=0.3289
- Zhang_external_resnext101: n=152, MAE_raw=5.3421, MAE_norm_p1=0.8705, OBO=0.3355
- FSM_baseline: n=152, MAE_raw=7.0592, MAE_norm_p1=0.4453, OBO=0.4934
- Phase_native_online: n=152, MAE_raw=6.6382, MAE_norm_p1=0.6686, OBO=0.1579
- Phase_vote: n=152, MAE_raw=6.4868, MAE_norm_p1=0.4356, OBO=0.2697

## PartA test 3 actions (53 videos: squat/push_up/pull_up)
- RepNet_external: n=53, MAE_raw=2.3208, MAE_norm_p1=0.3060, OBO=0.3208
- Zhang_external_resnext101: n=53, MAE_raw=4.1509, MAE_norm_p1=0.7180, OBO=0.3396
- FSM_baseline: n=53, MAE_raw=5.2075, MAE_norm_p1=0.2938, OBO=0.6415, EventF1=0.0337
- Phase_native_online: n=53, MAE_raw=6.1321, MAE_norm_p1=0.6657, OBO=0.1321, EventF1=0.2736
- Phase_vote: n=53, MAE_raw=4.0566, MAE_norm_p1=0.3594, OBO=0.3962, EventF1=0.1582
