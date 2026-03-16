# Expanded Comparison on RepCount-A Test (152)

## Ours Reproduced Runs

| Method | MAE_norm_p1 | OBO | Notes |
|---|---:|---:|---|
| FSM_baseline | 0.4453 | 0.4934 | pose-signal method |
| RepNet_external_multi_stride_full | 0.4885 | 0.3289 | external weights + multi-stride search |
| ViewpointFFT_similarity | 0.5022 | 0.1645 | skeleton cosine similarity + sliding FFT integration |
| TransRAC_official_ckpt | 0.5826 | 0.2829 | official checkpoint inference |
| Phase_native_softpenalty | 0.6016 | 0.2500 | added var/flip/pause soft penalties |
| Phase_native_peak_online_hybrid | 0.6016 | 0.2500 | confidence gate: mean_confidence<0.0000, fallback=0.000 |
| RepNet_external_paper64 | 0.7994 | 0.2500 | single 64-frame protocol |
| Zhang_external_resnext101 | 0.8705 | 0.3355 | external checkpoint |
| Baseline_peak_online | 1.2517 | 0.2632 | causal trough-peak-trough with low lookahead |

## Ours Reproduced Cached-Embedding Proxies (n=151)

| Method | MAE_norm_p1 | OBO | Notes |
|---|---:|---:|---|
| X3D_cached16k_proxy | 0.8663 | 0.0927 | method-inspired proxy on cached embeddings |
| Huang_cached16k_proxy | 1.0593 | 0.0993 | action-seg inspired proxy on cached embeddings |
| VideoSwinT_cached16k_proxy | 1.0738 | 0.0662 | method-inspired proxy on cached embeddings |
| TANet_cached16k_proxy | 1.3151 | 0.0397 | method-inspired proxy on cached embeddings |

## Paper-Reported Methods (TransRAC CVPR22 Table-2)

| Method | MAE_norm_p1 | OBO |
|---|---:|---:|
| TransRAC (paper) | 0.4431 | 0.2913 |
| Huang et al. | 0.5267 | 0.1589 |
| Video SwinT | 0.5756 | 0.1324 |
| TANet | 0.6624 | 0.0993 |
| Zhang et al. | 0.8786 | 0.1554 |
| X3D | 0.9105 | 0.1059 |
| RepNet | 0.9950 | 0.0134 |
