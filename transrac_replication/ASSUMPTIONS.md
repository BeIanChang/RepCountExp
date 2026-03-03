# Replication Assumptions

This document logs implementation choices that may differ from unclear paper details.

## Metrics

- OBO is implemented as `%(|pred - gt| <= 1)`.
- MAE in internal scripts is currently standard absolute count error mean unless a paper-specific normalized MAE adapter is explicitly enabled during final report generation.

## Sampling

- Inference/training default frame count is 64 (paper setting).
- Uniform temporal sampling is used; short clips are padded by index replication.

## Multi-scale Sequences

- `V1`: kernel 1, stride 1
- `V4`: kernel 4, stride 2
- `V8`: kernel 8, stride 4
- All scales are aligned to temporal length 64 via tail padding.

## Density Ground Truth

- Cycle pair `[start, end]` maps to Gaussian parameters:
  - `mu = (start + end) / 2`
  - `sigma = (end - start) / 6`
- Per-bin density value is computed from Gaussian CDF difference over frame bin boundaries.

## Encoder Freeze

- Backbone weights are frozen by default in replication experiments.
