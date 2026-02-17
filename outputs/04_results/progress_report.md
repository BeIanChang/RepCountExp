# RepCountA Progress Report

Date: 2026-02-16

## Scope

This report summarizes the current implementation and debug-subset results for the RepCountA pipeline (indexing -> period parsing -> pose extraction -> signal generation -> rep counting -> evaluation).

## What is completed

### 1) Data ingestion and indexing

- Parsed all dataset splits and built a unified index.
- Generated video metadata (`fps`, `n_frames`, `duration_sec`) from actual files.
- Canonicalized noisy action labels.

Outputs:

- `outputs/00_index/master_index.csv`
- `outputs/00_index/data_quality_report.json`

### 2) Period parsing

- Converted `L1..L302` annotations into normalized frame-pair periods.
- Added parsed period fields and parse-status columns.
- Logged mismatch/invalid cases in quality report.

Outputs:

- `outputs/00_index/master_index.parsed.csv`
- `outputs/00_index/data_quality_report.json`

### 3) Debug subset creation

- Built a debug subset for `squat`, `push_up`, `pull_up`.
- Sampling target met exactly: train 20, valid 10, test 10.
- Excluded problematic rows (empty periods/missing count) by default.

Outputs:

- `outputs/00_index/subset_debug.csv`
- `outputs/00_index/subset_debug_report.json`

### 4) MediaPipe pose extraction

- Extracted per-frame pose landmarks on all 40 debug videos.
- Saved per-video pose arrays and run-level quality report.

Outputs:

- `outputs/02_pose/{split}/{video_id}.npz`
- `outputs/02_pose/pose_run_report.csv`

Pose quality snapshot:

- mean valid-frame ratio: **0.9763**
- minimum valid-frame ratio: **0.7767**
- videos with valid ratio < 0.90: **2**

### 5) Signal computation and previews

- Computed per-signal `theta`, `omega`, `phi`, `r` for each debug video.
- Generated preview figures (`theta(t)`, `omega(t)`, phase portrait, `phi/r`).

Outputs:

- `outputs/03_signals/{split}/{video_id}.npz`
- `outputs/03_signals/signal_run_report.csv`
- `outputs/01_preview/{split}/{video_id}_plots.png`

### 6) Counting methods + evaluation

- Implemented an FSM baseline (FitCoach-style, calibration-free).
- Implemented **phase-native online counting** (moving-onset, causal per-frame update, no FSM component).
- Evaluated count metrics and event metrics at `K=10`.

Outputs:

- `outputs/04_results/baseline_fsm.csv`
- `outputs/04_results/phase_native_online.csv`
- `outputs/04_results/metrics_table_streaming_phase.csv`
- `outputs/04_results/metrics_summary_streaming_phase.md`
- `outputs/04_results/failure_cases/baseline_fsm_top_failures.csv`
- `outputs/04_results/failure_cases/phase_native_online_top_failures.csv`

## Counting logic in detail

### Important note on execution mode

- `baseline_fsm` and `phase_native_online` both run as **streaming replay**: per-frame causal updates over precomputed signals.
- This mirrors real-time logic behavior (no future-frame access inside counting decisions).
- Metrics are still computed offline after replay, by comparing outputs against GT counts/periods.

### A) FSM baseline (`scripts/06_baseline_fsm.py`)

Input per video:

- `theta_*` primary signal from `outputs/03_signals/{split}/{video_id}.npz`
- `fps` for time conversion and duration checks

Algorithm:

1. **Progress normalization (no external calibration)**
   - Estimate `theta_low` and `theta_high` from the video itself (5th and 95th percentile of finite `theta`).
   - Compute normalized progress: `p = (theta - theta_low) / (theta_high - theta_low)`.
   - Clip progress to a safe range to reduce outlier spikes.

2. **State machine transitions**
   - States: `BOTTOM -> UP_PHASE -> TOP -> DOWN_PHASE`.
   - Enter `UP_PHASE` when `p` rises above `start_th` and direction is positive.
   - Enter `TOP` when `p >= top_th`.
   - Enter `DOWN_PHASE` when descending from top.
   - Return to `BOTTOM` when `p <= bottom_th`.

3. **Rep counting condition**
   - Count one rep only when a full cycle reaches `DOWN_PHASE -> BOTTOM` and:
     - `reached_top = True`
     - rep duration satisfies `t_min <= duration <= t_max`.

4. **Failure guards**
   - Early-fail reset if motion falls back to bottom too early (`early_fail_th`).
   - Timeout reset when stuck too long (`t_max`).

Output per video:

- `pred_count`
- `pred_periods_json` (frame ranges from rep start to completion)
- diagnostic values (`theta_low_est`, `theta_high_est`, `status`)

### B) Phase-native online method (`scripts/07_phase_native_online.py`)

Input per video:

- per-signal raw angle streams (`theta_raw_*`) from `outputs/03_signals/{split}/{video_id}.npz`
- `fps`
- signal names from action config (3 signals per action, primary = first)

Algorithm:

1. **Online feature update per frame (for each signal)**
   - `theta_f[t] = EMA(theta_raw[t])`
   - `omega[t] = EMA((theta_f[t] - theta_f[t-1]) * fps)`
   - robust online normalization using sliding median/IQR
   - `psi[t] = atan2(omega', theta')` (wrapped phase)
   - `phi[t] = unwrap_online(psi[t])` (for continuity)
   - `r[t] = sqrt(theta'^2 + omega'^2)`

2. **Phase-crossing segmentation (primary signal)**
   - estimate adaptive phase landmark `psi0` from warm-up frames
   - compute relative phase `rel[t] = wrap_to_pi(psi_primary[t] - psi0)`
   - detect crossing with hysteresis:
     - `prev_rel <= -hyst` and `rel >= +hyst`
   - crossing-to-crossing defines candidate window `[start_cross, end_cross]`
   - duration gate: `t_min <= duration <= t_max`

3. **Online multi-signal vote at trigger time (Fix1)**
   - **Primary signal (strict):**
     - `abs(delta_phi_primary - 2*pi) / (2*pi) < eps_phi`
   - **Secondary signals (weaker):**
     - `delta_phi_secondary > pi`
     - `median(r_window_secondary) > r_min_secondary`
   - accept rep if:
     - primary passes AND required secondary support is met (`vote_k=2` setting)

4. **Reject/cooldown handling**
   - rejected candidate is logged (fail reason counters)
   - short cooldown prevents immediate retrigger noise

5. **Streaming outputs**
   - running count
   - accepted periods `(start_frame, end_frame)`
   - rep confidence and reject statistics
   - diagnostic counters: `n_crossings`, `n_candidates`, `n_accepted`, `reject_fail_phi`, `reject_fail_r`, `reject_fail_vote`, `moving_fraction`

Output per video:

- `pred_count`
- `pred_periods_json`
- `mean_confidence`, `n_rejects`
- `n_crossings`, `n_candidates`, `n_accepted`
- `reject_fail_phi`, `reject_fail_r`, `reject_fail_vote`
- `moving_fraction`, `status`

## Data quality summary (full dataset)

- total rows: **1041**
- split counts: train **758**, valid **131**, test **152**
- missing videos: **0**
- missing count rows: **1**
- rows without locations: **3**
- rows with parsed-count mismatch: **4**

Reference: `outputs/00_index/data_quality_report.json`

## Current debug-subset metrics (K=10)

### baseline_fsm

- overall: MAE **5.825**, RMSE **11.882**, OBOA **0.525**, event F1 **0.019**

### phase_native_online_phase_crossing (Fix1 best)

- overall: MAE **4.025**, RMSE **5.189**, OBOA **0.200**, event F1 **0.280**

- Parameters:
  - r_quantile = 0.05
  - r_floor = 0.05
  - eps_phi = 0.45
  - cooldown = 0.20s
  - vote_k = 2

Interpretation:

- Compared with moving-onset, phase-crossing + Fix1 improved count accuracy substantially (MAE/RMSE).
- Event-level alignment also improved, but is still below target (`F1 < 0.35`).
- Current bottleneck is mainly **phase rejection (`fail_phi` dominates)**, not radius rejection (`fail_r` is minimal).

## Known issues and risks

- The debug subset is **small** (40 videos), so metrics are not yet stable.
- Event F1 is still below the threshold for boundary post-alignment (`0.35`).
- Undercount remains the dominant error mode (negative signed error).

## Recommended next steps

1. Refine primary phase progress computation (**local phase integration**)
2. **Per-action** eps_phi tuning (push_up likely needs looser threshold)
3. Apply **snap-to-trough boundary alignment** for F1 improvement
4. **Expand beyond debug subset** once stable
