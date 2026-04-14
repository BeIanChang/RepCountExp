# Phase-Based Method Explanation

## Goal

Given a video, estimate repetition count $\hat{C}$ from pose-derived motion signals.


## Signals

For each action, we define a few biomechanical signals from pose landmarks.

For one signal $s$:

- angle trajectory: $\theta_s(t)$
- temporal derivative: $\omega_s(t) = d\theta_s / dt$
- normalized state: $\theta'_s(t), \omega'_s(t)$
- phase: $\phi_s(t) = \mathrm{unwrap}(\arctan2(\omega'_s(t), \theta'_s(t)))$
- radius: $r_s(t) = \sqrt{\theta'_s(t)^2 + \omega'_s(t)^2}$

Interpretation:

- $\theta$: where the motion is
- $\omega$: how fast it is moving
- $\phi$: cycle progress
- $r$: motion strength / reliability



## State machine

The counter maintains two states:

- `IDLE`
- `ACTIVE`

At each frame:

1. update all signal trackers,
2. test if the motion is active,
3. track the primary-signal phase relative to a reference,
4. detect candidate windows from repeated phase crossings.

## Method 1: Phase Native Online

### Candidate generation

Let

$$
rel(t) = wrap_\pi(\phi_{primary}(t) - \psi_0)
$$

where $\psi_0$ is the reference phase.

A crossing is detected when:

$$
rel(t-1) \le -h, \quad rel(t) \ge h
$$

with hysteresis threshold $h$.

The first crossing opens a candidate, the next crossing closes it.

So each candidate window is:

$$
[a,b]
$$

### Hard validation

For signal $s$, define accumulated wrapped phase advance on a candidate:

$$
\Delta_s(a,b) = \left|\sum_{t=a+1}^{b} wrap_\pi(\phi_s(t)-\phi_s(t-1))\right|
$$

Primary signal must satisfy approximately one full cycle:

$$
\frac{|\Delta_{primary} - 2\pi|}{2\pi} < \epsilon_{\phi}
$$

Secondary signals are weaker supports:

$$
\Delta_{secondary} > \pi
$$

Also require sufficient radius and enough supporting signals.

Accept if:

- primary full-cycle condition passes,
- enough secondary signals pass,
- duration is valid.

## Soft-Penalty Native Update

### Why it helped

Comprehensively evaluate and reject candidates

### In-window stability statistics

For per-step wrapped phase increments:

$$
d\phi_t = wrap_\pi(\phi(t)-\phi(t-1))
$$

we compute:

$$
var_{norm} = \frac{\mathrm{std}(d\phi)}{\max(\mathrm{mean}(|d\phi|), \epsilon)}
$$

$$
flip\_rate = \mathrm{mean}(\mathrm{sign}(d\phi_t) \ne \mathrm{sign}(d\phi_{t-1}))
$$

$$
pause\_rate = \mathrm{mean}(|d\phi_t| < \tau_{pause})
$$



### Confidence components

For one accepted candidate, define:

$$
s_{phase} = \exp\left(-\frac{|\Delta - 2\pi|}{\sigma_{phase}}\right)
$$

$$
s_r = \mathrm{clip}(r_{med}/r_{ref}, 0, 1)
$$

$$
s_{var} = \exp\left(-\frac{var_{norm}}{\sigma_{var}}\right)
$$

$$
s_{flip} = \exp\left(-\frac{flip\_rate}{\sigma_{flip}}\right)
$$

$$
s_{pause} = \exp\left(-\frac{pause\_rate}{\sigma_{pause}}\right)
$$

Final confidence:

$$
conf = 0.40 s_{phase} + 0.20 s_{radius} + 0.15 s_{variance} + 0.15 s_{flip} + 0.10 s_{pause}
$$



## Method 2: Baseline Peak Online

Implementation:

- `scripts/06_baseline_peak_online.py`

This is the online-compatible transfer of the old peak detector.

### Signal used

It uses only the primary 1D signal:

$$
\theta_{primary}(t)
$$

### Online extrema detection

Instead of full-sequence `find_peaks`, it uses slope sign change:

- peak candidate when slope changes $+ \to -$
- trough candidate when slope changes $- \to +$

Then each candidate is confirmed after a short lookahead buffer.

### Confirmation conditions

An extremum candidate is accepted only if:

- it remains a local max/min in a small local window,
- local prominence exceeds threshold,
- min-distance from previous same-type extremum is respected.

### Cycle definition

One repetition candidate is built as trough-to-trough, with:

- duration in valid range,
- at least one peak between the troughs.

So it is causal with bounded latency, but no longer identical to offline global peak finding.

## Method 3: Native + Peak Online Aggregation

Implementation:

- `scripts/12_aggregate_phase_native_peak_online.py`

The current rule is:

$$
\hat{C} =
\begin{cases}
C_{peak}, & conf < \tau \\
C_{phase}, & conf \ge \tau
\end{cases}
$$

where:

- $C_{phase}$ is phase-native count,
- $C_{peak}$ is online-peak count,
- $conf$ is native confidence.

## Method 4: Phase Omega

Implementation:

- `scripts/17_phase_omega.py`

This method follows the idea that zero-crossing / zero-velocity moments are often noisy. Instead of detecting repetitions from crossings around zero, it constructs a causal state machine from the sign and magnitude of angular velocity.

### Omega states

Let angular velocity be:

$$
\omega_s(t)
$$

Then define states:

- `U`: strong positive angular velocity
- `D`: strong negative angular velocity
- `P`: pause / near-stationary
- `M`: middle / uncertain otherwise

The thresholds are computed online from the recent history of $|\omega|$, not from the full future sequence.

For a recent window of length `window_sec`, let:

- $\tau_{enter}$ = quantile of recent $|\omega|$ (with a floor),
- $\tau_{exit} = \tau_{enter} \cdot exit\_ratio$,
- $\tau_{pause} = \tau_{enter} \cdot pause\_ratio$.

Then the online state logic is:

- enter `U` if $\omega \ge \tau_{enter}$
- enter `D` if $\omega \le -\tau_{enter}$
- keep `U` while $\omega \ge \tau_{exit}$
- keep `D` while $\omega \le -\tau_{exit}$
- enter `P` if $|\omega| \le \tau_{pause}$

This gives a local adaptive threshold + hysteresis design.

After that, the state sequence is run-length encoded, and only stable segments longer than `min_state_sec` are kept.

### Counting logic

The method counts one repetition when the state returns to the same anchor state after passing through the opposite state:

- `U -> D -> U`, or
- `D -> U -> D`

Long pause segments reset the anchor.

### Secondary support

For each candidate repetition window from the primary signal, secondary signals provide weak confirmation:

- a candidate is kept if at least one secondary signal contains both a strong positive and a strong negative omega segment inside the same window.

## Overall Result

The final statistics are grouped into three tables and placed together at the end of the report:

1. MAE-style counting metrics
2. candidate statistics
3. IoU-based overlap metrics

The goal is to separate:

- count accuracy,
- candidate generation behavior,
- temporal localization quality.

## Method 5: Phase Omega Interval

This is the interval-based version of the omega idea. The simplest way to understand it is:

- `phase_omega` works on frame-by-frame states.
- `phase_omega_interval` works on interval-by-interval events.

The reason for introducing this version is very practical: in real videos, the exact frame where omega crosses a threshold is often noisy, but the whole "high positive omega" band or "high negative omega" band is usually much more stable. So instead of trusting one frame, this method first finds a stable interval, then uses one representative point inside that interval as the event.

### Step 1: assign local omega labels

For every frame, use recent history of `|omega|` to compute an adaptive threshold. Then label the frame as:

- `U`: strong positive omega
- `D`: strong negative omega
- `P`: pause / near-zero omega
- `M`: middle / uncertain

Important point: this is still online-compatible, because the threshold only uses past local history, not future frames.

More explicitly, for frame $t$, let the recent history window be:

$$
\mathcal{H}_t = \{ |\omega(\tau)| : \tau \in [t-W+1, t] \}
$$

Then the adaptive entry threshold is:

$$
\tau_{enter}(t) = \max\big(\tau_{floor},\ Q_q(\mathcal{H}_t)\big)
$$

where $Q_q$ is the $q$-quantile of the recent history.

The pause threshold is:

$$
\tau_{pause}(t) = \max(0.05\,\tau_{floor},\ \tau_{enter}(t) \cdot r_{pause})
$$

Then the frame label is assigned by:

$$
label(t) =
\begin{cases}
U, & \omega(t) \ge \tau_{enter}(t) \\
D, & \omega(t) \le -\tau_{enter}(t) \\
P, & |\omega(t)| \le \tau_{pause}(t) \\
M, & \text{otherwise}
\end{cases}
$$

### Step 2: convert labels into intervals

After frame labels are produced:

1. consecutive equal labels are merged into runs,
2. very short noisy runs are merged away,
3. only stable `U` and `D` runs are kept,
4. extremely short or extremely long runs are removed.

At the end of this step, the algorithm no longer reasons on individual frames. It reasons on stable signed omega intervals.

If the frame labels are

$$
\{label(t)\}_{t=1}^{T},
$$

then each interval is a maximal contiguous segment

$$
I_k = [s_k, e_k]
$$

such that all frames in that segment have the same retained label (`U` or `D`).

### Step 3: score each interval

Each interval gets a score based on three ideas:

- strong enough (`mean |omega|` and `peak |omega|` are large),
- stable enough (variance is not too high),
- long enough (duration is not too short).

Plain-language meaning:

- if an interval has large average speed, it is more likely to be a real motion phase,
- if that speed is steady, it is less likely to be noise,
- if the interval is too short, it is probably just a noisy fragment.

So this score is just ranking intervals from "good candidate motion phase" to "bad/noisy interval".

For an interval $I_k = [s_k, e_k]$, define:

$$
\mu_k = \mathrm{mean}_{t \in I_k} |\omega(t)|,
\qquad
\sigma_k = \mathrm{std}_{t \in I_k}(\omega(t)),
\qquad
\bar{\tau}_k = \mathrm{mean}_{t \in I_k} \tau_{enter}(t)
$$

where:

- $\mu_k$ = the average magnitude of omega inside the interval; this measures how strong the interval is
- $\sigma_k$ = the standard deviation of omega inside the interval; this measures how unstable / noisy the interval is
- $\bar{\tau}_k$ = the average adaptive threshold inside that interval

and interval length

$$
L_k = e_k - s_k + 1.
$$

Then the current implementation computes:

$$
strength_k = \frac{\mathrm{clip}(\mu_k / \bar{\tau}_k, 0, 3)}{3}
$$

Meaning:

- if $\mu_k$ is close to or below the local threshold, then the interval is not very convincing
- if $\mu_k$ is much larger than the local threshold, then the interval is strong
- `clip(..., 0, 3)` prevents a very extreme interval from dominating the score too much
- dividing by 3 normalizes the term into roughly `[0,1]`

$$
stability_k = \exp\left(-\frac{\sigma_k}{\max(\mu_k, \epsilon)}\right)
$$

Meaning:

- the numerator $\sigma_k$ is how much omega fluctuates inside the interval
- the denominator $\max(\mu_k, \epsilon)$ makes this a relative instability measure
- if an interval is strong but also smooth, then $\sigma_k / \mu_k$ is small, so `stability_k` stays close to 1
- if an interval is noisy and shaky, then $\sigma_k / \mu_k$ becomes larger, so the exponential term shrinks toward 0

Why use the exponential:

- it gives a smooth penalty rather than a hard cutoff
- small instability only reduces the score a little
- large instability is penalized more strongly
- it keeps the score bounded in `(0,1]`

$$
duration_k = \frac{\mathrm{clip}(L_k / L_{min}, 0, 2)}{2}
$$

Meaning:

- if the interval is shorter than the minimum meaningful duration, this term is small
- if the interval is long enough, this term approaches 1
- again, clipping prevents extremely long intervals from being rewarded too much

and final interval score:

$$
score_k = 0.55\,strength_k + 0.30\,stability_k + 0.15\,duration_k
$$

Interpretation of the three terms:

- `strength_k`: is the interval strong compared with the local threshold?
- `stability_k`: does omega stay relatively steady inside the interval?
- `duration_k`: is the interval long enough to be meaningful?

So a high `score_k` means: the interval is strong, steady, and not too short.

So the role of Step 3 is:

- not to make the final counting decision directly,
- but to rank intervals by how much we trust them as true motion phases.

### Step 4: choose one representative point

- the frame inside the interval where `|omega|` is maximal.

Formally, the representative point of interval $I_k$ is:

$$
c_k = \arg\max_{t \in I_k} |\omega(t)|
$$

This is meant to capture the most informative point inside the interval.

### Step 5: merge nearby same-sign intervals

If two nearby intervals have the same sign, for example:

- `U ... U`
- or `D ... D`

they are treated as fragmented pieces of the same motion phase. In that case:

- the intervals are merged logically,
- the stronger one keeps the representative point.

This is necessary because real data often splits one plateau into several pieces.

Plain-language meaning:

Suppose the true motion should contain one long positive-velocity phase, but noise breaks it into:

- one `U` interval,
- then a tiny gap,
- then another `U` interval.

If we keep them separate, the algorithm may think they are two different positive events.

Step 5 fixes this by saying:

- if two nearby intervals have the same sign,
- and the gap between them is short,
- treat them as one motion phase instead of two.

The stronger interval keeps the representative point, so we do not double-count the same phase.

### Step 6: count with representative-point alternation

After interval detection, the method works on a simplified event sequence. Example:

- `U, D, U, D, U`

Then it counts one repetition from:

- `U-D-U`, or
- `D-U-D`

The repetition window is defined from the first representative point to the third representative point. A candidate repetition is kept only if the total span is inside the allowed cycle-duration range.

So if three consecutive representative events are

$$
(U, c_i), (D, c_{i+1}), (U, c_{i+2})
$$

or

$$
(D, c_i), (U, c_{i+1}), (D, c_{i+2}),
$$

then one repetition candidate is formed with temporal support

$$
[c_i, c_{i+2}]
$$

and accepted only if

$$
T_{min} \le c_{i+2} - c_i \le T_{max}.
$$

This means the method is now much closer to:

- interval detection -> event extraction -> repetition counting

instead of directly counting noisy threshold crossings.

### Step 7: secondary-signal support

As in the simpler omega version, secondary signals are only used as weak confirmation.

For a repetition candidate to survive, the method checks whether at least one secondary signal also shows both positive and negative omega evidence inside that candidate region.

## Final Summary Tables

### Table 1. Count-level metrics

| Method | MAE | MAE_norm_p1 | OBO | Event-F1 |
|---|---:|---:|---:|---:|
| phase_native_soft | 5.1711 | 0.6016 | 0.2500 | 0.4540 |
| baseline_peak_online | 6.3224 | 1.2517 | 0.2632 | 0.4310 |
| phase_omega | 6.9539 | 0.6925 | 0.2961 | 0.2485 |
| phase_omega_interval | 8.559 |       0.782 |  0.283 | 0.1853 |

### Table 2. Candidate statistics

| Method | candidate_total | accepted_total | survival_ratio | pred_total | pred_mean | pred_median | pred_std | pred_min | pred_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase_native_soft | 3572 | 1822 | 0.5101 | 1822 | 11.9868 | 7.0000 | 13.6401 | 0 | 87 |
| baseline_peak_online | 2968 | 2607 | 0.8784 | 2607 | 17.1513 | 10.5000 | 17.2389 | 0 | 92 |
| phase_omega | - | 1785 |              - | 1785 | 11.7434 | 9.0000 | 9.6173 | 0 | 59 |
| phase_omega_interval | - | 2746 | - | 2746 | 18.0658 | 18.0000 | 11.0826 | 0 | 59 |

Interpretation:

- `baseline_peak_online` produces a candidate count closest to GT on average, but that does not translate to the best error metrics.
- `phase_native_soft` starts with many raw candidates and keeps about half of them.
- `phase_omega` is more conservative than peak online, but still produces a reasonable number of candidates.
- `phase_omega_interval_phasefilter` is currently far too aggressive and collapses candidate survival.

### Table 3. IoU overlap metrics

| Method | pred_gt_ratio | mean_best_iou_gt | mean_best_iou_pred | F1_iou_03 | F1_iou_05 |
|---|---:|---:|---:|---:|---:|
| phase_native_soft | 0.7419 | 0.4034 | 0.5296 | 0.6980 | 0.5171 |
| baseline_peak_online | 1.0615 | 0.5260 | 0.5105 | 0.7106 | 0.4835 |
| phase_omega | 0.7268 | 0.3562 | 0.4326 | 0.6022 | 0.3942 |
| phase_omega_interval | 1.1181 | 0.2954 | 0.3053 | 0.4587 | 0.2026 |

How to read this table:

- `pred_gt_ratio > 1`: over-proposing windows.
- `pred_gt_ratio < 1`: under-proposing windows.
- `mean_best_iou_gt`: how well GT windows are covered by nearest predictions.
- `mean_best_iou_pred`: how well predictions align with some GT window.
- `F1_iou_03` is a loose localization quality score.
- `F1_iou_05` is a stricter localization quality score.

Interpretation:

- `baseline_peak_online` has the best loose-overlap recovery (`F1_iou_03`), which matches its broad candidate generation style.
- `phase_native_soft` has the best stricter-overlap score (`F1_iou_05`), which means its windows are cleaner and more selective.
- `phase_omega` is usable but still clearly behind native on localization quality.

## Future

- window fit: not only count, but find the most accurate window position
- Use confidence as threshold
- hard vote mechanism
- in method 5, use K means like methods to hold and adaptively find the intervals
- use different low band configurations
