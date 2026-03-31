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

## Overall Result

### Metrics used

- `MAE`: mean absolute error in repetition count. Lower is better.
- `MAE_norm_p1`: normalized count error using denominator `(GT + 0.1)`. Lower is better.
- `OBO`: fraction of videos whose count error is at most 1. Higher is better.
- `Event-F1`: period-window matching score. A predicted repetition window is matched to a GT repetition window if endpoint matches within 10 frames.

### Results

| Method | Candidate source | MAE | MAE_norm_p1 | OBO | Event-F1 |
|---|---|---:|---:|---:|---:|
| zero_crossing | phase-crossing | 5.1711 | 0.6016 | 0.2500 | 0.4540 |
| baseline_peak_online | peak/trough with low lookahead | 6.3224 | 1.2517 | 0.2632 | 0.4310 |

### Interpretation

- `zero_crossing` has better `MAE` and much better `MAE_norm_p1`, which means its per-video count estimates are **more stable overall**.
- `baseline_peak_online` has slightly higher `OBO`, which means it hits the exact-or-near-exact count on a few more videos, but it also has **several large outliers** (suffers from **heavier-tailed errors** on difficult videos).
- In particular, some `bench_pressing` and `pommel` videos show strong peak overcounting, which hurts `MAE` much more than it hurts `OBO`.

## Candidate Statistics Report

### A. Per-video count statistics

| Quantity | Total accepted / GT | Total produced | Mean | Median | Std | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| GT periods per video | 2456 | - | 16.1579 | 10.5000 | 15.3816 | 0 | 92 |
| baseline_peak_online predicted periods per video | 2607 | 2968 | 17.1513 | 10.5000 | 17.2389 | 0 | 92 |
| zero_crossing accepted candidates per video | 1822 | 3572 | 11.9868 | 7.0000 | 13.6401 | 0 | 87 |

Interpretation:

- `baseline_peak_online` produces on average 17.15 windows per video, which is **numerically close** to GT mean 16.16.
- `zero_crossing` starts from many raw candidates (mean 23.50) but only accepts about half of them. Its accepted window count (mean 11.99) is lower than GT mean, so it is conservative, but that conservativeness **reduces extreme overcounting**.

### B. IoU-based overlap with GT periods

We compare predicted windows against ground-truth periods using interval IoU.

For two intervals $A=[a_1,a_2]$ and $B=[b_1,b_2]$:

$$
IoU(A,B) = \frac{|A \cap B|}{|A \cup B|}
$$

where:

- intersection = overlapping duration,
- union = total covered duration.

Interpretation:

- high IoU means predicted window aligns well with GT window,
- low IoU means bad temporal localization even if the count is close.

We report precision / recall / F1 by matching predicted windows to GT windows with IoU thresholds 0.3 and 0.5.

| Method | Pred/GT ratio | Mean best IoU to GT | Mean best IoU to Pred | P@IoU0.3 | R@IoU0.3 | F1@IoU0.3 | P@IoU0.5 | R@IoU0.5 | F1@IoU0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_peak_online | 1.0615 | 0.5260 | 0.5105 | 0.6901 | 0.7325 | 0.7106 | 0.4695 | 0.4984 | 0.4835 |
| zero_crossing | 0.7419 | 0.4034 | 0.5296 | 0.8194 | 0.6079 | 0.6980 | 0.6070 | 0.4503 | 0.5171 |

How to read these numbers:

- `Pred/GT ratio > 1`: method tends to produce too many windows.
- `Pred/GT ratio < 1`: method tends to under-propose windows.
- `Mean best IoU to GT`: for each GT period, find the closest predicted window and measure overlap quality.
- `Mean best IoU to Pred`: for each predicted window, find the closest GT period and measure overlap quality.
- `P@IoU0.3`: among predicted windows, fraction that match some GT period with IoU >= 0.3.
- `R@IoU0.3`: among GT periods, fraction that are recovered by some prediction with IoU >= 0.3.
- `F1@IoU0.3`: harmonic mean of that precision and recall.
- IoU 0.5 is a stricter localization criterion than IoU 0.3.

Interpretation of the table:

- `baseline_peak_online` has higher recall to GT and higher `F1@IoU0.3`, meaning it **recovers more GT windows at a loose overlap threshold**.
- `zero_crossing` has higher precision and also higher `F1@IoU0.5`, meaning its **accepted windows are** **temporally cleaner and more selective**.
- This matches the count-level story: peak-online is **broader and noisier**, native-soft is **stricter and more conservative**.

## Future

- window fit: not only count, but find the most accurate window position
- Use confidence as threshold
- hard vote mechanism
