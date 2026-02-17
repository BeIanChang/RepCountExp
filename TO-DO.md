

# RepCountA Experiment Pipeline (Step-by-step)

## 0. Environment & Repo Layout

### Required tools

* Python 3.10+
* `opencv-python`
* `pandas`, `numpy`
* `matplotlib`
* Pose estimation backend (choose one):

  * **MediaPipe Pose** (fast, easy) **OR**
  * **OpenPose / MMPose** (optional, heavier)

### Recommended repo folders

```
repo/
  data/RepCountA/                  # dataset root (symlink ok)
  scripts/
  outputs/
    00_index/
    01_preview/
    02_pose/
    03_signals/
    04_results/
  configs/
    exercises.yaml                 # exercise families + joint signal definitions
```

---

## 1. Parse dataset splits + build a master index

### Input

* `RepCountA/annotation/train.csv`
* `RepCountA/annotation/valid.csv`
* `RepCountA/annotation/test.csv`
* videos under:

  * `RepCountA/video/train/*.mp4`
  * `RepCountA/video/valid/*.mp4`
  * `RepCountA/video/test/*.mp4`

### Steps

1. Read each csv, add a `split` column (`train|valid|test`).
2. For each row, resolve the video path:

   * `video_path = RepCountA/video/{split}/{video_name}.mp4`
3. Verify the file exists; drop or log missing entries.
4. Create `outputs/00_index/master_index.csv` with at least:

   * `video_id` (filename without extension)
   * `split`
   * `video_path`
   * `label_count` (if provided)
   * `periods` / `locations` (raw string from csv)
   * `action` (if provided; otherwise fill later)
   * `fps`, `n_frames`, `duration_sec` (computed from video)

### Output checks

* `master_index.csv` row count should match total annotations (minus missing videos).
* For 20 random rows, open the video and confirm fps/frames load correctly.

---

## 2. Parse period annotations into normalized format

### Goal

Convert the `locations` column into a consistent list of periods:

```
periods = [(start_frame, end_frame), ...]
```

### Steps

1. Inspect how `locations` are stored in the csv:

   * Could be: frame indices, normalized [0,1], or time in seconds.
2. Implement a parser that outputs one of:

   * `periods_frame`: list of (start_frame, end_frame)
3. If annotations are normalized:

   * `start_frame = round(start_norm * n_frames)`
   * `end_frame   = round(end_norm   * n_frames)`
4. Validate:

   * `0 <= start < end <= n_frames`
5. Save parsed periods into:

   * `outputs/00_index/master_index.parsed.csv`
   * include a column like `periods_json` (JSON string)

### Output checks

For 50 random videos:

* `len(periods)` should be close to `label_count` (allow small mismatch but log).
* Plot a simple timeline: marks for each period.

---

## 3. Map / infer action category (optional but recommended)

### Input

* `RepCountA/original_data/DownloadLink/*.txt` (lists of original video urls by action)
* `RepCountA/original_data/filename_mapping.xlsx` (maps IDs/filenames)

### Steps (recommended)

1. Use `filename_mapping.xlsx` to map each `video_id` to an action label if possible.
2. If mapping is not straightforward, skip for first run and do experiments on a subset by manual selection.

### Output

* Add `action` column to `master_index.parsed.csv`.

### Output checks

* Print distribution of `action` values (top 10).

---

## 4. Create a small “debug subset” for fast iteration

### Goal

Avoid running pose on all 1000+ videos initially.

### Steps

1. Choose 1–3 actions (e.g., `squat`, `push_up`, `pull_up`).
2. Sample:

   * train: 20 videos
   * valid: 10 videos
   * test: 10 videos
3. Save subset list:

   * `outputs/00_index/subset_debug.csv`

### Output checks

* Ensure each selected video plays.
* Ensure each has parsed `periods`.

---

## 5. Pose extraction (per video)

### Input

* videos from subset or full split

### Steps

For each video:

1. Run pose estimator per frame.
2. If multiple persons appear:

   * Select the main track by (a) largest bbox area, and/or (b) temporal continuity.
3. Save per-frame landmarks to a file:

   * `outputs/02_pose/{split}/{video_id}.npz`
   * store arrays:

     * `landmarks_3d`: shape (T, 33, 3) (or 2D if only 2D)
     * `visibility`: shape (T, 33)
     * `fps`, `n_frames`
     * `track_id` metadata if available

### Output checks

* For 5 random videos:

  * Confirm `T == n_frames` (or log dropped frames)
  * Plot a skeleton overlay for 10 frames (optional)
* Log pose failure rate:

  * % frames with missing keypoints

---

## 6. Convert pose to joint signals (angles + trunk)

### Goal

For each exercise family, compute 2–3 1D signals for redundancy voting.

### Define signals (example templates)

Put in `configs/exercises.yaml`:

**Squat**

* knee_flex = angle(hip, knee, ankle)
* hip_flex  = angle(shoulder, hip, knee) or pelvis-based variant
* trunk_pitch = angle(vector(hip→shoulder), vertical_axis)

**Push-up**

* elbow_flex = angle(shoulder, elbow, wrist)
* shoulder_ang = angle(elbow, shoulder, hip)
* trunk_pitch or trunk_alignment

**Pull-up**

* elbow_flex
* shoulder_ang
* trunk_vertical

### Steps

For each video:

1. Compute each signal `theta_j(t)` in radians or degrees.
2. Smooth `theta_j(t)` (Savitzky–Golay recommended).
3. Compute velocity `omega_j(t) = d(theta_j)/dt` (after smoothing).
4. Normalize within video (or within sliding window):

   * z-score or robust scale → `theta'_j`, `omega'_j`
5. Compute:

   * phase angle: `phi_j(t) = unwrap(arctan2(omega'_j, theta'_j))`
   * radius: `r_j(t) = sqrt(theta'^2 + omega'^2)`
6. Save features:

   * `outputs/03_signals/{split}/{video_id}.npz`
   * include raw theta, omega, phi, r for each signal

### Output checks (must do)

For 10 random videos, auto-generate plots:

* theta(t), omega(t)
* phase portrait (theta' vs omega')
* phi(t), r(t)
  Save to:
* `outputs/01_preview/{split}/{video_id}_plots.png`

---

## 7. Baseline rep counting methods

### Baseline A: Ground-truth count (upper bound)

* Use `len(periods)` as true count.
* This is for evaluation sanity.

### Baseline B: Simple peak/trough rep counter (signal-domain)

For a chosen primary signal (e.g., knee_flex):

1. Smooth theta
2. Find peaks/troughs with min distance / prominence
3. Count trough→peak→trough cycles
4. Apply duration gate using fps

Save predictions:

* predicted rep count per video
* predicted periods if possible

Output:

* `outputs/04_results/baseline_peak.csv`

### Baseline C: Existing FSM (if already implemented)

Run current FSM logic on the same theta(t) and output:

* `outputs/04_results/baseline_fsm.csv`

---

## 8. Proposed method: phase-cycle validation + redundancy voting

### Core idea

Use phase features to validate “real cycles” and reject partial/noisy reps.

### Method (per candidate rep window)

We need candidate windows first:

* Option 1 (recommended initially): use Baseline peak/trough windows.
* Option 2: use FSM’s bottom/top/bottom segmentation.

For each joint signal j in selected set:

1. Compute phase change in window:

   * `delta_phi_j = phi_j(t1) - phi_j(t0)`
2. Check loop completion:

   * `delta_phi_j` in `[1.7π, 2.5π]` (tunable)
3. Radius gate:

   * `median(r_j[t0:t1]) > r_min_j`

Joint passes if both checks pass.

### Voting rule

* Accept repetition if **at least 2 of 3** joints pass.
* Rep confidence = average of per-joint scores:

  * `score_phase = exp(-abs(delta_phi-2π)/σ)`
  * `score_radius = clamp(median(r)/target, 0..1)`

Output predictions:

* predicted rep count per video
* predicted periods per video
  Save:
* `outputs/04_results/proposed_phase_vote.csv`

---

## 9. Evaluation

### Count-level metrics (required)

For each split:

* MAE: mean(|pred_count - true_count|)
* RMSE
* Off-by-one accuracy (OBOA): %(|pred-true| <= 1)

### Event-level metrics (optional but recommended)

Compare predicted periods with GT `periods`:

* A prediction is correct if start/end are within ±K frames of a GT period (K=5 or 10)
* Compute precision/recall/F1

### Output

* `outputs/04_results/metrics_summary.md`
* `outputs/04_results/metrics_table.csv`

### Failure case report (high value)

Auto-collect top N failure videos and export:

* plots (theta/omega/phase portrait/phi)
* GT periods vs predicted periods overlay
  Save:
* `outputs/04_results/failure_cases/`

---

## 10. What to report in the next meeting (minimum deliverables)

1. Pose success rate (% frames with valid keypoints)
2. 6–12 example plots: clean rep / partial rep / pause / noisy pose
3. Count-level metrics on debug subset:

   * baseline_peak vs proposed_phase_vote
4. 3 failure cases with explanation

---

# Notes / Gotchas (RepCount-specific)

* RepCount videos may contain **multiple people**; selection strategy matters.
* Some actions are not “fitness reps” (e.g., soccer juggling). Start with workout actions first.
* Annotation `locations` format must be verified (frame vs normalized time). Do not assume.

​