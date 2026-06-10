# Quantitative Results

This directory contains the raw evaluation data, summary statistics, and
publication-quality plots from the 30-trial assessment of the v1 system.

## Headline Numbers

- **30 trials** total (10 per state)
- **66.7 % overall success rate** (20 of 30 trials)
  - Easy: **80 %** (8/10)
  - Medium: **60 %** (6/10)
  - Hard: **60 %** (6/10)
- **Mean wall-clock time**: 43.3 s (Easy) / 65.8 s (Medium and Hard)
- **Zero safety incidents** across all 30 trials

## File Index

| File | Description |
|---|---|
| `trial_results.csv` | Raw per-trial data: trial #, state, result, time, failure mode, notes |
| `failure_modes.csv` | Aggregated failure analysis by mode (CV / grasp / drop) |
| `timing_stats.csv` | Per-state success rate, mean time, std, range |
| `experiment_protocol.md` | Trial protocol: success criteria, termination conditions, reset procedure |
| `generate_plots.py` | Reproducibility script — regenerates all 3 plots from the CSVs |
| `plots/success_rate_by_state.png` | Bar chart of successes vs failures per state |
| `plots/failure_mode_distribution.png` | Donut chart of the 10 failure modes |
| `plots/timing_by_state.png` | Bar chart of mean completion time per state |

## Key Findings

### 1. Glue stick is the dominant grasp bottleneck
All 5 grasp failures (50 % of all failures) involve the cylindrical glue
stick. Its smooth, round geometry sits at the edge of the ACT training
distribution.

### 2. CV detection failures are state-agnostic
4 of 10 failures (40 %) are Grounding DINO mis-detections distributed
across all three states (Easy 2, Medium 1, Hard 1). Failure pattern is
random — small or partially-occluded targets get missed regardless of
scene clutter.

### 3. Bi-arm parallelism gives ~1.5× speedup
Easy (43.3 s) vs Medium/Hard (65.8 s): the 22.5 s overhead is the
sequential cross-side handoff that Medium and Hard require. When both
arms work independently on their own half (Easy), they run in true
parallel.

### 4. Timing is deterministic
Standard deviation across successful trials is exactly 0 because the
pipeline uses fixed-duration ramps (no closed-loop early termination on
success). CV failures terminate early at 43.0 s / 44.3 s.

## How to Read the Plots

- `success_rate_by_state.png` — primary results figure for the report.
  Green bars = successes, red bars = failures, percentages below state
  names show success rate.
- `failure_mode_distribution.png` — drill-down on the 10 failed trials.
  Use this to understand *which module is the bottleneck*.
- `timing_by_state.png` — efficiency story. Use this to discuss when
  parallel execution helps and when sequential handoff is unavoidable.

## Reproducing the Plots

From this directory:

```bash
pip install matplotlib pandas
python3 generate_plots.py
```

The script reads no external files — all numbers are hard-coded from
the CSVs and match the data in `trial_results.csv` exactly. Modify the
script to add new plots or change the styling.
