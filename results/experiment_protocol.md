# Experiment Protocol

Detailed methodology for the 30-trial quantitative evaluation. This is
the protocol description distilled from the final-presentation slide
deck (`quantitative_results_final.pptx`, slide 3).

## Task Description

The robot system must sort 4 target objects from a tabletop into 2 bins,
categorized by class:

- **Electronics bin**: battery, earbuds case
- **Stationery bin**: pen, glue stick

Three difficulty states are evaluated, each with 10 trials.

### State 1 — Easy: Spatial Separation
- Stationery objects on the left (`f7` side), electronics on the right (`f8` side)
- No cross-midline objects
- **Bi-arm strategy**: simultaneous bi-arm pick-and-place

### State 2 — Medium: Partial Interleaving
- 2 objects cross the midline (e.g., one electronic on the stationery side)
- **Bi-arm strategy**: sequential cross-side handoff first, then simultaneous same-side picks

### State 3 — Hard: Interleaving + Distractors
- All Medium conditions plus 2 distractor objects on the table (e.g., keys, coins)
- **Bi-arm strategy**: same as Medium plus CV rejection of distractors

## Success Criteria

A trial succeeds when:

- **All 4 target objects** are placed inside their assigned bins
- Each object is **fully contained within the bin's footprint** — no
  objects outside or balanced on the rim

## Termination Conditions (failure)

A trial terminates as failure if any of the following occurs:

| Condition | Description |
|---|---|
| **Timeout** | Total duration > 120 s |
| **Drop** | Object slips from gripper mid-transit |
| **Wrong bin** | Object placed in the incorrect bin |
| **Collision / safety stop** | Arm contact or manual abort |
| **Detection failure** | CV misses a visible target |
| **Missed pick** | Gripper closes but no object grasped |

## Starting State

Before each trial:

- Both arms at `safe_home` pose, grippers open
- 4 target objects placed on the tabletop:
  - Pen + glue stick on `f7` side (`-y`)
  - Battery + earbuds on `f8` side (`+y`)
  - All within IK workspace: `x ∈ [0.20, 0.28] m`, `|y| < 0.10 m`
  - **Medium**: items reassigned across sides
  - **Hard**: 2 visual distractors added
- Bins at calibrated positions:
  - Stationery (`f7`): `(0.155, -0.058, 0.10)`
  - Electronics (`f8`): `(0.155, +0.058, 0.10)`

## Reset Procedure (~30 s per reset)

After each trial:

1. Stop dispatcher (auto-exit on success; Ctrl+C on failure)
2. Kill ROS and dispatch processes to release hardware
3. Run `reset_to_safe_home.py` to return both arms
4. Retrieve all 4 objects from bins / table / floor
5. Re-place objects at new positions within the workspace
6. Reset USB permissions: `chmod 666 /dev/serial /dev/v4l`
7. Visually confirm clean workspace, all targets visible
8. Start next trial

## Timing Methodology

Wall-clock time is logged by a bash wrapper around
`dual_arm_full_dispatch.sh`. The wrapper records UNIX timestamps:
- **Start**: before the dispatcher invocation
- **End**: after both arms return to `safe_home`

This measurement covers the full execution from camera initialization
through both arms' final return — not just the active pick-and-place
phase. Reset time (~30 s between trials) is not included.

## Why Std = 0 in Reported Times

The pipeline uses fixed-duration ramps:

| Stage | Duration |
|---|---|
| Detection home + CV snapshot | ~5 s |
| Ramp to above-object pose | 2.5 s |
| ACT policy execution | 10.0 s |
| Ramp to bin | 3.0 s |
| Release + return to safe_home | 3.5 s |

All ramps run for their full duration regardless of whether the target
is reached earlier. Adding closed-loop early termination on grasp
success is listed in `Future Work` and would reduce successful-trial
wall-clock time by an estimated 20–30 %.

## Reporting Convention

- Failure modes are classified by the **module that produced the
  error**, not by visual outcome:
  - "CV detection" → Grounding DINO returned 0 boxes for a visible
    target
  - "Grasp failure" → gripper closed but object was not retained
  - "Object dropped" → object slipped during transit (ACT policy +
    transport)
- Times reported are wall-clock seconds, single-digit precision
- Success rates reported as integer percent of 10 per state

## Data Files

Raw per-trial data: `trial_results.csv` (30 rows, one per trial).
Aggregated stats: `timing_stats.csv` and `failure_modes.csv`.
