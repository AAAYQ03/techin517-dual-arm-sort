# techin517-dual-arm-sort

Dual-arm SO-101 LeRobot tabletop sorting system. Two arms cooperatively detect, pick, and sort 4 object classes into category bins using a hybrid CV → IK → ACT architecture.

**Authors:** [Team Member 1], [Team Member 2]
**Course:** TECHIN 517 (Spring 2026)

## Demo

[TODO: video link]

## Project Overview

The robot sorts 4 objects from a tabletop into 2 bins:
- **Electronics bin**: battery, earbuds case
- **Stationery bin**: pen, glue stick

Difficulty levels evaluated:
- **Easy**: each arm picks objects on its own half-table, delivers to its own-side bin
- **Medium**: cross-side delivery (some objects must travel across the table to the opposite bin)
- **Hard**: same as Medium plus distractor objects on the table

### Architecture

Pipeline:
1. D435i top camera (RGB + depth) captures a frame
2. Grounding DINO (open-vocabulary detection) produces candidate bboxes
3. Color prior + bbox-aspect-ratio shape prior filter false positives
4. CLIP fine-grained classifier (battery vs others)
5. IoU dedup across classes (handles same object detected as multiple types)
6. Dispatcher assigns picks to arms and schedules parallel vs sequential
7. lerobot_kinematics geometric IK (5DOF SO-101) computes joint targets
8. Smooth ramp trajectory drives the arm to a point 5cm above the object
9. ACT policy (per-object, per-arm) performs the final grasp
10. IK back to the destination box, release, return to safe_home

### Key features

- **Hybrid vision-IK-learning pipeline**: classical IK handles long-range positioning, learned ACT handles the final 5cm grasp where calibration drift dominates
- **6 specialized ACT policies** (one per arm x object): per-arm training to handle different wrist-cam viewpoints
- **Smart dispatcher** (parallel_demo/dual_arm_smart.py): parallelizes same-side picks, sequentializes cross-side picks to avoid arm collisions
- **Auto-assignment mode** (--auto flag): detects all objects in one snapshot and assigns to arms by bbox pixel coordinate

## Quantitative Results

[TODO: insert success rate table and timing chart]

Evaluation across 30 trials (10 per difficulty):

| Difficulty | Success rate | Mean time (s) |
|------------|--------------|----------------|
| Easy       | [TODO]       | [TODO]         |
| Medium     | [TODO]       | [TODO]         |
| Hard       | [TODO]       | [TODO]         |

## Team Contributions

[TODO: per-member contributions]

- **[Team Member 1]**: ...
- **[Team Member 2]**: ...

## Setup

### Hardware

- 2 x SO-101 LeRobot follower arms (designated follower7 and follower8)
- 2 x SO-101 LeRobot leader arms (for data collection only; not needed at deployment)
- 1 x Intel RealSense D435i (mounted overhead, centered between arms)
- 2 x USB wrist cameras (one per follower)
- Workstation with NVIDIA GPU (>=16 GB; tested on RTX 5090)

### Software environment

The repo includes a .devcontainer/ for VS Code Dev Container reproducibility. Anyone with Docker + VS Code installed should be able to clone and "Reopen in Container":

    git clone [TODO: github URL]
    cd techin517-dual-arm-sort
    code .

The container ships with ROS2 Humble, MoveIt2, PyTorch 2.7.1+cu128, LeRobot, and all Python dependencies.

### Required external dependencies (installed into the container by docker/setup.sh)

- LeRobot (https://github.com/huggingface/lerobot)
- lerobot-kinematics by box2ai-robotics (https://github.com/box2ai-robotics/lerobot-kinematics) — patch required: change `max_joint_change = 0.1` to `5.0` in lerobot_Kinematics.py smooth_joint_motion()
- ROS2 Humble + MoveIt2 + pymoveit2
- transformers (>=4.51 for Grounding DINO + CLIP)
- pyrealsense2, OpenCV

### Hand-eye calibration

Each follower arm requires its own hand-eye calibration. We provide our calibration values for follower7/follower8 in calibration_*.txt and embedded in ros2_ws/src/soa_ros2/soa_moveit_config/. You will need to redo the calibration for your hardware. See recalc_calib.py / recalc_calib2.py for the matrix transformation helper.

### Pre-trained ACT models

Download our 6 fine-tuned ACT checkpoints from Hugging Face:

| Object   | Arm | Hugging Face repo                                                   |
|----------|-----|---------------------------------------------------------------------|
| pen      | f7  | https://huggingface.co/ycui77/act_pen_f7                             |
| glue     | f7  | https://huggingface.co/ycui77/act_glue_f7                            |
| battery  | f7  | https://huggingface.co/ycui77/act_battery_f7                         |
| pen      | f8  | https://huggingface.co/ycui77/act_pen_f8_v1_final                    |
| battery  | f8  | https://huggingface.co/ycui77/act_battery_f8                         |
| earbuds  | f8  | https://huggingface.co/ycui77/act_earbuds_f8                         |

Place them under outputs/train/act_<object>_<arm>/checkpoints/last/pretrained_model/

## Usage

### 1. Bring up ROS

    cd ros2_ws && source install/setup.bash
    sudo chmod 666 /dev/serial/by-id/* /dev/v4l/by-path/*
    ros2 launch soa_moveit_config soa_moveit_bringup.launch.py

### 2. Run dispatch

Manual mode (you specify which objects each arm should look for):

    # Single-arm dispatch (f8 only)
    python3 dispatch_pick.py

    # Sequential dual-arm
    bash dual_arm_full_dispatch.sh

Smart dispatch (auto-parallelizes same-side picks):

    # Manual item specification
    python3 parallel_demo/dual_arm_smart.py --f8items battery earbuds --f7items pen glue

    # Auto-assignment (recommended) — robot decides who picks what
    python3 parallel_demo/dual_arm_smart.py --auto

### 3. Reset to safe pose between trials

    python3 reset_to_safe_home.py

### 4. Data collection (for retraining)

Workflow: CV+IK moves arm above the object, then leader-teleop records a 6-second grasp demo.

    ./do_one_demo.sh <object_name>      # for f8
    ./do_one_demo_f7.sh <object_name>   # for f7

See do_one_demo.sh and pipeline_to_above_only.py for details.

## Repository structure

    techin517-dual-arm-sort/
    |-- .devcontainer/           # VS Code Dev Container config
    |-- docker/                  # Dockerfile + setup script
    |-- cv_module/               # Grounding DINO wrapper, depth sampling
    |-- parallel_demo/           # Smart parallel/sequential dispatcher
    |   |-- dual_arm_smart.py
    |   `-- auto_assign_helper.py
    |-- ros2_ws/src/             # ROS2 workspace
    |   |-- soa_ros2/            # Our launch files, SRDF, hand-eye calibration
    |   |-- pymoveit2/           # MoveIt2 Python bindings
    |   `-- ...                  # Other lab-provided packages
    |-- dispatch_pick.py         # f8 single-arm dispatch
    |-- dispatch_pick_f7.py      # f7 single-arm dispatch
    |-- dual_arm_full_dispatch.sh
    |-- clip_classifier.py       # CLIP fine-grained classifier
    |-- pipeline_to_above_only.py # CV+IK above pose (for data collection)
    |-- do_one_demo*.sh          # Data recording wrappers
    |-- reset_to_safe_home.py
    |-- calibration_*.txt        # Recorded hand-eye calibration values
    |-- recalc_calib*.py         # Calibration matrix helpers
    `-- LICENSE                  # Apache 2.0

## License

Apache License 2.0 — see LICENSE.

This project builds on top of LeRobot (Apache 2.0), lerobot-kinematics, Grounding DINO, CLIP, and ROS2 Humble / MoveIt2.
