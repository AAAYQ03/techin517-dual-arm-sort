"""Single source of truth for SOA policy joint order, defaults, and frames.

Order MUST match the IsaacLab articulation order. Verified against
``model/soa_lab/source/soa_lab/soa_lab/robots/trs_so101/so_arm101.py``
and ``model/params/env.yaml``.
"""

ALL_JOINT_NAMES = (
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
    'gripper',
)

ARM_JOINT_NAMES = ALL_JOINT_NAMES[:5]
GRIPPER_JOINT_NAME = ALL_JOINT_NAMES[5]

# BASIC v1 (so_arm101.py:35-50). Same six values appear in env.yaml:167-172.
DEFAULT_JOINT_POS = (0.0, 0.323, -0.055, 1.33, 0.0, 0.2)

# env.yaml:587-611
ARM_ACTION_SCALE = 0.5
GRIPPER_OPEN_CMD = 1.0
GRIPPER_CLOSE_CMD = -0.2

# Defaults; overridable via ROS parameters.
ROBOT_BASE_FRAME = 'follower/base_link'
EE_FRAME = 'follower/gripper_frame_link'
CUBE_FRAME = 'aruco_cube'

OBS_DIM = 63
ACTION_DIM = 6
HISTORY_LEN = 3
