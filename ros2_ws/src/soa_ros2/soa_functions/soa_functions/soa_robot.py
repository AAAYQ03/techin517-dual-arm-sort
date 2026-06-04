"""SOA robot configuration for pymoveit2.

Single source of truth for robot constants used by action servers.
Follows the pymoveit2/robots/panda.py pattern.

Sources:
  - soa_moveit_config/config/soa.srdf (group names, gripper states)
  - soa_moveit_config/config/moveit_controllers.yaml (joint lists)
"""

from typing import List

MOVE_GROUP_ARM: str = "arm"
MOVE_GROUP_GRIPPER: str = "gripper"

OPEN_GRIPPER_JOINT_POSITIONS: List[float] = [1.7453]
CLOSED_GRIPPER_JOINT_POSITIONS: List[float] = [-0.1745]


def joint_names(prefix: str = "") -> List[str]:
    return [
        prefix + "shoulder_pan",
        prefix + "shoulder_lift",
        prefix + "elbow_flex",
        prefix + "wrist_flex",
        prefix + "wrist_roll",
    ]


def base_link_name(prefix: str = "") -> str:
    return prefix + "base_link"


def end_effector_name(prefix: str = "") -> str:
    return prefix + "gripper_link"


def gripper_joint_names(prefix: str = "") -> List[str]:
    return [prefix + "gripper"]
