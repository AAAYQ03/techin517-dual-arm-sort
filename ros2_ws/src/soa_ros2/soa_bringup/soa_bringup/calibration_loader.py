"""Calibration file loader for SOA robot arms.

This module provides utilities to load and parse calibration JSON files
for robot arms. Each calibration file contains servo IDs and homing offsets
for the arm's joints.
"""

import json
import os
from typing import Dict


def load_arm_calibration(calibration_dir: str, arm_id: str) -> Dict[str, Dict[str, int]]:
    """Load calibration data for an arm.

    Reads a JSON calibration file and extracts servo IDs and homing offsets
    for each joint. The calibration file should be located at
    {calibration_dir}/{arm_id}.json

    Args:
        calibration_dir: Directory containing calibration files
        arm_id: Arm identifier (filename without .json extension)

    Returns:
        Dictionary mapping joint names to {'id': int, 'offset': int}
        All joints use offset=2048 (servo center point) to shift the zero-point from
        tick 0 to tick 2048 (center), making tick 2048 → 0 radians as expected for servos.
        Example: {'shoulder_pan': {'id': 1, 'offset': 2048}, 'gripper': {'id': 6, 'offset': 2048}}

    Raises:
        FileNotFoundError: If calibration file doesn't exist
        json.JSONDecodeError: If JSON is malformed
        KeyError: If required fields ('id' or 'homing_offset') are missing
        ValueError: If data types are invalid (non-integer values)

    Example:
        >>> calib = load_arm_calibration('/path/to/calibrations', 'gix-follower1')
        >>> print(calib['shoulder_pan'])
        {'id': 1, 'offset': 2048}  # Center point offset
        >>> print(calib['gripper'])
        {'id': 6, 'offset': 2048}  # Same as other joints
    """
    # Construct calibration file path
    calibration_file = os.path.join(calibration_dir, f'{arm_id}.json')

    # Check if file exists
    if not os.path.exists(calibration_file):
        raise FileNotFoundError(
            f"Calibration file not found: {calibration_file}\n"
            f"Expected location: {calibration_dir}/{arm_id}.json"
        )

    # Load and parse JSON
    try:
        with open(calibration_file, 'r') as f:
            raw_calibration = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Failed to parse calibration file {calibration_file}: {str(e)}",
            e.doc,
            e.pos
        )

    # Extract ID and offset for each joint
    calibration = {}
    for joint_name, joint_data in raw_calibration.items():
        # Validate required fields exist
        if 'id' not in joint_data:
            raise KeyError(
                f"Missing required field 'id' for joint '{joint_name}' "
                f"in calibration file {calibration_file}"
            )
        if 'homing_offset' not in joint_data:
            raise KeyError(
                f"Missing required field 'homing_offset' for joint '{joint_name}' "
                f"in calibration file {calibration_file}"
            )

        # Validate data types
        try:
            servo_id = int(joint_data['id'])
            homing_offset = int(joint_data['homing_offset'])
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid data type for joint '{joint_name}' in {calibration_file}. "
                f"Expected integers for 'id' and 'homing_offset': {str(e)}"
            )

        # Store extracted values with offset=2048 for all joints
        #
        # The feetech_ros2_driver's to_radians() doesn't subtract a center point:
        #   tick 0 → 0 radians, tick 2048 → π radians (180°), tick 4095 → 2π radians (360°)
        #
        # Setting offset=2048 shifts the zero-point to center: to_radians(Present_Position - 2048)
        #   tick 2048 → 0 radians ✓ (center position reads as 0°)
        #   tick 0 → -π radians ✓ (minimum position)
        #   tick 4095 → +π radians ✓ (maximum position)
        #
        # This works for all joints including gripper, despite gripper having:
        #   - Different LeRobot normalization mode (RANGE_0_100 vs RANGE_M100_100)
        #   - Asymmetric URDF limits (-0.17 to +1.75 rad)
        #   - Servo homing_offset written to hardware registers
        # The key is that offset=2048 centers the tick range regardless of these differences.
        calibration[joint_name] = {
            'id': servo_id,
            'offset': 2048  # Universal center point for all joints
        }

    return calibration
