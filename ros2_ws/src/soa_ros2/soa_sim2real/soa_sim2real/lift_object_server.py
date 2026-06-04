#!/usr/bin/env python3
"""LiftObject action server for the SOA arm.

Closed-loop runner of the Isaac Lab-trained PyTorch JIT lift policy. One
ROS 2 node owns:

* the policy (TorchScript with EmpiricalNormalization baked in),
* the 63-dim observation assembler with 3-step history buffers,
* the 50 Hz inference loop driven by a sync ActionServer execute callback.

Observations come from ``/follower/joint_states`` (joint pos/vel) and tf2
(``follower/base_link`` -> ``follower/gripper_frame_link`` for the EE,
``follower/base_link`` -> ``aruco_cube`` for the object). Actions are
published as ``std_msgs/Float64MultiArray`` to the existing
forward-command controllers.
"""

import math
import threading
import time
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time

import tf2_ros
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from soa_interfaces.action import LiftObject

from soa_sim2real.joint_order import (
    ACTION_DIM,
    ALL_JOINT_NAMES,
    ARM_ACTION_SCALE,
    ARM_JOINT_NAMES,
    CUBE_FRAME,
    DEFAULT_JOINT_POS,
    EE_FRAME,
    GRIPPER_CLOSE_CMD,
    GRIPPER_JOINT_NAME,
    GRIPPER_OPEN_CMD,
    ROBOT_BASE_FRAME,
)
from soa_sim2real.observation_assembler import ObservationAssembler
from soa_sim2real.policy_runner import PolicyRunner
from soa_sim2real.urdf_limits import fetch_joint_limits


def _stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class LiftObjectServer(Node):

    def __init__(self):
        super().__init__('lift_object_server')

        # ----- Parameters -----
        self.declare_parameter('model_path', '')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('inference_rate_hz', 50.0)
        self.declare_parameter('goal_tolerance_m', 0.01)
        self.declare_parameter('cube_lost_timeout_s', 1.5)
        self.declare_parameter('cube_tf_max_age_s', 0.5)
        self.declare_parameter('joint_state_age_threshold_ms', 100.0)
        self.declare_parameter('pre_action_pose_tolerance_rad', -1.0)
        self.declare_parameter('max_action_duration_s', 8.0)
        self.declare_parameter('action_alpha', 1.0)
        self.declare_parameter('use_finite_difference_vel', False)
        self.declare_parameter('history_newest_first', False)
        self.declare_parameter('joint_states_topic', '/follower/joint_states')
        self.declare_parameter('arm_command_topic', '/follower/arm_fwd_controller/commands')
        self.declare_parameter('gripper_command_topic',
                               '/follower/gripper_fwd_controller/commands')
        self.declare_parameter('robot_description_topic', '/follower/robot_description')
        self.declare_parameter('base_frame', ROBOT_BASE_FRAME)
        self.declare_parameter('ee_frame', EE_FRAME)
        self.declare_parameter('cube_frame', CUBE_FRAME)
        self.declare_parameter('robot_description_timeout_s', 5.0)

        self._inference_rate_hz = float(self.get_parameter('inference_rate_hz').value)
        self._goal_tol = float(self.get_parameter('goal_tolerance_m').value)
        self._cube_lost_timeout = float(self.get_parameter('cube_lost_timeout_s').value)
        self._cube_tf_max_age = float(self.get_parameter('cube_tf_max_age_s').value)
        self._js_age_threshold = float(
            self.get_parameter('joint_state_age_threshold_ms').value) * 1e-3
        self._pre_pose_tol = float(self.get_parameter('pre_action_pose_tolerance_rad').value)
        self._max_duration = float(self.get_parameter('max_action_duration_s').value)
        self._action_alpha = float(self.get_parameter('action_alpha').value)
        self._use_fd_vel = bool(self.get_parameter('use_finite_difference_vel').value)

        self._base_frame = str(self.get_parameter('base_frame').value)
        self._ee_frame = str(self.get_parameter('ee_frame').value)
        self._cube_frame = str(self.get_parameter('cube_frame').value)

        # ----- Callback group / state -----
        self._cb_group = ReentrantCallbackGroup()
        self._js_lock = threading.Lock()
        self._latest_js: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
        self._prev_js: Optional[Tuple[np.ndarray, float]] = None
        self._active = False

        # ----- I/O -----
        self.create_subscription(
            JointState,
            str(self.get_parameter('joint_states_topic').value),
            self._joint_state_cb,
            10,
            callback_group=self._cb_group,
        )
        self._arm_pub = self.create_publisher(
            Float64MultiArray, str(self.get_parameter('arm_command_topic').value), 10)
        self._gripper_pub = self.create_publisher(
            Float64MultiArray, str(self.get_parameter('gripper_command_topic').value), 10)
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ----- Joint limits from URDF (pumps spin_once internally) -----
        self._joint_limits = fetch_joint_limits(
            self,
            joint_names=ALL_JOINT_NAMES,
            topic=str(self.get_parameter('robot_description_topic').value),
            timeout_s=float(self.get_parameter('robot_description_timeout_s').value),
        )
        self.get_logger().info(
            'Joint limits from URDF: '
            + ', '.join(f'{n}=[{lo:.3f},{hi:.3f}]'
                        for n, (lo, hi) in self._joint_limits.items())
        )

        # ----- Policy -----
        model_path = str(self.get_parameter('model_path').value)
        if not model_path:
            raise RuntimeError(
                'model_path parameter is empty (set it via launch or config)')
        self._policy = PolicyRunner(
            model_path,
            device=str(self.get_parameter('device').value),
        )
        self.get_logger().info(f'Policy loaded from {model_path}')

        # ----- Observation assembler -----
        self._obs_assembler = ObservationAssembler(
            DEFAULT_JOINT_POS,
            history_newest_first=bool(
                self.get_parameter('history_newest_first').value),
        )

        # Precomputed per-joint limit arrays (articulation order).
        self._arm_lo = np.array(
            [self._joint_limits[n][0] for n in ARM_JOINT_NAMES], dtype=np.float32)
        self._arm_hi = np.array(
            [self._joint_limits[n][1] for n in ARM_JOINT_NAMES], dtype=np.float32)
        self._grip_lo, self._grip_hi = self._joint_limits[GRIPPER_JOINT_NAME]
        self._arm_default = np.array(DEFAULT_JOINT_POS[:5], dtype=np.float32)

        # ----- Action server -----
        self._action_server = ActionServer(
            self,
            LiftObject,
            'lift_object',
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._cb_group,
        )
        self.get_logger().info('LiftObject action server ready')

    # ------------------------------------------------------------------
    # Subscriber callback
    # ------------------------------------------------------------------
    def _joint_state_cb(self, msg: JointState) -> None:
        try:
            jp, jv = self._remap_joint_state(msg)
        except KeyError as exc:
            self.get_logger().warn(
                f'JointState missing joint(s): {exc}', throttle_duration_sec=2.0)
            return
        stamp = _stamp_to_seconds(msg.header.stamp)
        if stamp <= 0.0:
            stamp = self.get_clock().now().nanoseconds * 1e-9
        with self._js_lock:
            if self._latest_js is not None:
                prev_jp, _, prev_stamp = self._latest_js
                self._prev_js = (prev_jp, prev_stamp)
            self._latest_js = (jp, jv, stamp)

    @staticmethod
    def _remap_joint_state(msg: JointState) -> Tuple[np.ndarray, np.ndarray]:
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        missing = [n for n in ALL_JOINT_NAMES if n not in name_to_idx]
        if missing:
            raise KeyError(', '.join(missing))
        jp = np.zeros(len(ALL_JOINT_NAMES), dtype=np.float32)
        jv = np.zeros(len(ALL_JOINT_NAMES), dtype=np.float32)
        has_vel = len(msg.velocity) >= len(msg.position)
        for k, n in enumerate(ALL_JOINT_NAMES):
            i = name_to_idx[n]
            jp[k] = float(msg.position[i])
            if has_vel:
                jv[k] = float(msg.velocity[i])
        return jp, jv

    # ------------------------------------------------------------------
    # Action callbacks
    # ------------------------------------------------------------------
    def _goal_callback(self, _goal_request) -> GoalResponse:
        if self._active:
            self.get_logger().warn('Rejecting goal: another LiftObject goal is active')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_callback(self, _goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _execute_callback(self, goal_handle):
        self._active = True
        result = LiftObject.Result()
        result.success = False
        t_start = self.get_clock().now()

        try:
            success, abort_msg = self._run(goal_handle, t_start)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Unhandled error in execute_callback: {exc!r}')
            success = False
            abort_msg = f'exception: {exc!r}'
            try:
                if goal_handle.is_active:
                    goal_handle.abort()
            except Exception:
                pass
        finally:
            self._active = False

        result.time_taken = (self.get_clock().now() - t_start).nanoseconds * 1e-9
        result.success = bool(success)
        if success:
            self.get_logger().info(f'LiftObject SUCCESS in {result.time_taken:.2f} s')
        else:
            self.get_logger().warn(
                f'LiftObject FAILED ({abort_msg}) after {result.time_taken:.2f} s')
        return result

    # ------------------------------------------------------------------
    # Core inference loop
    # ------------------------------------------------------------------
    def _run(self, goal_handle, t_start) -> Tuple[bool, str]:
        goal_height = float(goal_handle.request.goal_height)

        # 1. Wait for joint state
        if not self._wait_for_joint_state(timeout_s=1.0):
            goal_handle.abort()
            return False, 'no joint_state received'
        with self._js_lock:
            jp_now, jv_now, _ = self._latest_js  # type: ignore[misc]

        # 2. Pre-action pose check (arm joints only)
        if self._pre_pose_tol > 0.0:
            err = float(np.max(np.abs(jp_now[:5] - self._arm_default)))
            if err > self._pre_pose_tol:
                goal_handle.abort()
                return False, (
                    f'arm too far from default ({err:.3f} > {self._pre_pose_tol:.3f} rad); '
                    'please home the arm first'
                )

        # 3. tf base->EE
        ee_pos = self._lookup_pos(self._base_frame, self._ee_frame)
        if ee_pos is None:
            goal_handle.abort()
            return False, f'tf {self._base_frame} -> {self._ee_frame} lookup failed'

        # 4. tf base->cube (must succeed at goal accept)
        cube_pos_start, _ = self._lookup_pos_with_stamp(
            self._base_frame, self._cube_frame)
        if cube_pos_start is None:
            goal_handle.abort()
            return False, 'cube not visible at start (tf to aruco_cube failed)'

        target = cube_pos_start + np.array([0.0, 0.0, goal_height], dtype=np.float32)
        self.get_logger().info(
            f'goal: lift cube to z+={goal_height:.3f} '
            f'(target={target.tolist()}, cube_start={cube_pos_start.tolist()})'
        )

        # 5. Seed history; first inferred action will be sent on the first tick.
        self._obs_assembler.seed(jp_now, jv_now)
        last_raw_action = np.zeros(ACTION_DIM, dtype=np.float32)
        # Seed the EMA at zero so the first action is alpha * raw rather than the
        # unsmoothed raw. The trained policy emits aggressive open-loop reaches at
        # reset that the implicit sim actuator absorbs but real Feetech servos
        # don't; the zero-seed bounds the first commanded delta from default.
        ema_smoothed = np.zeros(ACTION_DIM, dtype=np.float32)
        last_cube_pos = cube_pos_start
        last_cube_seen_t = self._now_s()

        loop_dt = 1.0 / max(self._inference_rate_hz, 1.0)
        next_tick_ns = self.get_clock().now().nanoseconds
        t_start_ns = t_start.nanoseconds

        while True:
            # Cancellation
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return False, 'canceled'

            now_ns = self.get_clock().now().nanoseconds
            now_s = now_ns * 1e-9
            elapsed_s = (now_ns - t_start_ns) * 1e-9
            if elapsed_s > self._max_duration:
                goal_handle.abort()
                return False, f'timeout ({self._max_duration:.1f}s)'

            # Joint state
            with self._js_lock:
                snap = self._latest_js
                prev_snap = self._prev_js
            if snap is None:
                goal_handle.abort()
                return False, 'lost joint_state'
            jp, jv_msg, js_stamp = snap
            if (now_s - js_stamp) > self._js_age_threshold:
                goal_handle.abort()
                return False, f'stale joint_state ({(now_s - js_stamp) * 1000:.0f} ms)'

            if self._use_fd_vel and prev_snap is not None:
                prev_jp, prev_stamp = prev_snap
                dt = max(js_stamp - prev_stamp, 1e-3)
                jv = ((jp - prev_jp) / dt).astype(np.float32)
            else:
                jv = jv_msg

            # tf lookups
            ee_pos = self._lookup_pos(self._base_frame, self._ee_frame)
            if ee_pos is None:
                ee_pos = np.zeros(3, dtype=np.float32)  # last-resort fallback
            cube_pos, cube_stamp = self._lookup_pos_with_stamp(
                self._base_frame, self._cube_frame)
            if cube_pos is not None:
                last_cube_pos = cube_pos
                last_cube_seen_t = now_s
                cube_age = now_s - cube_stamp
            else:
                cube_age = math.inf
            cube_fresh = cube_age <= self._cube_tf_max_age

            if (now_s - last_cube_seen_t) > self._cube_lost_timeout:
                goal_handle.abort()
                return False, 'cube lost'

            # Build obs and run inference
            self._obs_assembler.push(jp, jv, last_raw_action)
            obs = self._obs_assembler.build_obs(ee_pos, last_cube_pos, target)
            raw = self._policy.infer(obs)

            # Save raw BEFORE post-processing (training feeds back raw outputs).
            last_raw_action = raw.copy()

            if self._action_alpha < 1.0:
                # ema_smoothed is now zero-initialized (see above), so the EMA
                # formula is well-defined on the first tick:
                #   tick 0:  shaped = alpha * raw + (1-alpha) * 0 = alpha * raw
                #   tick n:  shaped = alpha * raw + (1-alpha) * ema_smoothed
                ema_smoothed = (
                    self._action_alpha * raw
                    + (1.0 - self._action_alpha) * ema_smoothed
                )
                shaped = ema_smoothed
            else:
                shaped = raw

            # Decode -> joint position targets in RADIANS (matches Isaac Lab):
            #   arm:     JointPositionAction with scale=0.5, use_default_offset=true
            #            -> target_rad[i] = default_rad[i] + 0.5 * shaped[i]   (i in 0..4)
            #   gripper: BinaryJointPositionAction thresholds the *raw* policy
            #            output (sim feeds raw, never smoothed). open=1.0 rad,
            #            close=-0.2 rad. Smoothing only affects arm targets.
            arm_target = self._arm_default + ARM_ACTION_SCALE * shaped[:5]
            gripper_target = (
                GRIPPER_OPEN_CMD if raw[5] >= 0.0 else GRIPPER_CLOSE_CMD)

            # URDF joint-limit clamp in RADIANS (single safety net). Downstream:
            #   forward_command_controller -> position interface (radians) ->
            #   FeetechHardwareInterface::write converts via from_radians()
            #   and adds the per-joint calibration offset to produce STS3215
            #   ticks (4096 ticks per 2*pi).
            arm_target = np.clip(arm_target, self._arm_lo, self._arm_hi)
            gripper_target = float(np.clip(gripper_target, self._grip_lo, self._grip_hi))

            arm_msg = Float64MultiArray()
            arm_msg.data = arm_target.astype(float).tolist()
            self._arm_pub.publish(arm_msg)
            grip_msg = Float64MultiArray()
            grip_msg.data = [gripper_target]
            self._gripper_pub.publish(grip_msg)

            distance = (
                float(np.linalg.norm(target - last_cube_pos))
                if cube_fresh else math.inf
            )
            fb = LiftObject.Feedback()
            fb.distance_to_goal = distance
            goal_handle.publish_feedback(fb)

            if cube_fresh and distance <= self._goal_tol:
                goal_handle.succeed()
                return True, 'reached goal'

            # Drift-corrected sleep to next tick
            next_tick_ns += int(loop_dt * 1e9)
            sleep_s = (next_tick_ns - self.get_clock().now().nanoseconds) * 1e-9
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                # Loop is running behind; resync to avoid runaway catch-up.
                next_tick_ns = self.get_clock().now().nanoseconds

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _wait_for_joint_state(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._js_lock:
                if self._latest_js is not None:
                    return True
            time.sleep(0.02)
        return False

    def _lookup_pos(self, target_frame: str, source_frame: str) -> Optional[np.ndarray]:
        pos, _ = self._lookup_pos_with_stamp(target_frame, source_frame)
        return pos

    def _lookup_pos_with_stamp(
        self, target_frame: str, source_frame: str
    ) -> Tuple[Optional[np.ndarray], float]:
        try:
            t = self._tf_buffer.lookup_transform(
                target_frame, source_frame, Time(), timeout=Duration(seconds=0.0))
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
                tf2_ros.TransformException) as exc:
            self.get_logger().debug(f'tf lookup {target_frame}<-{source_frame}: {exc}')
            return None, 0.0
        tr = t.transform.translation
        pos = np.array([tr.x, tr.y, tr.z], dtype=np.float32)
        return pos, _stamp_to_seconds(t.header.stamp)


def main(args=None):
    rclpy.init(args=args)
    node = LiftObjectServer()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
