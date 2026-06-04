"""Assembles the 63-dim observation vector matching env.yaml term order.

Term layout (env.yaml:509-586):
    joint_pos_rel(18) | joint_vel_rel(18) | ee_pos(3) | obj_pos(3)
                      | tgt_obj_pos(3)    | actions(18)

Each history block is flattened oldest-first by default, matching how
IsaacLab's ``flatten_history_dim`` concatenates ``CircularBuffer`` slots
(``torch.cat(deque, dim=-1)`` -> oldest at index 0). Pass
``history_newest_first=True`` to flip if the deployed policy turns out to
expect the reverse convention; the unit test ``test_history_oldest_first``
catches a mismatch deterministically.

The ``actions`` term feeds back the raw, un-clamped policy outputs (training
uses ``isaaclab.envs.mdp.observations.last_action``).
"""

from collections import deque

import numpy as np

from soa_sim2real.joint_order import HISTORY_LEN, OBS_DIM


_FEETECH_TICK_TO_RAD = float(2.0 * np.pi / 4096.0)


# Boundary values for the buggy driver's output:
#   raw=0x0000  →  0           rad/s   (sign-bit clear, magnitude 0)
#   raw=0x7FFF  →  ~50.264     rad/s   (largest sign-bit-clear value)
#   raw=0x8000  →  ~50.265     rad/s   (sign-bit set, magnitude 0 = "negative zero")
#   raw=0xFFFF  →  ~100.529    rad/s   (largest possible buggy publish)
# Inputs in [50.265, 100.531) are exactly the buggy outputs with bit 15 set
# and need the sign-magnitude re-decode. Everything else passes through.
_FEETECH_SIGN_BIT_LOW_RAD = float(0x8000) * _FEETECH_TICK_TO_RAD
_FEETECH_FULL_RANGE_HIGH_RAD = float(0x10000) * _FEETECH_TICK_TO_RAD


def feetech_decode_signed_velocity(buggy_rad_s: np.ndarray) -> np.ndarray:
    """Recover signed STS3215 Present Speed from feetech_ros2_driver's
    unsigned-decoded radians/second value.
    """
    buggy = np.asarray(buggy_rad_s, dtype=np.float64)
    needs_decode = (buggy >= _FEETECH_SIGN_BIT_LOW_RAD) & (buggy < _FEETECH_FULL_RANGE_HIGH_RAD)
    ticks = np.rint(buggy / _FEETECH_TICK_TO_RAD).astype(np.int32)
    decoded_rad = (-(ticks & 0x7FFF)).astype(np.float64) * _FEETECH_TICK_TO_RAD
    return np.where(needs_decode, decoded_rad, buggy).astype(np.float32)


class ObservationAssembler:
    N_J = 6  # 5 arm + 1 gripper

    def __init__(self, default_joint_pos, history_newest_first: bool = False,
                 decode_feetech_velocity: bool = True):
        self._default = np.asarray(default_joint_pos, dtype=np.float32)
        if self._default.shape != (self.N_J,):
            raise ValueError(
                f'default_joint_pos must have shape ({self.N_J},), '
                f'got {self._default.shape}'
            )
        self._newest_first = bool(history_newest_first)
        self._decode_feetech_velocity = bool(decode_feetech_velocity)
        self._jp = deque(maxlen=HISTORY_LEN)
        self._jv = deque(maxlen=HISTORY_LEN)
        self._act = deque(maxlen=HISTORY_LEN)

    def seed(self, jp: np.ndarray, jv: np.ndarray) -> None:
        """Fill all history slots with the current joint state, zero past actions.

        Mirrors IsaacLab's ``ObservationManager`` reset behavior, where the
        per-term ``CircularBuffer`` is seeded with the current observation
        and the action history is zeroed.
        """
        if self._decode_feetech_velocity:
            jv = feetech_decode_signed_velocity(jv)
        jp_rel = (np.asarray(jp, dtype=np.float32) - self._default)
        jv_rel = np.asarray(jv, dtype=np.float32)
        zero_act = np.zeros(self.N_J, dtype=np.float32)
        self._jp.clear()
        self._jv.clear()
        self._act.clear()
        for _ in range(HISTORY_LEN):
            self._jp.append(jp_rel.copy())
            self._jv.append(jv_rel.copy())
            self._act.append(zero_act.copy())

    def push(self, jp: np.ndarray, jv: np.ndarray, last_raw_action: np.ndarray) -> None:
        if self._decode_feetech_velocity:
            jv = feetech_decode_signed_velocity(jv)

        # Add joint positions (relative), velocities, and last action to history queues
        jp_rel = np.asarray(jp, dtype=np.float32) - self._default
        jv_rel = np.asarray(jv, dtype=np.float32)
        act = np.asarray(last_raw_action, dtype=np.float32)

        self._jp.append(jp_rel)
        self._jv.append(jv_rel)
        self._act.append(act)

    def build_obs(
        self,
        ee: np.ndarray,
        obj: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        if len(self._jp) != HISTORY_LEN:
            raise RuntimeError('ObservationAssembler.seed() must be called first')

        def flat(dq):
            items = list(dq)
            if self._newest_first:
                items.reverse()
            return np.concatenate(items, axis=0)

        obs = np.concatenate([
            flat(self._jp),                                       # joint_pos_rel (18)
            flat(self._jv),                                       # joint_vel_rel (18)
            np.asarray(ee, dtype=np.float32).reshape(3),          # ee_pos (3)
            np.asarray(obj, dtype=np.float32).reshape(3),         # obj_pos (3)
            np.asarray(target, dtype=np.float32).reshape(3),      # tgt_obj_pos (3)
            flat(self._act),                                      # actions (18)
        ], axis=0).astype(np.float32, copy=False)
        if obs.shape != (OBS_DIM,):
            raise RuntimeError(f'assembled obs has shape {obs.shape}, expected ({OBS_DIM},)')
        return obs
