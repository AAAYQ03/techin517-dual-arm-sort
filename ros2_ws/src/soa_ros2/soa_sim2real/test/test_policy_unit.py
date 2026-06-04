"""Unit tests for the soa_sim2real package.

Run with::

    python3 -m pytest src/soa_ros2/soa_sim2real/test/test_policy_unit.py

These tests exercise the observation assembler and policy runner without
requiring a running ROS 2 graph or robot.
"""

import os

import numpy as np
import pytest

from soa_sim2real.joint_order import (
    ACTION_DIM,
    DEFAULT_JOINT_POS,
    GRIPPER_CLOSE_CMD,
    GRIPPER_OPEN_CMD,
    OBS_DIM,
)
from soa_sim2real.observation_assembler import ObservationAssembler


def _make_oa(history_newest_first: bool = False,
             decode_feetech_velocity: bool = False) -> ObservationAssembler:
    # Default to decode-OFF here because most of these tests pass synthetic jv
    # values (e.g. 1.0, 2.0, 3.0 rad/s) that don't round-trip through the
    # 4096-tick quantization. The velocity-decode behavior has its own tests.
    return ObservationAssembler(
        DEFAULT_JOINT_POS,
        history_newest_first=history_newest_first,
        decode_feetech_velocity=decode_feetech_velocity)


def test_obs_dim_and_dtype():
    oa = _make_oa()
    oa.seed(np.array(DEFAULT_JOINT_POS, dtype=np.float32),
            np.zeros(6, dtype=np.float32))
    obs = oa.build_obs(np.zeros(3), np.zeros(3), np.zeros(3))
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


def test_history_oldest_first():
    """LOAD-BEARING: if this fails, set history_newest_first=true ROS param."""
    oa = _make_oa(history_newest_first=False)
    default = np.array(DEFAULT_JOINT_POS, dtype=np.float32)
    oa.seed(default, np.zeros(6, dtype=np.float32))   # all rel-history zero
    # Push three sentinels; each push moves the deque one step forward.
    for v in (1.0, 2.0, 3.0):
        oa.push(
            default + v,
            np.full(6, v, dtype=np.float32),
            np.full(6, v, dtype=np.float32),
        )
    obs = oa.build_obs(np.zeros(3), np.zeros(3), np.zeros(3))

    # joint_pos_rel block 0:18 -> oldest (1.0) first, newest (3.0) last
    assert np.allclose(obs[0:6], 1.0)
    assert np.allclose(obs[6:12], 2.0)
    assert np.allclose(obs[12:18], 3.0)

    # joint_vel_rel block 18:36 follows the same ordering
    assert np.allclose(obs[18:24], 1.0)
    assert np.allclose(obs[24:30], 2.0)
    assert np.allclose(obs[30:36], 3.0)

    # actions block 45:63 also oldest-first
    assert np.allclose(obs[45:51], 1.0)
    assert np.allclose(obs[51:57], 2.0)
    assert np.allclose(obs[57:63], 3.0)


def test_history_newest_first_flag_flips_layout():
    oa = _make_oa(history_newest_first=True)
    default = np.array(DEFAULT_JOINT_POS, dtype=np.float32)
    oa.seed(default, np.zeros(6, dtype=np.float32))
    for v in (1.0, 2.0, 3.0):
        oa.push(default + v, np.full(6, v, dtype=np.float32),
                np.full(6, v, dtype=np.float32))
    obs = oa.build_obs(np.zeros(3), np.zeros(3), np.zeros(3))
    # With newest_first, freshest sentinel sits at index 0
    assert np.allclose(obs[0:6], 3.0)
    assert np.allclose(obs[12:18], 1.0)


def test_term_layout_offsets():
    oa = _make_oa()
    oa.seed(np.array(DEFAULT_JOINT_POS, dtype=np.float32),
            np.zeros(6, dtype=np.float32))
    ee = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    obj = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    tgt = np.array([0.7, 0.8, 0.9], dtype=np.float32)
    obs = oa.build_obs(ee, obj, tgt)
    assert np.allclose(obs[36:39], ee)
    assert np.allclose(obs[39:42], obj)
    assert np.allclose(obs[42:45], tgt)


def test_action_decode_gripper_threshold():
    raw_pos = np.array([0, 0, 0, 0, 0, +0.001], dtype=np.float32)
    raw_zero = np.array([0, 0, 0, 0, 0, 0.0], dtype=np.float32)
    raw_neg = np.array([0, 0, 0, 0, 0, -0.001], dtype=np.float32)
    decode = lambda r: GRIPPER_OPEN_CMD if r[5] >= 0.0 else GRIPPER_CLOSE_CMD
    assert decode(raw_pos) == GRIPPER_OPEN_CMD
    assert decode(raw_zero) == GRIPPER_OPEN_CMD
    assert decode(raw_neg) == GRIPPER_CLOSE_CMD


def test_seed_requires_six_joints():
    oa = _make_oa()
    with pytest.raises(Exception):
        oa.build_obs(np.zeros(3), np.zeros(3), np.zeros(3))  # not seeded


_TICK = float(2.0 * np.pi / 4096.0)  # one STS3215 position-tick in radians


def test_feetech_decode_idempotent_on_good_values():
    """Real signed arm velocities (in ±50 rad/s) must pass through untouched."""
    from soa_sim2real.observation_assembler import feetech_decode_signed_velocity
    good = np.array([0.0, _TICK, -_TICK, 1.5, -1.5, 5.0, -5.0, 50.0, -50.0],
                    dtype=np.float32)
    out = feetech_decode_signed_velocity(good)
    np.testing.assert_allclose(out, good, atol=1e-6)


def test_feetech_decode_inverts_bug():
    """Buggy driver outputs with bit 15 set must decode to the signed value."""
    from soa_sim2real.observation_assembler import feetech_decode_signed_velocity
    # (raw_uint16, signed_value): how the buggy driver publishes each, and what
    # we expect after decoding.
    cases = [
        (0x0001, +1),       # smallest positive tick — bit 15 clear
        (0x7FFF, +32767),   # largest positive — bit 15 clear
        (0x8000, 0),        # "negative zero" — bit 15 set, magnitude 0
        (0x8001, -1),       # one tick negative
        (0xFFFF, -32767),   # largest negative
    ]
    buggy = np.array([raw * _TICK for raw, _ in cases], dtype=np.float32)
    expected = np.array([signed * _TICK for _, signed in cases], dtype=np.float32)
    out = feetech_decode_signed_velocity(buggy)
    np.testing.assert_allclose(out, expected, atol=1e-4)


def test_observation_assembler_decodes_velocity_on_push():
    """seed() and push() must apply the decode by default."""
    oa = ObservationAssembler(DEFAULT_JOINT_POS)  # default decode_feetech_velocity=True
    # raw = 0x8001 → buggy publish ≈ 50.267 rad/s → decode = -1 tick ≈ -0.001534
    buggy = np.array([0x8001 * _TICK] * 6, dtype=np.float32)
    expected_jv_rel = np.array([-_TICK] * 6, dtype=np.float32)
    oa.seed(np.array(DEFAULT_JOINT_POS, dtype=np.float32), buggy)
    obs = oa.build_obs(np.zeros(3), np.zeros(3), np.zeros(3))
    # joint_vel_rel block is obs[18:36]; with all 3 history slots holding the same
    # seeded value, slot 0 (obs[18:24]) should equal the decoded value.
    np.testing.assert_allclose(obs[18:24], expected_jv_rel, atol=1e-4)


def test_observation_assembler_decode_opt_out():
    """If the caller passes decode_feetech_velocity=False, jv is forwarded verbatim."""
    oa = ObservationAssembler(DEFAULT_JOINT_POS, decode_feetech_velocity=False)
    buggy = np.array([0x8001 * _TICK] * 6, dtype=np.float32)
    oa.seed(np.array(DEFAULT_JOINT_POS, dtype=np.float32), buggy)
    obs = oa.build_obs(np.zeros(3), np.zeros(3), np.zeros(3))
    np.testing.assert_allclose(obs[18:24], buggy, atol=1e-4)


def test_policy_smoke():
    """Load policy.pt from the installed package share and run one inference."""
    try:
        from ament_index_python.packages import get_package_share_directory
        from soa_sim2real.policy_runner import PolicyRunner
    except ImportError:
        pytest.skip('ament_index or torch unavailable')

    try:
        pkg_share = get_package_share_directory('soa_sim2real')
    except Exception:
        pytest.skip('soa_sim2real not installed yet')

    model_path = os.path.join(pkg_share, 'models', 'policy.pt')
    if not os.path.isfile(model_path):
        pytest.skip(f'policy.pt not found at {model_path}')

    runner = PolicyRunner(model_path)
    oa = _make_oa()
    oa.seed(np.array(DEFAULT_JOINT_POS, dtype=np.float32),
            np.zeros(6, dtype=np.float32))
    obs = oa.build_obs(
        np.array([0.15, 0.0, 0.10], dtype=np.float32),  # ee
        np.array([0.20, 0.0, 0.05], dtype=np.float32),  # cube
        np.array([0.20, 0.0, 0.15], dtype=np.float32),  # target
    )
    action = runner.infer(obs)
    assert action.shape == (ACTION_DIM,)
    assert action.dtype == np.float32
    assert np.all(np.isfinite(action))
    assert np.max(np.abs(action)) < 50.0
