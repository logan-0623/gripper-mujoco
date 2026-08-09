import numpy as np
import pytest

from interaction_vla.config import PhysicsConfig
from interaction_vla.lerobot_bridge.codecs import (
    EndEffectorStateCodec,
    LocalCartesianActionCodec,
)
from interaction_vla.physics_env import FrankaContactEnv


def test_rotation_6d_round_trip_is_right_handed() -> None:
    rotation = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    encoded = EndEffectorStateCodec.encode_rotation(rotation)
    decoded = EndEffectorStateCodec.decode_rotation(encoded)

    np.testing.assert_array_equal(encoded, (0.0, 1.0, 0.0, -1.0, 0.0, 0.0))
    np.testing.assert_allclose(decoded, rotation, atol=1e-7)
    np.testing.assert_allclose(decoded.T @ decoded, np.eye(3), atol=1e-7)
    assert np.linalg.det(decoded) == pytest.approx(1.0)


def test_local_action_round_trip_preserves_controller_command() -> None:
    rotation = EndEffectorStateCodec.decode_rotation(
        np.asarray((0.0, 1.0, 0.0, -1.0, 0.0, 0.0), dtype=np.float32)
    )
    controller_action = np.asarray(
        (0.2, -0.4, 0.1, 0.3, -0.2, 0.5, 0.0), dtype=np.float32
    )

    stored = LocalCartesianActionCodec.encode(controller_action, rotation)
    restored = LocalCartesianActionCodec.decode(stored, rotation)

    np.testing.assert_allclose(restored, controller_action, atol=1e-6)
    assert stored[6] == 0.0


def test_public_snapshot_and_proprioception_encode_finite_10d_state() -> None:
    env = FrankaContactEnv(physics=PhysicsConfig(settle_steps=5))
    snapshot = env.reset(seed=11, object_count=2)

    state = EndEffectorStateCodec.encode_snapshot(snapshot, env.proprioception())

    assert state.shape == (10,)
    assert state.dtype == np.float32
    assert np.isfinite(state).all()
    assert 0.0 <= state[-1] <= 1.0
