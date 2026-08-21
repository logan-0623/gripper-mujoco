from __future__ import annotations

import numpy as np

from interaction_vla.config import PhysicsConfig
from interaction_vla.physics_env import FrankaContactEnv
from interaction_vla.physics_expert import PhysicsScriptedExpert
from interaction_vla.representation_study.rl.distributions import RecoveryCase
from interaction_vla.representation_study.rl.environment import prepare_interaction_start
from interaction_vla.representation_study.rl.oracle_state import (
    ORACLE_SLICES,
    CompactOracleStateCodec,
    OracleNormalization,
)


def _yaw(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def _scene() -> dict[str, object]:
    return {
        "gripper_position": np.asarray((0.35, -0.10, 0.45)),
        "gripper_rotation": _yaw(0.20),
        "target_position": np.asarray((0.40, -0.06, 0.30)),
        "target_rotation": _yaw(-0.30),
        "receptacle_position": np.asarray((0.62, 0.14, 0.24)),
        "receptacle_rotation": _yaw(0.05),
        "distractor_positions": np.asarray(((0.48, -0.02, 0.30),)),
        "gripper_open_fraction": 0.25,
        "bilateral_target_contact": True,
        "stable_target_grasp": False,
        "target_support": False,
        "phase": "grasp",
        "intervention_kind": "grasp_offset",
        "severity": 0.75,
        "progress": 0.10,
    }


def _globally_transform(
    scene: dict[str, object], *, yaw: float, translation: tuple[float, float, float]
) -> dict[str, object]:
    transformed = dict(scene)
    rotation = _yaw(yaw)
    offset = np.asarray(translation)
    for name in ("gripper", "target", "receptacle"):
        transformed[f"{name}_position"] = (
            rotation @ np.asarray(scene[f"{name}_position"]) + offset
        )
        transformed[f"{name}_rotation"] = rotation @ np.asarray(
            scene[f"{name}_rotation"]
        )
    distractors = np.asarray(scene["distractor_positions"])
    transformed["distractor_positions"] = distractors @ rotation.T + offset
    return transformed


def test_oracle_state_is_finite_width_36() -> None:
    encoded = CompactOracleStateCodec().encode(**_scene())
    assert encoded.shape == (36,)
    assert encoded.dtype == np.float32
    assert np.isfinite(encoded).all()


def test_oracle_state_is_global_frame_invariant() -> None:
    codec = CompactOracleStateCodec()
    original = codec.encode(**_scene())
    transformed = codec.encode(
        **_globally_transform(
            _scene(), yaw=0.7, translation=(1.0, -0.3, 0.0)
        )
    )
    np.testing.assert_allclose(original, transformed, atol=1e-5, rtol=1e-5)


def test_oracle_state_uses_registered_one_hot_slices() -> None:
    encoded = CompactOracleStateCodec().encode(**_scene())
    phase = encoded[ORACLE_SLICES["phase"]]
    intervention = encoded[ORACLE_SLICES["intervention"]]
    assert phase.sum() == 1.0
    assert phase.tolist() == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert intervention.sum() == 1.0
    assert intervention.tolist() == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_oracle_normalization_is_json_serializable() -> None:
    payload = OracleNormalization().to_json()
    assert payload["schema_version"] == "compact_oracle_normalization_v1"
    assert payload["oracle_state_width"] == 36
    assert payload["gripper_target_translation_scale"] > 0.0


def test_oracle_codec_encodes_a_real_runtime_start() -> None:
    env = FrankaContactEnv(
        max_steps=180,
        physics=PhysicsConfig(settle_steps=100),
        workspace_low=(0.25, -0.35, 0.23),
        workspace_high=(0.78, 0.35, 0.75),
        crowded_anchor_min_distance=0.055,
        crowded_anchor_max_distance=0.075,
    )
    case = RecoveryCase(
        case_id="training:11:nominal:0:nominal",
        partition="training",
        family="nominal",
        source_seed=11,
        variant_id=0,
        object_count=2,
        layout="normal",
        phase="approach",
        intervention_kind="nominal",
        severity=0.0,
    )
    prepared = prepare_interaction_start(
        env, PhysicsScriptedExpert(env.physics), case=case
    )
    encoded = CompactOracleStateCodec().encode_runtime(env, prepared, case)
    assert encoded.shape == (36,)
    assert np.isfinite(encoded).all()
    assert encoded[ORACLE_SLICES["phase"]].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
