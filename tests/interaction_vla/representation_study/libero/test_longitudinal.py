import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from interaction_vla.representation_study.libero.latents import _tree_sha256
from interaction_vla.representation_study.libero.longitudinal import (
    CONDITION_SPECS,
    load_longitudinal_plan,
    plan_longitudinal_conditions,
    validate_runtime_bindings,
)


def _checkpoint(root: Path, stage: str, step: int) -> Path:
    if stage == "pretrained":
        path = root / "stages" / stage / "checkpoint"
    else:
        path = root / "stages" / stage / "run" / "checkpoints" / f"{step:06d}" / "pretrained_model"
        state = path.parent / "training_state"
        state.mkdir(parents=True)
        (state / "training_step.json").write_text(
            json.dumps({"step": step}), encoding="utf-8"
        )
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(str(step).encode())
    if stage != "pretrained":
        total_steps = {"sft_25": 16070, "sft_50": 32650, "sft_100": 66470}[stage]
        (path / "train_config.json").write_text(
            json.dumps(
                {
                    "dataset": {
                        "repo_id": "lerobot/libero",
                        "revision": "r" * 40,
                        "episodes": [stage],
                    },
                    "seed": 7,
                    "batch_size": 8,
                    "steps": total_steps,
                }
            ),
            encoding="utf-8",
        )
    return path


def _complete_stages(root: Path) -> None:
    by_stage: dict[str, list[int]] = {}
    for spec in CONDITION_SPECS:
        by_stage.setdefault(spec.stage, []).append(spec.step)
    for stage, steps in by_stage.items():
        paths = [_checkpoint(root, stage, step) for step in steps]
        final = paths[-1]
        manifest = {
            "stage": stage,
            "status": "complete",
            "checkpoint": str(final),
            "checkpoint_sha256": _tree_sha256(final),
        }
        if stage != "pretrained":
            manifest.update(
                {
                    "dataset_repo_id": "lerobot/libero",
                    "dataset_revision": "r" * 40,
                    "episode_indices": [stage],
                    "seed": 7,
                    "training_steps": max(steps),
                }
            )
        manifest_path = root / "stages" / stage / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_longitudinal_plan_binds_saved_steps_and_matched_update_contrasts(
    tmp_path: Path,
) -> None:
    _complete_stages(tmp_path)
    report = plan_longitudinal_conditions(SimpleNamespace(output_dir=tmp_path))

    assert report["passed"] is True
    assert [row["condition"] for row in report["conditions"]] == [
        spec.condition for spec in CONDITION_SPECS
    ]
    assert report["contrasts"]["matched_update_16k"] == [
        "d25_u16070",
        "d50_u16324",
        "d100_u16617",
    ]
    assert report["contrasts"]["matched_update_32k"] == [
        "d50_u32650",
        "d100_u33234",
    ]
    assert all(len(row["checkpoint_sha256"]) == 64 for row in report["conditions"])


def test_longitudinal_plan_rejects_mislabeled_training_step(tmp_path: Path) -> None:
    _complete_stages(tmp_path)
    state = (
        tmp_path
        / "stages/sft_50/run/checkpoints/016324/training_state/training_step.json"
    )
    state.write_text(json.dumps({"step": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="training step"):
        plan_longitudinal_conditions(SimpleNamespace(output_dir=tmp_path))


def test_longitudinal_plan_rejects_foreign_same_step_checkpoint(tmp_path: Path) -> None:
    _complete_stages(tmp_path)
    foreign = (
        tmp_path
        / "stages/sft_50/run/checkpoints/016324/pretrained_model/train_config.json"
    )
    foreign.write_text(json.dumps({"stage": "foreign"}), encoding="utf-8")

    with pytest.raises(ValueError, match="training config"):
        plan_longitudinal_conditions(SimpleNamespace(output_dir=tmp_path))


def test_load_longitudinal_plan_rejects_truncated_condition_grid(tmp_path: Path) -> None:
    path = tmp_path / "protocol_v3/conditions/manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "libero_smolvla_longitudinal_plan_v3",
                "passed": True,
                "conditions": [{"condition": "pretrained"}],
                "contrasts": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="condition grid"):
        load_longitudinal_plan(SimpleNamespace(output_dir=tmp_path))


def test_runtime_binding_gate_rejects_cross_environment_latents() -> None:
    reports = [
        {
            "condition": "pretrained",
            "runtime_fingerprint_sha256": "a" * 64,
            "state_bank_sha256": "b" * 64,
            "implementation_sha256": "c" * 64,
        },
        {
            "condition": "d25_u16070",
            "runtime_fingerprint_sha256": "d" * 64,
            "state_bank_sha256": "b" * 64,
            "implementation_sha256": "c" * 64,
        },
    ]

    gate = validate_runtime_bindings(reports)
    assert gate["passed"] is False
    assert gate["runtime_fingerprints"] == 2


def test_runtime_binding_gate_rejects_missing_hashes() -> None:
    gate = validate_runtime_bindings(
        [{"condition": "pretrained", "runtime_fingerprint_sha256": None}]
    )
    assert gate["passed"] is False
    assert gate["invalid_bindings"] == 3
