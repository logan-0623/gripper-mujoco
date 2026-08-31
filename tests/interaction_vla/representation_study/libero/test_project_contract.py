from pathlib import Path

import yaml


def test_ccfa_registry_preserves_evidence_and_stops_before_rl() -> None:
    config = yaml.safe_load(Path("ccfa.yaml").read_text(encoding="utf-8"))
    registry = {row["id"]: row for row in config["experiment_registry"]}
    assert registry["ACT_GRAPH_V2"]["status"] == "formal_evidence"
    assert registry["REFLECT_GRAPH_PRETRAIN"]["status"] == "pilot_complete"
    assert registry["RECOVERY_RL_V2_CALIBRATION"]["status"] == "failed_gate"
    state_bank = registry["LIBERO_STATE_BANK"]
    assert state_bank["status"] == "formal_evidence"
    assert state_bank["evidence"]["audit_passed"] is True
    assert state_bank["evidence"]["manual_timeline_review_passed"] is True
    assert Path(state_bank["evidence"]["artifact"]).is_dir()
    longitudinal = registry["SMOLVLA_LONGITUDINAL"]
    assert longitudinal["status"] == "formal_evidence"
    formal_artifact = Path(longitudinal["evidence"]["artifact"])
    assert (formal_artifact / "protocol_v3/probes/crossfit_v1/report.json").is_file()
    assert registry["SMOLVLA_FACTOR_INTERVENTION"]["status"] == "implementation_only"
    assert registry["SMOLVLA_FUNCTIONAL_RECRUITMENT"]["status"] == "not_run"
    assert registry["SMOLVLA_PAIRED_CLOSED_LOOP"]["status"] == "not_run"
    assert config["scientific_contract"]["rl_scope"] == "conditional_future_work"
    graph = {row["id"]: row for row in config["dependency_graph"]}
    assert graph["P0_G"]["status"] == "formal_evidence"
    assert graph["RL_EXTENSION"]["status"] == "frozen_not_blocking"
    assert graph["RL_EXTENSION"]["execution_allowed"] is False


def test_linux_requirements_and_readme_expose_the_formal_libero_path() -> None:
    requirements = Path("requirements-lerobot-linux-cuda.txt").read_text(encoding="utf-8")
    assert "h5py" in requirements
    assert "smolvla" in requirements
    assert "libero" in requirements
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "libero state-bank collect" in readme
    assert "libero stages snapshot" in readme
    assert "libero probes run" in readme
    assert "libero longitudinal plan" in readme
    assert "当前不要运行或调优 PPO/SAC" in readme
