from types import SimpleNamespace as NS

import numpy as np
import pytest

from interaction_vla.representation_study.libero import checkpoint_readouts as cr


def record(index, partition):
    return NS(state_id=f"{partition}-{index}", suite="spatial", task_id=index % 2,
              source_episode_id=f"{partition}-{index // 2}", frame_index=index,
              labels=NS(contact=NS(gripper_target=bool(index % 2)), stable_grasp=bool(index % 2),
                        geometry=NS(gripper_target_distance=float(index), target_goal_distance=float(index))),
              observation=NS(action=[0] * 6 + [float(index % 2)], robot_state=[float(index), 0]))


def test_cka_rotation_constant_and_weighting():
    x = np.random.default_rng(2).normal(size=(12, 4))
    rotation = np.linalg.qr(np.random.default_rng(3).normal(size=(4, 4)))[0]
    assert cr.linear_cka(x, x @ rotation * 3 + 7, np.ones(12)) == pytest.approx(1)
    assert cr.linear_cka(x, np.ones_like(x), np.ones(12)) is None
    with pytest.raises(ValueError, match="weights"):
        cr.linear_cka(x, x, np.zeros(12))
    rows = [record(0, "train"), record(1, "train"), record(1, "train")]
    assert cr.weights(rows) == pytest.approx([.5, .25, .25])


def test_transfer_keeps_source_scaler_and_readout():
    x = np.arange(12, dtype=float)[:, None]
    rows = [record(i, "train") for i in range(12)]
    valid = [record(i, "validation") for i in range(12)]
    result = cr.probe_matrix({"early": (x, x), "late": (-x, -x)}, x[:, 0], x[:, 0], rows, valid, .01)
    scores = {(r["source"], r["destination"]): r["mse"] for r in result["rows"] if r["task"] == "macro"}
    assert scores["early", "early"] < .01
    assert scores["late", "late"] < .01
    assert scores["early", "late"] > 10
    assert cr.probe_matrix({"early": (x, x)}, np.ones(12), np.ones(12), rows, valid, 1)["status"] == "not_estimable"


def test_full_report_and_contract_failures(tmp_path, monkeypatch):
    records = [record(i, p) for p in ("train", "validation") for i in range(8)]
    split = NS(assignments={r.state_id: r.state_id.split("-")[0] for r in records})
    monkeypatch.setattr(cr, "load_state_bank", lambda _: (records, {"audit_passed": True}, None, split))
    monkeypatch.setattr(cr, "file_hash", lambda _: "bank")
    corrupt = {}

    def trace(path):
        name, partition = path.name.split("_")
        ids = np.array([r.state_id for r in records if r.state_id.startswith(partition)])
        values = np.random.default_rng(4).normal(size=(8, 1, 2, 3, 4))
        binding = dict.fromkeys(cr.CONTRACT, "same")
        binding.update(state_bank_sha256="bank", partition=partition, split_group="episode",
                       query_mode="natural_integration", checkpoint_tree_sha256=name,
                       flow_edit=None)
        arrays = {"expert_middle": values, "expert_late": values,
                  "epsilon": np.zeros((8, 1, 3, 4)), "sigma": np.array([[1, .5]])}
        if name == "late":
            binding.update(corrupt)
        return ids, arrays, binding, {"binding_sha256": path.name}

    monkeypatch.setattr(cr, "load_trace", trace)
    checkpoints = [(n, f"{n}_train", f"{n}_validation") for n in ("early", "late")]
    result = cr.run(tmp_path, checkpoints, tmp_path / "output")
    assert len(result["cka"]) == 24
    assert len(result["readouts"]) == 20
    assert result["controls"]
    assert (tmp_path / "output/report.json").is_file()
    with pytest.raises(FileExistsError):
        cr.run(tmp_path, checkpoints, tmp_path / "output")
    corrupt["image_binding"] = "changed"
    with pytest.raises(ValueError, match="contracts"):
        cr.run(tmp_path, checkpoints, tmp_path / "bad")
    corrupt.clear()
    split.assignments["validation-0"] = "test"
    with pytest.raises(ValueError, match="state IDs"):
        cr.run(tmp_path, checkpoints, tmp_path / "bad")
