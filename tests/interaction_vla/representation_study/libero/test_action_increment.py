from dataclasses import asdict
import io
import json

import numpy as np
import pytest

from interaction_vla.representation_study.libero.action_increment import ACTION, analyze, paired_summary, run
from interaction_vla.representation_study.libero.predictive_states import Protocol, TARGETS, file_hash, sequence_index, target
from interaction_vla.representation_study.libero.state_bank import finalize_state_bank, load_state_bank
from interaction_vla.representation_study.state_bank.io import write_bytes_atomic, write_json_atomic
from .helpers import make_record


def test_paired_gain_is_task_macro_not_independent_frame_average():
    scores = {"a": np.array([0.5, 0, 0, 0]), "z": np.zeros(4), "za": np.array([0, 0.5, 0.5, 0.5])}
    tasks = np.array(["a", "b", "b", "b"])
    result = paired_summary(scores, np.zeros(4), tasks, tasks, np.ones(4, dtype=bool))
    assert result["task_macro"]["gain_a_minus_za"] == 0
    assert result["sample_weighted"]["gain_a_minus_za"] == -0.125
    assert result["tasks_positive_gain"] == 1
    assert len(result["per_episode"]) == 2
    empty = paired_summary(scores, np.zeros(4), tasks, tasks, np.zeros(4, dtype=bool))
    assert empty["status"] == "not_estimable" and empty["n"] == 0


@pytest.fixture
def saved_run(tmp_path):
    records = [make_record(task_id=t, episode=t, frame=f, contact=f % 3 == 0, stable=f % 3 == 1)
               for t in range(6) for f in range(8)]
    bank = tmp_path / "bank"
    finalize_state_bank(records, output_dir=bank, source_binding_sha256="a" * 64,
                        ontology_sha256="b" * 64, split_seed=42, task_ratios=(0.5, 0.25, 0.25),
                        episode_ratios=(0.5, 0.25, 0.25),
                        replay_statistics={"episodes": 6, "accepted": 6, "l2_p95": 0., "max_abs": 0.})
    records, _, split, _ = load_state_bank(bank)
    cfg = Protocol(history=2, horizons=(0, 1), z_dim=8)
    index, partitions, _ = sequence_index(records, split, cfg)
    index = index[partitions == "test"]
    anchors = [records[i] for i in index[:, 1]]
    predictions, rows = {}, []
    feature = "hidden_mean_current"
    for label in TARGETS:
        for horizon in cfg.horizons:
            y = np.array([target(records[i], label) for i in index[:, 1 + horizon]])
            for method, error in ((ACTION, 0.5), (feature, 0.25), (feature + "+" + ACTION, 0.1)):
                key = f"{method}/{label}/{horizon}"
                predictions.update({key + "/state_id": np.array([r.state_id for r in anchors]),
                                    key + "/task": np.array([f"{r.suite}/{r.task_id}" for r in anchors]),
                                    key + "/episode": np.array([f"{r.suite}/{r.task_id}/{r.source_episode_id}" for r in anchors]),
                                    key + "/target": y, key + "/score": y * (1 - 2 * error) + error})
                rows.append({"method": method, "target": label, "horizon": horizon, "status": "measured",
                             "n": len(y), "brier": error ** 2, "task_macro_brier": error ** 2, "alpha": 1.0})
    folder = tmp_path / "readouts"
    write_json_atomic(folder / "report.json", {
        "schema": "smolvla_token_readouts_v1", "state_bank_sha256": file_hash(bank / "manifest.json"),
        "cache_binding_sha256": "c" * 64, "protocol": asdict(cfg), "features": {feature: [len(index), 2]},
        "controls": {ACTION: [len(index), 350], feature + "+" + ACTION: [len(index), 352]},
        "results": rows, "limitations": ["synthetic test, not policy evidence"]})
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **predictions)
    write_bytes_atomic(folder / "test_predictions.npz", buffer.getvalue())
    return folder, bank


def test_saved_run_analysis_writes_paired_results_without_training(saved_run, tmp_path):
    folder, bank = saved_run
    original_hash = file_hash(folder / "test_predictions.npz")
    result = run(folder, bank, tmp_path / "analysis")
    assert len(result["comparisons"]) == 4
    for entry in result["comparisons"]:
        assert entry["subsets"]["all"]["task_macro"]["gain_a_minus_za"] == pytest.approx(0.24)
        if entry["horizon"] == 0:
            assert entry["subsets"]["changed"]["status"] == "not_estimable"
        else:
            assert entry["subsets"]["changed"]["n"] > 0
    assert (tmp_path / "analysis/summary.csv").is_file()
    assert result["training"] == "not_run"
    assert file_hash(folder / "test_predictions.npz") == original_hash
    with pytest.raises(FileExistsError):
        run(folder, bank, tmp_path / "analysis")


@pytest.mark.parametrize("fault", ["reordered_ids", "wrong_target", "nan_score", "wrong_report", "wrong_bank"])
def test_saved_run_rejects_misalignment_nonfinite_and_wrong_provenance(saved_run, fault):
    folder, bank = saved_run
    if fault in {"wrong_report", "wrong_bank"}:
        report = json.loads((folder / "report.json").read_text())
        if fault == "wrong_report":
            report["results"][0]["brier"] = 0.9
        else:
            report["state_bank_sha256"] = "wrong"
        write_json_atomic(folder / "report.json", report)
    else:
        with np.load(folder / "test_predictions.npz", allow_pickle=False) as saved:
            data = dict(saved)
        key = f"{ACTION}/contact/0/"
        if fault == "reordered_ids":
            data[key + "state_id"] = data[key + "state_id"][::-1]
        elif fault == "wrong_target":
            data[key + "target"] = 1 - data[key + "target"]
        else:
            data[key + "score"][0] = np.nan
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **data)
        write_bytes_atomic(folder / "test_predictions.npz", buffer.getvalue())
    with pytest.raises(ValueError):
        analyze(folder, bank)
