import io

import numpy as np
import pytest

from interaction_vla.representation_study.state_bank.io import write_bytes_atomic, write_json_atomic
from interaction_vla.representation_study.libero.predictive_states import Protocol, file_hash, sequence_index
from interaction_vla.representation_study.libero.splits import build_task_group_split
from interaction_vla.representation_study.libero.token_cache import load_summaries, pilot_records, read_shard
from interaction_vla.representation_study.libero.token_readouts import build_features
from .helpers import make_record


def test_token_shards_bind_ids_shape_finiteness_and_hash(tmp_path):
    path = tmp_path / "part.npz"
    tokens = np.arange(2*50*480, dtype=np.float32).reshape(2, 50, 480)
    actions = np.ones((2, 50, 7), dtype=np.float32)
    buffer = io.BytesIO()
    np.savez(buffer, state_ids=["a", "b"], tokens=tokens, actions=actions)
    write_bytes_atomic(path, buffer.getvalue())
    receipt = {"sha256": file_hash(path), "binding_sha256": "binding"}
    write_json_atomic(path.with_suffix(".json"), receipt)
    actual, plan, _ = read_shard(path, ["a", "b"], "binding")
    np.testing.assert_array_equal(actual, tokens)
    np.testing.assert_array_equal(plan, actions)
    with pytest.raises(ValueError, match="IDs/order"):
        read_shard(path, ["b", "a"], "binding")
    with pytest.raises(ValueError, match="differently bound"):
        read_shard(path, ["a", "b"], "wrong")
    write_bytes_atomic(path, buffer.getvalue() + b"tampered")
    with pytest.raises(ValueError, match="Corrupt"):
        read_shard(path, ["a", "b"], "binding")


def test_pilot_selection_is_train_only_and_does_not_depend_on_labels():
    rows = [make_record(task_id=t, episode=t, frame=f) for t in range(10) for f in range(20)]
    split = build_task_group_split(rows, ratios=(0.6, 0.2, 0.2), seed=42)
    chosen = pilot_records(rows, split)
    assert len(chosen) == 64
    assert len({(r.suite, r.task_id) for r in chosen}) == 4
    assert all(split.assignments[r.state_id] == "train" for r in chosen)
    assert [r.state_id for r in chosen] == [r.state_id for r in pilot_records(list(reversed(rows)), split)]


def test_pca_is_train_only_and_plans_come_from_current_not_future_observation():
    rows = [make_record(task_id=t, episode=t, frame=f) for t in range(6) for f in range(18)]
    split = build_task_group_split(rows, ratios=(0.5, 0.25, 0.25), seed=42)
    cfg = Protocol(history=3, horizons=(0, 1, 3), z_dim=8)
    index, partitions, _ = sequence_index(rows, split, cfg)
    rng = np.random.default_rng(5)
    mean, first = [rng.normal(size=(len(rows), 8)).astype(np.float32) for _ in range(2)]
    actions = np.broadcast_to(np.arange(len(rows), dtype=np.float32)[:, None, None], (len(rows), 50, 7)).copy()
    features, controls, fitted = build_features(mean, first, actions, rows, split, index, cfg)
    expected = fitted["mean"]["pca"].transform(fitted["mean"]["scaler"].transform(mean))
    np.testing.assert_allclose(features["pca_mean_current"], expected[index[:, 2]], atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(controls["policy_first_action"][:, 0], index[:, 2])
    np.testing.assert_array_equal(controls["policy_action_chunk"][:, -1], index[:, 2])
    assert features["hidden_first_token_history"].shape == (len(index), 24)
    test = np.array([split.assignments[r.state_id] == "test" for r in rows])
    mean[test] += 100
    first[test] -= 100
    _, _, changed = build_features(mean, first, actions, rows, split, index, cfg)
    for name in fitted:
        np.testing.assert_array_equal(fitted[name]["scaler"].mean_, changed[name]["scaler"].mean_)
        np.testing.assert_array_equal(fitted[name]["pca"].components_, changed[name]["pca"].components_)


def test_summary_loader_rejects_pilot_as_full_and_checks_shard_coverage(tmp_path):
    bank, cache = tmp_path / "bank", tmp_path / "cache"
    write_json_atomic(bank / "manifest.json", {"synthetic": True})
    binding = {"schema": "smolvla_tokens_actions_v1", "state_ids": ["a", "b"], "batch_size": 2,
               "state_bank_sha256": file_hash(bank / "manifest.json")}
    write_json_atomic(cache / "binding.json", binding)
    binding_hash = file_hash(cache / "binding.json")
    path = cache / "shards/000000.npz"
    tokens = np.arange(2*50*480, dtype=np.float32).reshape(2, 50, 480)
    buffer = io.BytesIO()
    np.savez(buffer, state_ids=["a", "b"], tokens=tokens, actions=np.zeros((2, 50, 7), dtype=np.float32))
    write_bytes_atomic(path, buffer.getvalue())
    write_json_atomic(path.with_suffix(".json"), {"sha256": file_hash(path), "binding_sha256": binding_hash})
    manifest = {"schema": binding["schema"], "complete": True, "full_state_bank": False,
                "states": 2, "binding_sha256": binding_hash, "shards": ["shards/000000.npz"]}
    write_json_atomic(cache / "manifest.json", manifest)
    mean, first, actions, _ = load_summaries(cache, bank, require_full=False)
    np.testing.assert_array_equal(mean, tokens.mean(axis=1))
    np.testing.assert_array_equal(first, tokens[:, 0])
    assert actions.shape == (2, 50, 7)
    with pytest.raises(ValueError, match="full-StateBank"):
        load_summaries(cache, bank)
    manifest["shards"] *= 2
    write_json_atomic(cache / "manifest.json", manifest)
    with pytest.raises(ValueError, match="coverage"):
        load_summaries(cache, bank, require_full=False)
