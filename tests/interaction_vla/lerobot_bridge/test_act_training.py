from dataclasses import replace

import pytest

from interaction_vla.lerobot_bridge.act_smoke import (
    bounded_batches,
    iter_seeded_batches,
    optimizer_metric_from_loss_dict,
    run_training_with_fallback,
    train_once,
)
from interaction_vla.lerobot_bridge.config import load_bridge_config


def test_oom_restarts_once_at_batch_one(monkeypatch) -> None:
    attempts: list[int] = []

    def fake_train(*, batch_size: int, **kwargs):
        attempts.append(batch_size)
        if batch_size == 2:
            raise RuntimeError("MPS backend out of memory")
        return {
            "steps": 3,
            "losses": [1.0, 0.9, 0.8],
            "initial_state_hash": "fresh",
        }

    monkeypatch.setattr(
        "interaction_vla.lerobot_bridge.act_smoke.train_once", fake_train
    )
    result = run_training_with_fallback(object(), batch_size=2)

    assert attempts == [2, 1]
    assert result["fallback_from_batch_size"] == 2
    assert result["batch_size"] == 1


def test_non_oom_error_is_not_retried(monkeypatch) -> None:
    attempts = 0

    def fake_train(**kwargs):
        nonlocal attempts
        attempts += 1
        raise ValueError("bad schema")

    monkeypatch.setattr(
        "interaction_vla.lerobot_bridge.act_smoke.train_once", fake_train
    )
    with pytest.raises(ValueError, match="bad schema"):
        run_training_with_fallback(object(), batch_size=2)
    assert attempts == 1


def test_bounded_batches_restart_loader_and_stop_exactly() -> None:
    values = list(bounded_batches(lambda: iter(("a", "b", "c")), steps=5))

    assert values == [(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "b")]


def test_bounded_batches_reject_an_empty_loader() -> None:
    with pytest.raises(ValueError, match="empty"):
        list(bounded_batches(lambda: iter(()), steps=1))


def test_seeded_train_loader_is_shuffled_and_reproducible() -> None:
    dataset = list(range(24))
    first = list(iter_seeded_batches(dataset, batch_size=3, seed=17))
    second = list(iter_seeded_batches(dataset, batch_size=3, seed=17))
    different = list(iter_seeded_batches(dataset, batch_size=3, seed=18))

    flatten = lambda batches: [int(value) for batch in batches for value in batch]
    assert flatten(first) == flatten(second)
    assert flatten(first) != list(range(24))
    assert flatten(first) != flatten(different)


def test_optimizer_update_records_real_kld_loss() -> None:
    metric = optimizer_metric_from_loss_dict(
        total_loss=3.0,
        loss_dict={"l1_loss": 1.5, "kld_loss": 0.25},
    )

    assert metric == {"loss": 3.0, "l1_loss": 1.5, "kld_loss": 0.25}


def test_formal_training_summary_contains_reload_check(
    tiny_lerobot_dataset, tmp_path
) -> None:
    dataset_root, repo_id = tiny_lerobot_dataset
    bridge = load_bridge_config("configs/lerobot_act_smoke_macos.yaml")
    binding_files = []
    for name in ("bridge.yaml", "source.yaml", "expert.json"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        binding_files.append(path)
    bridge = replace(
        bridge,
        config_path=binding_files[0],
        source_config_path=binding_files[1],
        expert_gate=binding_files[2],
        dataset=replace(bridge.dataset, root=dataset_root, repo_id=repo_id),
        act=replace(bridge.act, steps=1, batch_size=1),
    )

    summary = train_once(
        bridge,
        batch_size=1,
        output_dir=tmp_path / "trained",
        architecture="test",
    )

    assert summary["reload_max_abs_error"] <= 1e-5
