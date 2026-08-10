import pytest

from interaction_vla.lerobot_bridge.act_smoke import (
    bounded_batches,
    run_training_with_fallback,
)


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
