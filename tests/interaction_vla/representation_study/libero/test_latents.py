from pathlib import Path

import numpy as np
import pytest
import torch

from interaction_vla.representation_study.libero import latents as latents_module
from interaction_vla.representation_study.libero.latents import (
    LatentCacheWriter,
    deterministic_inference_noise,
    load_latent_cache,
)


def test_latent_cache_resumes_and_finalizes_exact_coverage(tmp_path: Path) -> None:
    writer = LatentCacheWriter(
        tmp_path / "latents",
        checkpoint_id="sft_25:abc",
        checkpoint_sha256="a" * 64,
        state_bank_sha256="b" * 64,
        tap="pre_action",
        pooling="valid_token_mean",
        expected_state_ids=("state-1", "state-2"),
    )
    writer.add("state-1", np.asarray([1.0, 2.0], dtype=np.float32))
    writer.add("state-1", np.asarray([1.0, 2.0], dtype=np.float32))
    with pytest.raises(ValueError, match="missing"):
        writer.finalize()
    writer.add("state-2", np.asarray([3.0, 4.0], dtype=np.float32))
    manifest = writer.finalize()
    state_ids, values, loaded = load_latent_cache(tmp_path / "latents")
    assert state_ids == ("state-1", "state-2")
    assert np.array_equal(values, [[1.0, 2.0], [3.0, 4.0]])
    assert loaded == manifest


def test_latent_cache_rejects_stale_checkpoint_binding(tmp_path: Path) -> None:
    LatentCacheWriter(
        tmp_path / "latents",
        checkpoint_id="sft_25:abc",
        checkpoint_sha256="a" * 64,
        state_bank_sha256="b" * 64,
        tap="pre_action",
        pooling="valid_token_mean",
        expected_state_ids=("state-1",),
    )
    with pytest.raises(ValueError, match="scientific binding"):
        LatentCacheWriter(
            tmp_path / "latents",
            checkpoint_id="sft_25:def",
            checkpoint_sha256="c" * 64,
            state_bank_sha256="b" * 64,
            tap="pre_action",
            pooling="valid_token_mean",
            expected_state_ids=("state-1",),
        )


def test_inference_noise_is_state_keyed_and_batch_order_invariant() -> None:
    first = deterministic_inference_noise(
        ("state-a", "state-b"),
        checkpoint_id="sft_25:abc",
        row_shape=(3, 7),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    reverse = deterministic_inference_noise(
        ("state-b", "state-a"),
        checkpoint_id="sft_25:abc",
        row_shape=(3, 7),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.equal(first[0], reverse[1])
    assert torch.equal(first[1], reverse[0])
    assert not torch.equal(first[0], first[1])


def test_libero_smolvla_camera_binding_matches_formal_training() -> None:
    from interaction_vla.representation_study.libero.feature_binding import (
        LIBERO_SMOLVLA_RENAME_MAP,
    )

    assert LIBERO_SMOLVLA_RENAME_MAP == {
        "observation.images.image": "observation.images.camera1",
        "observation.images.image2": "observation.images.camera2",
    }


def test_latent_implementation_binding_includes_dataset_bound_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = latents_module._latent_implementation_source_paths()
    relative_names = {path.as_posix().split("representation_study/")[-1] for path in paths}
    assert relative_names == {
        "libero/latents.py",
        "libero/taps.py",
        "libero/feature_binding.py",
        "backends/lerobot.py",
    }
    digests = {
        path.as_posix(): f"{index + 1:064x}" for index, path in enumerate(paths)
    }
    monkeypatch.setattr(
        latents_module, "_file_sha256", lambda path: digests[Path(path).as_posix()]
    )
    first = latents_module._latent_implementation_sha256()
    backend_path = next(path for path in paths if path.name == "lerobot.py")
    digests[backend_path.as_posix()] = "f" * 64
    second = latents_module._latent_implementation_sha256()
    assert first != second
