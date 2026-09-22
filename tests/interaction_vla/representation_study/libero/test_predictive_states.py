from dataclasses import replace
import csv
import json

import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")
from interaction_vla.representation_study.libero.latents import LatentCacheWriter
from interaction_vla.representation_study.libero.predictive_states import (
    Protocol, ROOT, SparseReadoutRidge, bound_latents, evaluate, file_hash, fit_readout, paired_action_effects, readout_scores,
    run, sequence_index, upstream,
)
from interaction_vla.representation_study.libero.splits import build_task_group_split
from interaction_vla.representation_study.libero.state_bank import finalize_state_bank, load_state_bank
from .helpers import make_record


def records():
    return [make_record(task_id=t, episode=t, frame=f, contact=f % 4 > 0, stable=f % 4 == 2)
            for t in range(6) for f in range(18)]


def test_windows_preserve_time_and_task_split_and_reject_gaps():
    rows = records()
    split = build_task_group_split(rows, ratios=(0.5, 0.25, 0.25), seed=42)
    cfg = Protocol(history=3, horizons=(0, 1, 3))
    index, partitions, audit = sequence_index(rows, split, cfg)
    assert len(index) == 6 * 13
    assert set(partitions) == {"train", "validation", "test"}
    assert not audit["discarded_windows"]
    for window in index:
        assert len({rows[i].task_id for i in window}) == 1
        assert len({rows[i].source_episode_id for i in window}) == 1
        assert np.all(np.diff([rows[i].frame_index for i in window]) == 1)
    gapped = [r for r in rows if not (r.task_id == 0 and r.frame_index == 8)]
    split_gapped = split.with_assignments({r.state_id: split.assignments[r.state_id] for r in gapped})
    _, _, report = sequence_index(gapped, split_gapped, cfg)
    assert report["discarded_windows"]["frame_gap"] > 0
    bad = list(rows)
    bad[8] = replace(rows[8], observation=replace(rows[8].observation, timestamp=0.85))
    assert sequence_index(bad, split, cfg)[2]["discarded_windows"]["time_gap"] > 0
    leaky = split.with_assignments({**split.assignments, rows[0].state_id: "validation" if
                                   split.assignments[rows[0].state_id] != "validation" else "train"})
    with pytest.raises(ValueError, match="multiple partitions"):
        sequence_index(rows, leaky, cfg)


def test_readout_never_fits_scaler_or_selects_alpha_on_test():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(60, 5))
    y = (x[:, 0] > 0).astype(float)
    p = np.repeat(["train", "validation", "test"], 20)
    first, alpha = fit_readout(x, y, p)
    x[40:] += 10000
    y[40:] = 1 - y[40:]
    second, changed_alpha = fit_readout(x, y, p)
    assert alpha == changed_alpha
    np.testing.assert_array_equal(first[0].mean_, second[0].mean_)
    np.testing.assert_array_equal(first[-1].coef_, second[-1].coef_)
    assert first[-1].coef_.dtype == np.float32
    first[-1].intercept_ = np.float32(np.inf)
    with pytest.raises(ValueError, match="before clipping"):
        readout_scores(first, x[:2])


@pytest.mark.parametrize("width", [70, 1920])
def test_large_readout_uses_finite_sparse_matvec_without_changing_ridge(width, tmp_path):
    import joblib
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(3)
    x = rng.normal(size=(8677, width)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.float32)
    partitions = np.repeat(["train", "validation", "test"], [8477, 100, 100])
    with np.errstate(all="raise"):
        model, alpha = fit_readout(x, y, partitions)
        assert np.isfinite(readout_scores(model, x[-100:])).all()
    # Independently check the same Ridge objective on a small well-conditioned
    # case using the dense solver, where native matmul does not raise FP flags.
    small = x[:60, :5]
    split = np.repeat(["train", "validation", "test"], 20)
    fitted, alpha = fit_readout(small, y[:60], split)
    scaled = StandardScaler().fit_transform(small[:20])
    dense = Ridge(alpha=alpha, solver="lsqr").fit(scaled, y[:20])
    np.testing.assert_allclose(fitted[-1].coef_, dense.coef_, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(fitted[-1].intercept_, dense.intercept_, rtol=1e-4, atol=1e-5)
    with np.errstate(all="raise"):
        saved = tmp_path / "readout.joblib"
        joblib.dump(model, saved)
        restored = joblib.load(saved)
        np.testing.assert_array_equal(readout_scores(restored, x[-100:]),
                                      readout_scores(model, x[-100:]))


@pytest.mark.parametrize("multi_target", [False, True])
@pytest.mark.parametrize("with_scale", [False, True])
def test_readout_intercept_preserves_offset_scale_and_dtype(multi_target, with_scale):
    rng = np.random.default_rng(19)
    offset = rng.normal(size=1920).astype(np.float32)
    coef = rng.normal(size=(2, 1920) if multi_target else 1920)
    scale = rng.uniform(0.5, 2, size=1920).astype(np.float32) if with_scale else None
    y_offset = np.array([0.3, 0.7] if multi_target else 0.3, dtype=np.float32)
    model = SparseReadoutRidge(solver="lsqr")
    model.coef_ = coef.copy()
    expected_coef = coef.astype(np.float32)
    if scale is not None:
        expected_coef = expected_coef / scale
    expected = y_offset - np.sum(expected_coef.astype(np.float64) * offset, axis=-1)
    with np.errstate(all="raise"):
        model._set_intercept(offset, y_offset, scale)
    np.testing.assert_array_equal(model.coef_, expected_coef)
    np.testing.assert_allclose(model.intercept_, expected, rtol=1e-5, atol=1e-5)
    model.fit_intercept = False
    model._set_intercept(offset, y_offset, scale)
    assert model.intercept_ == 0.0


def test_time_and_action_controls_do_not_use_future_episode_length(tmp_path, monkeypatch):
    rows = [replace(r, observation=replace(r.observation,
                timestamp=r.observation.timestamp + 100 * (r.task_id + 1),
                action=(float(r.frame_index),) + (0.,) * 6)) for r in records()]
    split = build_task_group_split(rows, ratios=(0.5, 0.25, 0.25), seed=42)
    cfg = Protocol(history=3, horizons=(0, 1, 3))
    index, partitions, _ = sequence_index(rows, split, cfg)
    anchor_frames = np.array([rows[i].frame_index for i in index[:, 2]])
    observed = []
    def inspect_control(x, y, p):
        observed.append(x.copy())
        return None
    monkeypatch.setattr("interaction_vla.representation_study.libero.predictive_states.fit_readout",
                        inspect_control)
    evaluate({}, rows, index, partitions, cfg, tmp_path)
    assert {x.shape[1] for x in observed} == {1, 7, 21}
    for x in observed:
        if x.shape[1] == 1:
            np.testing.assert_allclose(x[:, 0], anchor_frames / 10, atol=1e-6)
        else:
            horizon = x.shape[1] // 7
            np.testing.assert_array_equal(x[:, ::7], anchor_frames[:, None] + np.arange(horizon))


def test_official_sae_ret_end_to_end_and_cache_binding(tmp_path, monkeypatch):
    if not (ROOT / "research/ret/ret").is_dir() or not (ROOT / "research/action-atlas/experiments").is_dir():
        pytest.skip("official source checkouts required; see research/predictive-states.md")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    import transformers
    def no_network(*args, **kwargs):
        raise AssertionError("Cached robot training must not load tokenizer/model")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", no_network)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", no_network)
    rows = records()
    bank = tmp_path / "bank"
    finalize_state_bank(rows, output_dir=bank, source_binding_sha256="a" * 64,
                        ontology_sha256="b" * 64, split_seed=42, task_ratios=(0.5, 0.25, 0.25),
                        episode_ratios=(0.5, 0.25, 0.25),
                        replay_statistics={"episodes": 6, "accepted": 6, "l2_p95": 0., "max_abs": 0.})
    cache = tmp_path / "cache"
    rng = np.random.default_rng(3)
    values = rng.normal(size=(len(rows), 8)).astype(np.float32)
    writer = LatentCacheWriter(cache, checkpoint_id="synthetic-not-a-policy", checkpoint_sha256="c" * 64,
                               state_bank_sha256=file_hash(bank / "manifest.json"), tap="synthetic",
                               pooling="synthetic", expected_state_ids=[r.state_id for r in reversed(rows)])
    for r, value in zip(rows, values):
        writer.add(r.state_id, value)
    writer.finalize()
    aligned, _ = bound_latents(bank, rows, cache)
    np.testing.assert_array_equal(aligned, values)
    cfg = Protocol(history=3, horizons=(0, 1), z_dim=8, ret_steps=20, sae_epochs=1, sae_k=2, batch_size=32)
    controls = run(bank, None, tmp_path / "controls", cfg, "controls-only")
    assert controls["representation_comparison"] == "not_run"
    assert controls["latent_binding"] is None and controls["upstream"] == {}
    assert controls["state_bank_sha256"] == file_hash(bank / "manifest.json")
    assert {r["method"] for r in controls["results"]} == {
        "elapsed_time", "realized_actions_only", "train_prevalence", "privileged_persistence"}
    with pytest.raises(ValueError, match="require a bound latent cache"):
        run(bank, None, tmp_path / "missing", cfg)
    assert not (tmp_path / "missing").exists()
    report = run(bank, cache, tmp_path / "run", cfg)
    assert report["causal_control"] == "not_run"
    assert report["diagnostics"]["ret_steps"] == 20
    assert "ret_history" in report["methods"] and "sae_history" in report["methods"]
    assert (tmp_path / "run/ret/checkpoint_final.pt").is_file()
    assert json.loads((tmp_path / "run/report.json").read_text())["status"] == "offline_diagnostic"
    with pytest.raises(FileExistsError):
        run(bank, cache, tmp_path / "run", cfg)
    # Official saved RET encoder is causal and reloadable.
    upstream("ret")
    from ret.models.encoders import build_encoder
    from ret.config import load_config
    encoder = build_encoder(8, load_config(tmp_path / "run/ret/config.yaml").encoder).eval()
    state = torch.load(tmp_path / "run/ret/checkpoint_final.pt", weights_only=True)
    assert all(torch.isfinite(v).all() for v in state["encoder"].values())
    assert state["optimizer"]["state"]
    with (tmp_path / "run/ret/train_metrics.csv").open() as stream:
        logs = list(csv.DictReader(stream))
    losses = np.array([float(row["train/total_loss"]) for row in logs])
    assert np.isfinite(losses).all() and losses[-1] < losses[0]
    encoder.load_state_dict(state["encoder"])
    h = torch.randn(2, 4, 8)
    a = encoder(h)[0]
    h[:, 3] += 100
    torch.testing.assert_close(a[:, :3], encoder(h)[0][:, :3])


def test_actual_forward_pairing_uses_atlas_hooks_and_cleans_up():
    if not (ROOT / "research/action-atlas/experiments").is_dir():
        pytest.skip("official Action Atlas checkout required")
    layer = torch.nn.Linear(8, 8, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(8))
    x = torch.ones(1, 3, 8)
    def forward():
        return layer(x).mean(dim=(0, 1))[:7] + torch.rand(7) * 0.01
    delta = np.zeros(8, dtype=np.float32)
    delta[0] = 0.1
    report = paired_action_effects(forward, layer, {"target": delta}, seed=42, reference=np.ones(8))
    assert report["conditions"]["zero"]["translation_l2"] == 0
    assert report["conditions"]["target"]["translation_l2"] == pytest.approx(0.1)
    assert not layer._forward_hooks
    zero_only = paired_action_effects(forward, layer, {}, seed=42, reference=np.ones(8))
    assert set(zero_only["conditions"]) == {"zero"}
    assert zero_only["conditions"]["zero"]["translation_l2"] == 0
    assert not layer._forward_hooks

    calls = 0
    def extra_call_on_replay():
        nonlocal calls
        calls += 1
        result = forward()
        if calls > 1:
            layer(x)  # Extra unused call used to evade injection_count checks.
        return result
    with pytest.raises(ValueError, match="call count/shape"):
        paired_action_effects(extra_call_on_replay, layer, {"target": delta},
                              seed=42, reference=np.ones(8))
    assert not layer._forward_hooks
    with pytest.raises(ValueError, match="cached reference"):
        paired_action_effects(forward, layer, {"target": delta}, seed=42, reference=np.zeros(8))
    assert not layer._forward_hooks
