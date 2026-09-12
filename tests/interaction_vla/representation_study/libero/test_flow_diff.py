import numpy as np

from interaction_vla.representation_study.libero.flow_diff import compare


def test_paired_flow_diff_checks_noise_and_summarizes_stages():
    rng = np.random.default_rng(4)
    n, r, s, t, d, h = 2, 3, 2, 4, 8, 5
    arrays = {
        "epsilon": rng.normal(size=(n, r, t, d)).astype("float32"),
        "sigma": np.tile(np.array([[1.0, 0.5]], dtype="float32"), (r, 1)),
        "x_sigma": rng.normal(size=(n, r, s, t, d)).astype("float32"),
        "velocity": rng.normal(size=(n, r, s, t, d)).astype("float32"),
        "expert_middle": rng.normal(size=(n, r, s, t, h)).astype("float32"),
        "expert_late": rng.normal(size=(n, r, s, t, h)).astype("float32"),
        "action_postprocessed": rng.normal(size=(n, r, t, 7)).astype("float32"),
    }
    after = {key: value.copy() for key, value in arrays.items()}
    after["velocity"] += 2
    after["expert_late"] += 1
    after["action_postprocessed"] += 0.5
    binding = {key: value for key, value in {
        "state_bank_sha256": "a", "dataset_revision": "r", "dataset_scientific_sha256": "b",
        "contract_tree_sha256": "c", "num_steps": s, "chunk_size": t, "max_action_dim": d,
    }.items()}
    ids = np.array(["a", "b"])
    manifest = {"binding_sha256": "hash"}

    report, evidence = compare(
        (ids, arrays, binding, manifest), (ids.copy(), after, binding.copy(), manifest.copy())
    )
    np.testing.assert_allclose(report["stage_metrics"]["velocity"]["paired_rms"], [2, 2])
    np.testing.assert_allclose(report["velocity_component_paired_rms"]["executed"], [2, 2])
    assert evidence["first_action_delta"].shape == (n, r, 7)
