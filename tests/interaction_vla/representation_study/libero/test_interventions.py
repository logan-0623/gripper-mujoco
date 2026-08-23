import numpy as np

from interaction_vla.representation_study.libero.interventions import (
    factor_rowspace_intervention,
    intervention_diagnostics,
    instruction_shuffle_indices,
    matched_donor_indices,
    matched_mean_intervention,
    matched_random_subspace_intervention,
    zero_ood_intervention,
)


def test_factor_intervention_changes_only_probe_rowspace() -> None:
    source = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    donor = np.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    weight = np.asarray([[1.0, 0.0, 0.0]])
    changed = factor_rowspace_intervention(source, donor, weight)
    assert np.array_equal(changed[:, 0], donor[:, 0])
    assert np.array_equal(changed[:, 1:], source[:, 1:])


def test_matched_donors_differ_on_factor_and_match_nuisance() -> None:
    donors = matched_donor_indices(
        factor_labels=np.asarray([0, 1, 0, 1]),
        nuisance_groups=np.asarray([10, 10, 20, 20]),
        seed=5,
    )
    assert np.array_equal(donors, [1, 0, 3, 2])


def test_intervention_report_separates_target_disruption_and_distribution_shift() -> None:
    source = np.asarray([[1.0, 0.0], [-1.0, 0.0]])
    changed = np.asarray([[-1.0, 0.0], [1.0, 0.0]])
    diagnostics = intervention_diagnostics(
        source,
        changed,
        target_weight=np.asarray([[1.0, 0.0]]),
        non_target_weights={"geometry": np.asarray([[0.0, 1.0]])},
    )
    assert diagnostics["target_probe_change"] > 0
    assert diagnostics["non_target_probe_change"]["geometry"] == 0
    assert "activation_norm_ratio" in diagnostics


def test_required_controls_preserve_shape_and_random_control_is_deterministic() -> None:
    source = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    donor = source[::-1].copy()
    weight = np.asarray([[1.0, 0.0, 0.0]])
    mean = matched_mean_intervention(source, source.mean(axis=0), weight)
    first = matched_random_subspace_intervention(source, donor, rank=1, seed=9)
    second = matched_random_subspace_intervention(source, donor, rank=1, seed=9)
    assert mean.shape == source.shape
    assert np.allclose(first, second)
    assert np.array_equal(zero_ood_intervention(source), np.zeros_like(source))
    shuffled = instruction_shuffle_indices(
        np.asarray(["task a", "task a", "task b", "task b"]), seed=4
    )
    instructions = np.asarray(["task a", "task a", "task b", "task b"])
    assert np.all(instructions[shuffled] != instructions)
