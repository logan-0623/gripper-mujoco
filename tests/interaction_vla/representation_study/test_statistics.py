import numpy as np
import pytest

from interaction_vla.representation_study.statistics import (
    benjamini_hochberg,
    clustered_bootstrap_mean,
    paired_sign_flip_pvalue,
    spearman_correlation,
)


def test_clustered_bootstrap_reports_cluster_unit() -> None:
    result = clustered_bootstrap_mean(
        [0.0, 1.0, 2.0, 3.0], ["a", "a", "b", "b"],
        samples=200, confidence=0.95, seed=7,
    )
    assert result["estimate"] == 1.5
    assert result["clusters"] == 2
    assert result["ci_low"] <= result["estimate"] <= result["ci_high"]


def test_benjamini_hochberg_is_order_preserving_after_unshuffle() -> None:
    adjusted = benjamini_hochberg([0.04, 0.001, 0.02])
    assert adjusted == pytest.approx([0.04, 0.003, 0.03])


def test_spearman_handles_ties() -> None:
    assert spearman_correlation([1, 2, 2, 4], [4, 3, 3, 1]) == pytest.approx(-1.0)
    assert np.isnan(spearman_correlation([1, 1, 1], [1, 2, 3]))


def test_paired_sign_flip_detects_consistent_direction() -> None:
    assert paired_sign_flip_pvalue([1.0] * 12, samples=2000, seed=3) < 0.01
