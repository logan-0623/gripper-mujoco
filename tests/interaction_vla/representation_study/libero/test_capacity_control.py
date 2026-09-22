import numpy as np

from interaction_vla.representation_study.libero.capacity_control import random_features


def test_random_control_is_deterministic_and_dimension_matched():
    features = {"current": np.zeros((7, 3), dtype=np.float32),
                "history": np.zeros((7, 12), dtype=np.float32)}
    first = random_features(features, 7, 123)
    second = random_features(features, 7, 123)
    assert first.keys() == features.keys()
    for name in features:
        assert first[name].shape == features[name].shape
        np.testing.assert_array_equal(first[name], second[name])
    assert not np.array_equal(first["current"], random_features(features, 7, 124)["current"])
