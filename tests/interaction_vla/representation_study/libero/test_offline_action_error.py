import numpy as np

from interaction_vla.representation_study.libero.offline_action_error import per_state_errors


def test_errors_separate_translation_rotation_gripper_and_noise():
    demo = np.zeros((1, 2, 7))
    demo[..., 6] = -1.0
    policy = np.zeros((1, 2, 2, 7))
    policy[..., 0] = 3.0
    policy[..., 4] = 4.0
    policy[:, 0, :, 6], policy[:, 1, :, 6] = -1.0, 1.0
    errors = per_state_errors(policy, demo)
    np.testing.assert_allclose(errors["translation_l2"], [3.0])
    np.testing.assert_allclose(errors["rotation_l2"], [4.0])
    np.testing.assert_allclose(errors["gripper_agreement"], [0.5])
    assert errors["noise_spread"][0] > 0
