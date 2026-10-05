from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from interaction_vla.representation_study.libero.stage_patching import (
    cluster_ci, decisions, flip_metrics, match_pairs, patched_actions)
from lerobot.policies.common.flow_matching import euler_integrate

STEPS = 5


class _Flow(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        layers = [SimpleNamespace(mlp=nn.Linear(4, 4)) for _ in range(4)]
        self._mlps = nn.ModuleList([layer.mlp for layer in layers])
        self.vlm_with_expert = SimpleNamespace(lm_expert=SimpleNamespace(layers=layers))

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        hidden = x_t
        for layer in self.vlm_with_expert.lm_expert.layers:
            hidden = hidden + torch.tanh(layer.mlp(hidden))
        return hidden - x_t + timestep[:, None, None]


class _Policy:
    def __init__(self):
        self.model = _Flow()
        self.config = SimpleNamespace(num_steps=STEPS)

    def reset(self):
        pass

    def predict_action_chunk(self, processed, noise):
        flow = self.model
        return euler_integrate(lambda x, t: flow.denoise_step(prefix_pad_masks=None, past_key_values=None,
                                                              x_t=x, timestep=t), noise, STEPS)


def _natural_path(policy, noise):
    """Flow states x_0..x_STEPS and per-call late-MLP outputs of an unpatched run."""
    states, late = [noise], []
    handle = policy.model.vlm_with_expert.lm_expert.layers[-1].mlp.register_forward_hook(
        lambda _m, _i, out: late.append(out.detach().clone()))
    x = noise
    with torch.no_grad():
        for step in range(STEPS):
            t = torch.full((noise.shape[0],), 1.0 - step / STEPS)
            x = x - policy.model.denoise_step(None, None, x_t=x, timestep=t) / STEPS
            states.append(x)
    handle.remove()
    return states, late


def test_unpatched_run_matches_native_integration():
    policy = _Policy()
    noise = torch.randn(2, 3, 4)
    states, _ = _natural_path(policy, noise)
    actions = patched_actions(policy, None, noise, site=None, stage=-1, donor=None)
    torch.testing.assert_close(actions, states[-1])


def test_x_t_patch_continues_integration_from_the_donor_point():
    policy = _Policy()
    recipient, donor_noise = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    donor_states, _ = _natural_path(policy, donor_noise)
    for stage in (0, 2, STEPS - 1):
        patched = patched_actions(policy, None, recipient, site="x_t", stage=stage, donor=donor_states[stage])
        # From the donor's stage-k point the recipient's dynamics (identical here) reproduce the donor path.
        torch.testing.assert_close(patched, donor_states[-1], atol=1e-5, rtol=1e-5)
    own_states, _ = _natural_path(policy, recipient)
    torch.testing.assert_close(
        patched_actions(policy, None, recipient, site="x_t", stage=2, donor=own_states[2]),
        own_states[-1], atol=1e-5, rtol=1e-5)


def test_expert_patch_replaces_only_the_requested_call():
    policy = _Policy()
    noise = torch.randn(2, 3, 4)
    states, late = _natural_path(policy, noise)
    torch.testing.assert_close(
        patched_actions(policy, None, noise, site="expert_late", stage=3, donor=late[3]), states[-1])
    changed = patched_actions(policy, None, noise, site="expert_late", stage=3, donor=late[3] + 5.0)
    assert not torch.allclose(changed, states[-1])
    # Only the final Euler update can differ when the last stage is patched.
    last = patched_actions(policy, None, noise, site="expert_late", stage=STEPS - 1, donor=late[-1] + 1.0)
    shift = (last - states[-1]).abs().max()
    assert 0 < shift <= 1.0 / STEPS * 2 + 1e-6


def _rows(n):
    return [SimpleNamespace(suite="libero_spatial", task_id=0, source_episode_id=i % 4, frame_index=i,
                            observation=SimpleNamespace(robot_state=[float(i), 0.0])) for i in range(n)]


def test_pairs_are_confident_same_task_cross_episode_and_label_blind():
    gripper = np.asarray([[1.0, 0.9], [-1.0, -0.8], [0.1, -0.2], [0.9, 1.0], [-0.9, -1.0], [1.0, 1.0]])
    opposite, same = match_pairs(_rows(6), gripper, margin=0.5, max_pairs=10, seed=0)
    rows = _rows(6)
    for recipient, donor, _ in opposite:
        assert np.sign(gripper[recipient].mean()) != np.sign(gripper[donor].mean())
        assert rows[recipient].source_episode_id != rows[donor].source_episode_id
        assert 2 not in (recipient, donor)
    for recipient, donor, _ in same:
        assert np.sign(gripper[recipient].mean()) == np.sign(gripper[donor].mean())


def test_decision_flip_and_shift_metrics():
    actions = np.zeros((1, 1, 50, 7))
    actions[..., :10, 6] = -1.0
    assert decisions(actions, 10)[0, 0] == -1.0
    natural, donor = np.asarray([[-1.0, -1.0]]), np.asarray([[1.0, 1.0]])
    flip, shift = flip_metrics(natural, np.asarray([[[0.5, -0.5]]]), donor)
    np.testing.assert_allclose(flip, [[0.5]])
    np.testing.assert_allclose(shift, [[0.5]])
    ci = cluster_ci([1.0, 1.0, 0.0, 0.0], ["a", "a", "b", "b"], draws=200)
    assert ci["clusters"] == 2 and ci["ci_low"] <= 0.5 <= ci["ci_high"]
    tuples = cluster_ci([1.0, 0.0, 1.0], [("s", 0, 1), ("s", 0, 2), ("s", 0, 1)], draws=50)
    assert tuples["clusters"] == 2 and tuples["pairs"] == 3
