import numpy as np
import torch
from types import SimpleNamespace

from interaction_vla.representation_study.libero.flow_trace import (
    FlowEdit,
    _edited_tensor,
    bind_policy_images,
    paired_inference_noise,
    query_action_flow_at_points,
    install_flow_edit,
    select_records,
    trace_action_flow,
)


def test_suppression_and_matched_suppression_remove_target_coefficient():
    tensor = torch.tensor([[[3.0, 4.0]]])
    center = torch.zeros(2)
    target = torch.tensor([1.0, 0.0])
    direct = FlowEdit("expert_late", (3,), target, 1.0, "suppress", center, target)
    np.testing.assert_allclose(_edited_tensor(tensor, direct), [[[0.0, 4.0]]])
    control = FlowEdit("expert_late", (3,), torch.tensor([0.0, 1.0]), 1.0,
                       "matched_suppress", center, target)
    np.testing.assert_allclose(_edited_tensor(tensor, control), [[[3.0, 1.0]]])


def test_select_records_filters_suite_and_tasks_before_balancing():
    records = [
        SimpleNamespace(state_id=f"s{i}", suite=suite, task_id=task,
                        lerobot_episode_index=i)
        for i, (suite, task) in enumerate((
            ("libero_spatial", 0), ("libero_spatial", 3), ("libero_spatial", 4),
            ("libero_object", 0),
        ))
    ]
    split = SimpleNamespace(assignments={row.state_id: "train" for row in records})
    selected = select_records(records, split, partition="train", max_states=10,
                              suite="libero_spatial", task_ids=(0, 3))
    assert {(row.suite, row.task_id) for row in selected} == {
        ("libero_spatial", 0), ("libero_spatial", 3)
    }


class FakeFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.middle = torch.nn.Identity()
        self.late = torch.nn.Identity()

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        hidden = self.middle(x_t + timestep[:, None, None])
        hidden = self.late(hidden)
        return hidden * 0.1


class FakePolicy:
    def __init__(self):
        self.model = FakeFlow()
        self.config = type("Config", (), {"num_steps": 3})()

    def reset(self):
        pass

    def predict_action_chunk(self, processed, noise):
        x = noise
        for step in range(self.config.num_steps):
            sigma = 1 - step / self.config.num_steps
            timestep = torch.full((len(x),), sigma)
            velocity = self.model.denoise_step(None, None, x, timestep)
            x = x - velocity / self.config.num_steps
        return x[..., :2]


def test_flow_trace_uses_checkpoint_independent_noise_and_captures_every_stage():
    ids = ["state-a", "state-b"]
    noise = paired_inference_noise(ids, 1, (2, 4))
    assert torch.equal(noise, paired_inference_noise(ids, 1, (2, 4)))
    assert not torch.equal(noise, paired_inference_noise(ids, 2, (2, 4)))

    policy = FakePolicy()
    original = policy.model.denoise_step
    trace = trace_action_flow(policy, {}, noise, policy.model.middle, policy.model.late)
    assert trace["x_sigma"].shape == (2, 3, 2, 4)
    assert trace["expert_middle"].shape == (2, 3, 2, 4)
    np.testing.assert_allclose(trace["sigma"], [1, 2 / 3, 1 / 3], atol=1e-6)
    np.testing.assert_allclose(trace["final_x0"][..., :2], trace["action_normalized"], atol=1e-6)
    assert policy.model.denoise_step == original


def test_bind_policy_images_maps_two_libero_cameras_and_zeros_the_third():
    image = torch.ones(1, 3, 4, 4)
    wrist = torch.full_like(image, 2)
    batch = {"observation.images.image": image, "observation.images.image2": wrist}
    expected = {f"observation.images.camera{i}": object() for i in (1, 2, 3)}

    result, binding = bind_policy_images(batch, expected)

    assert result["observation.images.camera1"] is image
    assert result["observation.images.camera2"] is wrist
    assert torch.count_nonzero(result["observation.images.camera3"]) == 0
    assert binding["observation.images.camera3"] == "zero_like:observation.images.image"


def test_fixed_point_query_reuses_reference_path_without_following_native_path():
    policy = FakePolicy()
    noise = torch.ones(2, 2, 4)
    reference = torch.stack((noise * 3, noise * 2, noise), dim=1)
    result = query_action_flow_at_points(
        policy, {}, noise, reference, np.array([1, 2 / 3, 1 / 3]),
        policy.model.middle, policy.model.late,
    )

    np.testing.assert_allclose(result["x_sigma"], reference.numpy())
    np.testing.assert_allclose(result["velocity"], reference.numpy() * 0.1 + np.array(
        [1.0, 2 / 3, 1 / 3], dtype=np.float32)[None, :, None, None] * 0.1,
        atol=1e-6,
    )


def test_flow_edit_changes_only_the_selected_tap_and_stage():
    policy = FakePolicy()
    noise = torch.ones(2, 2, 4)
    edit = FlowEdit("expert_middle", (1,), torch.ones(4), 2.0)
    result = trace_action_flow(
        policy, {}, noise, policy.model.middle, policy.model.late, edit
    )
    baseline_policy = FakePolicy()
    baseline = trace_action_flow(
        baseline_policy, {}, noise, baseline_policy.model.middle, baseline_policy.model.late
    )
    assert result["expert_middle"][:, 1].mean() > result["expert_middle"][:, 0].mean()
    assert not np.allclose(result["action_normalized"], baseline["action_normalized"])


def test_live_flow_edit_restarts_stage_count_for_each_action_chunk():
    policy = FakePolicy()
    edit = FlowEdit("expert_middle", (0,), torch.ones(4), 1.0)
    handle = install_flow_edit(policy, policy.model.middle, edit)
    try:
        first = policy.predict_action_chunk({}, noise=torch.ones(1, 2, 4))
        second = policy.predict_action_chunk({}, noise=torch.ones(1, 2, 4))
    finally:
        handle.remove()
    torch.testing.assert_close(first, second)
