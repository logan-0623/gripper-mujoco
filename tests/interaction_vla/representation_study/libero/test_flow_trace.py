import numpy as np
import torch

from interaction_vla.representation_study.libero.flow_trace import (
    bind_policy_images,
    paired_inference_noise,
    trace_action_flow,
)


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
