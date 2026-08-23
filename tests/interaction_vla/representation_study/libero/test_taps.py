import torch
from torch import nn

from interaction_vla.representation_study.libero.taps import (
    SmolVLASemanticTapCapture,
    valid_token_mean,
)


def test_valid_token_mean_excludes_padding() -> None:
    values = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
    mask = torch.tensor([[True, True, False]])
    assert torch.equal(valid_token_mean(values, mask), torch.tensor([[2.0, 4.0]]))


class FakeVLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.connector = nn.Identity()

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.connector(image)

    def forward(self, *, inputs_embeds, **_kwargs):
        prefix, suffix = inputs_embeds
        return ([None if prefix is None else prefix + 10.0, None if suffix is None else suffix + 20.0], object())


class FakeFlow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vlm_with_expert = FakeVLM()
        self.action_time_mlp_out = nn.Identity()
        self.action_out_proj = nn.Linear(4, 7, bias=False)

    def embed_prefix(self, images, img_masks, *_args, **_kwargs):
        image_values = [self.vlm_with_expert.embed_image(image) for image in images]
        prefix = torch.cat(image_values, dim=1)
        masks = torch.cat(
            [mask[:, None].expand(-1, image.shape[1]) for image, mask in zip(images, img_masks, strict=True)],
            dim=1,
        )
        return prefix, masks, masks

    def run(self, images, img_masks):
        prefix, pad, _ = self.embed_prefix(images, img_masks, None, None)
        self.vlm_with_expert.forward(inputs_embeds=[prefix, None])
        for value in (1.0, 2.0, 3.0):
            expert = self.action_time_mlp_out(torch.full((2, 3, 4), value))
            suffix, _ = self.vlm_with_expert.forward(inputs_embeds=[None, expert])
            self.action_out_proj(suffix[1])
        return torch.zeros(2, 7)


class FakePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeFlow()


def test_smolvla_taps_select_prefix_and_final_denoising_call() -> None:
    policy = FakePolicy()
    images = [
        torch.ones(2, 2, 4),
        torch.full((2, 2, 4), 2.0),
    ]
    masks = [torch.tensor([True, True]), torch.tensor([True, False])]
    capture = SmolVLASemanticTapCapture(policy)
    output, taps, metadata = capture.capture(lambda: policy.model.run(images, masks))
    assert output.shape == (2, 7)
    assert taps["vision_output"].shape == (2, 8)
    assert torch.equal(taps["vision_output"][0], torch.tensor([1.0] * 4 + [2.0] * 4))
    assert torch.equal(taps["vision_output"][1, 4:], torch.zeros(4))
    assert torch.equal(taps["multimodal_fusion"][0], torch.full((4,), 11.5))
    assert torch.equal(taps["action_expert_input"], torch.full((2, 4), 3.0))
    assert torch.equal(taps["pre_action"], torch.full((2, 4), 23.0))
    assert metadata["pre_action"]["call_selection"] == "final_denoising"
