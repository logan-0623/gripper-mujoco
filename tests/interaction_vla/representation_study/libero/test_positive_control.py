import pytest

from interaction_vla.representation_study.libero.latents import validate_requested_taps
from interaction_vla.representation_study.libero.taps import SEMANTIC_TAPS


def test_requested_taps_default_to_all_semantic_taps() -> None:
    assert validate_requested_taps(None) == tuple(SEMANTIC_TAPS)


def test_requested_taps_are_unique_known_and_ordered() -> None:
    assert validate_requested_taps(("action_expert_input",)) == (
        "action_expert_input",
    )
    with pytest.raises(ValueError, match="unknown semantic tap"):
        validate_requested_taps(("missing",))
    with pytest.raises(ValueError, match="duplicate semantic tap"):
        validate_requested_taps(("pre_action", "pre_action"))
