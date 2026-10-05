import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from interaction_vla.representation_study.libero.module_transplant import build, side, summarize


def _checkpoint(root, value):
    root.mkdir()
    save_file({"model.vlm_with_expert.vlm.model.text_model.w": torch.full((2,), value),
               "model.state_proj.weight": torch.full((2,), value),
               "model.vlm_with_expert.lm_expert.layers.0.mlp.w": torch.full((2,), value),
               "model.action_out_proj.weight": torch.full((2,), value)}, str(root / "model.safetensors"))
    (root / "config.json").write_text("{}")
    return root


def test_sides_cover_prefix_and_action_and_reject_unknown():
    assert side("model.state_proj.weight") == "prefix"
    assert side("model.vlm_with_expert.lm_expert.norm.weight") == "action"
    with pytest.raises(ValueError):
        side("model.unknown")


def test_hybrid_takes_prefix_and_action_sides_from_their_sources(tmp_path):
    early, late = _checkpoint(tmp_path / "5k", 5.0), _checkpoint(tmp_path / "25k", 25.0)
    record = build(early, late, tmp_path / "hybrid")
    weights = load_file(str(tmp_path / "hybrid" / "model.safetensors"))
    assert weights["model.state_proj.weight"][0] == 5.0
    assert weights["model.vlm_with_expert.vlm.model.text_model.w"][0] == 5.0
    assert weights["model.action_out_proj.weight"][0] == 25.0
    assert (tmp_path / "hybrid" / "config.json").is_file()
    assert record["parameters"] == {"prefix": 4, "action": 4}


def test_summary_pairs_shared_cells(tmp_path):
    for name, wins in (("a", [1, 1]), ("b", [0, 1])):
        (tmp_path / name / "task0").mkdir(parents=True)
        (tmp_path / name / "task0" / "physical_events.json").write_text(json.dumps(
            {"episodes": [{"initial_state_id": i, "success": bool(w)} for i, w in enumerate(wins)]}))
    report = summarize({"a": tmp_path / "a", "b": tmp_path / "b"}, [0], tmp_path / "out")
    assert report["success"]["a"]["success"] == 2 and report["success"]["b"]["success"] == 1
    assert report["paired_differences"]["a - b"]["mean"] == 0.5
