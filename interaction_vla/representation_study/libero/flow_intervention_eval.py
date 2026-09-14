"""Run LIBERO evaluation with a frozen candidate edit at selected flow stages."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .flow_trace import FlowEdit, action_atlas_provenance, file_hash, install_flow_edit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--dose", type=float, required=True)
    parser.add_argument("--edit-stage", type=int, action="append", required=True)
    parser.add_argument("--allow-control", action="store_true")
    args, forwarded = parser.parse_known_args()
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    candidate_artifact = json.loads(args.candidates.read_text(encoding="utf-8"))
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    if gate.get("candidate_sha256") != file_hash(args.candidates):
        raise ValueError("offline gate and candidate artifact differ")
    row = next((item for item in candidate_artifact["candidates"]
                if item["id"] == args.candidate_id), None)
    if row is None:
        raise ValueError(f"candidate not found: {args.candidate_id}")
    if row["role"] == "formation" and args.candidate_id not in gate["passed_candidate_ids"]:
        raise ValueError("formation candidate did not pass the offline action gate")
    if row["role"] != "formation" and not args.allow_control:
        raise ValueError("control interventions require --allow-control")

    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    edit = FlowEdit(row["tap"], tuple(sorted(set(args.edit_stage))),
                    torch.tensor(row["direction"], dtype=torch.float32), args.dose)
    action_atlas_provenance()
    from experiments.model_adapters import SmolVLAAdapter

    original = SmolVLAPolicy.from_pretrained

    def load(*load_args, **load_kwargs):
        policy = original(*load_args, **load_kwargs)
        adapter = SmolVLAAdapter(); adapter.policy = policy
        layers = adapter.get_layer_groups()["expert"]
        index = len(layers) // 2 if edit.tap == "expert_middle" else len(layers) - 1
        policy._acquisition_flow_edit_handle = install_flow_edit(policy, layers[index], edit)
        return policy

    SmolVLAPolicy.from_pretrained = staticmethod(load)
    from . import capability_events
    sys.argv = [sys.argv[0], *forwarded]
    capability_events.main()


if __name__ == "__main__":
    main()
