"""Run LIBERO evaluation with a frozen candidate edit at selected flow stages."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .flow_trace import action_atlas_provenance, file_hash, install_flow_edit, load_flow_edit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--dose", type=float, required=True)
    parser.add_argument("--edit-stage", type=int, action="append", required=True)
    parser.add_argument("--edit-mode", choices=("additive", "suppress", "matched_suppress"),
                        default="additive")
    parser.add_argument("--match-candidate-id")
    parser.add_argument("--allow-control", action="store_true")
    parser.add_argument("--exploratory", action="store_true")
    parser.add_argument("--environment-phase", choices=("all", "pre_contact", "contact"))
    parser.add_argument("--max-edited-chunks", type=int, default=1)
    parser.add_argument("--observe-only", action="store_true")
    args, forwarded = parser.parse_known_args()
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    candidate_artifact = json.loads(args.candidates.read_text(encoding="utf-8"))
    gate = json.loads(args.gate.read_text(encoding="utf-8")) if args.gate else None
    if not args.exploratory and gate is None:
        parser.error("--gate is required unless --exploratory")
    if gate is not None and gate.get("candidate_sha256") != file_hash(args.candidates):
        raise ValueError("offline gate and candidate artifact differ")
    row = next((item for item in candidate_artifact["candidates"]
                if item["id"] == args.candidate_id), None)
    if row is None:
        raise ValueError(f"candidate not found: {args.candidate_id}")
    if not args.exploratory and row["role"] == "formation" and args.candidate_id not in gate["passed_candidate_ids"]:
        raise ValueError("formation candidate did not pass the offline action gate")
    if row["role"] != "formation" and not args.allow_control:
        raise ValueError("control interventions require --allow-control")

    controller = None
    if args.environment_phase is not None:
        if not args.exploratory:
            parser.error("phase-triggered edits are currently exploratory only")
        required = {"--eval.batch_size=1", "--eval.use_async_envs=false", "--env.max_parallel_tasks=1"}
        if not required.issubset(forwarded):
            parser.error("phase editing requires explicit synchronous batch-one evaluation")
        from .phase_edit import PhaseEdit
        controller = PhaseEdit(args.environment_phase, args.max_edited_chunks, args.observe_only)
        controller.binding = {"candidate_sha256":file_hash(args.candidates),
                              "candidate_id":args.candidate_id,"dose":args.dose,
                              "flow_stages":args.edit_stage,"mode":args.edit_mode,
                              "match_candidate_id":args.match_candidate_id,"exploratory":True}
    elif args.observe_only:
        parser.error("--observe-only requires --environment-phase")
    if args.exploratory:
        episodes = [x.split("=", 1)[1] for x in forwarded if x.startswith("--eval.n_episodes=")]
        if len(episodes) != 1 or not 1 <= int(episodes[0]) <= 4:
            parser.error("exploratory entry requires an explicit budget of 1--4 episodes")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    edit, _ = load_flow_edit(
        args.candidates, args.candidate_id, args.dose, args.edit_stage,
        mode=args.edit_mode, match_candidate_id=args.match_candidate_id,
    )
    action_atlas_provenance()
    from experiments.model_adapters import SmolVLAAdapter

    original = SmolVLAPolicy.from_pretrained

    def load(*load_args, **load_kwargs):
        policy = original(*load_args, **load_kwargs)
        if candidate_artifact.get("center_checkpoint_sha256") is not None:
            from .latents import _tree_sha256
            path = load_args[0] if load_args else load_kwargs.get("pretrained_name_or_path")
            if path is None or _tree_sha256(Path(path)) != candidate_artifact["center_checkpoint_sha256"]:
                raise ValueError("candidate center was fitted on a different checkpoint")
        adapter = SmolVLAAdapter(); adapter.policy = policy
        layers = adapter.get_layer_groups()["expert"]
        index = len(layers) // 2 if edit.tap == "expert_middle" else len(layers) - 1
        policy._acquisition_flow_edit_handle = install_flow_edit(
            policy, layers[index], edit,
            begin_chunk=controller.begin_chunk if controller else None,
            on_edit=controller.on_edit if controller else None,
        )
        return policy

    SmolVLAPolicy.from_pretrained = staticmethod(load)
    from . import capability_events
    sys.argv = [sys.argv[0], *forwarded]
    capability_events.main(phase_controller=controller)


if __name__ == "__main__":
    main()
