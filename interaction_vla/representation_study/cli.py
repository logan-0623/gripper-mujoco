from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import load_study_config
from .extraction import extract_latents, inspect_latents
from .evaluation import evaluate_policy_stage
from .probes import train_probe_suite
from .interventions import run_closed_loop_interventions, run_interventions
from .measurement import run_stage_measurement
from .rl.training import evaluate_residual_rl, train_residual_rl
from .report import build_study_report
from .sft import train_sft
from .state_bank.builder import collect_state_bank, inspect_state_bank


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m interaction_vla.representation_study",
        description="Stage-wise interaction-representation study",
    )
    families = parser.add_subparsers(dest="family", required=True)
    state_bank = families.add_parser("state-bank", help="fixed held-out State Bank")
    state_commands = state_bank.add_subparsers(dest="command", required=True)
    for name in ("collect", "inspect"):
        command = state_commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
    latents = families.add_parser("latents", help="extract fixed-stage policy latents")
    latent_commands = latents.add_subparsers(dest="command", required=True)
    for name in ("extract", "inspect"):
        command = latent_commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--backend", choices=("act", "smolvla", "pi0"), required=True)
        command.add_argument(
            "--stage",
            choices=("pretrained", "sft", "continued_sft", "rl_head", "rl_representation"),
            required=True,
        )
        command.add_argument(
            "--partition", choices=("all", "train", "validation", "test"), default="all"
        )
        command.add_argument("--limit", type=int)
    probes = families.add_parser("probes", help="frozen representation probes")
    probe_commands = probes.add_subparsers(dest="command", required=True)
    train_probe = probe_commands.add_parser("train")
    train_probe.add_argument("--config", type=Path, required=True)
    train_probe.add_argument("--backend", choices=("act", "smolvla", "pi0"), required=True)
    train_probe.add_argument(
        "--stage",
        choices=("pretrained", "sft", "continued_sft", "rl_head", "rl_representation"),
        required=True,
    )
    train_probe.add_argument("--model", choices=("linear", "shallow_mlp"), default="linear")
    interventions = families.add_parser("interventions", help="functional latent-use tests")
    intervention_commands = interventions.add_subparsers(dest="command", required=True)
    for name in ("run", "rollout"):
        run_intervention = intervention_commands.add_parser(name)
        run_intervention.add_argument("--config", type=Path, required=True)
        run_intervention.add_argument("--backend", choices=("act", "smolvla", "pi0"), required=True)
        run_intervention.add_argument(
            "--stage",
            choices=("pretrained", "sft", "continued_sft", "rl_head", "rl_representation"),
            required=True,
        )
    policy = families.add_parser("policy", help="fixed-case closed-loop stage utility")
    policy_commands = policy.add_subparsers(dest="command", required=True)
    policy_evaluate = policy_commands.add_parser("evaluate")
    policy_evaluate.add_argument("--config", type=Path, required=True)
    policy_evaluate.add_argument("--backend", choices=("act", "smolvla", "pi0"), required=True)
    policy_evaluate.add_argument(
        "--stage",
        choices=("pretrained", "sft", "continued_sft", "rl_head", "rl_representation"),
        required=True,
    )
    policy_evaluate.add_argument("--force", action="store_true")
    measure = families.add_parser("measure", help="run the full stage evidence bundle")
    measure_commands = measure.add_subparsers(dest="command", required=True)
    measure_run = measure_commands.add_parser("run")
    measure_run.add_argument("--config", type=Path, required=True)
    measure_run.add_argument("--backend", choices=("act", "smolvla", "pi0"), required=True)
    measure_run.add_argument(
        "--stage",
        choices=("pretrained", "sft", "continued_sft", "rl_head", "rl_representation"),
        required=True,
    )
    measure_run.add_argument("--secondary-probe", action="store_true")
    measure_run.add_argument("--closed-loop-intervention", action="store_true")
    sft = families.add_parser("sft", help="LeRobotDataset supervised continuation")
    sft_commands = sft.add_subparsers(dest="command", required=True)
    sft_train = sft_commands.add_parser("train")
    sft_train.add_argument("--config", type=Path, required=True)
    sft_train.add_argument("--backend", choices=("act", "smolvla", "pi0"), required=True)
    sft_train.add_argument("--stage", choices=("sft", "continued_sft"), required=True)
    sft_train.add_argument("--resume", action="store_true")
    rl = families.add_parser("rl", help="online residual PPO plasticity study")
    rl_commands = rl.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        command = rl_commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--backend", choices=("act", "smolvla"), required=True)
        command.add_argument(
            "--stage", choices=("rl_head", "rl_representation"), required=True
        )
        if name == "train":
            command.add_argument("--resume", action="store_true")
    recovery_rl = families.add_parser(
        "recovery-rl",
        help="gated Recovery RL v2 foundation protocol",
    )
    recovery_commands = recovery_rl.add_subparsers(dest="command", required=True)
    for name in ("calibrate", "screen", "oracle-gate", "anchor-screen"):
        command = recovery_commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--resume", action="store_true")
    report = families.add_parser("report", help="aggregate C/U/P study evidence")
    report_commands = report.add_subparsers(dest="command", required=True)
    report_build = report_commands.add_parser("build")
    report_build.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.family == "recovery-rl":
            from .rl.protocol import run_recovery_command
            from .rl.v2_config import load_recovery_rl_v2_config

            recovery_config = load_recovery_rl_v2_config(args.config)
            result = run_recovery_command(
                recovery_config,
                args.command,
                resume=args.resume,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return
        config = load_study_config(args.config)
        if args.family == "state-bank" and args.command == "collect":
            result = collect_state_bank(config)
        elif args.family == "state-bank" and args.command == "inspect":
            result = inspect_state_bank(config)
        elif args.family == "latents" and args.command == "extract":
            result = extract_latents(
                config,
                backend=args.backend,
                stage=args.stage,
                partition=args.partition,
                limit=args.limit,
            )
        elif args.family == "latents" and args.command == "inspect":
            result = inspect_latents(
                config,
                backend=args.backend,
                stage=args.stage,
                partition=args.partition,
                limit=args.limit,
            )
        elif args.family == "probes" and args.command == "train":
            result = train_probe_suite(
                config, backend=args.backend, stage=args.stage, model_kind=args.model
            )
        elif args.family == "interventions" and args.command == "run":
            result = run_interventions(config, backend=args.backend, stage=args.stage)
        elif args.family == "interventions" and args.command == "rollout":
            result = run_closed_loop_interventions(
                config, backend=args.backend, stage=args.stage
            )
        elif args.family == "policy" and args.command == "evaluate":
            result = evaluate_policy_stage(
                config, backend=args.backend, stage=args.stage, force=args.force
            )
        elif args.family == "measure" and args.command == "run":
            result = run_stage_measurement(
                config,
                backend=args.backend,
                stage=args.stage,
                secondary_probe=args.secondary_probe,
                closed_loop_intervention=args.closed_loop_intervention,
            )
        elif args.family == "sft" and args.command == "train":
            result = train_sft(
                config,
                backend=args.backend,
                stage=args.stage,
                resume=args.resume,
            )
        elif args.family == "rl" and args.command == "train":
            result = train_residual_rl(
                config,
                backend=args.backend,
                stage=args.stage,
                resume=args.resume,
            )
        elif args.family == "rl" and args.command == "evaluate":
            result = evaluate_residual_rl(
                config, backend=args.backend, stage=args.stage
            )
        elif args.family == "report" and args.command == "build":
            result = build_study_report(config)
        else:  # pragma: no cover - guarded by argparse
            raise RuntimeError("unreachable representation-study command")
        print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as error:
        print(
            json.dumps(
                {
                    "passed": False,
                    "error": type(error).__name__,
                    "message": str(error),
                },
                sort_keys=True,
            )
        )
        raise SystemExit(1) from error
