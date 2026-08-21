from __future__ import annotations

from .config import RepresentationStudyConfig
from .evaluation import evaluate_policy_stage
from .extraction import extract_latents
from .interventions import run_closed_loop_interventions, run_interventions
from .probes import train_probe_suite


def run_stage_measurement(
    config: RepresentationStudyConfig,
    *,
    backend: str,
    stage: str,
    secondary_probe: bool,
    closed_loop_intervention: bool,
) -> dict[str, object]:
    artifacts: dict[str, object] = {}
    artifacts["latents"] = extract_latents(
        config, backend=backend, stage=stage, partition="all", limit=None
    )
    artifacts["linear_probes"] = train_probe_suite(
        config, backend=backend, stage=stage, model_kind="linear"
    )
    if secondary_probe:
        artifacts["shallow_mlp_probes"] = train_probe_suite(
            config, backend=backend, stage=stage, model_kind="shallow_mlp"
        )
    artifacts["offline_interventions"] = run_interventions(
        config, backend=backend, stage=stage
    )
    artifacts["closed_loop_utility"] = evaluate_policy_stage(
        config, backend=backend, stage=stage
    )
    if closed_loop_intervention:
        artifacts["closed_loop_interventions"] = run_closed_loop_interventions(
            config, backend=backend, stage=stage
        )
    return {
        "passed": True,
        "backend": backend,
        "stage": stage,
        "secondary_probe": secondary_probe,
        "closed_loop_intervention": closed_loop_intervention,
        "artifacts": artifacts,
    }
