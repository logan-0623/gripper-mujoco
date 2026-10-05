"""Locate flow stages whose fixed velocity response is context-dependent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..state_bank.io import write_json_atomic
from .context_dependence import _bootstrap, _match
from .flow_trace import file_hash
from .state_bank import load_state_bank


def run(effects: Path, bank: Path, output: Path, *, seed=20260929,
        bootstrap_samples=10000, same_episode_only=False, max_match_distance=None):
    if output.exists():
        raise FileExistsError(output)
    report_path, effects_path = effects / "report.json", effects / "effects.npz"
    report = json.loads(report_path.read_text())
    if not report.get("complete") or report.get("effects_sha256") != file_hash(effects_path):
        raise ValueError("incomplete or unbound effects report")
    if report.get("target_id") != "formation_0":
        raise ValueError("stage contract is frozen to formation_0")
    with np.load(effects_path, allow_pickle=False) as arrays:
        state_ids = arrays["state_ids"].tolist()
        random_keys = sorted(key for key in arrays.files
                             if key.startswith("expert_late:matched_random_")
                             and key.endswith("/fixed_velocity_rms_by_stage"))
        target_key = "expert_late:formation_0/fixed_velocity_rms_by_stage"
        if len(random_keys) < 4 or target_key not in arrays:
            raise ValueError("formation_0 and at least four matched-random stage arrays are required")
        target = arrays[target_key].mean(axis=1)
        random = np.stack([arrays[key].mean(axis=1) for key in random_keys], axis=1)
    if target.ndim != 2 or random.shape[0] != target.shape[0] or random.shape[2] != target.shape[1]:
        raise ValueError("expected state x flow_stage velocity arrays")
    records, _, _, _ = load_state_bank(bank)
    _, pairs, matching = _match(records, state_ids, same_episode_only=same_episode_only,
                                 max_match_distance=max_match_distance)
    index = {state_id: i for i, state_id in enumerate(state_ids)}
    stages = []
    for stage in range(target.shape[1]):
        rows = []
        for pair in pairs:
            i, j = index[pair["interaction_state_id"]], index[pair["reference_state_id"]]
            interaction = float(target[i, stage] - np.median(random[i, stage]))
            reference = float(target[j, stage] - np.median(random[j, stage]))
            rows.append({**pair, "stage": stage, "interaction_effect": interaction,
                         "reference_effect": reference, "difference": interaction - reference})
        boot = _bootstrap(rows, seed, bootstrap_samples)
        stages.append({"stage": stage, "matched_pairs": len(rows),
                       "same_episode_pairs": int(sum(row["same_episode"] for row in rows)),
                       "bootstrap": boot, "rows": rows})
    output.mkdir(parents=True)
    write_json_atomic(output / "report.json", {
        "schema": "smolvla_stage_context_dependence_v1", "complete": True, "exploratory": True,
        "effects_sha256": file_hash(effects_path), "state_bank_sha256": file_hash(bank / "manifest.json"),
        "states": len(state_ids), "target": "formation_0", "controls": random_keys,
        "metric": "fixed_velocity_rms_by_stage; candidate/control effect at fixed natural flow points",
        "matching": matching, "matching_options": {"same_episode_only": same_episode_only,
                                                       "max_match_distance": max_match_distance},
        "stages": stages, "bootstrap_seed": seed,
        "bootstrap_unit": "source interaction episode within task",
        "limits": ["offline fixed-point velocity response, not closed-loop utility",
                    "stage localization does not identify semantic planning or causal policy use"]})
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effects", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--same-episode-only", action="store_true")
    parser.add_argument("--max-match-distance", type=float)
    args = parser.parse_args()
    run(args.effects, args.bank, args.output, seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        same_episode_only=args.same_episode_only,
        max_match_distance=args.max_match_distance)
    print(f"ALL_DONE output={args.output}", flush=True)


if __name__ == "__main__":
    main()
