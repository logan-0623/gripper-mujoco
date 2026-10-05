"""Test whether a frozen candidate's action effect depends on physical contact context."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..state_bank.io import write_json_atomic
from .checkpoint_readouts import episode
from .flow_trace import file_hash
from .state_bank import load_state_bank


METRICS = ("deployed_full_rms", "deployed_translation_rms", "deployed_rotation_rms",
           "deployed_gripper_rms", "plan_full_rms")


def _finite_context(record):
    return (record.labels.contact is not None
            and record.labels.geometry is not None
            and np.isfinite(record.labels.geometry.gripper_target_distance))


def _features(record):
    """Nuisance matching variables; no action effect or stable-grasp label."""
    return np.asarray([record.frame_index, *record.observation.robot_state,
                       record.labels.geometry.gripper_target_distance], dtype=float)


def _match(records, state_ids, *, same_episode_only=False, max_match_distance=None):
    by_id = {record.state_id: record for record in records}
    selected = [by_id[state_id] for state_id in state_ids]
    usable = [record for record in selected if _finite_context(record)]
    if not usable:
        raise ValueError("no selected states have finite contact and geometry measurements")
    scale = np.asarray([np.nanstd(np.stack([_features(row) for row in usable]), axis=0)], dtype=float)[0]
    scale[scale < 1e-8] = 1.0
    interactions = [row for row in usable if bool(row.labels.contact.gripper_target)]
    references = [row for row in usable if not bool(row.labels.contact.gripper_target)]
    if not interactions or not references:
        raise ValueError("both contact-present and contact-absent states are required")
    pairs = []
    for source in interactions:
        same_episode = [row for row in references if episode(row) == episode(source)]
        pool = same_episode or [row for row in references
                                if (row.suite, row.task_id) == (source.suite, source.task_id)]
        if same_episode_only and not same_episode:
            continue
        if not pool:
            continue
        source_x = _features(source) / scale
        distances = [float(np.linalg.norm(source_x - _features(row) / scale)) for row in pool]
        index = int(np.argmin(distances))
        if max_match_distance is not None and distances[index] > max_match_distance:
            continue
        donor = pool[index]
        pairs.append({"interaction_state_id": source.state_id,
                      "reference_state_id": donor.state_id,
                      "task": [source.suite, source.task_id],
                      "interaction_episode": source.source_episode_id,
                      "reference_episode": donor.source_episode_id,
                      "same_episode": episode(source) == episode(donor),
                      "match_distance": distances[index]})
    if not pairs:
        raise ValueError("no task-comparable reference state exists")
    return selected, pairs, {"scale": scale.tolist(),
                             "interaction_states": len(interactions),
                             "reference_states": len(references)}


def _bootstrap(pair_values, seed, samples):
    grouped = defaultdict(list)
    for row in pair_values:
        grouped[tuple(row["task"])].append(row)
    if len(grouped) < 2:
        return {"status": "inconclusive", "reason": "fewer than two tasks with matched pairs"}
    episode_means = {}
    for task, rows in grouped.items():
        by_episode = defaultdict(list)
        for row in rows:
            by_episode[row["interaction_episode"]].append(row["difference"])
        episode_means[task] = {ep: float(np.mean(values)) for ep, values in by_episode.items()}
    if any(len(values) < 2 for values in episode_means.values()):
        status = "inconclusive"
        reason = "at least one task has fewer than two interaction episodes"
    else:
        status, reason = "measured", None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        task_values = []
        for episodes in episode_means.values():
            names = tuple(episodes)
            draw = rng.choice(names, size=len(names), replace=True)
            task_values.append(float(np.mean([episodes[name] for name in draw])))
        values.append(float(np.mean(task_values)))
    observed = float(np.mean([np.mean(list(episodes.values())) for episodes in episode_means.values()]))
    interval = np.quantile(values, [0.025, 0.975]).tolist()
    if interval[0] > 0:
        decision = "supports_context_dependence"
    elif interval[1] < 0:
        decision = "supports_reverse_context_dependence"
    else:
        decision = "inconclusive"
    return {"status": status, "decision": decision, "reason": reason,
            "observed_task_macro": observed, "bootstrap_ci_95": interval,
            "bootstrap_samples": samples, "tasks": len(grouped),
            "episodes": int(sum(len(values) for values in episode_means.values())),
            "episode_means": {str(task): values for task, values in episode_means.items()}}


def run(effects, bank, output, *, seed=20260929, bootstrap_samples=10000,
        max_states=None, same_episode_only=False, max_match_distance=None):
    if output.exists():
        raise FileExistsError(output)
    report_path = effects / "report.json"
    effects_path = effects / "effects.npz"
    report = json.loads(report_path.read_text())
    if not report.get("complete") or report.get("effects_sha256") != file_hash(effects_path):
        raise ValueError("incomplete or unbound effects report")
    if report.get("target_id") != "formation_0":
        raise ValueError("context contract is frozen to formation_0")
    state_ids = np.load(effects_path, allow_pickle=False)["state_ids"].tolist()
    if max_states is not None:
        if max_states < 4 or max_states > len(state_ids):
            raise ValueError("invalid max_states")
    records, manifest, _, _ = load_state_bank(bank)
    by_id = {record.state_id: record for record in records}
    if any(state_id not in by_id for state_id in state_ids):
        raise ValueError("effects contain a state outside the StateBank")
    if max_states is not None:
        # Smoke selection is only an engineering check: seed it with one
        # contact/non-contact pair per task when the prefix is unbalanced.
        ordered = [by_id[state_id] for state_id in state_ids]
        chosen = []
        tasks = sorted({(row.suite, row.task_id) for row in ordered})
        for task in tasks:
            contact = next((row for row in ordered
                            if (row.suite, row.task_id) == task
                            and _finite_context(row)
                            and row.labels.contact.gripper_target), None)
            reference = next((row for row in ordered
                              if (row.suite, row.task_id) == task
                              and _finite_context(row)
                              and not row.labels.contact.gripper_target), None)
            for row in (contact, reference):
                if row is not None and row.state_id not in chosen:
                    chosen.append(row.state_id)
        chosen.extend(row.state_id for row in ordered if row.state_id not in chosen)
        state_ids = chosen[:max_states]
    selected, pairs, matching = _match(
        records, state_ids, same_episode_only=same_episode_only,
        max_match_distance=max_match_distance)
    with np.load(effects_path, allow_pickle=False) as arrays:
        random_keys = sorted(key for key in arrays.files
                             if key.startswith("expert_late:matched_random_")
                             and key.endswith("/deployed_full_rms"))
        if len(random_keys) < 4:
            raise ValueError("at least four matched-random directions are required")
        arm_values = {}
        for metric in METRICS:
            target_key = f"expert_late:formation_0/{metric}"
            controls = [f"expert_late:{key.split(':', 1)[1].split('/', 1)[0]}/{metric}"
                        for key in random_keys]
            if target_key not in arrays or any(key not in arrays for key in controls):
                raise ValueError(f"missing effect metric: {metric}")
            arm_values[metric] = (arrays[target_key].mean(axis=1),
                                  np.stack([arrays[key].mean(axis=1) for key in controls], axis=1))
    index = {state_id: i for i, state_id in enumerate(state_ids)}
    rows = []
    for pair in pairs:
        i, j = index[pair["interaction_state_id"]], index[pair["reference_state_id"]]
        pair["interaction_contact"] = True
        pair["reference_contact"] = False
        rows.append(pair)
    results = {}
    for metric, (target, random) in arm_values.items():
        values = []
        for pair in rows:
            i, j = index[pair["interaction_state_id"]], index[pair["reference_state_id"]]
            interaction_effect = float(target[i] - np.median(random[i]))
            reference_effect = float(target[j] - np.median(random[j]))
            values.append({**pair, "interaction_effect": interaction_effect,
                           "reference_effect": reference_effect,
                           "difference": interaction_effect - reference_effect})
        results[metric] = {"matched_pairs": len(values),
                           "same_episode_pairs": int(sum(row["same_episode"] for row in values)),
                           "mean_match_distance": float(np.mean([row["match_distance"] for row in values])),
                           "by_task": {str(task): {
                               "pairs": int(sum(tuple(row["task"]) == task for row in values)),
                               "interaction_effect": float(np.mean([row["interaction_effect"] for row in values if tuple(row["task"]) == task])),
                               "reference_effect": float(np.mean([row["reference_effect"] for row in values if tuple(row["task"]) == task])),
                               "context_gap": float(np.mean([row["difference"] for row in values if tuple(row["task"]) == task]))}
                               for task in sorted({tuple(row["task"]) for row in values})},
                           "bootstrap": _bootstrap(values, seed, bootstrap_samples),
                           "rows": values}
    output.mkdir(parents=True)
    write_json_atomic(output / "report.json", {
        "schema": "smolvla_context_dependence_v1", "complete": True, "exploratory": True,
        "effects_report_sha256": file_hash(report_path), "effects_sha256": file_hash(effects_path),
        "state_bank_sha256": file_hash(bank / "manifest.json"), "state_ids": state_ids,
        "states": len(state_ids), "target": "formation_0", "controls": random_keys,
        "context_definition": "interaction = simulator gripper_target contact; reference = contact absent",
        "matching": "same episode when available, otherwise same task; nearest standardized frame_index + robot_state + gripper_target_distance",
        "matching_options": {"same_episode_only": same_episode_only,
                             "max_match_distance": max_match_distance},
        "matching_diagnostics": matching, "candidate_selection_uses_context_labels": False,
        "metrics": results, "bootstrap_seed": seed, "bootstrap_unit": "source interaction episode within task",
        "limits": ["development data", "matched references may be reused", "action RMS is not closed-loop utility",
                   "contact is an external simulator measurement, not a claim of an internal named feature"]})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effects", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--same-episode-only", action="store_true")
    parser.add_argument("--max-match-distance", type=float)
    args = parser.parse_args()
    run(args.effects, args.bank, args.output, seed=args.seed,
        bootstrap_samples=args.bootstrap_samples, max_states=args.max_states,
        same_episode_only=args.same_episode_only,
        max_match_distance=args.max_match_distance)
    print(f"ALL_DONE output={args.output}", flush=True)


if __name__ == "__main__":
    main()
