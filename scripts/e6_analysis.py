"""Summarise E6 (flow-stage / module mixing inside the grasp window) against its preregistered criteria.

Arms (flow_stage_mix.py): N 10k, A 25k, E steps 0-4 from 25k, L steps 5-9 from 25k,
V 25k VLM side + 10k expert, X 10k VLM side + 25k expert. Unpushed: tasks 0-3,
initial states 0-39; pushed 8 steps: N/A/E/L on states 0-19. Also completes E3b
on tasks 0-3 (10k/25k alone, pushed 8 steps; tasks 0-1 from E3, 2-3 refilled here).
Pairs are (task, initial state); bootstrap resamples pairs, 5000 draws.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from e4_analysis import E3, paired_ci

E6 = E3.parent / "e6_20261012"
ARMS = ("N", "A", "E", "L", "V", "X")


def outcomes(dirs: list[Path]) -> dict[tuple[int, int], bool]:
    """Per-task outputs count only when complete (eval_info.json written)."""
    cells = {}
    for d in dirs:
        for f in d.glob("task*/physical_events.json"):
            if (f.parent / "eval_info.json").is_file():
                for e in json.loads(f.read_text())["episodes"]:
                    cells[int(f.parent.name[4:]), int(e["initial_state_id"])] = bool(e["success"])
    return cells


def arm(name: str, push: int) -> dict:
    return outcomes([E6 / f"{name}_push{push}_t{t}" for t in range(4)])


def recovered(a: dict, n: dict, full: dict, rng) -> list[float]:
    """(arm - N) / (A - N) with a paired bootstrap CI."""
    keys = sorted(set(a) & set(n) & set(full))
    x = np.asarray([[a[k], n[k], full[k]] for k in keys], float)
    def ratio(m):
        gap = m[:, 2].mean() - m[:, 1].mean()
        return (m[:, 0].mean() - m[:, 1].mean()) / gap if gap else np.nan
    boots = [ratio(x[i]) for i in rng.integers(0, len(x), (5000, len(x)))]
    return [round(float(ratio(x)), 2), round(float(np.nanquantile(boots, 0.025)), 2),
            round(float(np.nanquantile(boots, 0.975)), 2), len(keys)]


def retention(plain: dict, pushed: dict) -> list[int]:
    kept = [pushed[k] for k in pushed if plain.get(k)]
    return [int(sum(kept)), len(kept)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rng = np.random.default_rng(0)
    plain = {a: arm(a, 0) for a in ARMS}
    pushed = {a: arm(a, 8) for a in ARMS[:4]}
    report = {"unpushed": {}, "pushed8": {}, "decision": {}}
    for a in ARMS:
        report["unpushed"][a] = {
            "success": [sum(plain[a].values()), len(plain[a])],
            "by_task": {t: sum(v for k, v in plain[a].items() if k[0] == t) for t in range(4)},
            "minus_N": paired_ci(plain[a], plain["N"], rng) if a != "N" else None,
            "recovered": recovered(plain[a], plain["N"], plain["A"], rng) if a not in ("N", "A") else None}
    for a in ARMS[:4]:
        report["pushed8"][a] = {
            "success": [sum(pushed[a].values()), len(pushed[a])],
            "retention": retention(plain[a], pushed[a]),
            "minus_N": paired_ci(pushed[a], pushed["N"], rng) if a != "N" else None}
    e_minus_l = paired_ci(plain["E"], plain["L"], rng)
    rec_e, rec_l = report["unpushed"]["E"]["recovered"][0], report["unpushed"]["L"]["recovered"][0]
    gap = report["unpushed"]["A"]["minus_N"]
    report["decision"] = {
        "A_minus_N": gap, "gap_measurable": gap[1] > 0,
        "E_minus_L": e_minus_l, "E_minus_L_pushed8": paired_ci(pushed["E"], pushed["L"], rng),
        "V_minus_X": paired_ci(plain["V"], plain["X"], rng),
        "early": gap[1] > 0 and e_minus_l[1] > 0 and rec_e >= 2 / 3,
        "late": gap[1] > 0 and e_minus_l[2] < 0 and rec_l >= 2 / 3,
        "distributed": gap[1] > 0 and rec_e < 2 / 3 and rec_l < 2 / 3 and e_minus_l[1] <= 0 <= e_minus_l[2]}
    # E3b on all four tasks: tasks 0-1 from E3, 2-3 refilled here; unpushed from E3 (states 0-19)
    e3b = {}
    for label, m in (("10k", "010000"), ("25k", "025000")):
        base = outcomes([E3 / f"k{label[:2]}_push0"])
        push = outcomes([E3 / f"k{label[:2]}_push8"])
        push = {k: v for k, v in push.items() if k[0] in (0, 1)}
        push.update(outcomes([E6 / f"e3b_{m}_push8_t{t}" for t in (2, 3)]))
        e3b[label] = {"unpushed": base, "pushed8": push}
    report["e3b_tasks0_3"] = {
        label: {"pushed8": [sum(d["pushed8"].values()), len(d["pushed8"])],
                "retention": retention(d["unpushed"], d["pushed8"])} for label, d in e3b.items()}
    both = {k for k in e3b["10k"]["pushed8"] if e3b["10k"]["unpushed"].get(k) and e3b["25k"]["unpushed"].get(k)}
    report["e3b_tasks0_3"]["25k_minus_10k_pushed8_both_succeed_unpushed"] = paired_ci(
        {k: e3b["25k"]["pushed8"][k] for k in both}, {k: e3b["10k"]["pushed8"][k] for k in both}, rng)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
