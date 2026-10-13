"""Summarise stage 0 rollouts against their preregistered criteria (docs/research-questions.md).

* E6c: XE (25k expert on flow steps 0-4, 10k prefix cache) and XL (steps 5-9),
  against E6's N/A and next to E6's E/L (same steps, 25k cache) as a 2x2.
* 0b: E6's N/A/E/L with E4 C@2500 as the base instead of 10k.
* Matched-state retention: E6 pushed-8 outcomes restricted to the initial
  states where N, A, E and L all succeed unpushed (descriptive).

All rollouts: Spatial tasks 0-3. Bootstrap resamples (task, initial state)
pairs within each task (task-stratified, as preregistered for stage 0), 5000 draws.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from e6_analysis import E6, outcomes

STAGE0 = E6.parent / "stage0_20261012"


def arm(root: Path, prefix: str) -> dict:
    return outcomes([root / f"{prefix}_t{t}" for t in range(4)])


def draws(keys, rng, n=5000) -> np.ndarray:
    """[n, len(keys)] indices, resampled within each task."""
    tasks = np.asarray([k[0] for k in keys])
    out = np.empty((n, len(keys)), int)
    for t in np.unique(tasks):
        where = np.flatnonzero(tasks == t)
        out[:, where] = rng.choice(where, (n, len(where)))
    return out


def ci(stat, columns, keys, rng) -> list:
    x = np.asarray([[c[k] for c in columns] for k in keys], float)
    boots = [stat(x[i]) for i in draws(keys, rng)]
    return [round(float(stat(x)), 3), round(float(np.nanquantile(boots, 0.025)), 3),
            round(float(np.nanquantile(boots, 0.975)), 3), len(keys)]


def diff(a: dict, b: dict, rng) -> list:
    keys = sorted(set(a) & set(b))
    return ci(lambda m: m[:, 0].mean() - m[:, 1].mean(), [a, b], keys, rng)


def recovered(a: dict, n: dict, full: dict, rng) -> list:
    """(arm - N) / (A - N)."""
    def ratio(m):
        gap = m[:, 2].mean() - m[:, 1].mean()
        return (m[:, 0].mean() - m[:, 1].mean()) / gap if gap else np.nan
    keys = sorted(set(a) & set(n) & set(full))
    return ci(ratio, [a, n, full], keys, rng)


def block(arms: dict, rng) -> dict:
    n, a = arms["N"], arms["A"]
    out = {name: {"success": [sum(v.values()), len(v)],
                  "by_task": [sum(x for k, x in v.items() if k[0] == t) for t in range(4)]}
           for name, v in arms.items()}
    out["A"]["minus_N"] = diff(a, n, rng)
    for name in arms:
        if name not in ("N", "A"):
            out[name]["minus_N"] = diff(arms[name], n, rng)
            out[name]["recovered"] = recovered(arms[name], n, a, rng)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rng = np.random.default_rng(0)
    report = {}

    e6 = {name: arm(E6, f"{name}_push0") for name in ("N", "A", "E", "L", "X")}
    e6c = {**e6, "XE": arm(STAGE0, "e6c_XE_push0"), "XL": arm(STAGE0, "e6c_XL_push0")}
    rec = block(e6c, rng)
    gap_ok = rec["A"]["minus_N"][1] > 0
    xe, xe_xl = rec["XE"]["recovered"][0], diff(e6c["XE"], e6c["XL"], rng)
    report["e6c"] = {"arms": rec, "XE_minus_XL": xe_xl, "E_minus_XE": diff(e6c["E"], e6c["XE"], rng),
                     "L_minus_XL": diff(e6c["L"], e6c["XL"], rng),
                     "decision": {"gap_measurable": gap_ok,
                                  "early_expert_sufficient": gap_ok and xe >= 2 / 3 and xe_xl[1] > 0,
                                  "needs_synergy": gap_ok and xe < 1 / 3}}

    b = {name: arm(STAGE0, f"b0b_{name}_push0") for name in ("N", "A", "E", "L")}
    rec = block(b, rng)
    gap_ok = rec["A"]["minus_N"][1] > 0
    e_l = diff(b["E"], b["L"], rng)
    report["0b"] = {"arms": rec, "E_minus_L": e_l,
                    "decision": {"gap_measurable": gap_ok,
                                 "early_holds": gap_ok and e_l[1] > 0 and rec["E"]["recovered"][0] >= 2 / 3,
                                 "early_fails": gap_ok and (rec["E"]["recovered"][0] < 1 / 3
                                                            or diff(b["L"], b["E"], rng)[1] > 0)}}

    pushed = {name: arm(E6, f"{name}_push8") for name in ("N", "A", "E", "L")}
    common = sorted(k for k in pushed["N"] if all(e6[a].get(k) for a in ("N", "A", "E", "L")))
    matched = {name: {k: pushed[name][k] for k in common if k in pushed[name]} for name in pushed}
    report["matched_retention"] = {
        "states_all_four_succeed_unpushed": len(common),
        "pushed8_success": {name: [sum(v.values()), len(v)] for name, v in matched.items()},
        "minus_N": {name: diff(matched[name], matched["N"], rng) for name in ("A", "E", "L")},
        "E_minus_L": diff(matched["E"], matched["L"], rng)}

    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
