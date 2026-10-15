"""Seed 2000 P1: success per checkpoint (10k from the replication N arm) and the stage where each episode ends:
never entered the grasp window / entered but never lifted / lifted then failed / success."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e6_analysis import outcomes
A = Path("/root/autodl-tmp/smolvla-official-reproduction-v2/acquisition")
P1, G = A / "s2000_p1_20261014", A / "s2000_e6_20261014"
dirs = {m: [P1 / f"k{m}_t{t}" for t in range(4)] for m in
        ("002500", "005000", "007500", "012500", "015000", "017500", "020000", "022500", "025000")}
dirs["010000"] = [G / f"N_push0_t{t}" for t in range(4)]
out = {}
for m in sorted(dirs):
    o = outcomes(dirs[m])
    out[m] = {"success": [sum(o.values()), len(o)], "by_task": [sum(v for k, v in o.items() if k[0] == t) for t in range(4)]}
    # stage of each episode: never entered window / entered but never lifted / lifted then failed / success
    st = {"no_window": 0, "window_no_lift": 0, "lift_then_fail": 0, "success": 0}
    for d in dirs[m]:
        for meta in d.glob("task*/window_metadata.json"):
            t = int(meta.parent.name[4:])
            for e in json.loads(meta.read_text())["episodes"]:
                ok = o.get((t, e["initial_state_id"]))
                st["success" if ok else "no_window" if e["window_opened_step"] is None else
                   "window_no_lift" if e["window_closed_step"] is None else "lift_then_fail"] += 1
    out[m]["stages"] = st
    print(m, out[m])
(P1 / "p1_summary.json").write_text(json.dumps(out, indent=2) + "\n")
