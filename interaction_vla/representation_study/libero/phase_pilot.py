"""Six-rollout exploratory contact-triggered pilot; never a confirmation gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from ..state_bank.io import write_json_atomic
from .acquisition import _evaluation_command
from .flow_trace import file_hash


def run(offline: Path, checkpoint_root: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    specs=[]
    for step in ("005000","025000"):
        report=json.loads((offline/step/"report.json").read_text())
        if not report.get("complete") or report["effects_sha256"] != file_hash(offline/step/"effects.npz"):
            raise ValueError("offline engineering check incomplete")
        candidates=offline/step/"expert_late/candidates.json"
        for arm in ("baseline","formation_0","matched_random_0"):
            child=output/step/arm
            command=list(_evaluation_command(checkpoint_root/step/"pretrained_model",0,child,10,1,50))
            module=command.index("interaction_vla.representation_study.libero.capability_events")
            command[module]="interaction_vla.representation_study.libero.flow_intervention_eval"
            options=["--exploratory","--candidates",str(candidates),"--candidate-id",
                     "formation_0" if arm=="baseline" else arm,"--dose","0.5",
                     "--edit-mode","suppress" if arm!="matched_random_0" else "matched_suppress",
                     "--environment-phase","contact","--max-edited-chunks","1"]
            for stage in range(10): options.extend(("--edit-stage",str(stage)))
            if arm=="baseline": options.append("--observe-only")
            if arm=="matched_random_0": options.extend(("--allow-control","--match-candidate-id","formation_0"))
            command[module+1:module+1]=options
            specs.append({"checkpoint":step,"arm":arm,"command":command,"events":str(child/"physical_events.json"),
                          "candidate_sha256":file_hash(candidates)})
    write_json_atomic(output/"plan.json",{"schema":"smolvla_contact_pilot_v1","exploratory":True,
                                         "rollout_budget":6,"initial_state":10,"task":0,
                                         "note":"previously observed state; no independent confirmation; contact trigger may never fire",
                                         "runs":specs})
    rows=[]
    for spec in specs:
        print(f"ROLLOUT {spec['checkpoint']} {spec['arm']} task=0 initial_state=10",flush=True)
        subprocess.run(spec["command"],check=True)
        result=json.loads(Path(spec["events"]).read_text())
        episodes=result["episodes"]
        if len(episodes)!=1 or episodes[0]["initial_state_id"]!=10 or episodes[0]["task_id"]!=0:
            raise ValueError("pilot execution identity differs from plan")
        row=episodes[0]
        if "phase_edit" not in row or len(row["phase_edit"]["trigger_steps"])>1:
            raise ValueError("missing phase audit or exposure exceeded")
        rows.append({"checkpoint":spec["checkpoint"],"arm":spec["arm"],**row})
        write_json_atomic(output/"report.json",{"complete":len(rows)==6,"exploratory":True,
                                              "episodes":rows,"plan_sha256":file_hash(output/"plan.json"),
                                              "limits":"one paired initial state per checkpoint; engineering/pilot evidence only; no CI or gate"})
    print("ALL_DONE phase pilot",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("offline","checkpoint-root","output"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    run(**vars(parser.parse_args()))


if __name__=="__main__": main()
