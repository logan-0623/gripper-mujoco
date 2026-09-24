"""Combine independent input, latent and behavior evidence without inventing SR."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..state_bank.io import write_json_atomic
from .flow_trace import file_hash


def _macro(summary, suffix):
    values=[]
    for key, row in summary.items():
        if key.endswith(suffix) and "/" in key:
            values.append(float(row["task_macro"]))
    return float(np.mean(values)) if values else None


def _success(events):
    rows=json.loads(events.read_text())["episodes"]
    return float(np.mean([bool(row["success"]) for row in rows])) if rows else None


def run(latent_reports, *, mask_reports=(), interaction_events=None, background_events=None, output):
    if output.exists(): raise FileExistsError(output)
    latent=[]
    for path in latent_reports:
        report=json.loads(path.read_text())
        if not report.get("complete") or report["effects_sha256"] != file_hash(path.parent/"effects.npz"):
            raise ValueError(f"incomplete latent report: {path}")
        target=report.get("target_id", "formation_0")
        controls=report.get("control_ids", ["matched_random_0"])
        target_vals=[]; control_vals=[]
        for key,row in report["summary"].items():
            if key.endswith("/deployed_full_rms") and target in key:
                target_vals.append(float(row["task_macro"]))
            if key.endswith("/deployed_full_rms") and any(c in key for c in controls):
                control_vals.append(float(row["task_macro"]))
        latent.append({"path":str(path),"checkpoint":report["checkpoint_sha256"],
                       "delta_A_contact_minus_random":float(np.mean(target_vals)-np.mean(control_vals)) if target_vals and control_vals else None,
                       "target_action_rms":float(np.mean(target_vals)) if target_vals else None,
                       "control_action_rms":float(np.mean(control_vals)) if control_vals else None})
    masks=[]
    for path in mask_reports:
        report=json.loads(path.read_text())
        if not report.get("complete") or report.get("mask_spec_sha256") is None:
            raise ValueError(f"mask report lacks explicit annotation provenance: {path}")
        summary=report["summary"]
        interaction=[float(row["task_macro"]) for key,row in summary.items() if key.startswith("mask:interaction:") and key.endswith("/deployed_full_rms")]
        background=[float(row["task_macro"]) for key,row in summary.items() if key.startswith("mask:background:") and key.endswith("/deployed_full_rms")]
        interaction_z=[float(row["task_macro"]) for key,row in summary.items() if key.startswith("mask:interaction:") and key.endswith("/expert_middle_projection_delta_by_stage")]
        background_z=[float(row["task_macro"]) for key,row in summary.items() if key.startswith("mask:background:") and key.endswith("/expert_middle_projection_delta_by_stage")]
        masks.append({"path":str(path),"delta_z_interaction_minus_background":
                      (float(np.mean(interaction_z)-np.mean(background_z)) if interaction_z and background_z else None),
                      "action_interaction":float(np.mean(interaction)) if interaction else None,
                      "action_background":float(np.mean(background)) if background else None})
    result={"schema":"smolvla_joint_evidence_v1","latent":latent,"mask":masks,
            "behavior":{
                "interaction_success_rate":_success(interaction_events) if interaction_events else None,
                "background_success_rate":_success(background_events) if background_events else None,
                "delta_SR_interaction_minus_background":(
                    _success(interaction_events)-_success(background_events)
                    if interaction_events and background_events else None),
            },"interpretation":{
                "status":"descriptive evidence alignment; no mediation or causal-path claim",
                "required_for_behavior_delta":"paired interaction/background closed-loop events with independent episodes",
                "missing":[]}}
    if not interaction_events or not background_events:
        result["interpretation"]["missing"].append("closed-loop interaction/background success events")
    write_json_atomic(output,result); return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--latent",type=Path,action="append",required=True); p.add_argument("--mask",type=Path,action="append",default=[]); p.add_argument("--interaction-events",type=Path); p.add_argument("--background-events",type=Path); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); print(json.dumps(run(a.latent,mask_reports=a.mask,interaction_events=a.interaction_events,background_events=a.background_events,output=a.output),indent=2))


if __name__=="__main__": main()
