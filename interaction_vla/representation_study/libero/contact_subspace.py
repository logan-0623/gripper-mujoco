"""Fit Contact readout directions on train-only expert activations."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler

from ..state_bank.io import write_bytes_atomic, write_json_atomic
from .checkpoint_readouts import episode
from .flow_diff import load_trace
from .flow_trace import file_hash
from .latents import _tree_sha256
from .state_bank import load_state_bank


def _direction(values, labels, alpha):
    keep = np.isfinite(labels)
    if keep.sum() < 8 or np.unique(labels[keep]).size < 2:
        raise ValueError("Contact training labels are missing or constant")
    scaler = StandardScaler().fit(values[keep])
    # Sparse LSQR follows the repository's macOS-safe Ridge path.
    model = Ridge(alpha=alpha, solver="lsqr").fit(csr_matrix(scaler.transform(values[keep])), labels[keep])
    raw = model.coef_ / scaler.scale_
    raw = np.asarray(raw, dtype=np.float64)
    norm = np.linalg.norm(raw)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Contact readout direction is degenerate")
    return raw / norm, float(model.score(scaler.transform(values[keep]), labels[keep]))


def run(bank, train_trace, output, *, alpha=10.0, rank=1, random_count=2):
    if output.exists():
        raise FileExistsError(output)
    ids, arrays, binding, manifest = load_trace(train_trace)
    if binding.get("partition") != "train" or binding.get("split_group") != "episode" or binding.get("flow_edit") is not None:
        raise ValueError("Contact subspace requires an unedited episode-split train trace")
    records, bank_manifest, _, split = load_state_bank(bank)
    if not bank_manifest.get("audit_passed"):
        raise ValueError("StateBank audit has not passed")
    by_id = {r.state_id:r for r in records}
    rows = [by_id[i] for i in ids.tolist()]
    if any(split.assignments.get(i) != "train" for i in ids.tolist()):
        raise ValueError("trace contains non-train state")
    episodes = {episode(r) for r in rows}
    labels = np.asarray([float(r.labels.contact.gripper_target) if r.labels.contact is not None else np.nan for r in rows])
    candidates=[]; directions={}
    for tap in ("expert_middle","expert_late"):
        values = arrays[tap].mean(axis=(1,3), dtype=np.float64)
        stage_dirs=[]; scores=[]
        for stage in range(values.shape[1]):
            direction, score = _direction(values[:,stage], labels, alpha)
            stage_dirs.append(direction); scores.append(score)
        direction = np.mean(stage_dirs, axis=0); direction /= np.linalg.norm(direction)
        directions[tap]=direction
        candidates.append({"id":"contact_0","role":"contact","tap":tap,"direction":direction.tolist(),
                           "stage_scores":scores,"label":"contact.gripper_target","fit_partition":"train"})
        rng=np.random.default_rng(2057736129)
        randoms=[]
        for _ in range(random_count):
            value=rng.normal(size=direction.shape)
            value-=np.dot(value,direction)*direction
            value/=np.linalg.norm(value); randoms.append(value)
        for index,value in enumerate(randoms):
            candidates.append({"id":f"matched_random_{index}","role":"matched_random","tap":tap,"direction":value.tolist(),"matched_to":"contact_0"})
        # A low-change control is the lowest empirical stage-averaged variance direction.
        pooled=values.reshape(-1,values.shape[-1]); _,_,vh=np.linalg.svd(pooled-pooled.mean(0),full_matrices=False)
        low=vh[-1]; low/=np.linalg.norm(low)
        candidates.append({"id":"low_change_0","role":"low_change","tap":tap,"direction":low.tolist(),"matched_to":"contact_0"})
    output.mkdir(parents=True)
    for tap in ("expert_middle","expert_late"):
        tap_rows=[x for x in candidates if x["tap"]==tap]
        basis=io.BytesIO(); np.savez(basis,mean=arrays[tap].mean(axis=(0,1,3)),state_ids=ids)
        write_bytes_atomic(output/f"{tap}/shared_basis.npz",basis.getvalue())
        payload={"schema":"smolvla_contact_subspace_v1","kind":"frozen_contact_candidates","tap":tap,
                 "selection_uses":"train Contact labels only","label":"contact.gripper_target",
                 "train_trace_binding_sha256":manifest["binding_sha256"],"state_bank_sha256":file_hash(bank/"manifest.json"),
                 "independent_episodes":len(episodes),"states":len(ids),"alpha":alpha,"rank":rank,
                 "shared_basis_sha256":file_hash(output/f"{tap}/shared_basis.npz"),"candidates":tap_rows}
        write_json_atomic(output/f"{tap}/candidates.json",payload)
    report={"schema":"smolvla_contact_subspace_v1","complete":True,"train_trace":str(train_trace.resolve()),
            "train_trace_binding_sha256":manifest["binding_sha256"],"state_bank_sha256":file_hash(bank/"manifest.json"),
            "states":len(ids),"independent_episodes":len(episodes),"label":"contact.gripper_target",
            "candidate_paths":[str((output/tap/"candidates.json").resolve()) for tap in ("expert_middle","expert_late")],
            "limits":["linear Ridge readout direction, not proof of a Contact neuron","labels select directions on train only","validation effects must be separate"]}
    write_json_atomic(output/"report.json",report); return report


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--bank",type=Path,required=True); p.add_argument("--train-trace",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--alpha",type=float,default=10.0); p.add_argument("--rank",type=int,default=1); p.add_argument("--random-count",type=int,default=2); a=p.parse_args(); print(json.dumps(run(**vars(a)),indent=2))


if __name__=="__main__": main()
