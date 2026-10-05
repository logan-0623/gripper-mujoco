"""Merge compatible candidate-effect arrays for a context sensitivity audit."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def merge(base: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    for step in ("005000", "025000"):
        old = base / "candidate_controls_random16_late_20260928" / f"full_{step}"
        new = base / "context_effects_spatial9_20260929" / step
        with np.load(old / "effects.npz", allow_pickle=False) as a, np.load(new / "effects.npz", allow_pickle=False) as b:
            if set(a.files) != set(b.files):
                raise ValueError("control contract mismatch")
            old_ids = set(a["state_ids"].tolist())
            keep = np.asarray([state_id not in old_ids for state_id in b["state_ids"]])
            ids = np.concatenate([a["state_ids"], b["state_ids"][keep]])
            out = output / f"full_{step}"
            shutil.rmtree(out, ignore_errors=True)
            out.mkdir()
            arrays = {"state_ids": ids}
            for key in a.files:
                if key == "state_ids":
                    continue
                if a[key].shape[1:] != b[key].shape[1:] or not np.isfinite(a[key]).all() or not np.isfinite(b[key]).all():
                    raise ValueError(f"invalid array: {key}")
                arrays[key] = np.concatenate([a[key], b[key][keep]], axis=0)
        np.savez_compressed(out / "effects.npz", **arrays)
        report = json.loads((old / "report.json").read_text())
        extra = json.loads((new / "report.json").read_text())
        report.update(states=len(ids), episodes=report["episodes"] + extra["episodes"],
                      merged_sources=[str(old), str(new)],
                      effects_sha256=hashlib.sha256((out / "effects.npz").read_bytes()).hexdigest())
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"MERGED {step} states={len(ids)}", flush=True)


if __name__ == "__main__":
    merge(Path("/root/autodl-tmp/smolvla-official-reproduction-v2/acquisition"),
          Path("/root/autodl-tmp/smolvla-official-reproduction-v2/acquisition/context_effects_merged_20260929"))
