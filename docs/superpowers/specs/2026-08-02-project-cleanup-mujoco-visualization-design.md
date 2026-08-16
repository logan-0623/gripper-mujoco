# Project Cleanup and MuJoCo Visualization Design

## Goal

Turn the repository into a focused interaction-Graph/VLA experiment, while preserving every current dataset, checkpoint, evaluation report, configuration, test, and Python environment. Add two user-facing commands: one for native MuJoCo rollout visualization and one for exporting a Flat-versus-Graph animated GIF that is embedded in the README.

## Conservative and Legacy Cleanup

Delete generated caches and unmistakable debris:

- `.pytest_cache/`;
- project-owned `__pycache__/` directories and `.pyc` files, excluding `.venv/`;
- every project `.DS_Store`;
- the accidental root file `=0.4.3`.

Delete the legacy tutorial surface that is unrelated to the current interaction-representation experiment:

- notebooks `1.collect_data.ipynb` through `8.smolvla.ipynb`;
- `asset/`, `media/`, `demo_data_example/`, and the root legacy `mujoco_env/` package;
- `train_model.py`, `pi0_omy.yaml`, `smolvla_omy.yaml`, and the CUDA-oriented `requirements.txt`.

Retain `.venv/`, `.gitignore`, `requirements-macos.txt`, `interaction_vla/`, `tests/`, `configs/`, `docs/`, and all of `outputs/`. Rewrite the README around the retained Graph/VLA experiment and its reproducible commands.

## Native Viewer Command

Create `interaction_vla.visualize` with a `viewer` subcommand. On macOS it is launched through `.venv/bin/mjpython` and uses MuJoCo's passive native viewer over the existing `MujocoTabletopEnv`; it does not introduce another simulator.

The command accepts:

- `--controller expert`, `flat`, or `graph`;
- `--checkpoint` for learned controllers;
- `--layout normal` or `crowded`;
- `--object-count`, `--seed`, `--max-steps`, and `--fps`;
- optional `--recovery-kind` for an expert recovery demonstration.

The target object is green, the nearest target distractor is orange, other active objects are blue, the gripper is dark gray, and the receptacle remains visually distinct. The rollout stops on success, wrong-object selection, drop, timeout, viewer close, or keyboard interrupt. Learned-controller inference reuses checkpoint normalization, graph construction, and policy code from the experiment rather than creating a second inference implementation.

Example:

```bash
.venv/bin/mjpython -m interaction_vla.visualize viewer \
  --controller graph \
  --checkpoint outputs/interaction_vla/recovery/graph/seed_0/checkpoint.pt \
  --layout crowded --object-count 5 --seed 100042
```

## GIF Export Command

Add an `export-gif` subcommand using `mujoco.Renderer` for offscreen RGB frames and Pillow for animated GIF encoding. Flat and Graph receive the exact same initial case. Their rollouts run independently and are rendered side by side with labels, step counters, and final termination status. Shorter completed rollouts hold their final frame while the other side finishes.

The command accepts Flat and Graph checkpoints, the same deterministic scene arguments as the viewer, frame size, FPS, maximum steps, and an output path. The normal documentation output is `docs/media/flat_vs_graph_crowded.gif`.

Example:

```bash
.venv/bin/python -m interaction_vla.visualize export-gif \
  --flat-checkpoint outputs/interaction_vla/recovery/flat/seed_0/checkpoint.pt \
  --graph-checkpoint outputs/interaction_vla/recovery/graph/seed_0/checkpoint.pt \
  --layout crowded --object-count 5 --seed 100042 \
  --output docs/media/flat_vs_graph_crowded.gif
```

The README embeds the generated GIF with a relative path and also documents both commands so a user can reproduce or replace it.

## Dependencies and Errors

Add `Pillow>=10.0` to `requirements-macos.txt`; do not add FFmpeg or notebook dependencies. Missing checkpoint files, representation mismatches, unsupported expert/recovery combinations, invalid output extensions, render failures, and non-macOS-style viewer launch errors must fail with actionable messages. Parent directories for GIF output are created safely.

## Testing and Verification

Unit tests cover deterministic controller rollout, checkpoint argument validation, target/anchor color assignment, side-by-side image composition, and a short real GIF export with valid dimensions and multiple frames. GUI tests never open a native window automatically. A manual smoke command verifies the native viewer.

After implementation, run the full test suite and static compilation without recreating cleanup artifacts by setting `PYTHONDONTWRITEBYTECODE=1` and disabling pytest's cache provider. Finally verify that all approved legacy targets and project-owned cache files are absent, all retained experiment artifacts still exist, and the README GIF can be opened by Pillow.

## Scope Boundaries

This change does not retrain policies, alter evaluation metrics, change physics or task thresholds, introduce RGB observations into training, or modify existing checkpoint/data formats. The visualization is a verification and communication layer over the already completed experiment.
