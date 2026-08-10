# TorchCodec Compatibility Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pair Torch 2.10 with TorchCodec 0.10 so the documented ACT rollout command completes without the fatal-looking TorchCodec native-library traceback.

**Architecture:** Treat the supported Torch/TorchCodec pair as an explicit macOS environment contract. Protect that contract in the existing configuration test, constrain both the direct requirements and frozen lock, then repair the local LeRobot environment and exercise the unchanged rollout entry point end to end.

**Tech Stack:** Python 3.12, pytest, pip requirements, PyTorch 2.10, TorchCodec 0.10, LeRobot 0.6.1, MuJoCo, Pillow.

---

### Task 1: Lock the Supported TorchCodec Pair

**Files:**
- Modify: `tests/interaction_vla/lerobot_bridge/test_config.py`
- Modify: `requirements-lerobot-macos.txt`
- Modify: `requirements-lerobot-macos.lock.txt`

- [ ] **Step 1: Write the failing dependency-contract test**

Append this test beside the existing validated-MuJoCo runtime test:

```python
def test_lerobot_environment_pairs_torch_2_10_with_torchcodec_0_10() -> None:
    requirements = Path("requirements-lerobot-macos.txt").read_text(encoding="utf-8")
    lock = Path("requirements-lerobot-macos.lock.txt").read_text(encoding="utf-8")

    assert "torch>=2.10,<2.11" in requirements.splitlines()
    assert "torchcodec>=0.10,<0.11" in requirements.splitlines()
    assert "torch==2.10.0" in lock.splitlines()
    assert "torchcodec==0.10.0" in lock.splitlines()
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gripper-mujoco-pycache-current \
  .venv/bin/python -m pytest \
  tests/interaction_vla/lerobot_bridge/test_config.py::test_lerobot_environment_pairs_torch_2_10_with_torchcodec_0_10 \
  -q
```

Expected: FAIL because `requirements-lerobot-macos.txt` has no TorchCodec constraint and the lock currently contains `torchcodec==0.11.1`.

- [ ] **Step 3: Add the minimal compatible constraints**

Place the direct TorchCodec constraint immediately after the Torch constraint:

```text
torch>=2.10,<2.11
torchcodec>=0.10,<0.11
torchvision>=0.25,<0.26
```

Replace the frozen lock entry with:

```text
torchcodec==0.10.0
```

Do not change any other dependency.

- [ ] **Step 4: Run the dependency-contract tests and verify GREEN**

Run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gripper-mujoco-pycache-current \
  .venv/bin/python -m pytest \
  tests/interaction_vla/lerobot_bridge/test_config.py \
  -q
```

Expected: all tests in `test_config.py` PASS.

- [ ] **Step 5: Commit the contract repair**

```bash
git add requirements-lerobot-macos.txt \
  requirements-lerobot-macos.lock.txt \
  tests/interaction_vla/lerobot_bridge/test_config.py
git commit -m "fix: pair TorchCodec 0.10 with Torch 2.10"
```

### Task 2: Repair and Validate the Local LeRobot Environment

**Files:**
- Verify only: `.venv-lerobot/`

- [ ] **Step 1: Install the corrected locked package**

Run:

```bash
.venv-lerobot/bin/python -m pip install torchcodec==0.10.0
```

Expected: pip uninstalls TorchCodec 0.11.1 and installs TorchCodec 0.10.0 without changing Torch 2.10.0.

- [ ] **Step 2: Verify package metadata and native import**

Run:

```bash
.venv-lerobot/bin/python -c \
  "import importlib.metadata as m; import torchcodec; print({'torch': m.version('torch'), 'torchcodec': m.version('torchcodec')})"
```

Expected:

```text
{'torch': '2.10.0', 'torchcodec': '0.10.0'}
```

The command must exit zero without `Could not load libtorchcodec`.

- [ ] **Step 3: Check the complete installed dependency graph**

Run:

```bash
.venv-lerobot/bin/python -m pip check
```

Expected: `No broken requirements found.`

### Task 3: Regression and Real Rollout Verification

**Files:**
- Verify: `outputs/lerobot/act_smoke/rollout.gif`
- Verify: `outputs/lerobot/act_smoke/rollout.json`

- [ ] **Step 1: Run the complete LeRobot bridge test suite**

Run:

```bash
HF_HOME=/private/tmp/gripper-mujoco-hf-cache \
PYTHONPYCACHEPREFIX=/private/tmp/gripper-mujoco-pycache-lerobot \
  .venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/lerobot_bridge \
  -q
```

Expected: all LeRobot bridge tests PASS, with only documented optional skips.

- [ ] **Step 2: Run the user's command unchanged**

Run:

```bash
.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge rollout \
  --config configs/lerobot_act_smoke_macos.yaml \
  --checkpoint outputs/lerobot/act_smoke/checkpoint \
  --object-count 2 \
  --gif outputs/lerobot/act_smoke/rollout.gif
```

Expected: exit code 0, no `Could not load libtorchcodec` output, and final JSON fields `passed=true`, `finite_rollout=true`, `steps=180`, and `gif_frames=91`. `task_success=false` with `terminal_reason=timeout` remains an allowed smoke-policy result.

- [ ] **Step 3: Audit the generated GIF and rollout JSON**

Run:

```bash
.venv-lerobot/bin/python -c \
  "import json; from pathlib import Path; from PIL import Image; p=Path('outputs/lerobot/act_smoke/rollout.gif'); im=Image.open(p); d=json.loads(Path('outputs/lerobot/act_smoke/rollout.json').read_text()); print({'size': im.size, 'frames': im.n_frames, 'duration_ms': im.info.get('duration'), 'passed': d['passed'], 'terminal_reason': d['terminal_reason']})"
```

Expected:

```text
{'size': (512, 296), 'frames': 91, 'duration_ms': 100, 'passed': True, 'terminal_reason': 'timeout'}
```

- [ ] **Step 4: Check repository integrity**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors and no uncommitted tracked changes.
