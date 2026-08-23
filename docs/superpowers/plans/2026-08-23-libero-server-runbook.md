# LIBERO Server Runbook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one server-focused, copyable runbook for publishing the current implementation and running the LIBERO–SmolVLA smoke and formal pipelines on Linux/CUDA.

**Architecture:** Keep scientific definitions and detailed experiment rationale in the existing README/audit. Add a root-level operational document that separates repository publication, CUDA environment setup, the two required data sources, smoke gates, formal execution, resume behavior, and artifact checks. Link it from the README without changing any experiment code or configuration.

**Tech Stack:** Markdown, Git, Python 3.12, CUDA 12.8, LeRobot 0.6.1, LIBERO, Hugging Face Hub, tmux.

---

### Task 1: Add the server execution runbook

**Files:**
- Create: `SERVER_RUNBOOK.md`

- [x] **Step 1: Document publication state and safe Git commands**

State explicitly that uncommitted files are not transferred by `git push`, provide an allow-listed `git add`, and exclude `outputs/`, `data/`, virtual environments, and the user PDF.

- [x] **Step 2: Document the Linux/CUDA environment and data binding**

Provide exact commands for Python 3.12, `requirements-lerobot-linux-cuda.txt`, CUDA verification, `MUJOCO_GL=egl`, the original LIBERO HDF5 download, and the repository-relative `data/libero/raw` symlink.

- [x] **Step 3: Document smoke and formal experiment commands**

Use only the existing `interaction_vla.representation_study libero` CLI and the two committed configs. Keep State Bank collection, timeline approval, stage planning/snapshot/training, latent extraction, and probe reports in gate order.

- [x] **Step 4: Document restart behavior and artifact checks**

Distinguish idempotent State Bank/latent commands from SFT `--resume`, list expected output files, and state that intervention and RL remain blocked.

### Task 2: Add the README entry point

**Files:**
- Modify: `README.md`

- [x] **Step 1: Link the runbook near the Linux execution section**

Add a prominent relative link to `SERVER_RUNBOOK.md` while keeping the README scientific overview and existing command reference intact.

### Task 3: Verify documentation consistency

**Files:**
- Verify: `SERVER_RUNBOOK.md`
- Verify: `README.md`
- Verify: `configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml`
- Verify: `configs/representation_study/libero_smolvla_linux_cuda.yaml`

- [x] **Step 1: Check all referenced files and CLI commands**

Run repository searches for every config, requirements file, and command family referenced by the runbook.

- [x] **Step 2: Run formatting checks**

Run `git diff --check` and the LIBERO project-contract test. Expected result: no whitespace errors and the project-contract test passes.
