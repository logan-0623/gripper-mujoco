"""E6: which flow stages (and which side, VLM or expert) carry the 10k->25k gap inside the grasp window?

Extends E2 (``grasp_window_switch.py``): base (10k) and donor (25k) run in
lockstep on every step, and outside the window the base acts alone. Inside the
window the base's action chunk is integrated from a *mixed* velocity field:
flow step k (of ``num_steps`` Euler steps, k=0 at pure noise) uses the donor's
expert if k is in ``--donor-stages``, otherwise the base's. Each step reads a
prefix key/value cache (the VLM side, including the state projection) chosen by
``--kv``: ``own`` (the cache of whichever model computes that step), ``base``
or ``donor``. Arms used in E6:

* ``--donor-stages none --kv own``: 10k everywhere (base control).
* ``--donor-stages all --kv own``: 25k chunks in the window (E2 through this path).
* ``--donor-stages 0-4`` / ``5-9`` with ``--kv own``: early or late flow steps from 25k.
* ``--donor-stages none --kv donor``: 25k VLM side + 10k expert in the window.
* ``--donor-stages all --kv base``: 10k VLM side + 25k expert in the window.

Both models see the same preprocessed observation (processors are hash-checked
identical) and the same initial noise. Mixing is decided when a chunk is
computed, so it takes effect from the first chunk computed after window entry;
every arm shares this timing. ``--push-steps`` applies the E3b push at window
entry (``window_entry_probe.push_action``). Outcomes pair by task and
initial-state ID across arms.

E7 (noise scaling): ``--noise-scale`` multiplies the base's initial flow noise,
on every chunk (``--noise-scope all``) or only on chunks computed inside the
window (``window``). Noise is drawn before scaling, so random-number use is the
same for every scale. With ``--donor-stages none`` the base acts alone, so an
E7 arm is just ``--base <checkpoint> --donor-stages none --noise-scale s``.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/gripper-mujoco-hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from grasp_window_switch import GraspWindow, _write, summarize
from window_entry_probe import push_action
from interaction_vla.representation_study.libero.acquisition import _evaluation_command
from interaction_vla.representation_study.libero.flow_trace import file_hash
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks


def parse_stages(text: str, steps: int) -> frozenset[int]:
    if text == "none":
        return frozenset()
    if text == "all":
        return frozenset(range(steps))
    out = set()
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    if not out <= set(range(steps)):
        raise ValueError(f"stages {sorted(out)} outside 0..{steps - 1}")
    return frozenset(out)


def prefix_cache(flow, images, img_masks, lang_tokens, lang_masks, state):
    """The prefix pass of VLAFlowMatching.sample_actions: (prefix_pad_masks, past_key_values)."""
    embs, pad, att = flow.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
    _, cache = flow.vlm_with_expert.forward(
        attention_mask=make_att_2d_masks(pad, att), position_ids=torch.cumsum(pad, dim=1) - 1,
        past_key_values=None, inputs_embeds=[embs, None], use_cache=flow.config.use_cache)
    return pad, cache


def install_mix(base, donor, window: GraspWindow, stages: frozenset[int], kv: str, live: dict,
                noise_scale: float = 1.0, noise_scope: str = "all") -> None:
    flow, other = base.model, donor.model
    real_sample, real_denoise = flow.sample_actions, flow.denoise_step

    def sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        # sample noise here in every arm so random-number use does not depend on the window
        if noise is None:
            noise = flow.sample_noise((state.shape[0], flow.config.chunk_size, flow.config.max_action_dim),
                                      state.device)
        if noise_scale != 1.0 and (noise_scope == "all" or window.active):
            noise = noise * noise_scale
        if not window.active:
            return real_sample(images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs)
        caches = {"donor": prefix_cache(other, images, img_masks, lang_tokens, lang_masks, state)}
        calls = [0]

        def denoise_step(prefix_pad_masks, past_key_values, x_t, timestep):
            k = calls[0]
            calls[0] += 1
            model = other if k in stages else flow
            if kv == "donor" or (kv == "own" and model is other):
                prefix_pad_masks, past_key_values = caches["donor"]
            return (other.denoise_step if model is other else real_denoise)(
                prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values, x_t=x_t, timestep=timestep)

        flow.denoise_step = denoise_step
        try:
            out = real_sample(images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs)
        finally:
            flow.denoise_step = real_denoise
        if calls[0] != flow.config.num_steps:
            raise RuntimeError(f"expected {flow.config.num_steps} denoise calls, saw {calls[0]}")
        for item in live.values():
            item["mixed_chunks"] += 1
        return out

    flow.sample_actions = sample_actions


def _run_task(args: argparse.Namespace) -> None:
    from lerobot.envs.libero import LiberoEnv
    import lerobot.scripts.lerobot_eval as lerobot_eval
    from interaction_vla.representation_study.libero import capability_events
    from interaction_vla.representation_study.libero.runtime import LiberoOffscreenSimulator

    task = args.task[0]
    window = GraspWindow("grasp", args.approach_distance, args.lift)
    episodes, live, adapters = [], {}, {}

    def finish(env) -> None:
        item = live.pop(id(env), None)
        if item is not None:
            episodes.append({**item, "window_opened_step": window.opened_at,
                             "window_closed_step": window.closed_at})
            _write(args.output / "window_metadata.json", {"arm": args.arm, "episodes": episodes})

    original_install = capability_events.install_libero_event_recorder

    def install(**kwargs):
        real_reset, real_step, real_close = LiberoEnv.reset, LiberoEnv.step, LiberoEnv.close

        def reset(env, *call_args, **call_kwargs):
            finish(env)
            state_id = int(env.init_state_id) % len(env._init_states)  # same ID the recorder reports
            result = real_reset(env, *call_args, **call_kwargs)
            adapters[id(env)] = LiberoOffscreenSimulator.from_live_env(
                env._env, suite=args.suite, task_id=int(env.task_id),
                task_name=str(env.task), language=str(env.task_description))
            window.reset(adapters[id(env)]._privileged_frame())
            live[id(env)] = {"initial_state_id": state_id, "steps": 0, "donor_steps": 0, "mixed_chunks": 0,
                             "pushed_steps": 0, "push_start_xyz": None, "push_displacement_m": None}
            return result

        def step(env, action):
            item = live[id(env)]
            if window.opened_at is not None and item["pushed_steps"] < args.push_steps:
                if item["push_start_xyz"] is None:
                    item["push_start_xyz"] = window.frames[-1].gripper_pose[:3, 3].tolist()
                action = push_action(task, item["initial_state_id"], args.push_action).astype(
                    np.asarray(action).dtype)
                item["pushed_steps"] += 1
            result = real_step(env, action)
            window.observe(adapters[id(env)]._privileged_frame())
            if item["pushed_steps"] == args.push_steps and args.push_steps and item["push_displacement_m"] is None:
                item["push_displacement_m"] = float(np.linalg.norm(
                    window.frames[-1].gripper_pose[:3, 3] - np.asarray(item["push_start_xyz"])))
            return result

        def close(env):
            finish(env)
            return real_close(env)

        LiberoEnv.reset, LiberoEnv.step, LiberoEnv.close = reset, step, close
        original_install(**kwargs)

    real_make_policy = lerobot_eval.make_policy

    def make_policy(cfg, **kwargs):
        base = real_make_policy(cfg=cfg, **kwargs)
        donor_cfg = copy.deepcopy(cfg)
        donor_cfg.pretrained_path = args.donor
        donor = real_make_policy(cfg=donor_cfg, **kwargs).eval()
        install_mix(base, donor, window, parse_stages(args.donor_stages, base.config.num_steps), args.kv, live,
                    args.noise_scale, args.noise_scope)
        base_select, base_reset = base.select_action, base.reset

        def select_action(observation):
            own = base_select(observation)
            donor.select_action(observation)  # lockstep, as in E2: identical random-number use in every arm
            for item in live.values():
                item["steps"] += 1
                item["donor_steps"] += int(window.active)
            return own

        def reset():
            base_reset()
            donor.reset()

        base.select_action, base.reset = select_action, reset
        return base

    lerobot_eval.make_policy = make_policy
    capability_events.install_libero_event_recorder = install
    forwarded = list(_evaluation_command(args.base, task, args.output, args.initial_state_offset,
                                         args.episodes, args.initial_state_count, args.suite))
    separator = forwarded.index("--")
    forwarded[separator:separator] = ["--policy-noise-seed-base", "2057736129"]
    sys.argv = [sys.argv[0], *forwarded[5:]]
    capability_events.main()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--donor", type=Path, required=True)
    p.add_argument("--donor-stages", required=True, help="none | all | e.g. 0-4 or 0,3,5-9")
    p.add_argument("--kv", choices=("own", "base", "donor"), default="own")
    p.add_argument("--suite", default="libero_spatial", help="any suite the task registry supports")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", type=int, action="append", required=True)
    p.add_argument("--initial-state-offset", type=int, default=0)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--initial-state-count", type=int, default=50)
    p.add_argument("--approach-distance", type=float, default=0.10)
    p.add_argument("--lift", type=float, default=0.02)
    p.add_argument("--push-steps", type=int, default=0)
    p.add_argument("--push-action", type=float, default=1.0, help="E3b used 1.0")
    p.add_argument("--noise-scale", type=float, default=1.0, help="E7: multiply the initial flow noise")
    p.add_argument("--noise-scope", choices=("all", "window"), default="all")
    p.add_argument("--single-task", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    args.arm = (f"stages={args.donor_stages},kv={args.kv},push={args.push_steps},"
                f"noise={args.noise_scale}@{args.noise_scope}")
    parse_stages(args.donor_stages, 10)
    if args.single_task:
        _run_task(args)
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    for checkpoint in (args.base, args.donor):
        if not (checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(checkpoint / "model.safetensors")
    processors = {name: [file_hash(c / name) for c in (args.base, args.donor)]
                  for name in sorted(q.name for q in args.base.glob("policy_*processor*"))}
    if any(a != b for a, b in processors.values()):
        raise ValueError("base and donor use different pre/post-processors")
    args.output.mkdir(parents=True)
    _write(args.output / "contract.json", {
        "suite": args.suite,
        "base": str(args.base.resolve()), "donor": str(args.donor.resolve()), "arm": args.arm,
        "donor_stages": args.donor_stages, "kv": args.kv, "tasks": args.task,
        "initial_state_offset": args.initial_state_offset, "episodes": args.episodes,
        "initial_state_count": args.initial_state_count, "approach_distance_m": args.approach_distance,
        "lift_m": args.lift, "push_steps": args.push_steps, "push_action": args.push_action,
        "noise_scale": args.noise_scale, "noise_scope": args.noise_scope,
        "seed_base": 2057736129, "shared_processors_sha256": processors})
    for task in args.task:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--single-task", "--suite", args.suite,
                        "--base", str(args.base), "--donor", str(args.donor),
                        "--donor-stages", args.donor_stages, "--kv", args.kv,
                        "--output", str(args.output / f"task{task}"), "--task", str(task),
                        "--initial-state-offset", str(args.initial_state_offset),
                        "--episodes", str(args.episodes), "--initial-state-count", str(args.initial_state_count),
                        "--approach-distance", str(args.approach_distance), "--lift", str(args.lift),
                        "--push-steps", str(args.push_steps), "--push-action", str(args.push_action),
                        "--noise-scale", str(args.noise_scale), "--noise-scope", args.noise_scope],
                       check=True)
    print(json.dumps({k: v for k, v in summarize(args.output, args.task).items() if k != "rows"}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
