"""0c: how much of the remaining flow-matching loss does the VLM side still carry at each checkpoint?

Method-1 metric (preregistered in docs/research-questions.md, section 0c). For a
lineage c_0 < c_1 < ... (c_0 may be the initialisation, e.g. smolvla_base), and
window frames of the training demonstrations:

    L(E, V)  flow-matching loss when expert E reads the prefix cache of model V
             (the VLM side incl. the state projection, as in E6's ``--kv``)
    M_V(c) = [L(E_c, V_prev) - L(E_c, V_c)] / L(E_c, V_c)
    M_E(c) = [L(E_prev, V_c) - L(E_c, V_c)] / L(E_c, V_c)
    R(c)   = M_V / (M_V + M_E)

reported for the high-noise band (tau in [0.5, 1], tau = 1 is pure noise; the
early flow steps of F15) and the low-noise band (tau in [0, 0.5), control).
The switch point s* is the first checkpoint of the first two consecutive ones
with R < rho. Velocities go through ``denoise_step`` on a prefix cache built by
``flow_stage_mix.prefix_cache``, the same path the E6 arms used.

Frames: the E4 window ([first close - 20, first close + 25]) of the demos of
the chosen suite tasks, ``--frames`` sampled with a fixed seed; each frame gets
``--draws`` (tau, noise) pairs per band, shared by every (E, V) pair. CIs
bootstrap over frames. All checkpoints are held on the GPU at once.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/gripper-mujoco-hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BANDS = {"high": (0.5, 1.0), "low": (0.0, 0.5)}


def shares(own, old_v, old_e):
    """Arrays of per-frame losses -> (M_V, M_E, R) from the means."""
    own, old_v, old_e = (float(np.mean(x)) for x in (own, old_v, old_e))
    m_v, m_e = (old_v - own) / own, (old_e - own) / own
    return m_v, m_e, m_v / (m_v + m_e) if m_v + m_e else float("nan")


def switch_point(names, ratios, rho):
    """First checkpoint of the first two consecutive ones with R < rho, else None."""
    for i in range(len(ratios) - 1):
        if ratios[i] < rho and ratios[i + 1] < rho:
            return names[i]
    return None


def bind_cameras(batch: dict, image_features) -> dict:
    """Dataset cameras (image, image2) -> the policy's first two image features; later ones stay
    missing so ``prepare_images`` fills them as masked empty cameras, as in closed-loop eval."""
    sources = [k for k in ("observation.images.image", "observation.images.image2") if k in batch]
    if not sources:  # the lineage pre-processor already renamed them (rename_map in training)
        return batch
    out = {k: v for k, v in batch.items() if k not in sources}
    out.update({target: batch[source] for source, target in zip(sources, image_features)})
    return out


def load(path: Path, device: str, empty_cameras: int, vlm: Path | None):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = PreTrainedConfig.from_pretrained(str(path))
    config.device, config.empty_cameras = device, empty_cameras  # eval runs with empty_cameras=1
    if vlm is not None:
        config.vlm_model_name, config.load_vlm_weights = str(vlm), False
    return SmolVLAPolicy.from_pretrained(str(path), config=config).to(device).eval().requires_grad_(False)


def window_frames(dataset_root: Path, revision: str, suite: str, tasks, count: int, seed: int):
    """Global frame indices inside the E4 window of the chosen tasks' demos."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from libero.libero import benchmark
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from e4_train import window_table
    from interaction_vla.representation_study.libero.sources import _normalize_language

    wanted = {_normalize_language(benchmark.get_benchmark_dict()[suite]().get_task(t).language) for t in tasks}
    meta = LeRobotDatasetMetadata("lerobot/libero", root=dataset_root, revision=revision)
    language = {int(i): _normalize_language(str(t)) for t, i in zip(meta.tasks.index, meta.tasks["task_index"])}
    rows = pa.concat_tables([pq.read_table(f, columns=["episode_index", "task_index"])
                             for f in sorted(dataset_root.glob("data/chunk-*/*.parquet"))])
    task_of = dict(zip(rows["episode_index"].to_pylist(), rows["task_index"].to_pylist()))
    episodes = sorted(e for e, t in task_of.items() if language[t] in wanted)
    table = window_table(dataset_root, 20, 25)
    pool = np.concatenate([np.arange(int(meta.episodes[i]["dataset_from_index"]),
                                     int(meta.episodes[i]["dataset_to_index"])) for i in episodes])
    pool = pool[table[pool]]
    if len(pool) < count:
        raise ValueError(f"only {len(pool)} window frames for {suite} tasks {tasks}")
    return episodes, np.sort(np.random.default_rng(seed).choice(pool, count, replace=False)), float(meta.fps)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", action="append", required=True,
                   help="NAME=PATH, in training order; the first one only serves as 'prev'")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--revision", default="a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4")
    p.add_argument("--processor-from", type=Path, required=True,
                   help="checkpoint whose pre-processor normalises every model's input (lineage processors are identical)")
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task", type=int, action="append", required=True)
    p.add_argument("--frames", type=int, default=256)
    p.add_argument("--draws", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--rho", type=float, default=0.25)
    p.add_argument("--empty-cameras", type=int, default=1)
    p.add_argument("--vlm-model-name", type=Path, help="local SmolVLM2 directory when the Hub cache lacks it")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    import torch
    from torch.utils.data import default_collate
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from flow_stage_mix import prefix_cache
    from interaction_vla.representation_study.libero.flow_trace import file_hash

    names, paths = zip(*(c.split("=", 1) for c in args.checkpoint))
    models = {n: load(Path(path), args.device, args.empty_cameras, args.vlm_model_name)
              for n, path in zip(names, paths)}
    first = models[names[-1]]
    overrides = {"device_processor": {"device": args.device}}
    if args.vlm_model_name is not None:
        overrides["tokenizer_processor"] = {"tokenizer_name": str(args.vlm_model_name)}
    pre, _ = make_pre_post_processors(first.config, str(args.processor_from), preprocessor_overrides=overrides)
    episodes, frames, fps = window_frames(args.dataset_root, args.revision, args.suite, args.task, args.frames,
                                          args.seed)
    dataset = LeRobotDataset("lerobot/libero", root=args.dataset_root, revision=args.revision, episodes=episodes,
                             delta_timestamps={"action": [i / fps for i in range(first.config.chunk_size)]})
    position = {int(g): i for i, g in enumerate(dataset.hf_dataset["index"])}

    pairs = [(e, v) for i in range(1, len(names)) for e, v in
             ((names[i], names[i]), (names[i], names[i - 1]), (names[i - 1], names[i]))]
    pairs = list(dict.fromkeys(pairs))
    losses = {band: {pair: [] for pair in pairs} for band in BANDS}
    gen = torch.Generator().manual_seed(args.seed)
    with torch.no_grad():
        for start in range(0, len(frames), args.batch_size):
            items = [dataset[position[int(g)]] for g in frames[start:start + args.batch_size]]
            batch = pre(default_collate(items))
            valid = (~batch["action_is_pad"]).float()  # [B, chunk]; the policy's loss masks padding the same way
            caches = {}
            for n, m in models.items():
                images, img_masks = m.prepare_images(bind_cameras(batch, list(m.config.image_features)))
                caches[n] = prefix_cache(m.model, images, img_masks, batch["observation.language.tokens"],
                                         batch["observation.language.attention_mask"], m.prepare_state(batch))
            actions = first.prepare_action(batch)
            dim = batch["action"].shape[-1]
            bsize = actions.shape[0]
            for band, (lo, hi) in BANDS.items():
                per_frame = {pair: torch.zeros(bsize, device=args.device) for pair in pairs}
                for _ in range(args.draws):
                    tau = (lo + (hi - lo) * torch.rand(bsize, generator=gen)).to(args.device)
                    noise = torch.randn(actions.shape, generator=gen).to(args.device)
                    x_t = tau[:, None, None] * noise + (1 - tau[:, None, None]) * actions
                    target = (noise - actions)[..., :dim]
                    for e, v in pairs:
                        pad, cache = caches[v]
                        vel = models[e].model.denoise_step(prefix_pad_masks=pad, past_key_values=cache,
                                                           x_t=x_t, timestep=tau)[..., :dim]
                        err = ((vel - target) ** 2).mean(-1).mul(valid).sum(1) / valid.sum(1)
                        per_frame[(e, v)] += err / args.draws
                for pair in pairs:
                    losses[band][pair].extend(per_frame[pair].cpu().tolist())
            print(f"frames {start + bsize}/{len(frames)}", flush=True)

    rng = np.random.default_rng(args.seed)
    boot = rng.integers(0, len(frames), (2000, len(frames)))
    report = {"contract": {"checkpoints": dict(zip(names, map(str, paths))), "suite": args.suite, "tasks": args.task,
                           "frames": args.frames, "draws": args.draws, "bands": BANDS, "rho": args.rho,
                           "window": "E4 [first close - 20, first close + 25]", "seed": args.seed,
                           "empty_cameras": args.empty_cameras, "dataset_revision": args.revision,
                           "processor_from": str(args.processor_from),
                           "preprocessor_sha256": {q.name: file_hash(q) for q in
                                                   sorted(args.processor_from.glob("policy_preprocessor*"))}},
              "bands": {}}
    for band in BANDS:
        rows, ratios = [], []
        for i in range(1, len(names)):
            c, prev = names[i], names[i - 1]
            x = [np.asarray(losses[band][pair]) for pair in ((c, c), (c, prev), (prev, c))]
            m_v, m_e, r = shares(*x)
            ci = np.nanquantile([shares(*(a[b] for a in x))[2] for b in boot], [0.025, 0.975])
            rows.append({"checkpoint": c, "prev": prev, "L_own": float(x[0].mean()), "L_prev_vlm": float(x[1].mean()),
                         "L_prev_expert": float(x[2].mean()), "M_V": m_v, "M_E": m_e, "R": r,
                         "R_ci": [float(ci[0]), float(ci[1])]})
            ratios.append(r)
        report["bands"][band] = {"rows": rows, "switch_point": switch_point(names[1:], ratios, args.rho)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["bands"], indent=2))


if __name__ == "__main__":
    main()
