"""E4: continue SmolVLA training with extra loss weight on the approach-to-grasp window.

Wraps the native ``lerobot.scripts.lerobot_train`` entry point. With
``--window-weight w > 1`` it installs a sample weighter (LeRobot's built-in
per-sample loss weighting path) that gives weight ``w`` to frames inside the
approach-to-grasp window and 1 elsewhere, renormalised to mean 1 per batch, so
the effective learning rate is unchanged and only where the gradient comes from
shifts. ``--window-weight 1`` trains with uniform weights through the same path.

The window is derived from actions alone, so every training episode is covered
without simulation: from ``--pre`` frames before the episode's first gripper
close command to ``--post`` frames after it. On the 100 annotated StateBank
demonstrations, [close-20, close+25] overlaps the label window
[gripper-target distance <= 0.10 m, stable grasp + 10] with median IoU 0.80
(10th percentile 0.66).

Usage:
    python scripts/e4_train.py --window-weight 3 -- <lerobot_train arguments>
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

GRIPPER = 6


def window_mask(actions: np.ndarray, pre: int, post: int) -> np.ndarray:
    """Boolean per-frame window for one episode's [T, 7] actions; empty if the gripper never closes."""
    closes = np.flatnonzero(actions[:, GRIPPER] > 0)
    mask = np.zeros(len(actions), dtype=bool)
    if len(closes):
        first = int(closes[0])
        mask[max(first - pre, 0):first + post + 1] = True
    return mask


def window_table(dataset_root: Path, pre: int, post: int) -> np.ndarray:
    """Boolean array indexed by the dataset's global frame ``index``."""
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(dataset_root / "data" / "chunk-*" / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet data under {dataset_root}")
    columns = {"index": [], "episode_index": [], "frame_index": [], "action": []}
    for path in files:
        table = pq.read_table(path, columns=list(columns))
        for name in columns:
            columns[name].append(table.column(name).to_numpy() if name != "action"
                                 else np.stack(table.column(name).to_pylist()))
    index = np.concatenate(columns["index"])
    episode = np.concatenate(columns["episode_index"])
    frame = np.concatenate(columns["frame_index"])
    action = np.concatenate(columns["action"])
    table = np.zeros(int(index.max()) + 1, dtype=bool)
    for ep in np.unique(episode):
        rows = np.flatnonzero(episode == ep)
        rows = rows[np.argsort(frame[rows])]
        table[index[rows]] = window_mask(action[rows], pre, post)
    return table


def install_window_weighter(table: np.ndarray, weight: float) -> None:
    import torch
    import lerobot.utils.sample_weighting as sample_weighting

    class WindowWeighter(sample_weighting.SampleWeighter):
        def __init__(self, device):
            self.device = device
            self.mask = torch.from_numpy(table)
            self.weights = torch.where(self.mask, weight, 1.0).float()
            self.seen = self.in_window = 0

        def compute_batch_weights(self, batch):
            index = batch["index"].detach().long().cpu().view(-1)
            weights = self.weights[index]
            in_window = int(self.mask[index].sum())
            self.seen += len(index)
            self.in_window += in_window
            weights = weights / weights.mean()
            return weights.to(self.device), {"window_fraction": in_window / len(index)}

        def get_stats(self):
            return {"window_fraction_total": self.in_window / max(self.seen, 1)}

    def make(config, policy, device, dataset_root=None, dataset_repo_id=None):
        if config is None:
            return None
        if config.type != "window":
            raise ValueError("e4_train only installs the 'window' sample weighter")
        return WindowWeighter(device)

    sample_weighting.make_sample_weighter = make


def main() -> None:
    if "--" not in sys.argv:
        raise SystemExit("usage: e4_train.py [options] -- <lerobot_train arguments>")
    split = sys.argv.index("--")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--window-weight", type=float, required=True)
    p.add_argument("--pre", type=int, default=20)
    p.add_argument("--post", type=int, default=25)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--table-report", type=Path, required=True)
    args = p.parse_args(sys.argv[1:split])
    forwarded = sys.argv[split + 1:]
    if args.window_weight < 1.0:
        raise ValueError("window weight must be >= 1")
    table = window_table(args.dataset_root, args.pre, args.post)
    args.table_report.parent.mkdir(parents=True, exist_ok=True)
    args.table_report.write_text(json.dumps({
        "frames": int(len(table)), "window_frames": int(table.sum()), "window_fraction": float(table.mean()),
        "pre": args.pre, "post": args.post, "window_weight": args.window_weight}, indent=2) + "\n")
    install_window_weighter(table, args.window_weight)
    import lerobot.scripts.lerobot_train as lerobot_train

    sys.argv = [sys.argv[0], *forwarded, "--sample_weighting.type=window"]
    lerobot_train.main()


if __name__ == "__main__":
    main()
