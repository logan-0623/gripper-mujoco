"""Small local GUI for sending language instructions to SmolVLA in MuJoCo."""
from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import tkinter as tk
from tkinter import ttk

import numpy as np

from interaction_vla.lerobot_bridge.capture import DualViewCapture
from interaction_vla.physics_env import FrankaContactEnv
from interaction_vla.smolvla_mujoco import SmolVLAMuJoCo, observation_batch, smolvla_state


def _photo(root: tk.Tk, path: Path, rgb: np.ndarray) -> tk.PhotoImage:
    values = np.asarray(rgb, dtype=np.uint8)
    if values.shape != (256, 256, 3):
        raise ValueError("GUI expects 256x256 RGB frames")
    path.write_bytes(b"P6\n256 256\n255\n" + values.tobytes())
    return tk.PhotoImage(master=root, file=str(path), format="PPM")


class SmolVLAGUI:
    def __init__(self, root: tk.Tk, checkpoint: Path, device: str) -> None:
        self.root = root
        root.title("SmolVLA · MuJoCo object selection")
        root.geometry("640x620")
        ttk.Style(root).configure("Title.TLabel", font=("Helvetica", 16, "bold"))
        ttk.Style(root).configure("ViewTitle.TLabel", font=("Helvetica", 11, "bold"))
        self.policy = SmolVLAMuJoCo(checkpoint, device=device)
        self.env = FrankaContactEnv(max_steps=180)
        self.capture = DualViewCapture(self.env.model, width=256, height=256)
        self.frame_dir = Path(tempfile.mkdtemp(prefix="smolvla-gui-"))
        self.agent_image: tk.PhotoImage | None = None
        self.wrist_image: tk.PhotoImage | None = None

        ttk.Label(root, text="SmolVLA MuJoCo control panel", style="Title.TLabel").grid(
            row=0, column=0, padx=12, pady=(12, 2), sticky="w"
        )
        controls = ttk.Frame(root, padding=12)
        controls.grid(row=1, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="Pick instruction").grid(row=0, column=0, sticky="w")
        self.task = tk.StringVar(value="pick up the green object and place it in the receptacle")
        ttk.Entry(controls, textvariable=self.task, width=62).grid(row=0, column=1, columnspan=3, sticky="ew")
        ttk.Label(controls, text="objects").grid(row=1, column=0, sticky="w")
        self.objects = tk.IntVar(value=3)
        ttk.Spinbox(controls, from_=2, to=5, textvariable=self.objects, width=5).grid(row=1, column=1, sticky="w")
        ttk.Label(controls, text="target index").grid(row=1, column=2, sticky="e")
        self.target = tk.IntVar(value=0)
        ttk.Spinbox(controls, from_=0, to=4, textvariable=self.target, width=5).grid(row=1, column=3, sticky="w")
        ttk.Button(controls, text="Reset", command=self.reset).grid(row=2, column=0, pady=(8, 0), sticky="w")
        ttk.Button(controls, text="Step policy", command=self.step).grid(row=2, column=1, pady=(8, 0), sticky="w")
        self.status = tk.StringVar(value="Press Reset to create a scene.")
        ttk.Label(controls, textvariable=self.status).grid(row=2, column=2, columnspan=2, sticky="e")

        views = ttk.Frame(root, padding=12)
        views.grid(row=2, column=0)
        agent_panel = ttk.Frame(views)
        agent_panel.grid(row=0, column=0, padx=8)
        ttk.Label(agent_panel, text="Agent view", style="ViewTitle.TLabel").pack(pady=(0, 4))
        self.agent = ttk.Label(agent_panel)
        self.agent.pack()
        wrist_panel = ttk.Frame(views)
        wrist_panel.grid(row=0, column=1, padx=8)
        ttk.Label(wrist_panel, text="Wrist view", style="ViewTitle.TLabel").pack(pady=(0, 4))
        self.wrist = ttk.Label(wrist_panel)
        self.wrist.pack()
        root.protocol("WM_DELETE_WINDOW", self.close)

    def _show(self) -> None:
        frame = self.capture.capture(self.env, include_teacher=False)
        self.agent_image = _photo(self.root, self.frame_dir / "agent.ppm", frame.views["agent"].rgb)
        self.wrist_image = _photo(self.root, self.frame_dir / "wrist.ppm", frame.views["wrist"].rgb)
        self.agent.configure(image=self.agent_image, text="")
        self.wrist.configure(image=self.wrist_image, text="")

    def reset(self) -> None:
        try:
            count = int(self.objects.get())
            target = int(self.target.get())
            if not 0 <= target < count:
                raise ValueError("target index must be smaller than object count")
            self.env.reset(seed=0, object_count=count, target_index=target)
            self._show()
            self.status.set(f"ready · target=object_{target} · step=0")
        except Exception as error:
            self.status.set(f"reset failed: {error}")

    def step(self) -> None:
        try:
            if not getattr(self.env, "_initialized", False):
                raise RuntimeError("press Reset first")
            frame = self.capture.capture(self.env, include_teacher=False)
            batch = observation_batch(
                image=frame.views["agent"].rgb,
                image2=frame.views["wrist"].rgb,
                state=smolvla_state(self.env),
                task=self.task.get(),
                device=self.policy.device,
            )
            action = self.policy.action(batch)
            transition = self.env.step(action)
            self._show()
            self.status.set(
                f"step={self.env.step_count} · reason={transition.reason.value} · "
                f"action={np.round(action, 3).tolist()}"
            )
        except Exception as error:
            self.status.set(f"step failed: {error}")

    def close(self) -> None:
        self.capture.close()
        shutil.rmtree(self.frame_dir, ignore_errors=True)
        self.root.destroy()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/pretrained/smolvla_libero"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    root = tk.Tk()
    SmolVLAGUI(root, args.checkpoint, args.device)
    root.mainloop()


if __name__ == "__main__":
    main()
