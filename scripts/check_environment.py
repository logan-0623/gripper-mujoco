"""Offline runtime checks. Run with: bash scripts/python.sh scripts/check_environment.py.

No model downloads, training, dataset changes or rendering window are required.
"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


def main() -> int:
    packages = ("torch", "torchvision", "torchcodec", "mujoco", "lerobot", "datasets", "transformers", "h5py")
    print(json.dumps({"python": platform.python_version(), "packages": {
        name: importlib.metadata.version(name) for name in packages
    }}, indent=2), flush=True)
    # Separate imports expose lazy optional-dependency errors without loading weights.
    checks = {
        "SmolVLA import (no weights)": "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy",
        "LeRobotDataset import": "from lerobot.datasets import LeRobotDataset",
        "TorchCodec native libraries": "from torchcodec.decoders import VideoDecoder",
        "MuJoCo simulation step": (
            "import mujoco, numpy as np; "
            "m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body pos=\"0 0 1\"><freejoint/><geom type=\"sphere\" size=\"0.1\"/></body></worldbody></mujoco>'); "
            "d=mujoco.MjData(m); mujoco.mj_step(m,d); "
            "assert np.isfinite(d.qpos).all() and d.time > 0"
        ),
    }
    failures = []
    for name, code in checks.items():
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        print(f"{'PASS' if result.returncode == 0 else 'FAIL'}: {name}", flush=True)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode:
            failures.append(name)
    if "TorchCodec native libraries" not in failures:
        try:
            with tempfile.TemporaryDirectory(prefix="gripper-video-check-") as directory:
                video = Path(directory) / "synthetic.mp4"
                subprocess.run([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                    "-i", "color=c=red:s=32x32:r=5", "-frames:v", "3",
                    "-c:v", "mpeg4", str(video),
                ], check=True)
                from torchcodec.decoders import VideoDecoder

                decoder = VideoDecoder(str(video))
                assert len(decoder) == 3 and tuple(decoder[0].shape) == (3, 32, 32)
                del decoder
            print("PASS: synthetic video encode/decode (3 RGB frames)")
        except Exception as error:
            failures.append("video roundtrip")
            print(f"FAIL: video roundtrip: {error}", file=sys.stderr)
    print("NOT CHECKED: model weights/inference, GPU, real LIBERO replay, closed-loop success")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
