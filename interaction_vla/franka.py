from __future__ import annotations

from pathlib import Path

import numpy as np


FRANKA_ASSET_ROOT = Path(__file__).parent / "assets"
FRANKA_MODEL_ROOT = FRANKA_ASSET_ROOT / "franka_emika_panda"
FRANKA_SCENE_PATH = FRANKA_MODEL_ROOT / "franka_tabletop.xml"
FRANKA_COMMIT = "71f066ad0be9cd271f7ed58c030243ef157af9f4"

ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
ARM_ACTUATOR_NAMES = tuple(f"actuator{index}" for index in range(1, 8))
FINGER_JOINT_NAMES = ("finger_joint1", "finger_joint2")
FINGER_ACTUATOR_NAME = "actuator8"
OBJECT_NAMES = tuple(f"object_{index}" for index in range(5))
CAMERA_NAMES = ("agentview", "wristview", "sideview", "topview")

HOME_QPOS = np.asarray(
    (0.0, -0.45, 0.0, -2.25, 0.0, 1.85, 0.785),
    dtype=np.float64,
)
TCP_OFFSET_IN_HAND = np.asarray((0.0, 0.0, 0.1034), dtype=np.float64)
