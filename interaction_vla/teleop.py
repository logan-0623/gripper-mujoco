from __future__ import annotations

import glfw
import numpy as np


class TeleopController:
    def __init__(self) -> None:
        self._pressed: set[int] = set()
        self._gripper_open = True
        self.discard_and_reset = False
        self.quit = False

    def handle_key(self, key: int, key_action: int) -> None:
        if key_action == glfw.PRESS:
            if key == glfw.KEY_SPACE:
                self._gripper_open = not self._gripper_open
            elif key == glfw.KEY_Z:
                self.discard_and_reset = True
            elif key == glfw.KEY_ESCAPE:
                self.quit = True
            self._pressed.add(key)
        elif key_action == glfw.RELEASE:
            self._pressed.discard(key)

    def action(self) -> np.ndarray:
        translation = np.asarray(
            (
                float(glfw.KEY_W in self._pressed) - float(glfw.KEY_S in self._pressed),
                float(glfw.KEY_A in self._pressed) - float(glfw.KEY_D in self._pressed),
                float(glfw.KEY_R in self._pressed) - float(glfw.KEY_F in self._pressed),
            ),
            dtype=np.float32,
        )
        rotation = np.asarray(
            (
                float(glfw.KEY_UP in self._pressed) - float(glfw.KEY_DOWN in self._pressed),
                float(glfw.KEY_LEFT in self._pressed) - float(glfw.KEY_RIGHT in self._pressed),
                float(glfw.KEY_Q in self._pressed) - float(glfw.KEY_E in self._pressed),
            ),
            dtype=np.float32,
        )
        translation = self._clip_by_norm(translation)
        rotation = self._clip_by_norm(rotation)
        return np.concatenate(
            (translation, rotation, np.asarray((float(self._gripper_open),), dtype=np.float32))
        ).astype(np.float32)

    def clear_reset_request(self) -> None:
        self.discard_and_reset = False

    @staticmethod
    def _clip_by_norm(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        return vector if norm <= 1.0 else vector / norm
