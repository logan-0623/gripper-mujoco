from __future__ import annotations

import glfw
import numpy as np

from interaction_vla.teleop import TeleopController


def press(controller: TeleopController, key: int) -> None:
    controller.handle_key(key, glfw.PRESS)


def release(controller: TeleopController, key: int) -> None:
    controller.handle_key(key, glfw.RELEASE)


def test_translation_and_rotation_keys_share_the_normalized_7d_contract() -> None:
    controller = TeleopController()
    for key in (glfw.KEY_W, glfw.KEY_A, glfw.KEY_R, glfw.KEY_Q, glfw.KEY_UP, glfw.KEY_LEFT):
        press(controller, key)

    action = controller.action()

    np.testing.assert_allclose(action[:3], np.asarray((1.0, 1.0, 1.0)) / np.sqrt(3.0))
    np.testing.assert_allclose(action[3:6], np.asarray((1.0, 1.0, 1.0)) / np.sqrt(3.0))
    assert action[6] == 1.0
    for key in (glfw.KEY_W, glfw.KEY_A, glfw.KEY_R, glfw.KEY_Q, glfw.KEY_UP, glfw.KEY_LEFT):
        release(controller, key)
    np.testing.assert_array_equal(controller.action()[:6], np.zeros(6))


def test_space_toggles_once_per_press_and_reset_escape_are_latched() -> None:
    controller = TeleopController()

    press(controller, glfw.KEY_SPACE)
    assert controller.action()[6] == 0.0
    controller.handle_key(glfw.KEY_SPACE, glfw.REPEAT)
    assert controller.action()[6] == 0.0
    release(controller, glfw.KEY_SPACE)
    press(controller, glfw.KEY_SPACE)
    assert controller.action()[6] == 1.0

    press(controller, glfw.KEY_Z)
    press(controller, glfw.KEY_ESCAPE)
    assert controller.discard_and_reset
    assert controller.quit
    controller.clear_reset_request()
    assert not controller.discard_and_reset
