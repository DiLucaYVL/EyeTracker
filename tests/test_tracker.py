"""Tests for the cursor/click logic and Windows input structures (no camera, no real clicks)."""
from __future__ import annotations

import ctypes
import queue
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from eyemouse import mouse
from eyemouse.calibration_ui import grid_points
from eyemouse.config import Config
from eyemouse.gaze_model import GazeModel
from eyemouse.tracker import Tracker
from tests.test_core import SCREEN, synthetic_dataset


def ready_model() -> GazeModel:
    rng = np.random.default_rng(0)
    model = GazeModel(SCREEN)
    x, y, g = synthetic_dataset(rng, grid_points(16))
    for gid in np.unique(g):
        model.add_samples(x[g == gid], y[g == gid][0])
    model.fit()
    return model


class CursorLogicTest(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.tracker = Tracker(self.cfg, ready_model(), queue.SimpleQueue())
        self.tracker.mouse_enabled = True
        patcher = mock.patch("eyemouse.tracker.mouse")
        self.mouse = patcher.start()
        self.addCleanup(patcher.stop)

    def drive(self, t, gaze, pinch=None, holding=False):
        self.tracker._drive_mouse(t, np.array(gaze, dtype=float) if gaze is not None else None, holding, pinch)

    def test_cursor_follows_gaze_outside_the_deadzone_only(self):
        self.drive(0.0, (100, 100))
        self.mouse.move_to.assert_called_with(100.0, 100.0)
        self.mouse.move_to.reset_mock()
        self.drive(0.1, (108, 104))            # inside the 20 px dead zone: cursor stays
        self.mouse.move_to.assert_not_called()
        self.drive(0.2, (200, 100))
        self.mouse.move_to.assert_called_with(200.0, 100.0)

    def test_left_pinch_clicks_at_the_locked_position_and_can_drag_after_hold(self):
        self.drive(0.0, (100, 100))
        self.mouse.move_to.reset_mock()
        self.drive(1.0, (100, 100), pinch="left")
        self.mouse.button_down.assert_called_once_with("left")
        self.mouse.move_to.assert_called_with(100.0, 100.0)          # click lands where the eyes were
        self.mouse.move_to.reset_mock()
        self.drive(1.1, (400, 400), pinch="left")                    # eyes wander during the precision lock
        self.mouse.move_to.assert_not_called()
        self.drive(1.5, (400, 400), pinch="left")                    # held past drag_hold_ms: drag
        self.mouse.move_to.assert_called_with(400.0, 400.0)
        self.drive(1.6, (400, 400), pinch=None)
        self.mouse.button_up.assert_called_once_with("left")

    def test_right_pinch_uses_the_right_button(self):
        self.drive(0.0, (300, 300))
        self.drive(0.5, (300, 300), pinch="right")
        self.mouse.button_down.assert_called_once_with("right")
        self.drive(0.6, (300, 300), pinch=None)
        self.mouse.button_up.assert_called_once_with("right")

    def test_disabling_control_while_pressed_releases_the_button(self):
        self.drive(0.0, (300, 300))
        self.drive(0.5, (300, 300), pinch="left")
        self.tracker.mouse_enabled = False
        self.drive(0.6, (300, 300), pinch="left")
        self.mouse.button_up.assert_called_once_with("left")

    def test_calibration_screen_suspends_all_output(self):
        self.tracker.control_suspended = True
        self.drive(0.0, (300, 300), pinch="left")
        self.mouse.move_to.assert_not_called()
        self.mouse.button_down.assert_not_called()

    def test_cursor_is_frozen_while_eyes_are_closed(self):
        self.drive(0.0, (100, 100))
        self.mouse.move_to.reset_mock()
        self.drive(0.1, (700, 500), holding=True)
        self.mouse.move_to.assert_not_called()

    def test_no_output_without_calibration(self):
        self.tracker.model.clear()
        self.drive(0.0, (100, 100), pinch="left")
        self.mouse.move_to.assert_not_called()
        self.mouse.button_down.assert_not_called()


class LearnFromClickTest(unittest.TestCase):
    def test_physical_click_adds_a_sample_only_when_near_the_prediction(self):
        tracker = Tracker(Config(), ready_model(), queue.SimpleQueue())
        tracker._schedule_refit = lambda: None
        feat = synthetic_dataset(np.random.default_rng(1), [(0.5, 0.5)], frames=1)[0][0]
        import time
        for _ in range(5):
            tracker._feat_hist.append((time.perf_counter(), feat))
        before = tracker.model.n_samples
        tracker.on_physical_click(50, 700)            # far from where the model thinks the user looks: ignored
        self.assertEqual(tracker.model.n_samples, before)
        tracker.on_physical_click(700, 400)           # close to the prediction: learned
        self.assertEqual(tracker.model.n_samples, before + 1)

    def test_no_learning_while_eye_control_is_on(self):
        tracker = Tracker(Config(), ready_model(), queue.SimpleQueue())
        tracker.mouse_enabled = True
        before = tracker.model.n_samples
        tracker.on_physical_click(700, 400)
        self.assertEqual(tracker.model.n_samples, before)


class ConfigTest(unittest.TestCase):
    def test_roundtrip_including_camera_props(self):
        cfg = Config()
        cfg.img_gamma = 1.4
        cfg.camera_props = {"saturation": 60, "gain": 12}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.json"
            cfg.save(path)
            loaded = Config.load(path)
        self.assertEqual(loaded.img_gamma, 1.4)
        self.assertEqual(loaded.camera_props, {"saturation": 60, "gain": 12})

    def test_unknown_and_missing_keys_are_tolerated(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.json"
            path.write_text('{"bubble_size": 120, "does_not_exist": 1}', encoding="utf-8")
            loaded = Config.load(path)
        self.assertEqual(loaded.bubble_size, 120)
        self.assertEqual(loaded.smoothing_beta, Config().smoothing_beta)


class WindowsInputTest(unittest.TestCase):
    def test_input_struct_matches_the_win32_layout(self):
        self.assertEqual(ctypes.sizeof(mouse._Input), 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28)

    def test_sendinput_accepts_a_zero_move(self):
        # MOUSEEVENTF_MOVE (0x1) with dx=dy=0: exercises SendInput without moving or clicking anything.
        inp = mouse._Input(type=0, u=mouse._InputUnion(mi=mouse._MouseInput(0, 0, 0, 0x0001, 0, 0)))
        self.assertEqual(mouse.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(mouse._Input)), 1)

    def test_screen_size_is_positive(self):
        w, h = mouse.screen_size()
        self.assertGreater(w, 0)
        self.assertGreater(h, 0)


if __name__ == "__main__":
    unittest.main()
