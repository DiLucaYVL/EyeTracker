"""Pinch calibration wizard logic (it sets the ball size), exercised without opening any window."""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from eyemouse.calibration_ui import PINCH_MIN_FRAMES, PINCH_STEPS, CalibrationScreen
from eyemouse.config import Config


def make_wizard() -> CalibrationScreen:
    screen = CalibrationScreen.__new__(CalibrationScreen)      # skip __init__: it creates a fullscreen Tk window
    screen.cfg = Config()
    screen.cfg.save = lambda *a, **k: None                     # never touch the user's config file
    screen.pinch_step, screen.pinch_t0, screen.pinch_buf, screen.pinch_vals = 0, 0.0, [], {}
    screen.msg, screen.msg_color = "", ""
    screen._set_phase = mock.Mock()
    return screen


def frames(index_value, middle_value, n=20, noise=0.02, seed=0):
    """Wizard samples: (thumb-index distance, thumb-middle distance) in hand sizes."""
    rng = np.random.default_rng(seed)
    return [(float(index_value + rng.normal(0, noise)), float(middle_value + rng.normal(0, noise))) for _ in range(n)]


def run_all_steps(screen, open_=(0.9, 0.85), index=(0.09, 0.85), middle=(0.88, 0.12)):
    for values in (open_, index, middle):
        screen.pinch_buf = frames(*values)
        screen._finish_pinch_step(PINCH_STEPS[screen.pinch_step][0])


class PinchWizardTest(unittest.TestCase):
    def test_the_ball_size_makes_the_balls_touch_when_the_users_fingertips_touch(self):
        screen = make_wizard()
        run_all_steps(screen)                                      # contact levels 0.09 (index) and 0.12 (middle)
        on, _ = screen.cfg.pinch_thresholds()                      # two radii
        self.assertGreaterEqual(on, 0.12)                          # the looser contact (middle, 0.12) still counts as touching
        self.assertLess(on, 0.25)                                  # ...but the balls are not oversized
        self.assertTrue(screen.msg.startswith("Pinça calibrada — "))
        self.assertIn("bolas de", screen.msg)
        screen._set_phase.assert_called_with("intro")

    def test_a_loose_pinch_gets_bigger_balls(self):
        screen = make_wizard()
        run_all_steps(screen, index=(0.45, 0.85))                  # a hand whose index pinch only reaches 0.45
        on, _ = screen.cfg.pinch_thresholds()
        self.assertGreater(on, 0.45)                               # a default 0.19 would never fire
        self.assertLess(on, 0.9)

    def test_a_finger_that_cannot_be_separated_does_not_set_the_size(self):
        screen = make_wizard()
        run_all_steps(screen, middle=(0.88, 0.80))                 # the middle "pinch" barely differs from open
        on, _ = screen.cfg.pinch_thresholds()
        self.assertAlmostEqual(on, 2 * ((0.09 + 0.03) / 2), delta=0.03)      # from the index alone
        self.assertIn("parcialmente", screen.msg)

    def test_nothing_changes_when_neither_finger_can_be_separated(self):
        screen = make_wizard()
        before = screen.cfg.pinch_ball_size
        run_all_steps(screen, index=(0.85, 0.85), middle=(0.88, 0.80))
        self.assertEqual(screen.cfg.pinch_ball_size, before)

    def test_step_is_repeated_when_the_hand_was_not_detected(self):
        screen = make_wizard()
        screen.pinch_buf = frames(0.9, 0.85, n=PINCH_MIN_FRAMES - 1)
        screen._finish_pinch_step("open")
        self.assertEqual(screen.pinch_step, 0)                     # same step again
        self.assertNotIn("open", screen.pinch_vals)
        self.assertEqual(screen.pinch_buf, [])
        self.assertIn("não foi detectada", screen.msg)


if __name__ == "__main__":
    unittest.main()
