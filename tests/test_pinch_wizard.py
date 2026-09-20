"""Pinch calibration wizard logic, exercised without opening any window."""
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


def frames(index_value, middle_value, n=20, noise=0.02, seed=0, curl=1.9):
    """Wizard samples: (index metric, middle metric, other-fingers curl for an index pinch, ... for a middle pinch)."""
    rng = np.random.default_rng(seed)
    return [(float(index_value + rng.normal(0, noise)), float(middle_value + rng.normal(0, noise)),
             float(curl + rng.normal(0, 0.05)), float(curl + rng.normal(0, 0.05))) for _ in range(n)]


def run_all_steps(screen, open_=(0.9, 0.85), index=(0.09, 0.85), middle=(0.88, 0.12), curl=1.9):
    for values in (open_, index, middle):
        screen.pinch_buf = frames(*values, curl=curl)
        screen._finish_pinch_step(PINCH_STEPS[screen.pinch_step][0])


class PinchWizardTest(unittest.TestCase):
    def test_full_run_sets_per_finger_thresholds_between_pinched_and_open_values(self):
        screen = make_wizard()
        run_all_steps(screen)
        on_i, off_i = screen.cfg.pinch_thresholds("index")
        on_m, off_m = screen.cfg.pinch_thresholds("middle")
        self.assertTrue(0.09 < on_i < off_i < 0.9)
        self.assertTrue(0.12 < on_m < off_m < 0.85)
        self.assertTrue(screen.msg.startswith("Pinça calibrada — "))
        screen._set_phase.assert_called_with("intro")

    def test_calibrated_thresholds_make_the_users_own_pinch_fire_but_not_the_open_hand(self):
        screen = make_wizard()
        run_all_steps(screen, index=(0.45, 0.85))                 # a hand whose index pinch only reaches 0.45
        on_i, _ = screen.cfg.pinch_thresholds("index")
        self.assertGreater(on_i, 0.45)                            # would never fire with the default 0.30
        self.assertLess(on_i, 0.9)

    def test_a_finger_that_cannot_be_separated_keeps_its_previous_thresholds(self):
        screen = make_wizard()
        run_all_steps(screen, middle=(0.88, 0.80))                # "pinch" barely differs from open
        self.assertEqual(screen.cfg.pinch_thresholds("middle"), (Config().pinch_on_ratio, Config().pinch_off_ratio))
        self.assertNotEqual(screen.cfg.pinch_thresholds("index"), (Config().pinch_on_ratio, Config().pinch_off_ratio))
        self.assertIn("parcialmente", screen.msg)

    def test_guard_limit_is_learned_from_how_extended_the_other_fingers_are_in_the_users_pinch(self):
        screen = make_wizard()
        run_all_steps(screen, curl=1.9)
        self.assertTrue(screen.cfg.pinch_fist_guard)
        self.assertTrue(1.4 <= screen.cfg.pinch_guard_curl <= 1.6)
        self.assertIn("guarda de punho: ligada", screen.msg)

    def test_guard_is_switched_off_for_someone_who_pinches_with_folded_fingers(self):
        screen = make_wizard()
        run_all_steps(screen, curl=0.9)
        self.assertFalse(screen.cfg.pinch_fist_guard)
        self.assertIn("guarda de punho: desligada", screen.msg)

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
