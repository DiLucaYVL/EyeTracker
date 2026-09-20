"""Unit tests for the pure-Python parts (no camera, no MediaPipe, no GUI)."""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from eyemouse import head3d
from eyemouse.blink import BlinkDetector
from eyemouse.calibration_ui import grid_points, pose_is_new
from eyemouse.config import Config
from eyemouse.features import N_FEATURES, eye_features, extract_features, head_pose
from eyemouse.filters import MedianFilter, OneEuroFilter
from eyemouse import imaging
from eyemouse.features import eye_region_box
from eyemouse.gaze_model import GazeModel, reject_outliers
from eyemouse.hands import FINGER_TIPS, PinchDetector, derive_ball_size, finger_curls, fingers_folded, pinch_metrics
from eyemouse.landmarks import HandData

SCREEN = (1366, 768)


def synthetic_dataset(rng, targets, frames=20, noise=0.002, head_wobble=0.0):
    """Features that depend (slightly non-linearly) on where the user looks."""
    xs, ys, gs = [], [], []
    for g, (tx, ty) in enumerate(targets):
        for _ in range(frames):
            u = 0.30 * (tx - 0.5) + 0.05 * (tx - 0.5) ** 2 + rng.normal(0, noise)
            v = 0.20 * (ty - 0.5) + rng.normal(0, noise)
            f = np.array([u, v, u * 0.9, v * 1.1, 0.3 - 0.05 * (ty - 0.5),
                          0.0, 0.0, 0.0, 0.0, 0.0, -40.0])
            f[5:8] += rng.normal(0, head_wobble, 3)
            xs.append(f)
            ys.append((tx, ty))
            gs.append(g)
    return np.array(xs), np.array(ys), np.array(gs)


class GazeModelTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def _fit(self, n_points=16, **kw):
        model = GazeModel(SCREEN)
        x, y, g = synthetic_dataset(self.rng, grid_points(n_points), **kw)
        for gid in np.unique(g):
            model.add_samples(x[g == gid], y[g == gid][0])
        return model, model.fit(cv=True)

    def test_recovers_mapping_on_unseen_points(self):
        model, report = self._fit()
        self.assertIsNotNone(report)
        targets = self.rng.uniform(0.1, 0.9, (30, 2))
        x, y, _ = synthetic_dataset(self.rng, [tuple(t) for t in targets], frames=1)
        errs = [np.linalg.norm((model.predict_norm(f) - t) * SCREEN) for f, t in zip(x, y)]
        self.assertLess(np.mean(errs), 0.02 * np.hypot(*SCREEN))

    def test_head_noise_is_tolerated(self):
        model, report = self._fit(head_wobble=0.02)
        self.assertLess(report.cv_frac, 0.06)

    def test_not_ready_with_too_few_targets(self):
        model = GazeModel(SCREEN)
        x, y, g = synthetic_dataset(self.rng, [(0.1, 0.1), (0.9, 0.9)])
        for gid in np.unique(g):
            model.add_samples(x[g == gid], y[g == gid][0])
        self.assertIsNone(model.fit())
        self.assertFalse(model.ready)

    def test_incremental_samples_improve_or_keep_accuracy(self):
        model, first = self._fit(9)
        x, y, g = synthetic_dataset(self.rng, grid_points(25))
        for gid in np.unique(g):
            model.add_samples(x[g == gid], y[g == gid][0])
        second = model.fit(cv=True)
        self.assertEqual(second.n_groups, 9 + 25)
        self.assertLessEqual(second.cv_px, first.cv_px * 1.5)

    def test_outlier_frames_do_not_ruin_the_fit(self):
        model = GazeModel(SCREEN)
        x, y, g = synthetic_dataset(self.rng, grid_points(16))
        bad = self.rng.random(len(x)) < 0.08
        x[bad, :4] += self.rng.normal(0, 0.15, (bad.sum(), 4))
        for gid in np.unique(g):
            model.add_samples(x[g == gid], y[g == gid][0])
        report = model.fit(cv=True)
        self.assertLess(report.cv_frac, 0.08)

    def test_save_load_roundtrip(self):
        model, _ = self._fit()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cal.json"
            model.save(path)
            other = GazeModel(SCREEN)
            self.assertTrue(other.load(path))
        f = synthetic_dataset(self.rng, [(0.3, 0.7)], frames=1)[0][0]
        np.testing.assert_allclose(model.predict_norm(f), other.predict_norm(f), atol=1e-4)

    def test_snapshot_restore(self):
        model, _ = self._fit()
        snap = model.snapshot()
        model.clear()
        self.assertFalse(model.ready)
        model.restore(snap)
        self.assertTrue(model.ready)

    def test_predict_px_stays_on_screen(self):
        model, _ = self._fit()
        f = np.full(N_FEATURES, 5.0)
        p = model.predict_px(f)
        self.assertTrue(0 <= p[0] <= SCREEN[0] - 1 and 0 <= p[1] <= SCREEN[1] - 1)

    def test_reject_outliers(self):
        rng = np.random.default_rng(0)
        f = np.zeros((30, N_FEATURES))
        f[:, :4] = rng.normal(0, 0.002, (30, 4))
        f[0, :4] = 0.5
        out = reject_outliers(f)
        self.assertLess(len(out), 30)
        self.assertNotIn(0.5, out[:, 0])


class FeatureTest(unittest.TestCase):
    def _face(self, iris_dx=0.0, iris_dy=0.0):
        pts = np.zeros((478, 3))
        # right eye (image left): corners 33 (left) / 133 (right)
        pts[33, :2], pts[133, :2] = (0.30, 0.40), (0.40, 0.40)
        pts[159, :2], pts[145, :2] = (0.35, 0.39), (0.35, 0.41)
        pts[468, :2] = (0.35 + iris_dx, 0.40 + iris_dy)
        # left eye (image right): 362 (left) / 263 (right)
        pts[362, :2], pts[263, :2] = (0.60, 0.40), (0.70, 0.40)
        pts[386, :2], pts[374, :2] = (0.65, 0.39), (0.65, 0.41)
        pts[473, :2] = (0.65 + iris_dx, 0.40 + iris_dy)
        return pts

    def test_iris_direction_signs(self):
        centre = extract_features(self._face(), 100, 100, None)
        right = extract_features(self._face(iris_dx=0.02), 100, 100, None)
        down = extract_features(self._face(iris_dy=0.01), 100, 100, None)
        self.assertAlmostEqual(centre[0], 0.0, places=6)
        self.assertGreater(right[0], 0.1)   # iris towards image right -> u > 0 for both eyes
        self.assertGreater(right[2], 0.1)
        self.assertGreater(down[1], 0.05)   # iris lower -> v > 0
        self.assertGreater(down[3], 0.05)

    def test_feature_is_scale_invariant(self):
        a = extract_features(self._face(0.02, 0.005), 100, 100, None)
        pts = self._face(0.02, 0.005)
        pts[:, :2] = 0.5 + (pts[:, :2] - 0.5) * 0.5  # face twice as far away
        b = extract_features(pts, 100, 100, None)
        np.testing.assert_allclose(a[:5], b[:5], atol=1e-6)

    def test_head_pose_roundtrip(self):
        yaw, pitch, roll = 0.3, -0.2, 0.1
        cy, sy, cp, sp, cr, sr = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
        m = np.eye(4)
        m[:3, :3] = ry @ rx @ rz * 1.7  # non-unit scale must be ignored
        m[:3, 3] = (1, 2, -40)
        np.testing.assert_allclose(head_pose(m), [yaw, pitch, roll, 1, 2, -40], atol=1e-9)

    def test_rejects_incomplete_landmarks(self):
        self.assertIsNone(extract_features(np.zeros((468, 3)), 100, 100, None))
        self.assertIsNotNone(eye_features(self._face()[:, :2], (33, 133, 159, 145, 468)))


class FilterTest(unittest.TestCase):
    def test_one_euro_smooths_jitter_but_follows_motion(self):
        rng = np.random.default_rng(0)
        f = OneEuroFilter(min_cutoff=0.6, beta=0.015)
        out = [f([500 + rng.normal(0, 8), 300 + rng.normal(0, 8)], i / 30) for i in range(120)]
        self.assertLess(np.std([o[0] for o in out[60:]]), 5)  # jitter reduced
        target = np.array([900.0, 300.0])
        for i in range(120, 180):
            last = f(target, i / 30)
        self.assertLess(np.linalg.norm(last - target), 15)    # converges after a jump

    def test_median_removes_single_outlier(self):
        m = MedianFilter(3)
        m([10, 10])
        m([10, 10])
        np.testing.assert_allclose(m([500, 500]), [10, 10])


IMG = (100, 100)


def hand_touching(index_r, middle_r, at=(0.5, 0.4), palm=(0.5, 0.6)):
    """A hand as MediaPipe reports it (image landmarks only: the pinch criterion is 2D).

    The thumb tip is at `at`; the index / middle tips are that many HAND SIZES away from it (the hand size is
    0.2 of the image here). `palm` shifts the whole hand vertically (to place two hands in one frame).
    """
    dx, dy = at[0] - 0.5, palm[1] - 0.6
    pts = np.zeros((21, 3))
    pts[0, :2], pts[9, :2] = (0.5, 0.8), (0.5, 0.6)              # wrist and middle knuckle: 0.2 apart = one hand size
    pts[5, :2], pts[17, :2] = (0.45, 0.62), (0.55, 0.62)         # narrower than the palm, so the palm sets the scale
    pts[4, :2] = (0.5, 0.4)
    pts[8, :2] = (0.5 + 0.2 * index_r, 0.4)
    pts[12, :2] = (0.5, 0.4 - 0.2 * middle_r)
    pts[:, 0] += dx
    pts[:, 1] += dy
    return HandData(pts, np.zeros((21, 3)))                       # no depth at all: the world coordinates must not matter


class PinchTest(unittest.TestCase):
    def run_seq(self, det, frames):
        return [det.update(i / 30, [hand_touching(*f)] if f else [], IMG) for i, f in enumerate(frames)]

    def test_distances_are_measured_in_hand_sizes(self):
        mi, mm = pinch_metrics(hand_touching(0.1, 0.8), *IMG)
        self.assertAlmostEqual(mi, 0.1, places=2)
        self.assertAlmostEqual(mm, 0.8, places=2)

    def test_the_only_criterion_is_that_the_two_balls_touch(self):
        cfg = Config()
        self.assertAlmostEqual(cfg.pinch_thresholds()[0], 2 * cfg.pinch_ball_size)     # two radii: the balls touch
        touching, apart = 2 * cfg.pinch_ball_size - 0.01, 2 * cfg.pinch_ball_size + 0.01
        self.assertEqual(self.run_seq(PinchDetector(cfg), [(touching, 0.9)])[-1], "left")
        self.assertIsNone(self.run_seq(PinchDetector(cfg), [(apart, 0.9)])[-1])

    def test_a_bigger_ball_makes_fingers_further_apart_count(self):
        cfg = Config()
        gap = 2 * cfg.pinch_ball_size + 0.06
        self.assertIsNone(self.run_seq(PinchDetector(cfg), [(gap, 0.9)])[-1])
        cfg.pinch_ball_size += 0.05
        self.assertEqual(self.run_seq(PinchDetector(cfg), [(gap, 0.9)])[-1], "left")

    def test_thumb_and_middle_balls_are_the_right_button(self):
        self.assertEqual(self.run_seq(PinchDetector(Config()), [(0.9, 0.1)])[-1], "right")

    def test_open_hand_never_clicks(self):
        self.assertEqual(self.run_seq(PinchDetector(Config()), [(0.9, 0.9)] * 10), [None] * 10)

    def test_the_click_follows_the_balls_frame_by_frame_with_no_lag(self):
        out = self.run_seq(PinchDetector(Config()), [(0.9, 0.9), (0.1, 0.9), (0.1, 0.9), (0.25, 0.9), (0.9, 0.9), (0.1, 0.9)])
        self.assertEqual(out, [None, "left", "left", None, None, "left"])       # pressed exactly while the balls touch

    def test_hand_lost_releases_the_button_after_a_moment(self):
        out = self.run_seq(PinchDetector(Config()), [(0.1, 0.9)] * 5 + [None] * 15)
        self.assertEqual(out[4], "left")
        self.assertIsNone(out[-1])

    def test_a_dropout_of_a_frame_or_two_does_not_release(self):
        out = self.run_seq(PinchDetector(Config()), [(0.1, 0.9)] * 3 + [None] * 2 + [(0.1, 0.9)] * 2)
        self.assertEqual(out, ["left"] * 7)

    def test_confirm_and_release_frames_are_optional_extras(self):
        cfg = Config()
        cfg.pinch_confirm_frames, cfg.pinch_release_frames = 3, 2
        out = self.run_seq(PinchDetector(cfg), [(0.1, 0.9)] * 4 + [(0.5, 0.9)] + [(0.1, 0.9)] + [(0.5, 0.9)] * 3)
        self.assertEqual(out[:2], [None, None])
        self.assertEqual(out[2], "left")
        self.assertEqual(out[4], "left")                  # one frame apart is not enough to release
        self.assertIsNone(out[-1])


class TwoHandsTest(unittest.TestCase):
    """Both hands count, not only the right one."""

    def update(self, det, hands, t=0.0):
        return det.update(t, hands, IMG)

    def test_either_hand_can_click(self):
        left_hand = hand_touching(0.1, 0.9, at=(0.25, 0.4))
        right_hand = hand_touching(0.1, 0.9, at=(0.75, 0.4))
        idle = lambda x: hand_touching(0.9, 0.9, at=(x, 0.4))
        for pinching, other, index in ((left_hand, idle(0.75), 0), (right_hand, idle(0.25), 1)):
            det = PinchDetector(Config())
            hands = [pinching, other] if index == 0 else [other, pinching]
            self.assertEqual(self.update(det, hands), "left")
            self.assertEqual(det.hand_index, index)

    def test_a_relaxed_second_hand_neither_holds_nor_cuts_the_click(self):
        det = PinchDetector(Config())
        pinch, idle = hand_touching(0.1, 0.9, at=(0.25, 0.4)), hand_touching(0.9, 0.9, at=(0.75, 0.4))
        self.assertEqual(self.update(det, [pinch, idle], 0.0), "left")
        self.assertEqual(self.update(det, [idle, pinch], 0.03), "left")                # same hands, other order: still pressed
        self.assertIsNone(self.update(det, [hand_touching(0.9, 0.9, at=(0.25, 0.4)), idle], 0.06))    # the pinching hand opened

    def test_a_hand_that_is_scrolling_cannot_click(self):
        det = PinchDetector(Config())
        touching = hand_touching(0.1, 0.9, at=(0.25, 0.4))
        self.assertIsNone(det.update(0.0, [touching], IMG, blocked=[True]))
        self.assertEqual(det.update(0.03, [touching], IMG, blocked=[False]), "left")

    def test_the_other_hand_can_still_click_while_one_scrolls(self):
        det = PinchDetector(Config())
        scrolling, pinch = hand_touching(0.1, 0.9, at=(0.25, 0.4)), hand_touching(0.1, 0.9, at=(0.75, 0.4))
        self.assertEqual(det.update(0.0, [scrolling, pinch], IMG, blocked=[True, False]), "left")
        self.assertEqual(det.hand_index, 1)


def hand_pose(curls=(1.9, 1.9, 1.9, 1.9)):
    """World landmarks of a hand (palm = 0.1 m) whose index/middle/ring/pinky are `curls` palm lengths from the wrist."""
    w = np.zeros((21, 3))
    w[9] = (0, 0.1, 0)
    dirs = [(-0.3, 1.0), (0.0, 1.0), (0.3, 1.0), (0.6, 1.0)]
    for (name, i), c, d in zip(FINGER_TIPS.items(), curls, dirs):
        w[i, :2] = np.array(d) / np.linalg.norm(d) * 0.1 * c
    w[4] = (0.12, 0.05, 0)                                            # the thumb: wherever, it is not part of the test
    pts = np.zeros((21, 3))
    pts[:, :2] = 0.5 + w[:, :2] * 2.0
    return HandData(pts, w)


class ScrollGestureTest(unittest.TestCase):
    """Scroll pose = index, middle, ring and pinky folded; the thumb is free (thumbs-up / "legal" sign or a fist)."""

    def test_curls_reflect_finger_extension_and_ignore_the_thumb(self):
        open_hand, folded = finger_curls(hand_pose()), finger_curls(hand_pose((0.8,) * 4))
        self.assertEqual(set(open_hand), {"index", "middle", "ring", "pinky"})
        self.assertGreater(min(open_hand.values()), 1.8)
        self.assertLess(max(folded.values()), 0.9)

    def test_four_folded_fingers_are_the_pose_with_the_thumb_anywhere(self):
        hand = hand_pose((0.8,) * 4)
        self.assertTrue(fingers_folded(hand))
        hand.world[4] = (0.02, 0.25, 0.0)                                # thumb up, far from the fingers: a thumbs-up
        self.assertTrue(fingers_folded(hand))

    def test_one_finger_that_stays_extended_is_not_the_pose(self):
        for i in range(4):
            curls = [0.8] * 4
            curls[i] = 1.9
            self.assertFalse(fingers_folded(hand_pose(curls)), FINGER_TIPS)
        self.assertFalse(fingers_folded(hand_pose()))                    # open hand
        self.assertFalse(fingers_folded(hand_pose((1.3,) * 4)))         # relaxed, half-folded


class DeriveBallSizeTest(unittest.TestCase):
    def test_balls_touch_when_the_users_fingertips_touch(self):
        ball = derive_ball_size(open_value=1.1, contact_value=0.13)
        self.assertGreaterEqual(2 * ball, 0.13)                            # the user's contact distance counts as touching
        self.assertLess(2 * ball, 0.30)                                    # ...without being oversized
        self.assertLess(2 * ball, 1.1 / 2)                                 # and far below the open hand

    def test_returns_none_when_open_and_pinched_are_not_separable(self):
        self.assertIsNone(derive_ball_size(open_value=0.5, contact_value=0.42))

    def test_a_pinch_that_never_reaches_zero_gets_a_bigger_ball(self):
        self.assertGreater(2 * derive_ball_size(open_value=1.0, contact_value=0.5), 0.5)

    def test_the_size_is_clamped(self):
        self.assertGreaterEqual(derive_ball_size(3.0, 0.0), 0.03)
        self.assertLessEqual(derive_ball_size(3.0, 1.9), 0.30)


class BlinkTest(unittest.TestCase):
    def test_hysteresis(self):
        det = BlinkDetector(Config())
        states = [det.update(i / 30, s, True) for i, s in enumerate([0.1, 0.6, 0.45, 0.4, 0.2, 0.1])]
        self.assertEqual(states, [False, True, True, True, False, False])
        self.assertTrue(det.settling(4 / 30 + 0.05))
        self.assertFalse(det.settling(4 / 30 + 0.5))


class CalibrationGridTest(unittest.TestCase):
    def test_grid_sizes_and_bounds(self):
        for n in (9, 16, 25):
            pts = grid_points(n)
            self.assertEqual(len(pts), n)
            self.assertTrue(all(0 < x < 1 and 0 < y < 1 for x, y in pts))


def head_dataset(rng, targets, poses_per_target, head_moves=True, noise=0.001):
    """Look at `targets` (normalised screen points) from random head poses.

    Physics: screen = 0.5 + 2 * (eye angle + head rotation + parallax * head shift), so the iris offset the camera
    sees is what is left after subtracting the head's contribution.
    """
    xs, ys, gs = [], [], []
    for g, (sx, sy) in enumerate(targets):
        for _ in range(poses_per_target):
            yaw, pitch = (rng.normal(0, 0.15), rng.normal(0, 0.10)) if head_moves else (0.0, 0.0)
            tx, ty = rng.normal(0, 3.0, 2) if head_moves else (0.0, 0.0)
            tz = -50.0 + (rng.normal(0, 5.0) if head_moves else 0.0)
            ex = (sx - 0.5) / 2 - yaw - 0.01 * tx
            ey = (sy - 0.5) / 2 - pitch - 0.01 * ty
            u, v = 0.6 * ex + rng.normal(0, noise), 0.6 * ey + rng.normal(0, noise)
            xs.append([u, v, u, v, 0.3, yaw, pitch, 0.0, tx, ty, tz])
            ys.append((sx, sy))
            gs.append(g)
    return np.array(xs), np.array(ys), np.array(gs)


class HeadMovementTest(unittest.TestCase):
    def _train(self, rng, head_moves):
        model = GazeModel(SCREEN)
        targets = grid_points(16)
        x, y, g = head_dataset(rng, targets, 1, head_moves=False)  # phase 1: still head
        for gid in np.unique(g):
            model.add_samples(np.repeat(x[g == gid], 10, axis=0), y[g == gid][0])
        if head_moves:  # phase 2: 9 targets, many head poses each
            nine = [(a, b) for a in (0.1, 0.5, 0.9) for b in (0.1, 0.5, 0.9)]
            x, y, g = head_dataset(rng, nine, 20, head_moves=True)
            for gid in np.unique(g):
                model.add_samples(x[g == gid], y[g == gid][0])
        return model, model.fit(cv=True)

    def _moving_head_error(self, rng, model):
        targets = [tuple(t) for t in rng.uniform(0.1, 0.9, (60, 2))]
        x, y, _ = head_dataset(rng, targets, 1, head_moves=True)
        return np.mean([np.linalg.norm((model.predict_norm(f) - t) * SCREEN) for f, t in zip(x, y)])

    def test_training_with_head_movement_compensates_the_head(self):
        rng = np.random.default_rng(7)
        still, _ = self._train(rng, head_moves=False)
        moving, report = self._train(rng, head_moves=True)
        diag = np.hypot(*SCREEN)
        err_still, err_moving = self._moving_head_error(rng, still), self._moving_head_error(rng, moving)
        self.assertGreater(err_still, 0.10 * diag)              # eye-only mapping breaks when the head moves
        self.assertLess(err_moving, 0.05 * diag)                # ...and is fixed by the head-movement phase
        self.assertLess(err_moving * 3, err_still)

    def test_head_spread_tells_whether_the_head_moved(self):
        rng = np.random.default_rng(3)
        still, _ = self._train(rng, head_moves=False)
        moving, _ = self._train(rng, head_moves=True)
        self.assertLess(still.head_spread(), 0.02)
        self.assertGreater(moving.head_spread(), 0.08)


def nose_tip_xy(pose, ref_dist=45.0):
    """Screen position of the nose tip (the last vertex of the first face polyline that ends at z = C + 2.6)."""
    for line, proj in zip(head3d.HEAD, head3d.project(pose, 300, ref_dist)):
        if line.kind == "face" and abs(line.pts[-1, 2] - (head3d.C + 2.6)) < 1e-6 and line.pts[-1, 1] == -1.6 and line.pts[-1, 0] == 0:
            return proj.xy[-1]
    raise AssertionError("nose tip not found")


class Head3DTest(unittest.TestCase):
    def test_frontal_nose_is_centred_horizontally(self):
        self.assertAlmostEqual(nose_tip_xy((0, 0, 0, 0, 0, -45.0))[0], 0.0, places=6)

    def test_positive_yaw_turns_to_the_subjects_left_which_is_screen_left_in_the_mirror(self):
        self.assertLess(nose_tip_xy((0.4, 0, 0, 0, 0, -45.0))[0], -10)
        self.assertGreater(nose_tip_xy((-0.4, 0, 0, 0, 0, -45.0))[0], 10)

    def test_positive_pitch_looks_down(self):
        frontal_y = nose_tip_xy((0, 0, 0, 0, 0, -45.0))[1]
        self.assertGreater(nose_tip_xy((0, 0.3, 0, 0, 0, -45.0))[1], frontal_y)

    def test_closer_head_is_drawn_larger(self):
        def height(tz):
            ys = np.concatenate([p.xy[:, 1] for p in head3d.project((0, 0, 0, 0, 0, tz), 300, 45.0) if p.kind == "skull"])
            return ys.max() - ys.min()
        self.assertGreater(height(-35.0), height(-45.0) * 1.2)
        self.assertLess(height(-60.0), height(-45.0) * 0.85)

    def test_live_pose_coincides_with_target_when_the_user_holds_the_reference_pose(self):
        ref = np.array([0.02, 0.10, -0.01, 1.0, 2.0, -33.0])
        feat = np.zeros(11)
        feat[5:11] = ref
        live = head3d.live_pose(feat, ref, 33.0)
        target = head3d.target_pose("still", 0.0, 33.0, base=tuple(ref[:3]))
        np.testing.assert_allclose(live, target, atol=1e-9)

    def test_every_prompt_animation_returns_to_the_reference_at_phase_zero(self):
        for key in head3d.HEAD_PROMPT_KEYS:
            if key == "circle":
                continue  # the circle starts at its rightmost point by design
            yaw, pitch, roll, dx, dy, dz = head3d.TARGET_POSES[key][1](0.0)
            self.assertAlmostEqual(abs(yaw) + abs(pitch) + abs(dz), 0.0, places=9)

    def test_runs_split_front_and_back(self):
        proj = head3d.project((0, 0, 0, 0, 0, -45.0), 300, 45.0)
        kinds = {front for p in proj if p.kind == "skull" for _, front in head3d.runs(p)}
        self.assertEqual(kinds, {True, False})


class PoseVarietyTest(unittest.TestCase):
    def test_keeps_only_poses_that_differ_enough(self):
        kept = []
        stream = [np.zeros(5), np.full(5, 0.005), np.array([0.2, 0, 0, 0, 0]), np.array([0.2, 0.01, 0, 0, 0]),
                  np.array([0, 0.3, 0, 0, 0])]
        for pose in stream:
            if pose_is_new(pose, kept):
                kept.append(pose)
        self.assertEqual(len(kept), 3)  # near-duplicates are dropped, real movements are kept


class ImagingTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.dark = (rng.normal(50, 8, (60, 200, 3))).clip(0, 255).astype(np.uint8)  # dull, dark eye region

    def test_neutral_returns_same_frame(self):
        cfg = Config()
        self.assertTrue(imaging.is_neutral(cfg))
        self.assertIs(imaging.adjust(self.dark, cfg), self.dark)

    def test_lut_is_monotonic_and_bounded(self):
        for b, c, g in ((0, 1, 1), (40, 1.5, 1.4), (-80, 2.5, 0.6)):
            lut = imaging.build_lut(b, c, g)
            self.assertTrue(np.all(np.diff(lut.astype(int)) >= 0))
            self.assertEqual(lut.dtype, np.uint8)

    def test_adjust_changes_brightness_and_contrast(self):
        cfg = Config()
        cfg.img_brightness, cfg.img_contrast, cfg.img_gamma = 30, 1.8, 1.4
        out = imaging.adjust(self.dark, cfg)
        self.assertGreater(out.mean(), self.dark.mean())
        self.assertGreater(out.std(), self.dark.std())

    def test_clahe_and_sharpen_keep_shape_and_dtype(self):
        cfg = Config()
        cfg.img_clahe, cfg.img_sharpen = 0.6, 0.8
        out = imaging.adjust(self.dark, cfg)
        self.assertEqual(out.shape, self.dark.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_auto_params_improve_a_dull_dark_roi(self):
        gray = self.dark[:, :, 0]
        before = imaging.eye_stats(gray)
        b, c, g = imaging.auto_params(gray)
        cfg = Config()
        cfg.img_brightness, cfg.img_contrast, cfg.img_gamma = b, c, g
        after = imaging.eye_stats(imaging.adjust(self.dark, cfg)[:, :, 0])
        self.assertGreater(after[1], before[1] * 2)          # much more spread
        self.assertGreater(after[0], before[0])              # brighter
        self.assertEqual(imaging.describe_eye_quality(*after)[1], "good")
        self.assertEqual(imaging.describe_eye_quality(*before)[1], "bad")

    def test_quality_verdicts(self):
        self.assertEqual(imaging.describe_eye_quality(200, 100, 0.0)[1], "bad")
        self.assertEqual(imaging.describe_eye_quality(120, 40, 0.0)[1], "warn")
        self.assertEqual(imaging.describe_eye_quality(120, 90, 0.01)[1], "good")

    def test_eye_region_box_is_inside_image(self):
        pts = np.zeros((478, 3))
        for i, (x, y) in {33: (.3, .4), 133: (.4, .4), 159: (.35, .39), 145: (.35, .41),
                          362: (.6, .4), 263: (.7, .4), 386: (.65, .39), 374: (.65, .41)}.items():
            pts[i, :2] = (x, y)
        x0, y0, x1, y1 = eye_region_box(pts, 640, 480)
        self.assertTrue(0 <= x0 < x1 <= 640 and 0 <= y0 < y1 <= 480)
        self.assertLess(x0, 0.3 * 640)
        self.assertGreater(x1, 0.7 * 640)


if __name__ == "__main__":
    unittest.main()
