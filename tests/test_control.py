"""Control modes (eye / head+eye / head, pinch / hand pointing), both hands, and the folded-fingers scroll."""
from __future__ import annotations

import ctypes
import queue
import unittest
from unittest import mock

import numpy as np

from eyemouse import control, mouse
from eyemouse.config import HAND_MODES, HEAD_MODES, Config
from eyemouse.control import HandRoles, ScrollController, hand_target_px, head_offset_px, head_target_px, scroll_rate
from eyemouse.hands import FINGER_TIPS
from eyemouse.landmarks import HandData
from eyemouse.tracker import Tracker
from tests.test_tracker import ready_model

SCREEN = (1366, 768)
IMG = (640, 480)


def feat_with(yaw=0.0, pitch=0.0):
    f = np.zeros(11)
    f[5], f[6], f[10] = yaw, pitch, -40.0
    return f


def hand_at(x, y, scroll=False, tip=None):
    """A hand whose palm is centred at normalised image position (x, y).

    The index fingertip (the pointer) is at `tip`, by default straight above the palm. Every finger is extended, unless
    `scroll`: then index, middle, ring and pinky are folded (the thumb is free and stays away from them).
    """
    curl = 0.8 if scroll else 1.9
    world = np.zeros((21, 3))
    world[9] = (0, 0.1, 0)                                            # palm length 0.1 m
    dirs = {"index": (-0.3, 1.0), "middle": (0.0, 1.0), "ring": (0.3, 1.0), "pinky": (0.6, 1.0)}
    for name, i in FINGER_TIPS.items():
        v = np.array(dirs[name]) / np.linalg.norm(dirs[name])
        world[i, :2] = v * 0.1 * curl
    pts = np.zeros((21, 3))
    pts[:, :2] = (x, y)
    for idx, (dx, dy) in {0: (-0.01, 0.08), 5: (-0.05, -0.02), 9: (-0.01, -0.04), 13: (0.02, -0.02), 17: (0.05, 0.0),
                          12: (0.0, -0.10), 16: (0.04, -0.09), 20: (0.07, -0.07), 4: (0.15, 0.05)}.items():
        pts[idx, :2] = (x + dx, y + dy)
    pts[8, :2] = tip if tip is not None else (x, y - 0.08)
    return HandData(pts, world)


class HeadPointingTest(unittest.TestCase):
    def test_neutral_pose_is_the_screen_centre(self):
        target = head_target_px(feat_with(0.1, 0.05), (0.1, 0.05), SCREEN)
        np.testing.assert_allclose(target, [SCREEN[0] / 2, SCREEN[1] / 2])

    def test_turning_to_the_users_left_moves_the_cursor_left_and_looking_down_moves_it_down(self):
        left = head_target_px(feat_with(yaw=0.2), (0, 0), SCREEN)
        down = head_target_px(feat_with(pitch=0.1), (0, 0), SCREEN)
        self.assertLess(left[0], SCREEN[0] / 2)
        self.assertGreater(down[1], SCREEN[1] / 2)

    def test_gain_scales_the_displacement_and_the_result_stays_on_screen(self):
        one = head_offset_px(0.1, 0.0, (0, 0), SCREEN, gain=1.0)
        two = head_offset_px(0.1, 0.0, (0, 0), SCREEN, gain=2.0)
        np.testing.assert_allclose(two, one * 2)
        far = head_target_px(feat_with(yaw=-3.0, pitch=3.0), (0, 0), SCREEN, gain=3.0)
        self.assertTrue(0 <= far[0] <= SCREEN[0] - 1 and 0 <= far[1] <= SCREEN[1] - 1)

    def test_comfortable_head_range_sweeps_the_whole_screen(self):
        edge = head_target_px(feat_with(yaw=-control.HEAD_YAW_RANGE), (0, 0), SCREEN)
        self.assertAlmostEqual(edge[0], SCREEN[0] - 1, delta=1.5)


class HandPointingTest(unittest.TestCase):
    """The pointer is the tip of the index finger (its ball), not the hand as a whole."""

    def test_centre_of_the_frame_is_the_centre_of_the_screen(self):
        target = hand_target_px(hand_at(0.5, 0.65, tip=(0.5, 0.5)), SCREEN)
        np.testing.assert_allclose(target, [SCREEN[0] / 2, SCREEN[1] / 2], atol=1.0)

    def test_only_the_index_fingertip_moves_the_cursor(self):
        a = hand_target_px(hand_at(0.4, 0.6, tip=(0.4, 0.5)), SCREEN)
        b = hand_target_px(hand_at(0.7, 0.3, tip=(0.4, 0.5)), SCREEN)      # whole hand elsewhere, same fingertip
        np.testing.assert_allclose(a, b)
        c = hand_target_px(hand_at(0.4, 0.6, tip=(0.45, 0.5)), SCREEN)              # same hand, fingertip moved
        self.assertNotEqual(a[0], c[0])

    def test_the_view_is_mirrored_like_a_mirror(self):
        to_users_right = hand_target_px(hand_at(0.3, 0.5, tip=(0.3, 0.5)), SCREEN)     # image left = the user's right
        to_users_left = hand_target_px(hand_at(0.7, 0.5, tip=(0.7, 0.5)), SCREEN)
        self.assertGreater(to_users_right[0], SCREEN[0] / 2)
        self.assertLess(to_users_left[0], SCREEN[0] / 2)

    def test_lower_in_the_frame_is_lower_on_the_screen(self):
        self.assertGreater(hand_target_px(hand_at(0.5, 0.5, tip=(0.5, 0.65)), SCREEN)[1],
                           hand_target_px(hand_at(0.5, 0.5, tip=(0.5, 0.35)), SCREEN)[1])

    def test_higher_gain_needs_less_hand_travel(self):
        h = hand_at(0.4, 0.5, tip=(0.4, 0.5))
        self.assertGreater(abs(hand_target_px(h, SCREEN, 2.0)[0] - SCREEN[0] / 2), abs(hand_target_px(h, SCREEN, 1.0)[0] - SCREEN[0] / 2))

    def test_pointing_hand_is_the_closest_one(self):
        small, big = hand_at(0.3, 0.5), hand_at(0.6, 0.5)
        big.pts[0, 1] += 0.2                                      # longer wrist-to-knuckle distance in the image
        self.assertIs(control.pointing_hand([small, big]), big)


class TrackerModesTest(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.tracker = Tracker(self.cfg, ready_model(), queue.SimpleQueue())
        self.feat = np.array([0.0, 0.0, 0.0, 0.0, 0.3, 0.02, -0.03, 0.0, 0.0, 0.0, -40.0])

    def select(self, feat=None, closed=False, hands=()):
        return self.tracker._select_target(self.feat if feat is None else feat, closed, list(hands), IMG)

    def close_hands(self, hands, frames=6):
        """Show scrolling hands for a few frames so the tracker's per-hand scroll state notices them."""
        for i in range(frames):
            self.tracker._scroll.update(i / 30, hands, IMG)

    def test_eye_mode_uses_only_the_gaze_model_never_the_head_offset(self):
        self.cfg.head_mode = "eye"
        target, source, hold = self.select()
        np.testing.assert_allclose(target, self.tracker.model.predict_px(self.feat))
        self.tracker.head_neutral = np.array([0.5, -0.4])           # a very different neutral must change nothing
        np.testing.assert_allclose(self.select()[0], target)
        self.assertEqual((source, hold), ("eye", False))

    def test_head_eye_mode_adds_the_head_offset_to_the_gaze_point(self):
        self.cfg.head_mode = "head_eye"
        self.tracker.head_neutral = np.array([0.0, 0.0])
        base = self.tracker.model.predict_px(self.feat)
        target, source, _ = self.select()
        offset = head_offset_px(self.feat[5], self.feat[6], (0.0, 0.0), self.tracker.model.screen, self.cfg.head_gain)
        np.testing.assert_allclose(target, np.clip(base + offset, 0, np.array(self.tracker.model.screen) - 1))
        self.assertEqual(source, "head_eye")
        turned = self.select(feat=self.feat + np.eye(11)[5] * 0.15)[0]      # turn further to the left: cursor moves left
        self.assertLess(turned[0] - base[0], target[0] - base[0])

    def test_head_mode_needs_no_calibration(self):
        self.tracker.model.clear()
        self.cfg.head_mode = "head"
        self.assertTrue(self.tracker.mode_ready())
        target, source, _ = self.select()
        self.assertEqual(source, "head")
        np.testing.assert_allclose(target, [self.tracker.model.screen[0] / 2, self.tracker.model.screen[1] / 2], atol=1.0)  # 1st frame = neutral

    def test_eye_modes_hold_while_the_eyes_are_closed_but_head_mode_does_not(self):
        self.cfg.head_mode = "eye"
        self.assertEqual(self.select(closed=True), (None, "eye", True))
        self.cfg.head_mode = "head"
        self.assertEqual(self.select(closed=True)[1:], ("head", False))

    def test_hand_mode_points_with_the_index_fingertip_and_falls_back_without_a_hand(self):
        self.cfg.hand_mode = "hand"
        hand = hand_at(0.4, 0.6, tip=(0.4, 0.5))
        target, source, hold = self.select(hands=[hand])
        np.testing.assert_allclose(target, hand_target_px(hand, self.tracker.model.screen, self.cfg.hand_gain))
        self.assertEqual((source, hold), ("hand", False))
        self.assertEqual(self.select(hands=[])[1], "eye")                    # no hand: the head mode takes over

    def test_a_scrolling_hand_freezes_the_cursor_instead_of_pointing(self):
        self.cfg.hand_mode = "hand"
        scrolling = hand_at(0.4, 0.6, scroll=True)
        self.close_hands([scrolling])
        self.assertEqual(self.select(hands=[scrolling]), (None, "hand", True))
        self.cfg.hand_scroll = False                                          # scrolling switched off: pointing again
        self.assertEqual(self.select(hands=[scrolling])[1], "hand")

    def test_the_hand_keeps_moving_the_cursor_while_pinching_but_not_while_scrolling(self):
        self.cfg.hand_mode = "hand"
        hand = hand_at(0.4, 0.6)
        self.assertEqual(self.select(hands=[hand])[1:], ("hand", False))              # visible: follows the fingertip
        self.tracker._pinch.active = "left"                                            # pinch held: still follows (drag)
        self.assertEqual(self.select(hands=[hand])[1:], ("hand", False))
        self.tracker._pinch.active = None
        scrolling = hand_at(0.4, 0.6, scroll=True)
        self.close_hands([scrolling])                                                       # folded fingers: scrolling, frozen
        self.assertEqual(self.select(hands=[scrolling]), (None, "hand", True))

    def test_both_hands_count_one_points_while_the_other_scrolls(self):
        self.cfg.hand_mode = "hand"
        pointer, scrolling = hand_at(0.3, 0.6, tip=(0.3, 0.45)), hand_at(0.7, 0.6, scroll=True)
        self.close_hands([pointer, scrolling])
        target, source, hold = self.select(hands=[pointer, scrolling])
        np.testing.assert_allclose(target, hand_target_px(pointer, self.tracker.model.screen, self.cfg.hand_gain))
        self.assertEqual((source, hold), ("hand", False))
        self.close_hands([scrolling, pointer])                                            # same hands, other list order
        target2, _, _ = self.select(hands=[scrolling, pointer])
        np.testing.assert_allclose(target2, target)

    def test_the_left_hand_works_exactly_like_the_right_one(self):
        self.cfg.hand_mode = "hand"
        for x in (0.25, 0.75):                                                       # either side of the frame
            hand = hand_at(x, 0.6, tip=(x, 0.5))
            target, source, _ = self.select(hands=[hand])
            self.assertEqual(source, "hand")
            np.testing.assert_allclose(target, hand_target_px(hand, self.tracker.model.screen, self.cfg.hand_gain))

    def test_pinch_mode_never_moves_the_cursor_with_the_hand(self):
        self.cfg.hand_mode = "pinch"
        self.assertEqual(self.select(hands=[hand_at(0.4, 0.6)])[1], "eye")

    def test_off_mode_disables_eye_and_head_control_completely(self):
        self.cfg.head_mode = "off"
        self.assertEqual(self.select(), (None, "off", False))                  # face visible, calibrated: still nothing
        self.assertEqual(self.select(closed=True), (None, "off", False))
        self.tracker.model.clear()
        self.assertTrue(self.tracker.mode_ready())                              # needs no calibration
        self.tracker.mouse_enabled = True
        self.assertTrue(self.tracker.mouse_active)                              # hand gestures (click/scroll) stay available

    def test_off_mode_still_lets_the_hand_move_the_cursor_when_the_hand_mode_says_so(self):
        self.cfg.head_mode, self.cfg.hand_mode = "off", "hand"
        hand = hand_at(0.4, 0.6)
        self.assertEqual(self.select(hands=[hand])[1:], ("hand", False))
        self.assertEqual(self.select(hands=[])[1:], ("off", False))            # no hand: nothing moves the cursor

    def test_mode_readiness_and_mouse_active(self):
        self.tracker.model.clear()
        self.tracker.mouse_enabled = True
        self.assertFalse(self.tracker.mouse_active)                            # eye + pinch need a calibration
        self.cfg.head_mode = "head"
        self.assertTrue(self.tracker.mouse_active)
        self.cfg.head_mode, self.cfg.hand_mode = "eye", "hand"
        self.assertTrue(self.tracker.mouse_active)

    def test_recenter_takes_the_recent_head_pose_as_neutral(self):
        import time
        self.tracker.recenter_head()
        self.assertIsNone(self.tracker.head_neutral)                           # no frames yet: captured on the next one
        for _ in range(5):
            self.tracker._feat_hist.append((time.perf_counter(), self.feat))
        self.tracker.recenter_head()
        np.testing.assert_allclose(self.tracker.head_neutral, [0.02, -0.03])

    def test_click_lands_where_the_hand_was_before_the_fingers_closed(self):
        self.tracker._src = "hand"
        for i in range(30):                                                    # fingertip drifting to the right while closing
            self.tracker._hand_hist.append((i / 30, np.array([500.0 + (i > 22) * 80.0, 300.0])))
        anchor = self.tracker._press_anchor(1.0)
        np.testing.assert_allclose(anchor, [500.0, 300.0])
        self.tracker._src = "eye"
        self.assertIsNone(self.tracker._press_anchor(1.0))


class ClickVersusScrollTest(unittest.TestCase):
    """Through the whole frame pipeline: a hand with the four fingers folded scrolls, it does not click."""

    def setUp(self):
        self.cfg = Config()
        self.cfg.head_mode, self.cfg.hand_mode = "off", "hand"
        self.tracker = Tracker(self.cfg, ready_model(), queue.SimpleQueue())
        self.tracker.mouse_enabled = True
        patcher = mock.patch("eyemouse.tracker.mouse")
        self.mouse = patcher.start()
        self.addCleanup(patcher.stop)
        self.frame = np.zeros((480, 640, 3), np.uint8)

    def feed(self, hand, frames=5, t0=0.0):
        for i in range(frames):
            self.tracker._process(t0 + i / 30, self.frame, self.frame, None, [hand], 30.0)
        return self.tracker.latest()

    def touching(self, scroll):
        hand = hand_at(0.5, 0.6, scroll=scroll, tip=(0.5, 0.5))
        hand.pts[4, :2] = hand.pts[8, :2]                         # the thumb ball sits on the index ball
        return hand

    def test_thumb_on_the_index_tip_clicks_when_the_other_fingers_are_extended(self):
        state = self.feed(self.touching(scroll=False))
        self.assertEqual(state.hand_states, ("pinch_left",))
        self.mouse.button_down.assert_called_once_with("left")

    def test_the_same_contact_with_the_four_fingers_folded_scrolls_instead_of_clicking(self):
        state = self.feed(self.touching(scroll=True))
        self.mouse.button_down.assert_not_called()
        self.assertEqual(state.hand_states, ("scroll",))
        self.assertTrue(state.scrolling)

    def test_the_thumb_is_free_in_the_scroll_pose(self):
        hand = hand_at(0.5, 0.6, scroll=True)                     # thumb resting away from the fingers: a thumbs-up
        self.assertEqual(self.feed(hand).hand_states, ("scroll",))
        self.mouse.button_down.assert_not_called()


class HandRolesTest(unittest.TestCase):
    """Both hands count: each has its own id and scroll state."""

    def setUp(self):
        self.cfg = Config()
        self.roles = HandRoles(self.cfg)

    def feed(self, hands, frames=8, t0=0.0):
        out = (0, 0)
        for i in range(frames):
            out = self.roles.update(t0 + i / 30, hands, IMG)
        return out

    def test_one_hand_scrolls_while_the_other_points(self):
        idle, scrolling = hand_at(0.3, 0.6), hand_at(0.7, 0.6, scroll=True)
        self.feed([idle, scrolling])
        self.assertTrue(self.roles.scrolling(1))
        self.assertFalse(self.roles.scrolling(0))
        self.assertIs(self.roles.pointing([idle, scrolling]), idle)
        self.assertTrue(self.roles.active)

    def test_ids_follow_the_hands_when_the_list_order_swaps(self):
        idle, scrolling = hand_at(0.3, 0.6), hand_at(0.7, 0.6, scroll=True)
        self.feed([idle, scrolling])
        for i in range(3):
            self.roles.update(1.0 + i / 30, [scrolling, idle], IMG)                 # same hands, other order
        self.assertTrue(self.roles.scrolling(0))
        self.assertFalse(self.roles.scrolling(1))

    def test_two_scrolling_hands_leave_nothing_to_point_with(self):
        a, b = hand_at(0.3, 0.6, scroll=True), hand_at(0.7, 0.6, scroll=True)
        self.feed([a, b])
        self.assertIsNone(self.roles.pointing([a, b]))

    def test_the_pointing_hand_is_sticky(self):
        a, b = hand_at(0.3, 0.6), hand_at(0.7, 0.6)
        b.pts[0, 1] += 0.2                                                     # b looks bigger (closer)
        self.feed([a, b])
        first = self.roles.pointing([a, b])
        self.assertIs(first, b)
        a.pts[0, 1] += 0.5                                                     # now a is bigger, but b already points
        self.feed([a, b], t0=1.0)
        self.assertIs(self.roles.pointing([a, b]), b)

    def test_the_wheel_comes_from_the_scrolling_hand_while_the_other_hand_is_idle(self):
        idle = hand_at(0.3, 0.6)
        total = 0
        positions = [(0.7, 0.3)] * 6 + [(0.7, 0.3 + 0.02 * i) for i in range(1, 25)]      # the hand moves down
        for i, (x, y) in enumerate(positions):
            dv, _ = self.roles.update(i / 30, [idle, hand_at(x, y, scroll=True)], IMG)
            total += dv
        self.assertLess(total, -200)                                            # scrolled down
        self.assertFalse(self.roles.scrolling(0))

    def test_a_vanished_hand_is_forgotten(self):
        scrolling = hand_at(0.5, 0.5, scroll=True)
        self.feed([scrolling])
        self.roles.update(0.5, [], IMG)
        self.roles.update(2.0, [], IMG)
        self.assertFalse(self.roles.active)


class ConfigModesTest(unittest.TestCase):
    def test_defaults_and_names(self):
        cfg = Config()
        self.assertEqual((cfg.head_mode, cfg.hand_mode), ("eye", "pinch"))
        self.assertEqual(set(HEAD_MODES), {"eye", "head_eye", "head", "off"})
        self.assertEqual(set(HAND_MODES), {"pinch", "hand"})

    def test_invalid_saved_modes_fall_back_to_the_defaults(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.json"
            path.write_text('{"head_mode": "banana", "hand_mode": "?", "head_gain": 1.7}', encoding="utf-8")
            cfg = Config.load(path)
        self.assertEqual((cfg.head_mode, cfg.hand_mode, cfg.head_gain), ("eye", "pinch", 1.7))


def drive(controller, positions, scroll=True, fps=30.0):
    """Feed a sequence of palm positions (normalised) with the four fingers folded; returns total (vertical, horizontal) wheel units."""
    v = h = 0
    for i, (x, y) in enumerate(positions):
        dv, dh = controller.update(i / fps, hand_at(x, y, scroll), IMG)
        v, h = v + dv, h + dh
    return v, h


def swipe(start, velocity, seconds=1.0, fps=30.0, settle=6):
    """`settle` still frames (enter the scroll state), then a constant-velocity motion in frame-height units/s."""
    n = int(seconds * fps)
    still = [start] * settle
    move = [(start[0] + velocity[0] * i / fps * (IMG[1] / IMG[0]), start[1] + velocity[1] * i / fps) for i in range(1, n + 1)]
    return still + move


class ScrollRateTest(unittest.TestCase):
    def test_nothing_below_the_deadzone_then_superlinear_and_capped(self):
        self.assertEqual(scroll_rate(0.05), 0.0)
        self.assertGreater(scroll_rate(1.0), 2.5 * scroll_rate(0.5))          # speed matters more than proportionally
        self.assertGreater(scroll_rate(2.0), 8 * scroll_rate(0.5))
        self.assertLessEqual(scroll_rate(50.0), control.SCROLL_MAX_NOTCHES_PER_S)
        self.assertTrue(all(scroll_rate(a) <= scroll_rate(b) for a, b in zip(np.linspace(0, 3, 30), np.linspace(0, 3, 30)[1:])))

    def test_gain_scales_the_rate(self):
        self.assertAlmostEqual(scroll_rate(0.8, gain=2.0), 2 * scroll_rate(0.8, gain=1.0))


class ScrollControllerTest(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.sc = ScrollController(self.cfg)

    def test_folded_fingers_enter_scroll_mode_without_scrolling(self):
        v, h = drive(self.sc, [(0.5, 0.5)] * 10)
        self.assertTrue(self.sc.active)
        self.assertEqual((v, h), (0, 0))

    def test_a_hand_without_the_touch_never_scrolls_however_fast_it_moves(self):
        v, h = drive(self.sc, swipe((0.5, 0.3), (0.0, 1.2)), scroll=False)
        self.assertFalse(self.sc.active)
        self.assertEqual((v, h), (0, 0))

    def test_moving_down_scrolls_down_and_moving_up_scrolls_up(self):
        down, _ = drive(ScrollController(self.cfg), swipe((0.5, 0.2), (0.0, 0.6)))
        up, _ = drive(ScrollController(self.cfg), swipe((0.5, 0.8), (0.0, -0.6)))
        self.assertLess(down, -300)                 # wheel deltas: negative = down (about 5.6 notches in a second)
        self.assertGreater(up, 300)

    def test_natural_scrolling_inverts_the_direction(self):
        self.cfg.scroll_natural = True
        down, _ = drive(ScrollController(self.cfg), swipe((0.5, 0.2), (0.0, 0.6)))
        self.assertGreater(down, 300)

    def test_faster_movement_scrolls_much_more(self):
        slow, _ = drive(ScrollController(self.cfg), swipe((0.5, 0.1), (0.0, 0.5), seconds=0.6))
        fast, _ = drive(ScrollController(self.cfg), swipe((0.5, 0.1), (0.0, 1.5), seconds=0.6))
        self.assertGreater(abs(fast), 3 * abs(slow))

    def test_the_gain_setting_changes_the_intensity(self):
        base, _ = drive(ScrollController(self.cfg), swipe((0.5, 0.2), (0.0, 0.6)))
        self.cfg.scroll_gain = 2.0
        more, _ = drive(ScrollController(self.cfg), swipe((0.5, 0.2), (0.0, 0.6)))
        self.assertGreater(abs(more), 1.7 * abs(base))

    def test_moving_to_the_users_right_scrolls_right(self):
        _, right = drive(ScrollController(self.cfg), swipe((0.7, 0.5), (-0.6, 0.0)))    # the image is mirrored: x decreases
        _, left = drive(ScrollController(self.cfg), swipe((0.3, 0.5), (0.6, 0.0)))
        self.assertGreater(right, 300)
        self.assertLess(left, -300)

    def test_axis_lock_keeps_only_the_dominant_direction(self):
        v, h = drive(self.sc, swipe((0.5, 0.2), (-0.1, 0.8)))                          # mostly vertical, a little sideways
        self.assertLess(v, -300)
        self.assertEqual(h, 0)

    def test_slow_drift_is_ignored(self):
        v, h = drive(self.sc, swipe((0.5, 0.5), (0.0, 0.04), seconds=2.0))
        self.assertEqual((v, h), (0, 0))

    def test_opening_the_fingers_ends_scroll_mode(self):
        drive(self.sc, [(0.5, 0.5)] * 8)
        self.assertTrue(self.sc.active)
        for i in range(4):
            self.sc.update(1.0 + i / 30, hand_at(0.5, 0.5), IMG)
        self.assertFalse(self.sc.active)

    def test_losing_the_hand_ends_scroll_mode_after_a_moment(self):
        drive(self.sc, [(0.5, 0.5)] * 8)
        last_seen = 7 / 30
        self.sc.update(last_seen + 0.3, None, IMG)
        self.assertTrue(self.sc.active)                                                # a 0.3 s dropout is tolerated
        self.sc.update(last_seen + 0.8, None, IMG)
        self.assertFalse(self.sc.active)

    def test_a_hand_that_reappears_elsewhere_after_a_dropout_does_not_cause_a_scroll_burst(self):
        drive(self.sc, [(0.5, 0.2)] * 8)
        before = self.sc.update(0.5, hand_at(0.5, 0.2, True), IMG)
        jump = self.sc.update(0.5 + 0.4, hand_at(0.5, 0.8, True), IMG)              # 0.4 s later, far away
        self.assertEqual((before, jump), ((0, 0), (0, 0)))
        self.assertTrue(self.sc.active)


class WheelOutputTest(unittest.TestCase):
    def test_wheel_events_are_accepted_by_sendinput(self):
        # a zero-delta MOUSEEVENTF_WHEEL / HWHEEL scrolls nothing but exercises the struct and the call
        for flag in (mouse._WHEEL, mouse._HWHEEL):
            inp = mouse._Input(type=0, u=mouse._InputUnion(mi=mouse._MouseInput(0, 0, 0, flag, 0, 0)))
            self.assertEqual(mouse.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(mouse._Input)), 1)

    def test_negative_deltas_are_encoded_as_twos_complement(self):
        self.assertEqual(-120 & 0xFFFFFFFF, 0xFFFFFF88)


if __name__ == "__main__":
    unittest.main()
