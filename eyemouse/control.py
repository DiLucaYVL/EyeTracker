"""Where the cursor comes from: eye gaze, head pointing or hand pointing.

Head modes (Config.head_mode):
  * "eye"      - the cursor follows the gaze point. The head pose is still an input of the gaze model (it compensates
                 head movement), but moving the head does NOT move the cursor.
  * "head_eye" - the gaze point places the cursor and turning the head shifts it as well (head = coarse, eyes = fine).
  * "head"     - the cursor follows where the nose points, relative to a neutral pose (see Tracker.recenter_head).
Hand modes (Config.hand_mode):
  * "pinch"    - the hand only clicks (thumb+index = left, thumb+middle = right).
  * "hand"     - the hand moves the cursor whenever it is visible and NOT performing another action (no pose is
                 required, e.g. no pointing finger). A pinch clicks and the hand keeps moving the cursor while pinching
                 (that is how you drag); a closed hand scrolls and the cursor stays put. Without a hand the head mode
                 takes over.
"""
from __future__ import annotations

import numpy as np

from .landmarks import HandData

INDEX_TIP, WRIST, MIDDLE_MCP = 8, 0, 9
PALM_POINTS = (0, 5, 9, 13, 17)          # wrist + the four knuckles (used to measure the fist's movement)

# Head rotation (radians, each side of the neutral pose) that sweeps the whole screen at gain 1: ~25 deg / ~15 deg.
HEAD_YAW_RANGE = 0.44
HEAD_PITCH_RANGE = 0.26
# Fraction of the camera frame (width, height) the palm travels to sweep the whole screen at gain 1.
HAND_RANGE = (0.55, 0.45)


def clip_to_screen(p, screen: tuple[int, int]) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), [0.0, 0.0], [screen[0] - 1.0, screen[1] - 1.0])


def head_offset_px(yaw: float, pitch: float, neutral, screen: tuple[int, int], gain: float = 1.0) -> np.ndarray:
    """Cursor displacement (px) for a head pose relative to `neutral` = (yaw, pitch).

    +yaw is the head turning to the user's left, which is the screen's left as the user faces it, so x decreases;
    +pitch is looking down, so y increases.
    """
    w, h = screen
    dx = -(yaw - float(neutral[0])) / HEAD_YAW_RANGE * (w / 2.0) * gain
    dy = (pitch - float(neutral[1])) / HEAD_PITCH_RANGE * (h / 2.0) * gain
    return np.array([dx, dy])


def head_target_px(feat: np.ndarray, neutral, screen: tuple[int, int], gain: float = 1.0) -> np.ndarray:
    """Pure head pointing: the screen centre plus the head offset."""
    centre = np.array([screen[0] / 2.0, screen[1] / 2.0])
    return clip_to_screen(centre + head_offset_px(feat[5], feat[6], neutral, screen, gain), screen)


def hand_position(hand: HandData) -> np.ndarray:
    """Where the hand is in the image (normalised): the centre of the box around all 21 landmarks."""
    pts = hand.pts[:, :2].astype(np.float64)
    return (pts.min(axis=0) + pts.max(axis=0)) / 2.0


def hand_target_px(hand: HandData, screen: tuple[int, int], gain: float = 1.0, centre=(0.5, 0.5)) -> np.ndarray:
    """The idle hand's position -> screen position (mirrored, like a mirror: moving the hand right moves the cursor
    right)."""
    x, y = hand_position(hand)
    gain = max(gain, 1e-3)
    u = 0.5 + (centre[0] - x) / (HAND_RANGE[0] / gain)
    v = 0.5 + (y - centre[1]) / (HAND_RANGE[1] / gain)
    return clip_to_screen((u * screen[0], v * screen[1]), screen)


def pointing_hand(hands: list[HandData]) -> HandData:
    """The hand used for pointing: the closest one (largest palm in the image)."""
    return max(hands, key=lambda h: float(np.linalg.norm(h.pts[WRIST, :2] - h.pts[MIDDLE_MCP, :2])))


# ---------------------------------------------------------------------------------------------- hand scroll
# A closed hand scrolls like a two-finger touchpad gesture: the faster the fist moves, the faster the page scrolls.
from .config import Config  # noqa: E402
from .hands import finger_curls  # noqa: E402

WHEEL_DELTA = 120                      # one wheel notch
FIST_ENTER, FIST_EXIT, FIST_FRAMES = 1.15, 1.4, 3   # wrist-to-fingertip curl of the *most extended* finger (measured on recordings)
SCROLL_DEADZONE = 0.08                 # fist speed (frame heights per second) below which nothing scrolls
SCROLL_BASE, SCROLL_EXPONENT = 14.0, 1.4
SCROLL_MAX_NOTCHES_PER_S = 60.0
LOST_HAND_S = 0.5                      # a closed hand is often lost for a moment while it moves: keep scroll mode
GAP_RESET_S = 0.3                      # after a dropout this long the hand may have jumped: restart the velocity estimate


def scroll_rate(speed: float, gain: float = 1.0) -> float:
    """Wheel notches per second for a fist moving at `speed` frame heights per second (superlinear: flicks scroll fast)."""
    over = max(speed - SCROLL_DEADZONE, 0.0)
    return min(gain * SCROLL_BASE * over ** SCROLL_EXPONENT, SCROLL_MAX_NOTCHES_PER_S)


def palm_center(hand: HandData, size: tuple[int, int]) -> np.ndarray:
    """Palm centre in units of the frame height (x is aspect-corrected), stable while the fingers close."""
    w, h = size
    c = hand.pts[list(PALM_POINTS), :2].astype(np.float64).mean(axis=0)
    return np.array([c[0] * w / h, c[1]])


class ScrollController:
    """Turns fist movement into wheel deltas. `active` is True while the hand is closed (scroll mode)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.speed = 0.0
        self._enter_n = 0
        self._exit_n = 0
        self._prev: tuple[float, np.ndarray] | None = None
        self._vel = np.zeros(2)
        self._acc = np.zeros(2)     # fractional wheel units not yet sent (vertical, horizontal)
        self._last_seen = -1e9

    def update(self, t: float, hand: HandData | None, size: tuple[int, int]) -> tuple[int, int]:
        """Returns (vertical, horizontal) wheel units to send now (WHEEL_DELTA per notch). Positive = up / right."""
        if hand is None:
            if self.active and t - self._last_seen > LOST_HAND_S:
                self.reset()
            self._enter_n = 0
            return 0, 0
        self._last_seen = t
        curl = max(finger_curls(hand).values())
        if not self.active:
            self._enter_n = self._enter_n + 1 if curl < FIST_ENTER else 0
            if self._enter_n >= FIST_FRAMES:
                self.active, self._exit_n, self._prev, self._vel, self._acc = True, 0, None, np.zeros(2), np.zeros(2)
            return 0, 0
        self._exit_n = self._exit_n + 1 if curl > FIST_EXIT else 0
        if self._exit_n >= 2:
            self.reset()
            return 0, 0

        pos = palm_center(hand, size)
        if self._prev is None:
            self._prev = (t, pos)
            return 0, 0
        dt = t - self._prev[0]
        if dt > GAP_RESET_S:
            self._prev, self._vel = (t, pos), np.zeros(2)
            return 0, 0
        if dt <= 1e-3:
            return 0, 0
        raw = (pos - self._prev[1]) / dt
        self._prev = (t, pos)
        self._vel = 0.5 * self._vel + 0.5 * raw                     # light smoothing of the hand velocity
        vx_user, vy = -self._vel[0], self._vel[1]                   # the camera image is mirrored: left = the user's right
        self.speed = float(np.hypot(vx_user, vy))
        if abs(vy) < 0.4 * abs(vx_user):                            # touchpad-like axis lock: keep the dominant direction
            vy = 0.0
        elif abs(vx_user) < 0.4 * abs(vy):
            vx_user = 0.0
        sign = -1.0 if self.cfg.scroll_natural else 1.0
        # hand moving down (image y grows) scrolls down = negative wheel delta; hand moving right scrolls right
        self._acc[0] += -sign * np.sign(vy) * scroll_rate(abs(vy), self.cfg.scroll_gain) * dt * WHEEL_DELTA
        self._acc[1] += sign * np.sign(vx_user) * scroll_rate(abs(vx_user), self.cfg.scroll_gain) * dt * WHEEL_DELTA
        out = [0, 0]
        for i in (0, 1):
            if abs(self._acc[i]) >= 20:                             # send in >= 1/6 notch steps (smooth in modern apps)
                out[i] = int(self._acc[i])
                self._acc[i] -= out[i]
        return out[0], out[1]
