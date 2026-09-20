"""Where the cursor comes from: eye gaze, head pointing or hand pointing.

Head modes (Config.head_mode):
  * "eye"      - the cursor follows the gaze point. The head pose is still an input of the gaze model (it compensates
                 head movement), but moving the head does NOT move the cursor.
  * "head_eye" - the gaze point places the cursor and turning the head shifts it as well (head = coarse, eyes = fine).
  * "head"     - the cursor follows where the nose points, relative to a neutral pose (see Tracker.recenter_head).
  * "off"      - neither the eyes nor the head move the cursor (the physical mouse, or the hand mode, does).
Hand modes (Config.hand_mode):
  * "pinch"    - the hand only clicks (thumb+index = left, thumb+middle = right).
  * "hand"     - the TIP OF THE INDEX FINGER (its ball) moves the cursor while a hand is visible. A pinch (the thumb
                 ball touching the index ball) clicks and the fingertip keeps moving the cursor while pinching (that is
                 how you drag); four folded fingers (thumb free, e.g. a thumbs-up) scroll (see below) and the cursor
                 stays put. Both hands count, and touching the index ball of one hand to the index ball of the other
                 hands the cursor over to that other hand. Without a hand the head mode takes over.
"""
from __future__ import annotations

import numpy as np

from .landmarks import HandData

INDEX_TIP, WRIST, MIDDLE_MCP = 8, 0, 9
PALM_POINTS = (0, 5, 9, 13, 17)          # wrist + the four knuckles (used to measure the hand's movement)

# Head rotation (radians, each side of the neutral pose) that sweeps the whole screen at gain 1: ~25 deg / ~15 deg.
HEAD_YAW_RANGE = 0.44
HEAD_PITCH_RANGE = 0.26
# Fraction of the camera frame (width, height) the fingertip travels to sweep the whole screen at gain 1.
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


def hand_target_px(hand: HandData, screen: tuple[int, int], gain: float = 1.0, centre=(0.5, 0.5)) -> np.ndarray:
    """Index fingertip -> screen position (mirrored, like a mirror: moving the hand right moves the cursor right)."""
    x, y = float(hand.pts[INDEX_TIP, 0]), float(hand.pts[INDEX_TIP, 1])
    gain = max(gain, 1e-3)
    u = 0.5 + (centre[0] - x) / (HAND_RANGE[0] / gain)
    v = 0.5 + (y - centre[1]) / (HAND_RANGE[1] / gain)
    return clip_to_screen((u * screen[0], v * screen[1]), screen)


def pointing_hand(hands: list[HandData]) -> HandData:
    """The hand used for pointing: the closest one (largest palm in the image)."""
    return max(hands, key=lambda h: float(np.linalg.norm(h.pts[WRIST, :2] - h.pts[MIDDLE_MCP, :2])))


# ---------------------------------------------------------------------------------------------- hand scroll
# Index, middle, ring and pinky folded (the thumb does not matter: thumbs-up or fist) = scroll mode; then moving the
# hand scrolls like a two-finger touchpad gesture: the faster the hand moves, the faster the page scrolls.
from .config import Config  # noqa: E402
from .hands import FOLDED_ENTER, FOLDED_EXIT, finger_curls, index_tips_touching  # noqa: E402

WHEEL_DELTA = 120                      # one wheel notch
SCROLL_ENTER_FRAMES, SCROLL_EXIT_FRAMES = 3, 2   # consecutive frames folded / open to enter / leave scroll mode
SCROLL_DEADZONE = 0.08                 # hand speed (frame heights per second) below which nothing scrolls
SCROLL_BASE, SCROLL_EXPONENT = 14.0, 1.4
SCROLL_MAX_NOTCHES_PER_S = 60.0
LOST_HAND_S = 0.5                      # a moving hand is often lost for a moment: keep scroll mode
GAP_RESET_S = 0.3                      # after a dropout this long the hand may have jumped: restart the velocity estimate


def scroll_rate(speed: float, gain: float = 1.0) -> float:
    """Wheel notches per second for a hand moving at `speed` frame heights per second (superlinear: flicks scroll fast)."""
    over = max(speed - SCROLL_DEADZONE, 0.0)
    return min(gain * SCROLL_BASE * over ** SCROLL_EXPONENT, SCROLL_MAX_NOTCHES_PER_S)


def palm_center(hand: HandData, size: tuple[int, int]) -> np.ndarray:
    """Palm centre in units of the frame height (x is aspect-corrected), stable while the fingers move."""
    w, h = size
    c = hand.pts[list(PALM_POINTS), :2].astype(np.float64).mean(axis=0)
    return np.array([c[0] * w / h, c[1]])


class ScrollController:
    """Turns hand movement into wheel deltas. `active` is True while the four fingers are folded (scroll mode)."""

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
        curl = max(finger_curls(hand).values())          # the most extended of the four fingers; the thumb is ignored
        if not self.active:
            self._enter_n = self._enter_n + 1 if curl < FOLDED_ENTER else 0
            if self._enter_n >= SCROLL_ENTER_FRAMES:
                self.active, self._exit_n, self._prev, self._vel, self._acc = True, 0, None, np.zeros(2), np.zeros(2)
            return 0, 0
        self._exit_n = self._exit_n + 1 if curl > FOLDED_EXIT else 0
        if self._exit_n >= SCROLL_EXIT_FRAMES:
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


class HandRoles:
    """Both hands count. Each visible hand gets a stable id and its own scroll state, so one hand can point (or stay
    idle) while the other one folds its fingers and scrolls. Pinching is tracked by PinchDetector."""

    ASSOC_DIST = 0.30      # frame heights: max movement between frames for a hand to keep its id
    FORGET_S = 1.0         # ids of hands that vanished are forgotten after this long
    SWITCH_COOLDOWN_S = 0.8  # the tips are still together right after touching: a switch cannot repeat sooner

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._scrolls: dict[int, ScrollController] = {}
        self._last: dict[int, tuple[float, np.ndarray]] = {}
        self._ids: list[int] = []
        self._next_id = 0
        self._pointer_id: int | None = None
        self._was_touching = False
        self._last_switch = -1e9
        self.switches = 0            # how many times the cursor was handed to the other hand

    @property
    def active(self) -> bool:
        return self.cfg.hand_scroll and any(c.active for c in self._scrolls.values())

    @property
    def speed(self) -> float:
        return max((c.speed for c in self._scrolls.values() if c.active), default=0.0) if self.cfg.hand_scroll else 0.0

    def scrolling(self, index: int) -> bool:
        """Is the `index`-th hand of the latest update() in scroll mode (four fingers folded)? Never when scroll is off."""
        if not self.cfg.hand_scroll or index >= len(self._ids):
            return False
        ctrl = self._scrolls.get(self._ids[index])
        return bool(ctrl and ctrl.active)

    def _associate(self, t: float, centres: list[np.ndarray]) -> list[int]:
        known = {i: p for i, (ts, p) in self._last.items() if t - ts <= self.FORGET_S}
        pairs = sorted(((float(np.linalg.norm(c - p)), k, i) for k, c in enumerate(centres) for i, p in known.items()))
        ids: list[int | None] = [None] * len(centres)
        used: set[int] = set()
        for dist, k, i in pairs:
            if dist <= self.ASSOC_DIST and ids[k] is None and i not in used:
                ids[k] = i
                used.add(i)
        for k in range(len(ids)):
            if ids[k] is None:
                ids[k], self._next_id = self._next_id, self._next_id + 1
        return ids  # type: ignore[return-value]

    def update(self, t: float, hands: list[HandData], size: tuple[int, int]) -> tuple[int, int]:
        """Feed every visible hand; returns the (vertical, horizontal) wheel units to send (sum over closed hands)."""
        centres = [palm_center(h, size) for h in hands]
        ids = self._associate(t, centres)
        wheel = [0, 0]
        for hid, hand, centre in zip(ids, hands, centres):
            self._last[hid] = (t, centre)
            ctrl = self._scrolls.setdefault(hid, ScrollController(self.cfg))
            if self.cfg.hand_scroll:
                dv, dh = ctrl.update(t, hand, size)
            else:
                ctrl.reset()
                dv = dh = 0
            wheel[0] += dv
            wheel[1] += dh
        for hid in list(self._scrolls):
            if hid not in ids:
                self._scrolls[hid].update(t, None, size)           # lets the dropout timer run
                if t - self._last.get(hid, (-1e9, None))[0] > self.FORGET_S:
                    del self._scrolls[hid]
                    self._last.pop(hid, None)
        self._ids = ids
        if self._pointer_id is not None and self._pointer_id not in ids and t - self._last.get(self._pointer_id, (-1e9, None))[0] > self.FORGET_S:
            self._pointer_id = None
        self._maybe_switch(t, hands, size)
        return wheel[0], wheel[1]

    def _maybe_switch(self, t: float, hands: list[HandData], size: tuple[int, int]) -> None:
        """Index ball of one hand touching the index ball of the other = the other hand takes over the cursor."""
        touching = (self.cfg.hand_mode == "hand" and len(hands) == 2 and not (self.scrolling(0) or self.scrolling(1))
                    and index_tips_touching(hands[0], hands[1], size, self.cfg.pinch_ball_size))
        if touching and not self._was_touching and t - self._last_switch >= self.SWITCH_COOLDOWN_S:
            current = self.pointing(hands)                     # also settles who is pointing right now
            if current is not None:
                k = next(i for i, h in enumerate(hands) if h is current)
                other = [j for j in range(len(hands)) if j != k and not self.scrolling(j)]
                if other:
                    self._pointer_id = self._ids[other[0]]
                    self._last_switch = t
                    self.switches += 1
        self._was_touching = touching

    def pointing(self, hands: list[HandData]) -> HandData | None:
        """The hand that moves the cursor: an idle one (not scrolling), preferring the one that was already pointing."""
        idle = [k for k in range(len(hands)) if not self.scrolling(k)]
        if not idle:
            return None
        keep = [k for k in idle if k < len(self._ids) and self._ids[k] == self._pointer_id]
        k = keep[0] if keep else max(idle, key=lambda j: float(np.linalg.norm(hands[j].pts[WRIST, :2] - hands[j].pts[MIDDLE_MCP, :2])))
        self._pointer_id = self._ids[k] if k < len(self._ids) else None
        return hands[k]
