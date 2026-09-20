"""Pinch gesture detection: thumb+index = left button, thumb+middle = right button."""
from __future__ import annotations

import numpy as np

from .config import Config
from .landmarks import HandData

WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, MIDDLE_MCP, INDEX_MCP, PINKY_MCP = 0, 4, 8, 12, 9, 5, 17
# 2D closeness is only trusted while the 3D estimate does not say the fingers are clearly far apart.
PINCH_3D_SANITY = 1.1
BUTTON_FINGER = {"left": "index", "right": "middle"}
FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
CURLED_BELOW = 1.3   # wrist-to-fingertip distance in palm lengths: ~1.5-2.0 extended, ~0.7-1.2 curled


def _d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def finger_curls(hand: HandData) -> dict[str, float]:
    """Wrist-to-fingertip distance (3D) in palm lengths for each finger: high = extended, low = curled."""
    world = hand.world
    palm = _d(world[WRIST], world[MIDDLE_MCP])
    if palm < 1e-6:
        return {k: 9.9 for k in FINGER_TIPS}
    return {k: _d(world[WRIST], world[i]) / palm for k, i in FINGER_TIPS.items()}


def other_fingers_curl(hand: HandData, finger: str) -> float:
    """Largest curl among the three fingers that are not `finger` (low = all of them are folded into the palm)."""
    return max(v for k, v in finger_curls(hand).items() if k != finger)


def fist_like(hand: HandData, finger: str, below: float = CURLED_BELOW) -> bool:
    """True when every finger except `finger` is curled: a fist or a relaxed hand, where the thumb rests near the
    fingertips without any intent to click. (On recorded data a real pinch never had all three other fingers curled.)"""
    return other_fingers_curl(hand, finger) < below


def pinch_components(hand: HandData, width: int = 640, height: int = 480) -> tuple[float, float, float, float]:
    """(3D index, 3D middle, 2D index, 2D middle): thumb-to-fingertip distances in units of hand size (see pinch_metrics)."""
    world = hand.world
    palm3 = _d(world[WRIST], world[MIDDLE_MCP])
    if palm3 < 1e-6:
        return 9.9, 9.9, 9.9, 9.9
    p = hand.pts[:, :2].astype(np.float64) * np.array([width, height], dtype=np.float64)
    scale = max(_d(p[WRIST], p[MIDDLE_MCP]), _d(p[INDEX_MCP], p[PINKY_MCP]), _d(p[WRIST], p[INDEX_MCP]), 1e-6)
    return (_d(world[THUMB_TIP], world[INDEX_TIP]) / palm3, _d(world[THUMB_TIP], world[MIDDLE_TIP]) / palm3,
            _d(p[THUMB_TIP], p[INDEX_TIP]) / scale, _d(p[THUMB_TIP], p[MIDDLE_TIP]) / scale)


def pinch_metrics(hand: HandData, width: int = 640, height: int = 480) -> tuple[float, float]:
    """(thumb-index, thumb-middle) fingertip closeness in units of hand size: ~0 = touching, ~1 = fully apart.

    Two estimates are computed and the smaller one wins:
      * 3D, from MediaPipe's metric world landmarks. It survives hand rotation, but its depth is a monocular guess and
        often reads 0.4-0.6 even when the tips visibly touch;
      * 2D, in the image plane, which is accurate when the tips touch on screen. It is normalised by the largest of
        several palm lengths so that a hand pointing at the camera (foreshortened) does not look "small".
    The 2D value is ignored when the 3D one is already large (a finger passing in front of the thumb must not click).
    """
    d3_i, d3_m, d2_i, d2_m = pinch_components(hand, width, height)
    return tuple(min(a, b) if a < PINCH_3D_SANITY else a for a, b in ((d3_i, d2_i), (d3_m, d2_m)))  # type: ignore[return-value]


def derive_guard_curl(pinch_other_curls: list[float] | np.ndarray) -> float | None:
    """Curl limit for the fist guard from the user's own pinches, or None if the guard cannot work for them.

    Uses 85% of the 30th percentile of "how extended are the other fingers while pinching": low enough that the user's
    own pinches pass with margin, high enough to ignore the frames where the finger is still closing in (the 5th
    percentile made the limit 1.35, which let semi-relaxed hands through). If they pinch with the other fingers folded,
    the guard would block them: disable it.
    """
    low = float(np.percentile(np.asarray(pinch_other_curls, dtype=np.float64), 30))
    limit = 0.85 * low
    return None if limit < 1.2 else round(min(limit, 1.65), 2)


def derive_thresholds(open_value: float, contact_value: float) -> tuple[float, float] | None:
    """(on, off) thresholds from a user's open-hand level and the level their fingertips reach when touching.

    The trigger sits just above the contact level (1.6x, measured on real pinches: a threshold halfway to the open hand
    made 25-30% of free-gesture frames fire), release at twice that. None if contact and open hand are not separable.
    """
    gap = open_value - contact_value
    if gap < 0.3:
        return None
    on = min(max(contact_value * 1.6 + 0.02, 0.15), contact_value + 0.4 * gap, 0.6)
    off = min(max(on * 2.0, on + 0.1), 0.9)
    return round(on, 3), round(off, 3)


class PinchDetector:
    """Debounced pinch state machine with hysteresis. `active` is None, "left" or "right"."""

    LOST_HAND_S = 0.25  # release when the hand disappears for this long

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.active: str | None = None
        self.ratios: tuple[float, float] | None = None  # metrics of the hand closest to pinching
        self._cand: str | None = None
        self._cand_n = 0
        self._release_n = 0
        self._last_seen = -1e9

    def reset(self) -> None:
        self.active = None
        self._cand, self._cand_n, self._release_n = None, 0, 0

    def update(self, t: float, hands: list[HandData], size: tuple[int, int] = (640, 480)) -> str | None:
        cfg = self.cfg
        if not hands:
            self.ratios = None
            self._cand, self._cand_n = None, 0
            if self.active and t - self._last_seen > self.LOST_HAND_S:
                self.active = None
            return self.active
        self._last_seen = t
        metrics = [pinch_metrics(h, *size) for h in hands]
        self.ratios = min(metrics, key=lambda r: min(r))
        on_i, off_i = cfg.pinch_thresholds("index")
        on_m, off_m = cfg.pinch_thresholds("middle")

        if self.active is None:
            best = None  # (relative closeness, button)
            for hand, (mi, mm) in zip(hands, metrics):
                if cfg.pinch_fist_guard:              # only guards the *start* of a click; a held pinch is never cut
                    mi = 9.9 if fist_like(hand, "index", cfg.pinch_guard_curl) else mi
                    mm = 9.9 if fist_like(hand, "middle", cfg.pinch_guard_curl) else mm
                ri, rm = mi / on_i, mm / on_m         # < 1 means "below its own threshold"
                if ri < 1.0 and ri <= rm and (best is None or ri < best[0]):
                    best = (ri, "left")
                elif rm < 1.0 and rm < ri and (best is None or rm < best[0]):
                    best = (rm, "right")
            cand = best[1] if best else None
            if cand and cand == self._cand:
                self._cand_n += 1
            else:
                self._cand, self._cand_n = cand, 1 if cand else 0
            if cand and self._cand_n >= cfg.pinch_confirm_frames:
                self.active, self._release_n = cand, 0
        else:
            idx, off = (0, off_i) if self.active == "left" else (1, off_m)
            if min(m[idx] for m in metrics) > off:
                self._release_n += 1
                if self._release_n >= cfg.pinch_confirm_frames:
                    self.active, self._cand, self._cand_n = None, None, 0
            else:
                self._release_n = 0
        return self.active
