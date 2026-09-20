"""Pinch gesture: thumb+index = left button, thumb+middle = right button, thumb+ring = scroll mode (see control.py).

The ONLY criterion is that the two fingertip "balls" touch. Each fingertip (thumb 4, index 8, middle 12) is a disc of
radius `Config.pinch_ball_size` (a fraction of the hand size) drawn on the camera view; two discs touch when the distance
between the tips, in hand sizes, is at most two radii. Touching = pinch, not touching = no pinch. No 3D estimate, no
hysteresis and no debounce by default (the settings exist but default to none).
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .landmarks import HandData

WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, MIDDLE_MCP, INDEX_MCP, PINKY_MCP = 0, 4, 8, 12, 16, 9, 5, 17
FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
NOT_TOUCHING = 9.9


def _d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def hand_scale_px(hand: HandData, size: tuple[int, int]) -> float:
    """The hand-size unit used everywhere (largest of three palm lengths in the image), in pixels of a `size` image."""
    p = hand.pts[:, :2].astype(np.float64) * np.array(size, dtype=np.float64)
    return max(_d(p[WRIST], p[MIDDLE_MCP]), _d(p[INDEX_MCP], p[PINKY_MCP]), _d(p[WRIST], p[INDEX_MCP]), 1e-6)


def ball_radius_px(ball_size: float, hand_scale: float) -> float:
    """Radius (px) of a fingertip ball for a hand whose size is `hand_scale` pixels."""
    return ball_size * hand_scale


def pinch_metrics(hand: HandData, width: int = 640, height: int = 480) -> tuple[float, float]:
    """(thumb-index, thumb-middle) distance between fingertips in hand sizes, measured in the image (2D).

    The balls of a pair touch when this is <= 2 * ball_size. 0 = tips on top of each other, ~1 = fully apart.
    """
    scale = hand_scale_px(hand, (width, height))
    p = hand.pts[:, :2].astype(np.float64) * np.array([width, height], dtype=np.float64)
    return _d(p[THUMB_TIP], p[INDEX_TIP]) / scale, _d(p[THUMB_TIP], p[MIDDLE_TIP]) / scale


def touch_metrics(hand: HandData, width: int = 640, height: int = 480) -> tuple[float, float, float]:
    """(thumb-index, thumb-middle, thumb-ring) fingertip distances in hand sizes (2D). A pair touches when <= 2 balls."""
    scale = hand_scale_px(hand, (width, height))
    p = hand.pts[:, :2].astype(np.float64) * np.array([width, height], dtype=np.float64)
    return tuple(_d(p[THUMB_TIP], p[i]) / scale for i in (INDEX_TIP, MIDDLE_TIP, RING_TIP))  # type: ignore[return-value]


def scroll_touch(hand: HandData, cfg: Config, size: tuple[int, int] = (640, 480), strict: bool = True) -> bool:
    """Is the thumb ball touching the ring-finger ball (the scroll gesture)?

    `strict` (used to START the gesture) also requires the ring finger to be the closest of the three fingers to the
    thumb, so a thumb that is between the middle and the ring finger is a click, not a scroll.
    """
    mi, mm, mr = touch_metrics(hand, *size)
    on = cfg.pinch_thresholds()[0]
    return mr <= on and (not strict or mr < min(mi, mm))


def derive_ball_size(open_value: float, contact_value: float) -> float | None:
    """Ball radius (fraction of the hand) from a user's measured open-hand distance and fingertip-contact distance.

    Two balls of radius r touch at 2r, so r = (contact + a small margin) / 2: the balls touch when the user's fingertips
    do. None if the open hand and the contact are not clearly different (nothing to calibrate).
    """
    if open_value - contact_value < 0.3:
        return None
    return round(min(max((contact_value + 0.03) / 2.0, 0.03), 0.30), 4)


def _palm(hand: HandData, size: tuple[int, int]) -> np.ndarray:
    """Palm centre in units of the frame height (used to follow a hand from frame to frame)."""
    c = hand.pts[[0, 5, 9, 13, 17], :2].astype(np.float64).mean(axis=0)
    return np.array([c[0] * size[0] / size[1], c[1]])


class PinchDetector:
    """Pinch = the two fingertip balls touch. `active` is None, "left" (thumb+index) or "right" (thumb+middle).

    Works with any number of hands: a pinch starts on whichever hand touches, and from then on that hand (followed by
    position) alone decides when it ends, so a relaxed second hand can neither hold the click nor cut it.
    """

    LOST_HAND_S = 0.25   # release when the pinching hand disappears for this long
    FOLLOW_DIST = 0.35   # how far (frame heights) the pinching hand may move between two frames and still be the same hand

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.active: str | None = None
        self.hand_index: int | None = None               # index (in the list given to update) of the pinching hand
        self.ratios: tuple[float, float] | None = None   # (thumb-index, thumb-middle) of the hand closest to pinching
        self._pos: np.ndarray | None = None
        self._cand: str | None = None
        self._cand_n = 0
        self._release_n = 0
        self._last_seen = -1e9

    def _clear(self) -> None:
        self.active, self.hand_index, self._pos = None, None, None
        self._cand, self._cand_n, self._release_n = None, 0, 0

    def reset(self) -> None:
        self._clear()

    def _tracked(self, hands: list[HandData], size: tuple[int, int]) -> int | None:
        """Index of the hand that started the pinch: the one nearest to where it was last seen."""
        if self._pos is None:
            return None
        dists = [float(np.linalg.norm(_palm(h, size) - self._pos)) for h in hands]
        k = int(np.argmin(dists))
        return k if dists[k] <= self.FOLLOW_DIST else None

    def update(self, t: float, hands: list[HandData], size: tuple[int, int] = (640, 480), blocked=()) -> str | None:
        """`blocked[k]` = the k-th hand is doing something else (a closed hand scrolling): it cannot pinch."""
        cfg = self.cfg
        if not hands:
            self.ratios = None
            self._cand, self._cand_n = None, 0
            if self.active and t - self._last_seen > self.LOST_HAND_S:
                self._clear()
            return self.active
        three = [touch_metrics(h, *size) for h in hands]
        raw = [(mi, mm) for mi, mm, _ in three]
        self.ratios = min(raw, key=lambda r: min(r))
        # a hand whose thumb is closest to the ring finger (scroll gesture) or that scrolls already cannot click
        metrics = [(NOT_TOUCHING, NOT_TOUCHING) if (k < len(blocked) and blocked[k]) or mr < min(mi, mm) else (mi, mm)
                   for k, (mi, mm, mr) in enumerate(three)]
        on, off = cfg.pinch_thresholds()

        if self.active is None:
            best = None  # (relative distance, button, hand index)
            for k, (mi, mm) in enumerate(metrics):
                ri, rm = mi / on, mm / on               # <= 1: the two balls touch
                if ri <= 1.0 and ri <= rm and (best is None or ri < best[0]):
                    best = (ri, "left", k)
                elif rm <= 1.0 and rm < ri and (best is None or rm < best[0]):
                    best = (rm, "right", k)
            cand = best[1] if best else None
            if cand and cand == self._cand:
                self._cand_n += 1
            else:
                self._cand, self._cand_n = cand, 1 if cand else 0
            if cand and self._cand_n >= cfg.pinch_confirm_frames:
                self.active, self._release_n = cand, 0
                self.hand_index, self._pos, self._last_seen = best[2], _palm(hands[best[2]], size), t
        else:
            col = 0 if self.active == "left" else 1
            k = self._tracked(hands, size)
            if k is None:                              # the pinching hand is not visible (another one is)
                if t - self._last_seen > self.LOST_HAND_S:
                    self._clear()
                return self.active
            self.hand_index, self._pos, self._last_seen = k, _palm(hands[k], size), t
            if metrics[k][col] > off:
                self._release_n += 1
                if self._release_n >= cfg.pinch_release_frames:
                    self._clear()
            else:
                self._release_n = 0
        return self.active
