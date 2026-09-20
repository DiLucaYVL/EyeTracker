"""Pinch gesture detection: thumb+index = left button, thumb+middle = right button."""
from __future__ import annotations

import numpy as np

from .config import Config
from .landmarks import HandData

WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, MIDDLE_MCP = 0, 4, 8, 12, 9


def pinch_ratios(world: np.ndarray) -> tuple[float, float]:
    """(thumb-index, thumb-middle) fingertip distances divided by palm length (wrist to middle knuckle)."""
    palm = float(np.linalg.norm(world[WRIST] - world[MIDDLE_MCP]))
    if palm < 1e-6:
        return 9.9, 9.9
    thumb = world[THUMB_TIP]
    return (float(np.linalg.norm(thumb - world[INDEX_TIP])) / palm,
            float(np.linalg.norm(thumb - world[MIDDLE_TIP])) / palm)


class PinchDetector:
    """Debounced pinch state machine with hysteresis. `active` is None, "left" or "right"."""

    LOST_HAND_S = 0.25  # release when the hand disappears for this long

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.active: str | None = None
        self.ratios: tuple[float, float] | None = None  # of the hand closest to pinching
        self._cand: str | None = None
        self._cand_n = 0
        self._release_n = 0
        self._last_seen = -1e9

    def reset(self) -> None:
        self.active = None
        self._cand, self._cand_n, self._release_n = None, 0, 0

    def update(self, t: float, hands: list[HandData]) -> str | None:
        cfg = self.cfg
        if not hands:
            self.ratios = None
            self._cand, self._cand_n = None, 0
            if self.active and t - self._last_seen > self.LOST_HAND_S:
                self.active = None
            return self.active
        self._last_seen = t
        all_ratios = [pinch_ratios(h.world) for h in hands]
        self.ratios = min(all_ratios, key=lambda r: min(r))

        if self.active is None:
            best = None  # (ratio, button)
            for ri, rm in all_ratios:
                if ri < cfg.pinch_on_ratio and ri <= rm and (best is None or ri < best[0]):
                    best = (ri, "left")
                elif rm < cfg.pinch_on_ratio and rm < ri and (best is None or rm < best[0]):
                    best = (rm, "right")
            cand = best[1] if best else None
            if cand and cand == self._cand:
                self._cand_n += 1
            else:
                self._cand, self._cand_n = cand, 1 if cand else 0
            if cand and self._cand_n >= cfg.pinch_confirm_frames:
                self.active, self._release_n = cand, 0
        else:
            idx = 0 if self.active == "left" else 1
            ratio = min(r[idx] for r in all_ratios)
            if ratio > cfg.pinch_off_ratio:
                self._release_n += 1
                if self._release_n >= cfg.pinch_confirm_frames:
                    self.active, self._cand, self._cand_n = None, None, 0
            else:
                self._release_n = 0
        return self.active
