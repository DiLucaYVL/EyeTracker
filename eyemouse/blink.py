"""Eye-closure detection with hysteresis (used to freeze the cursor, not to click)."""
from __future__ import annotations

from .config import Config


class BlinkDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.closed = False
        self.opened_at = -1e9

    def update(self, t: float, score: float, face_ok: bool) -> bool:
        """score = min(eyeBlinkLeft, eyeBlinkRight). Returns the current closed state."""
        if not face_ok:
            return self.closed
        if not self.closed:
            if score >= self.cfg.blink_close_thr:
                self.closed = True
        elif score <= self.cfg.blink_open_thr:
            self.closed = False
            self.opened_at = t
        return self.closed

    def settling(self, t: float, settle_s: float = 0.12) -> bool:
        """True while eyes are closed or just reopened (gaze estimate not yet reliable)."""
        return self.closed or (t - self.opened_at) < settle_s
