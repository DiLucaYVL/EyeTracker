"""Signal filters used to stabilise the gaze point."""
from __future__ import annotations

import math
from collections import deque

import numpy as np


class OneEuroFilter:
    """1€ filter (Casiez et al.) for 2D points: smooth at rest, responsive when moving fast."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: np.ndarray | None = None
        self._dx = np.zeros(2)
        self._t = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None:
            self._x, self._t = x.copy(), t
            return x
        dt = max(t - self._t, 1e-3)
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx))
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        self._t = t
        return self._x.copy()


class MedianFilter:
    """Sliding-window median, removes single-frame outliers."""

    def __init__(self, size: int = 3):
        self._buf: deque[np.ndarray] = deque(maxlen=size)

    def reset(self) -> None:
        self._buf.clear()

    def __call__(self, x) -> np.ndarray:
        self._buf.append(np.asarray(x, dtype=np.float64))
        return np.median(np.stack(self._buf), axis=0)
