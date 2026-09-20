"""Gaze model: ridge regression from eye/head features to normalised screen coordinates.

The model keeps every calibration sample it has ever seen (up to MAX_SAMPLES) and refits on demand, so
each new calibration, guided test or corrective click refines the previous knowledge instead of replacing it.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import N_FEATURES

N_EYE = 4  # uR, vR, uL, vL take part in the quadratic terms
# Fixed feature scaling so that every column is roughly unit range for the gaze span.
SCALE = np.array([10, 10, 10, 10, 5, 3, 3, 3, 0.1, 0.1, 0.1], dtype=np.float64)
MIN_GROUPS = 4
MAX_SAMPLES = 6000
ALPHAS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
DEGREES = (1, 2, 3)  # 1: linear, 2: + eye quadratics, 3: + head/depth interactions


@dataclass
class FitReport:
    degree: int
    alpha: float
    cv_px: float | None
    cv_frac: float | None  # cv_px as a fraction of the screen diagonal
    n_groups: int
    n_samples: int


def _design(z: np.ndarray, degree: int) -> np.ndarray:
    cols = [np.ones((z.shape[0], 1)), z]
    if degree >= 2:
        e = z[:, :N_EYE]
        cols.append(np.stack([e[:, i] * e[:, j] for i in range(N_EYE) for j in range(i, N_EYE)], axis=1))
    if degree >= 3:
        # Distance to the screen scales how far a given eye/head angle moves the gaze point.
        depth = z[:, 10:11]
        cols.append(np.hstack([e * depth, z[:, 5:7] * depth]))
    return np.hstack(cols)


def _penalty(degree: int) -> np.ndarray:
    pen = [0.0] + [1.0] * 4 + [1.0] + [4.0] * 6  # bias, eye, lid, head
    if degree >= 2:
        pen += [8.0] * 10
    if degree >= 3:
        pen += [6.0] * 6
    return np.array(pen)


def _group_weights(groups: np.ndarray) -> np.ndarray:
    """Every target counts the same regardless of how many frames were captured on it."""
    _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]
    return w / w.sum()


def _solve(phi: np.ndarray, y: np.ndarray, w: np.ndarray, lam: np.ndarray) -> np.ndarray:
    a = phi.T @ (phi * w[:, None]) + np.diag(lam + 1e-9)
    return np.linalg.solve(a, phi.T @ (y * w[:, None]))


def reject_outliers(feats: np.ndarray, z_max: float = 3.5, keep_min: int = 8) -> np.ndarray:
    """Drop frames whose eye features deviate strongly from the median (saccades, tracking glitches)."""
    if len(feats) <= keep_min:
        return feats
    eye = feats[:, :N_EYE]
    med = np.median(eye, axis=0)
    mad = np.median(np.abs(eye - med), axis=0) * 1.4826 + 1e-4
    score = np.max(np.abs(eye - med) / mad, axis=1)
    keep = score <= z_max
    if keep.sum() < keep_min:
        keep = score <= np.sort(score)[keep_min - 1]
    return feats[keep]


class GazeModel:
    def __init__(self, screen: tuple[int, int]):
        self.screen = (int(screen[0]), int(screen[1]))
        self._lock = threading.RLock()
        self._x = np.zeros((0, N_FEATURES))
        self._y = np.zeros((0, 2))
        self._g = np.zeros(0, dtype=np.int64)
        self._next_group = 0
        self._params: dict | None = None
        self._hyper: tuple[int, float] | None = None
        self.cv_px: float | None = None

    # ------------------------------------------------------------------ data
    @property
    def ready(self) -> bool:
        return self._params is not None

    @property
    def n_samples(self) -> int:
        return len(self._x)

    @property
    def n_groups(self) -> int:
        return len(np.unique(self._g))

    @property
    def cv_frac(self) -> float | None:
        if self.cv_px is None:
            return None
        return self.cv_px / float(np.hypot(*self.screen))

    def head_spread(self) -> float:
        """Std of the (scaled) head pose over the stored samples: ~0 means the head never moved during calibration."""
        with self._lock:
            if len(self._x) < 2:
                return 0.0
            return float(np.std(self._x[:, 5:11] * SCALE[5:11], axis=0).mean())

    def new_group(self) -> int:
        with self._lock:
            self._next_group += 1
            return self._next_group

    def add_samples(self, feats: np.ndarray, target_norm, group: int | None = None) -> int:
        """Add frames captured while the user looked at target_norm = (x/W, y/H)."""
        feats = np.atleast_2d(np.asarray(feats, dtype=np.float64))
        with self._lock:
            if group is None:
                group = self.new_group()
            y = np.tile(np.asarray(target_norm, dtype=np.float64), (len(feats), 1))
            self._x = np.vstack([self._x, feats])[-MAX_SAMPLES:]
            self._y = np.vstack([self._y, y])[-MAX_SAMPLES:]
            self._g = np.concatenate([self._g, np.full(len(feats), group, dtype=np.int64)])[-MAX_SAMPLES:]
        return group

    def clear(self) -> None:
        with self._lock:
            self._x = np.zeros((0, N_FEATURES))
            self._y = np.zeros((0, 2))
            self._g = np.zeros(0, dtype=np.int64)
            self._params = None
            self._hyper = None
            self.cv_px = None

    # ---------------------------------------------------------------- fitting
    def fit(self, cv: bool = True) -> FitReport | None:
        """Fit on all stored samples. cv=True also selects (degree, alpha) by leave-one-target-out CV."""
        with self._lock:
            x, y, g = self._x.copy(), self._y.copy(), self._g.copy()
        if len(np.unique(g)) < MIN_GROUPS:
            self._params = None
            return None
        center = x.mean(axis=0)
        z = (x - center) * SCALE
        w0 = _group_weights(g)

        if cv or self._hyper is None:
            best = None
            for degree in DEGREES:
                phi = _design(z, degree)
                pen = _penalty(degree)
                for alpha in ALPHAS:
                    err = self._cv_error(phi, y, g, pen * alpha)
                    if best is None or err < best[0] - 1e-6:
                        best = (err, degree, alpha)
            self.cv_px, degree, alpha = best
            self._hyper = (degree, alpha)
        degree, alpha = self._hyper

        phi = _design(z, degree)
        lam = _penalty(degree) * alpha
        w = w0.copy()
        for _ in range(3):  # Huber-style reweighting to resist stray samples
            beta = _solve(phi, y, w, lam)
            res = np.linalg.norm(phi @ beta - y, axis=1)
            c = 1.5 * float(np.median(res)) + 1e-6
            w = w0 * np.minimum(1.0, c / np.maximum(res, 1e-9))
            w /= w.sum()
        beta = _solve(phi, y, w, lam)

        self._params = {"center": center, "beta": beta, "degree": degree}
        frac = self.cv_frac
        return FitReport(degree, alpha, self.cv_px, frac, len(np.unique(g)), len(x))

    def _cv_error(self, phi: np.ndarray, y: np.ndarray, g: np.ndarray, lam: np.ndarray) -> float:
        groups = np.unique(g)
        if len(groups) <= 25:
            folds = [g == gid for gid in groups]
        else:
            idx = np.searchsorted(groups, g)
            folds = [(idx % 10) == f for f in range(10)]
        scr = np.array(self.screen, dtype=np.float64)
        errs = []
        for test in folds:
            train = ~test
            if not train.any() or not test.any():
                continue
            beta = _solve(phi[train], y[train], _group_weights(g[train]), lam)
            e = np.linalg.norm((phi[test] @ beta - y[test]) * scr, axis=1)
            errs.append(float(e.mean()))
        return float(np.mean(errs)) if errs else float("inf")

    # -------------------------------------------------------------- inference
    def predict_norm(self, feat) -> np.ndarray | None:
        p = self._params
        if p is None:
            return None
        z = (np.asarray(feat, dtype=np.float64) - p["center"]) * SCALE
        return (_design(z[None, :], p["degree"]) @ p["beta"])[0]

    def predict_px(self, feat) -> np.ndarray | None:
        n = self.predict_norm(feat)
        if n is None:
            return None
        n = np.clip(n, -0.03, 1.03)
        w, h = self.screen
        return np.clip(n * np.array([w, h]), [0, 0], [w - 1, h - 1])

    # ------------------------------------------------------------ persistence
    def snapshot(self):
        with self._lock:
            return (self._x.copy(), self._y.copy(), self._g.copy(), self._next_group, self._params, self._hyper, self.cv_px)

    def restore(self, snap) -> None:
        with self._lock:
            self._x, self._y, self._g, self._next_group, self._params, self._hyper, self.cv_px = (
                snap[0].copy(), snap[1].copy(), snap[2].copy(), snap[3], snap[4], snap[5], snap[6])

    def save(self, path: Path) -> None:
        with self._lock:
            data = {
                "version": 1,
                "screen": list(self.screen),
                "hyper": list(self._hyper) if self._hyper else None,
                "cv_px": self.cv_px,
                "next_group": self._next_group,
                "x": np.round(self._x, 6).tolist(),
                "y": np.round(self._y, 6).tolist(),
                "g": self._g.tolist(),
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(path)

    def load(self, path: Path) -> bool:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            x = np.array(data["x"], dtype=np.float64).reshape(-1, N_FEATURES)
            y = np.array(data["y"], dtype=np.float64).reshape(-1, 2)
            g = np.array(data["g"], dtype=np.int64)
        except (OSError, ValueError, KeyError):
            return False
        with self._lock:
            self._x, self._y, self._g = x, y, g
            self._next_group = int(data.get("next_group", int(g.max()) + 1 if len(g) else 0))
            hyper = data.get("hyper")
            self._hyper = (int(hyper[0]), float(hyper[1])) if hyper else None
            self.cv_px = data.get("cv_px")
        self.fit(cv=self._hyper is None)
        return self.ready
