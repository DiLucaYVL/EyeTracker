"""Procedural 3D wireframe head, used to show (and compare with) the head pose wanted during calibration.

Conventions match MediaPipe's facial transformation matrix (verified against real captures):
camera space is X right, Y up, looking down -Z; the face looks towards +Z. Pose = (yaw, pitch, roll, tx, ty, tz)
with R = Ry(yaw) @ Rx(pitch) @ Rz(roll): +yaw turns the head to the *subject's* left, +pitch looks down.
The view is mirrored horizontally (like the camera preview): turning to your left moves the nose to the screen's left.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Head half-extents (cm): width, height, depth of the skull ellipsoid.
A, B, C = 7.2, 9.6, 8.6
VIEW_HEIGHT_CM = 30.0        # how many cm of the scene fit in a viewport's height
VIEW_CENTER_Y_CM = -2.0      # head + neck are drawn slightly below the head centre


@dataclass
class Polyline:
    pts: np.ndarray   # (N, 3) points in head space
    kind: str         # "skull" | "face" | "neck"


def _surface(x: float, y: float) -> float:
    """z of the skull surface at (x, y) (front side)."""
    return C * math.sqrt(max(0.0, 1.0 - (x / A) ** 2 - (y / B) ** 2))


def build_head() -> list[Polyline]:
    lines: list[Polyline] = []
    th = np.linspace(0, np.pi, 25)
    for k in range(8):  # meridians
        ph = k * np.pi / 4
        lines.append(Polyline(np.stack([A * np.sin(th) * np.cos(ph), B * np.cos(th), C * np.sin(th) * np.sin(ph)], 1), "skull"))
    ph = np.linspace(0, 2 * np.pi, 49)
    for t in np.radians((35, 60, 90, 120, 145)):  # parallels
        lines.append(Polyline(np.stack([A * np.sin(t) * np.cos(ph), np.full_like(ph, B * np.cos(t)), C * np.sin(t) * np.sin(ph)], 1), "skull"))

    # face features (drawn brighter): eyes, brows, nose, mouth
    a = np.linspace(0, 2 * np.pi, 17)
    for sx in (-1, 1):
        ex, ey = sx * 3.1, 2.6
        pts = np.stack([ex + 1.5 * np.cos(a), ey + 0.75 * np.sin(a)], 1)
        lines.append(Polyline(np.array([[x, y, _surface(x, y) + 0.15] for x, y in pts]), "face"))
        brow = np.linspace(-1.8, 1.8, 7)
        lines.append(Polyline(np.array([[ex + t, 4.6 + 0.35 * (1 - (t / 1.8) ** 2), _surface(ex + t, 4.6) + 0.15] for t in brow]), "face"))
    tip = np.array([0.0, -1.6, C + 2.6])
    bridge = np.array([0.0, 2.4, _surface(0, 2.4) + 0.2])
    lines.append(Polyline(np.array([bridge, tip]), "face"))
    for sx in (-1, 1):
        lines.append(Polyline(np.array([tip, [sx * 1.5, -2.3, C + 1.2], [sx * 0.5, -2.6, C + 1.8]]), "face"))
    mouth = np.linspace(-2.4, 2.4, 9)
    lines.append(Polyline(np.array([[t, -5.2 - 0.25 * (1 - (t / 2.4) ** 2), _surface(t, -5.2) + 0.15] for t in mouth]), "face"))
    # ears
    e = np.linspace(-0.9, 0.9, 11)
    for sx in (-1, 1):
        lines.append(Polyline(np.stack([np.full_like(e, sx * (A + 0.2)), 0.5 + 2.2 * e, -0.5 + 1.6 * np.cos(e * 1.6)], 1), "face"))
    # neck
    for sx in (-1, 1):
        lines.append(Polyline(np.array([[sx * 3.4, -B * 0.82, -1.5], [sx * 3.6, -B - 6.0, -1.8]]), "neck"))
    return lines


HEAD = build_head()


def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy, cp, sp, cr, sr = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return ry @ rx @ rz


@dataclass
class Projected:
    xy: np.ndarray     # (N, 2) view-plane coordinates in pixels, relative to the viewport centre
    front: np.ndarray  # (N,) True where the point faces the camera
    kind: str


def project(pose, view_px: float, ref_dist_cm: float, mirror: bool = True) -> list[Projected]:
    """Project the head for `pose` = (yaw, pitch, roll, tx, ty, tz) with tz < 0 (in front of the camera).

    view_px: viewport height in pixels. ref_dist_cm: distance at which the head has its nominal size.
    """
    yaw, pitch, roll, tx, ty, tz = pose
    rot = rotation(yaw, pitch, roll)
    px_per_cm = view_px / VIEW_HEIGHT_CM
    focal = px_per_cm * ref_dist_cm
    out = []
    for line in HEAD:
        local = line.pts @ rot.T
        cam = local + np.array([tx, ty, tz])
        depth = np.maximum(-cam[:, 2], 5.0)
        x = cam[:, 0] / depth * focal
        y = (cam[:, 1] - VIEW_CENTER_Y_CM) / depth * focal
        out.append(Projected(np.stack([-x if mirror else x, -y], 1), local[:, 2] > 0.0 if line.kind != "neck" else np.zeros(len(local), bool), line.kind))
    return out


def runs(p: Projected):
    """Split a projected polyline into runs that are entirely front-facing or entirely back-facing."""
    start = 0
    for i in range(1, len(p.front) + 1):
        if i == len(p.front) or p.front[i] != p.front[start]:
            end = min(i + 1, len(p.front))  # share the boundary vertex so runs join up
            if end - start >= 2:
                yield p.xy[start:end], bool(p.front[start])
            start = i


def mix(fg: str, bg: str, t: float) -> str:
    a = [int(fg[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(bg[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(round(b[i] + (a[i] - b[i]) * t)):02x}" for i in range(3))


def draw_on_canvas(canvas, cx: float, cy: float, projected: list[Projected], color: str, bg: str,
                   width: int = 2, tag: str = "head") -> None:
    """Draw with depth cue: back-facing lines are faint, face features are emphasised."""
    for p in projected:
        for xy, front in runs(p):
            if p.kind == "face":
                col, w = mix(color, bg, 1.0), width + 1
            elif front:
                col, w = mix(color, bg, 0.75), width
            else:
                col, w = mix(color, bg, 0.22), 1
            coords = [c for pt in xy for c in (cx + float(pt[0]), cy + float(pt[1]))]
            canvas.create_line(*coords, fill=col, width=w, tags=tag)


# ---- target poses used by the calibration screens ---------------------------------------------------------------
# Each entry: (caption, function(phase in [0, 1)) -> (yaw, pitch, roll, dx, dy, dz) relative to the reference pose).
def _wave(phase: float) -> float:
    return math.sin(2 * math.pi * phase)


TARGET_POSES = {
    "still": ("Cabeça parada, de frente para a tela", lambda ph: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    "sides": ("Vire a cabeça devagar para os lados", lambda ph: (0.35 * _wave(ph), 0.0, 0.0, 0.0, 0.0, 0.0)),
    "updown": ("Incline a cabeça para cima e para baixo", lambda ph: (0.0, 0.25 * _wave(ph), 0.0, 0.0, 0.0, 0.0)),
    "depth": ("Aproxime e afaste um pouco o rosto", lambda ph: (0.0, 0.0, 0.0, 0.0, 0.0, 6.0 * _wave(ph))),
    "circle": ("Faça um pequeno círculo com a cabeça",
               lambda ph: (0.3 * math.cos(2 * math.pi * ph), 0.2 * math.sin(2 * math.pi * ph), 0.0, 0.0, 0.0, 0.0)),
}
HEAD_PROMPT_KEYS = ("sides", "updown", "depth", "circle")


def target_pose(key: str, phase: float, ref_dist_cm: float, base=(0.0, 0.0, 0.0)):
    """Absolute pose for the target model: the reference orientation/distance plus the animated deltas."""
    yaw, pitch, roll, dx, dy, dz = TARGET_POSES[key][1](phase)
    return (base[0] + yaw, base[1] + pitch, base[2] + roll, dx, dy, -ref_dist_cm + dz)


def live_pose(feat: np.ndarray, ref: np.ndarray, ref_dist_cm: float):
    """The user's current pose, translated so that their reference position coincides with the target's."""
    yaw, pitch, roll, tx, ty, tz = (float(v) for v in feat[5:11])
    return (yaw, pitch, roll, tx - float(ref[3]), ty - float(ref[4]), -ref_dist_cm + (tz - float(ref[5])))
