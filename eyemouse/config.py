"""User configuration and data paths."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Mapping

FROZEN = bool(getattr(sys, "frozen", False))   # running from the PyInstaller build / installed app
ROOT = Path(__file__).resolve().parent.parent
# Files bundled with the app (PyInstaller unpacks them here); in development this is the project folder.
RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", ROOT))


def resolve_data_dir(frozen: bool, env: Mapping[str, str], root: Path = ROOT) -> Path:
    """Where the user's calibration, config, log and downloaded models live.

    EYEMOUSE_DATA_DIR overrides everything (portable use, tests). The installed app must not write next to its
    executable (Program Files may be read-only), so it uses %APPDATA%\\EyeMouse; development keeps ./data.
    """
    if env.get("EYEMOUSE_DATA_DIR"):
        return Path(env["EYEMOUSE_DATA_DIR"])
    if frozen:
        return Path(env.get("APPDATA") or Path.home()) / "EyeMouse"
    return root / "data"


def find_model(name: str, data_dir: Path, resource_dir: Path) -> Path:
    """Prefer a model bundled in the installer; otherwise the data folder (where ensure_models downloads it)."""
    bundled = resource_dir / "models" / name
    return bundled if bundled.exists() else data_dir / name


DATA_DIR = resolve_data_dir(FROZEN, os.environ)
CONFIG_PATH = DATA_DIR / "config.json"
CALIBRATION_PATH = DATA_DIR / "calibration.json"
LOG_PATH = DATA_DIR / "eyemouse.log"
ICON_PATH = RESOURCE_DIR / "assets" / "eyemouse.ico"
FACE_MODEL_PATH = find_model("face_landmarker.task", DATA_DIR, RESOURCE_DIR)
HAND_MODEL_PATH = find_model("hand_landmarker.task", DATA_DIR, RESOURCE_DIR)


HEAD_MODES = {"eye": "Olho", "head_eye": "Cabeça + olho", "head": "Cabeça", "off": "Desativado"}
HAND_MODES = {"pinch": "Pinça (clique)", "hand": "Ponta do indicador move o cursor + pinça"}

# Bump when a *default* changes in a way old saved files must not override. Only the keys listed for a version are reset
# when loading a file written before it (everything the user chose deliberately is kept).
CONFIG_VERSION = 4
RESET_ON_UPGRADE = {2: ("pinch_on_ratio", "pinch_off_ratio", "pinch_confirm_frames"),
                    3: ("pinch_off_ratio", "pinch_release_frames"),
                    4: ("pinch_confirm_frames", "pinch_release_frames")}


def _migrate_pinch_ball(cfg: "Config", data: dict) -> None:
    """Version 4 replaced the per-finger pinch thresholds by one ball size (the click fires when the two drawn balls touch).
    A file written by the pinch calibration keeps the user's measured contact level: recover it from the rule that
    produced the stored threshold (version 2: 1.6 * contact + 0.02, version 3: 1.35 * contact + 0.03) and re-derive."""
    version = int(data.get("config_version", 0))
    scale, offset = (1.6, 0.02) if version < 3 else (1.35, 0.03)
    ons = []
    for finger in ("index", "middle"):
        stored = float(data.get(f"pinch_on_{finger}", 0) or 0)
        if stored > 0:
            contact = max((stored - offset) / scale, 0.0)
            ons.append(min(max(contact + 0.03, 0.15), 0.6))
    if ons:
        cfg.pinch_ball_size = round(min(max(max(ons) / 2.0, 0.03), 0.30), 4)


@dataclass
class Config:
    config_version: int = 0
    # camera
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    camera_props: dict = field(default_factory=dict)  # driver settings the user changed, e.g. {"saturation": 60}
    camera_original: dict = field(default_factory=dict)  # driver settings seen before the user changed anything
    camera_fourcc: str = ""           # e.g. "MJPG"; empty = driver default (forcing MJPG breaks some webcams)
    # image adjustments applied before detection (neutral = 0 / 1 / 1 / 0 / 0)
    img_brightness: int = 0
    img_contrast: float = 1.0
    img_gamma: float = 1.0
    img_clahe: float = 0.0
    img_sharpen: float = 0.0
    # control modes (see eyemouse/control.py)
    head_mode: str = "eye"            # "eye" | "head_eye" | "head"
    hand_mode: str = "pinch"          # "pinch" | "hand"
    head_gain: float = 1.0            # head pointing sensitivity
    hand_gain: float = 1.0            # hand pointing sensitivity
    hand_scroll: bool = True          # a closed hand scrolls (two-finger touchpad style)
    scroll_gain: float = 1.0          # scroll sensitivity (speed -> intensity)
    scroll_natural: bool = False      # False: hand down scrolls down (Windows default); True: inverted
    # cursor
    start_with_mouse: bool = False
    smoothing_min_cutoff: float = 0.6
    smoothing_beta: float = 0.015
    deadzone_px: float = 20.0
    # bubble
    show_bubble: bool = True
    bubble_size: int = 96
    # blink: only used to freeze the cursor while the eyes are closed
    blink_close_thr: float = 0.55
    blink_open_thr: float = 0.35
    # hand pinch clicks (ratio = thumb-to-fingertip distance / palm length)
    hand_clicks: bool = True
    hand_confidence: float = 0.5      # MediaPipe hand detection/tracking confidence (0.2-0.5 made no difference when the hand is in view)
    pinch_ball_size: float = 0.095    # radius of the fingertip "balls" as a fraction of the hand size: the click starts when they touch
    pinch_release_frames: int = 1     # frames the balls must be apart before the click is released (1 = immediately)
    pinch_confirm_frames: int = 1     # consecutive frames the balls must touch (1 = immediately: touching is the only criterion)
    pinch_release_margin: float = 0.0  # extra distance (fraction) before a held pinch is released (0 = as soon as they stop touching)
    drag_hold_ms: int = 350
    # learning / calibration
    learn_from_clicks: bool = True
    calibration_points: int = 16
    calibration_head: bool = True     # second calibration phase: look at the targets while moving the head

    def pinch_thresholds(self, finger: str = "index") -> tuple[float, float]:
        """(trigger, release) fingertip distance in hand sizes: the balls touch at two radii and, by default, the pinch
        ends at that same distance (touching is the only criterion)."""
        on = 2.0 * self.pinch_ball_size
        return on, on * (1.0 + max(self.pinch_release_margin, 0.0))

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Config":
        cfg = cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cfg
        known = {f.name: f.type for f in fields(cls)}
        stale = {k for v, keys in RESET_ON_UPGRADE.items() if int(data.get("config_version", 0)) < v for k in keys}
        for key, value in data.items():
            if key in known and key not in stale:
                setattr(cfg, key, type(getattr(cfg, key))(value))
        if int(data.get("config_version", 0)) < 4:
            _migrate_pinch_ball(cfg, data)
        if cfg.head_mode not in HEAD_MODES:
            cfg.head_mode = "eye"
        if cfg.hand_mode not in HAND_MODES:
            cfg.hand_mode = "pinch"
        cfg.config_version = CONFIG_VERSION
        return cfg

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.config_version = CONFIG_VERSION
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
