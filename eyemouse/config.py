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


@dataclass
class Config:
    # camera
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    camera_props: dict = field(default_factory=dict)  # driver settings the user changed, e.g. {"saturation": 60}
    camera_fourcc: str = ""           # e.g. "MJPG"; empty = driver default (forcing MJPG breaks some webcams)
    # image adjustments applied before detection (neutral = 0 / 1 / 1 / 0 / 0)
    img_brightness: int = 0
    img_contrast: float = 1.0
    img_gamma: float = 1.0
    img_clahe: float = 0.0
    img_sharpen: float = 0.0
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
    pinch_on_ratio: float = 0.30
    pinch_off_ratio: float = 0.45
    pinch_confirm_frames: int = 2
    drag_hold_ms: int = 350
    # learning / calibration
    learn_from_clicks: bool = True
    calibration_points: int = 16
    calibration_head: bool = True     # second calibration phase: look at the targets while moving the head

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Config":
        cfg = cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cfg
        known = {f.name: f.type for f in fields(cls)}
        for key, value in data.items():
            if key in known:
                setattr(cfg, key, type(getattr(cfg, key))(value))
        return cfg

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
