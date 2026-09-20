"""Software image adjustments applied to every camera frame before landmark detection."""
from __future__ import annotations

import cv2
import numpy as np

from .config import Config

_clahe_cache: dict[float, "cv2.CLAHE"] = {}


def is_neutral(cfg: Config) -> bool:
    return (cfg.img_brightness == 0 and abs(cfg.img_contrast - 1.0) < 1e-6 and abs(cfg.img_gamma - 1.0) < 1e-6
            and cfg.img_clahe <= 0 and cfg.img_sharpen <= 0)


def build_lut(brightness: float, contrast: float, gamma: float) -> np.ndarray:
    """256-entry table: gamma (>1 brightens midtones), then contrast around mid-grey, then brightness."""
    x = np.arange(256, dtype=np.float32) / 255.0
    x = np.power(x, 1.0 / max(gamma, 1e-3)) * 255.0
    x = (x - 128.0) * contrast + 128.0 + brightness
    return np.clip(x, 0, 255).astype(np.uint8)


def adjust(frame: np.ndarray, cfg: Config) -> np.ndarray:
    """Return the adjusted BGR frame (the input itself when every setting is neutral)."""
    if is_neutral(cfg):
        return frame
    out = frame
    if cfg.img_brightness != 0 or cfg.img_contrast != 1.0 or cfg.img_gamma != 1.0:
        out = cv2.LUT(out, build_lut(cfg.img_brightness, cfg.img_contrast, cfg.img_gamma))
    if cfg.img_clahe > 0:
        clip = round(1.0 + cfg.img_clahe * 5.0, 1)
        clahe = _clahe_cache.setdefault(clip, cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)))
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if cfg.img_sharpen > 0:
        blur = cv2.GaussianBlur(out, (0, 0), 1.5)
        out = cv2.addWeighted(out, 1.0 + cfg.img_sharpen, blur, -cfg.img_sharpen, 0)
    return out


def auto_params(gray_roi: np.ndarray, low: float = 25.0, high: float = 225.0) -> tuple[int, float, float]:
    """Brightness/contrast/gamma that make the eye region well exposed and use most of the tonal range.

    1. gamma moves the median luminance to ~45% grey (lifts dark eyes, tames blown-out ones);
    2. contrast then stretches the 2nd-98th percentile span towards [low, high];
    3. brightness centres the result.
    """
    med = float(np.clip(np.median(gray_roi) / 255.0, 0.02, 0.98))
    gamma = float(np.clip(np.log(med) / np.log(0.45), 0.5, 2.5))
    lifted = np.power(gray_roi.astype(np.float32) / 255.0, 1.0 / gamma) * 255.0
    p2, p98 = np.percentile(lifted, (2, 98))
    contrast = float(np.clip((high - low) / max(float(p98 - p2), 20.0), 0.5, 3.0))
    mid = (p2 + p98) / 2.0
    brightness = (low + high) / 2.0 - ((mid - 128.0) * contrast + 128.0)
    return int(np.clip(round(brightness), -100, 100)), round(contrast, 2), round(gamma, 2)


def eye_stats(gray_roi: np.ndarray) -> tuple[float, float, float]:
    """(mean luminance, p5-p95 spread, fraction of clipped pixels) of the eye region."""
    p5, p95 = np.percentile(gray_roi, (5, 95))
    clipped = float(np.mean((gray_roi >= 250) | (gray_roi <= 4)))
    return float(gray_roi.mean()), float(p95 - p5), clipped


def describe_eye_quality(mean: float, spread: float, clipped: float) -> tuple[str, str]:
    """Human-readable verdict and a level: 'good' | 'warn' | 'bad'."""
    if mean < 25:
        return "Imagem quase preta — algum ajuste (brilho, gama, exposição) está escurecendo demais. Aperte R para restaurar", "bad"
    if mean < 70:
        return "Olhos escuros demais — aumente brilho/gama ou acenda uma luz", "bad"
    if mean > 190 or clipped > 0.08:
        return "Imagem estourada — reduza brilho/contraste", "bad"
    if spread < 60:
        return "Contraste baixo — aumente contraste ou CLAHE", "warn"
    return "Bom contraste nos olhos ✓", "good"
