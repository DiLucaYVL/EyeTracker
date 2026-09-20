"""Background thread: camera -> face/hand landmarks -> gaze point -> cursor and pinch clicks."""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from . import imaging, mouse
from .blink import BlinkDetector
from .camera_props import CAMERA_DEFAULTS, CAMERA_PROPS
from .config import CALIBRATION_PATH, Config
from .control import ScrollController, head_offset_px, head_target_px, hand_target_px, pointing_hand
from .features import extract_features, eye_region_box, face_box
from .filters import MedianFilter, OneEuroFilter
from .gaze_model import GazeModel
from .hands import PinchDetector
from .landmarks import FaceHandTracker

PREVIEW_SIZE = (480, 360)
IMAGE_PREVIEW_SIZE = (480, 360)
EYE_PREVIEW_WIDTH = 480
FACE_IRIS_POINTS = (468, 473)
HAND_LINKS = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17))


@dataclass
class FrameState:
    t: float
    face_ok: bool
    feat: np.ndarray | None
    blink_score: float
    eyes_closed: bool
    gaze: np.ndarray | None          # filtered gaze point in screen pixels
    face_box: tuple[float, float, float] | None
    brightness: float
    fps: float
    hand_ok: bool
    pinch: str | None                # "left" / "right" / None
    ratios: tuple[float, float] | None
    preview: bytes | None            # PPM image (only while want_preview)
    eye_preview: bytes | None = None  # zoomed eye strip, PPM (only in the image-adjust screen)
    eye_stats: tuple[float, float, float] | None = None  # (mean, p5-p95 spread, clipped fraction)
    head_vis: tuple[float, float] | None = None  # (dx, dy) nose offset in the mirrored view, for on-screen guidance
    source: str = "none"              # what drives the cursor now: eye | head_eye | head | hand | none
    scrolling: bool = False           # closed hand = scroll mode
    scroll_speed: float = 0.0         # fist speed in frame heights per second


class Tracker(threading.Thread):
    def __init__(self, cfg: Config, model: GazeModel, events: "queue.SimpleQueue"):
        super().__init__(daemon=True, name="tracker")
        self.cfg = cfg
        self.model = model
        self.events = events
        self.mouse_enabled = False
        self.control_suspended = False   # True while the fullscreen calibration owns the screen
        self.want_preview = False
        self.preview_mode = "small"       # "small" (calibration intro) or "image" (image-adjust screen)
        self.show_original = False        # preview shows the raw camera frame instead of the adjusted one
        self._open_dialog = False
        self._pending_props: dict[int, float] = {}
        self.camera_props_now: dict[str, float] = {}   # values read back from the driver (image screen)
        self._eye_roi_raw: np.ndarray | None = None
        self.error: str | None = None
        self.started = threading.Event()
        self._stop_evt = threading.Event()
        self._state: FrameState | None = None
        self._feat_hist: deque[tuple[float, np.ndarray]] = deque(maxlen=45)
        self._median = MedianFilter(3)
        self._smooth = OneEuroFilter(cfg.smoothing_min_cutoff, cfg.smoothing_beta)
        self._blink = BlinkDetector(cfg)
        self._pinch = PinchDetector(cfg)
        self._cursor: np.ndarray | None = None
        self._pressed: str | None = None
        self._press_t = 0.0
        self._shown_pinch: str | None = None
        self._refit_timer: threading.Timer | None = None
        self._preview_n = 0
        self.head_neutral: np.ndarray | None = None   # (yaw, pitch) that maps to the screen centre in the head modes
        self._src = "none"
        self._hold_offset: np.ndarray | None = None   # hand pointing: cursor minus hand target, kept while pinching
        self._scroll = ScrollController(cfg)
        self.frame_sink = None                        # optional callable(t, raw_frame): diagnostics / clip recording
        self._hand_hist: deque[tuple[float, np.ndarray]] = deque(maxlen=60)
        self.last_hands: tuple[list, tuple[int, int]] = ([], (640, 480))

    # ---------------------------------------------------------------- public
    def mode_ready(self) -> bool:
        """Can the selected modes drive the cursor? The eye modes need a calibration; head/hand pointing do not."""
        cfg = self.cfg
        return self.model.ready or cfg.head_mode in ("head", "off") or cfg.hand_mode == "hand"

    @property
    def mouse_active(self) -> bool:
        return self.mouse_enabled and not self.control_suspended and self.mode_ready()

    def recenter_head(self) -> None:
        """Take the current head pose as the neutral one (screen centre) for the head modes."""
        feats = self.recent_features(0.4)
        self.head_neutral = np.median(np.stack(feats), axis=0)[5:7].copy() if feats else None   # None: next frame

    def latest(self) -> FrameState | None:
        return self._state

    def recent_features(self, seconds: float) -> list[np.ndarray]:
        now = time.perf_counter()
        return [f for ts, f in list(self._feat_hist) if now - ts <= seconds]

    def stop(self) -> None:
        self._stop_evt.set()

    def open_camera_dialog(self) -> None:
        """Ask the capture thread to open the driver's own camera settings dialog."""
        self._open_dialog = True

    def set_camera_prop(self, prop: int, value: float) -> None:
        """Queue a cv2.CAP_PROP_* change; it is applied by the capture thread (which owns the device)."""
        self._pending_props[prop] = value

    def restore_camera_defaults(self) -> None:
        """Put the camera driver back to its original settings and forget the user's overrides."""
        for key, value in (self.cfg.camera_original or CAMERA_DEFAULTS).items():
            if key in CAMERA_PROPS:
                self.set_camera_prop(CAMERA_PROPS[key][0], float(value))
        self.cfg.camera_props.clear()

    def auto_adjust(self) -> tuple[int, float, float] | None:
        """Brightness/contrast/gamma that stretch the eye region of the *raw* image; None without a face."""
        roi = self._eye_roi_raw
        return imaging.auto_params(roi) if roi is not None else None

    def on_physical_click(self, x: int, y: int) -> None:
        """Learn from a real mouse click: assume the user was looking where they clicked."""
        if not self.cfg.learn_from_clicks or not self.model.ready or self.mouse_enabled or self.control_suspended:
            return
        feats = self.recent_features(0.25)
        if len(feats) < 3:
            return
        feat = np.median(np.stack(feats), axis=0)
        pred = self.model.predict_px(feat)
        w, h = self.model.screen
        if pred is None or np.hypot(pred[0] - x, pred[1] - y) > 0.35 * np.hypot(w, h):
            return  # too far from the current belief: probably not looking at the click target
        self.model.add_samples(feat, (x / w, y / h))
        self._schedule_refit()

    # ------------------------------------------------------------------ loop
    def run(self) -> None:
        cap = None
        landmarker = None
        try:
            cap = self._open_camera()
            landmarker = FaceHandTracker(want_hands=self.cfg.hand_clicks, hand_confidence=self.cfg.hand_confidence)
        except Exception as exc:  # noqa: BLE001 - reported to the UI
            self.error = f"{type(exc).__name__}: {exc}"
            self.started.set()
            if cap is not None:
                cap.release()
            return
        self.started.set()

        last_ts, fps, frame_n = 0, 0.0, 0
        stamps: deque[float] = deque(maxlen=20)     # frame times: fps = frames / elapsed over the window
        try:
            while not self._stop_evt.is_set():
                frame_n += 1
                if self.want_preview and self.preview_mode == "image" and frame_n % 15 == 1:
                    self.camera_props_now = {k: cap.get(v[0]) for k, v in CAMERA_PROPS.items()}
                while self._pending_props:
                    prop, value = self._pending_props.popitem()
                    cap.set(prop, value)
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                t = time.perf_counter()
                ts = max(int(t * 1000), last_ts + 1)
                last_ts = ts
                stamps.append(t)
                if len(stamps) > 1 and stamps[-1] > stamps[0]:
                    fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
                if self._open_dialog:  # native DirectShow property page (blocks this thread until closed)
                    self._open_dialog = False
                    cap.set(cv2.CAP_PROP_SETTINGS, 1)
                    continue
                try:
                    proc = imaging.adjust(frame, self.cfg)
                    face, hands = landmarker.process(proc, ts)
                    self._process(t, proc, frame, face, hands, fps)
                except Exception as exc:  # noqa: BLE001 - keep the loop alive
                    self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self._release_button()
            landmarker.close()
            cap.release()

    def _open_camera(self):
        cfg = self.cfg
        cap = cv2.VideoCapture(cfg.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a câmera {cfg.camera_index}")
        if len(cfg.camera_fourcc) == 4:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.camera_fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera_height)
        cap.set(cv2.CAP_PROP_FPS, cfg.camera_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cfg.camera_original and not cfg.camera_props:   # first run: remember what the driver looked like
            cfg.camera_original = {k: float(cap.get(v[0])) for k, v in CAMERA_PROPS.items()}
        for key, value in cfg.camera_props.items():  # settings the user chose in the image screen
            if key in CAMERA_PROPS:
                cap.set(CAMERA_PROPS[key][0], float(value))
        return cap

    def _process(self, t: float, frame: np.ndarray, raw: np.ndarray, face, hands, fps: float) -> None:
        cfg = self.cfg
        h, w = frame.shape[:2]
        self._smooth.min_cutoff = cfg.smoothing_min_cutoff
        self._smooth.beta = cfg.smoothing_beta

        feat = box = head_vis = None
        score = 0.0
        if face is not None:
            feat = extract_features(face.pts, w, h, face.matrix)
            box = face_box(face.pts)
            head_vis = self._head_vis(face.pts)
            score = min(face.blink_left, face.blink_right)
        face_ok = feat is not None
        closed = self._blink.update(t, score, face_ok)
        if face_ok and not closed:
            self._feat_hist.append((t, feat))

        pinch = self._pinch.update(t, hands, (w, h)) if cfg.hand_clicks else None
        self.last_hands = (hands, (w, h))       # raw landmarks of the latest frame (diagnostics)

        # closed hand = scroll mode (wheel events are sent below, once we know the mouse is under our control)
        if cfg.hand_scroll:
            wheel = self._scroll.update(t, pointing_hand(hands) if hands else None, (w, h))
        else:
            self._scroll.reset()
            wheel = (0, 0)

        # cursor target: eye / head+eye / head / hand
        target, source, hold = self._select_target(feat, closed, hands, (w, h))
        if source != self._src:                      # different signal: forget the old filter state
            self._median.reset()
            self._smooth.reset()
            self._src = source
        eyes = source in ("eye", "head_eye")
        self._smooth.min_cutoff = cfg.smoothing_min_cutoff * (1.0 if eyes else 2.0)   # head/hand: less lag
        if target is not None:
            gaze = self._smooth(self._median(target), t)
            if source == "hand":
                self._hand_hist.append((t, gaze.copy()))
        elif hold and self._state is not None:
            gaze = self._state.gaze                  # eyes closed: hold the last point
        else:
            gaze = None
            self._median.reset()
            self._smooth.reset()
        holding = self._blink.settling(t) if eyes else False

        self._drive_mouse(t, gaze, holding, pinch)
        if (wheel[0] or wheel[1]) and self.mouse_active and pinch is None:
            mouse.wheel(*wheel)

        preview = eye_preview = eye_stats = None
        if self.want_preview:
            image_mode = self.preview_mode == "image"
            preview = self._make_preview(raw if self.show_original else frame, face, hands, pinch,
                                         self._scroll.active and cfg.hand_scroll)
            if image_mode and face is not None:
                eye_preview, eye_stats = self._make_eye_views(frame, raw, face)
        self._state = FrameState(
            t=t, face_ok=face_ok, feat=feat, blink_score=score, eyes_closed=closed, gaze=gaze, face_box=box,
            brightness=float(frame[::8, ::8].mean()), fps=fps, hand_ok=bool(hands), pinch=pinch,
            ratios=self._pinch.ratios, preview=preview, eye_preview=eye_preview, eye_stats=eye_stats,
            head_vis=head_vis, source=source, scrolling=self._scroll.active and cfg.hand_scroll,
            scroll_speed=self._scroll.speed)
        if self.frame_sink is not None:
            self.frame_sink(t, raw)

    # ---------------------------------------------------------------- targets
    def _select_target(self, feat, closed: bool, hands, size) -> tuple[np.ndarray | None, str, bool]:
        """(target px or None, source name, hold) for the current frame according to the head/hand modes."""
        cfg = self.cfg
        screen = self.model.screen
        if cfg.hand_mode == "hand" and hands:
            if self._scroll.active and cfg.hand_scroll:
                return None, "hand", True            # closed hand scrolls: the cursor stays where it is (a pinch does not freeze it)
            return hand_target_px(pointing_hand(hands), screen, cfg.hand_gain), "hand", False
        if feat is None:
            return None, "none", False
        mode = cfg.head_mode
        if mode == "off":
            return None, "off", False                # eyes and head are disabled: they never touch the cursor
        if mode == "head":
            return head_target_px(feat, self._neutral(feat), screen, cfg.head_gain), "head", False
        if not self.model.ready:
            return None, mode, False
        if closed:
            return None, mode, True                  # eyes closed: the gaze estimate is invalid, hold
        target = self.model.predict_px(feat)
        if mode == "head_eye":
            offset = head_offset_px(feat[5], feat[6], self._neutral(feat), screen, cfg.head_gain)
            target = np.clip(target + offset, [0, 0], [screen[0] - 1, screen[1] - 1])
        return target, mode, False

    def _neutral(self, feat: np.ndarray) -> np.ndarray:
        if self.head_neutral is None:
            self.head_neutral = np.asarray(feat[5:7], dtype=np.float64).copy()
        return self.head_neutral

    def _press_anchor(self, t: float) -> np.ndarray | None:
        """Hand pointing: the fingertip drifts while the fingers close for the pinch, so click where the hand was
        just before the pinch started (0.2-0.55 s earlier)."""
        if self._src != "hand":
            return None
        pts = [p for ts, p in self._hand_hist if t - 0.55 <= ts <= t - 0.2]
        return np.median(np.stack(pts), axis=0) if len(pts) >= 2 else None

    # ----------------------------------------------------------------- mouse
    def _drive_mouse(self, t: float, gaze, holding: bool, pinch: str | None) -> None:
        active = self.mouse_active
        if not active:
            self._release_button()
            if pinch != self._shown_pinch:  # visual feedback even when mouse control is off
                self._shown_pinch = pinch
                self.events.put(("pinch", pinch, None, None, False))
            return
        self._shown_pinch = pinch
        if self._src == "off":
            self._cursor = None                      # clicks/scroll happen wherever the physical cursor is: never warp it

        # button state follows the pinch
        if pinch and self._pressed is None:
            anchor = self._press_anchor(t)
            if anchor is not None:
                self._cursor = anchor
            if self._cursor is not None:
                mouse.move_to(*self._cursor)
            mouse.button_down(pinch)
            self._pressed, self._press_t = pinch, t
            # Hand pointing: keep following the hand *relatively* from here on. The box around the fingers changes
            # while they close, so following the absolute position would drag the cursor away from the click point.
            if self._src == "hand" and gaze is not None and self._cursor is not None:
                self._hold_offset = self._cursor - np.asarray(gaze, dtype=np.float64)
            self.events.put(("pinch", pinch, *(self._cursor if self._cursor is not None else (None, None)), True))
        elif not pinch and self._pressed is not None:
            self._release_button()
            self.events.put(("pinch", None, None, None, True))

        # cursor follows the target, except: eyes closing, or (eye/head sources) the first instants of a click
        # (precision lock: the eyes wander right after the click). The hand keeps moving the cursor while pinching.
        locked = self._pressed is not None and (t - self._press_t) * 1000 < self.cfg.drag_hold_ms and self._src != "hand"
        if gaze is None or holding or locked:
            return
        pos = np.asarray(gaze, dtype=np.float64)
        if self._hold_offset is not None:
            if self._pressed is None:                # pinch released: let the offset fade out instead of snapping
                self._hold_offset = self._hold_offset * 0.85
                if np.linalg.norm(self._hold_offset) < 1.0:
                    self._hold_offset = None
            if self._hold_offset is not None:
                pos = pos + self._hold_offset
        deadzone = self.cfg.deadzone_px * (1.0 if self._src in ("eye", "head_eye") else 0.5)
        if self._cursor is None or np.linalg.norm(pos - self._cursor) > deadzone or self._pressed:
            self._cursor = pos.copy()
            mouse.move_to(*self._cursor)

    def _release_button(self) -> None:
        if self._pressed is not None:
            mouse.button_up(self._pressed)
            self._pressed = None

    # -------------------------------------------------------------- learning
    def _schedule_refit(self) -> None:
        if self._refit_timer is not None and self._refit_timer.is_alive():
            return
        self._refit_timer = threading.Timer(1.0, self._refit)
        self._refit_timer.daemon = True
        self._refit_timer.start()

    def _refit(self) -> None:
        self.model.fit(cv=False)
        self.model.save(CALIBRATION_PATH)

    # --------------------------------------------------------------- preview
    def _make_preview(self, frame, face, hands, pinch, scrolling: bool = False) -> bytes | None:
        self._preview_n += 1
        if self._preview_n % 2:  # ~15 fps is plenty for the setup screen
            return self._state.preview if self._state else None
        small = cv2.flip(cv2.resize(frame, PREVIEW_SIZE), 1)
        pw, ph = PREVIEW_SIZE

        def px(p):
            return int((1.0 - p[0]) * pw), int(p[1] * ph)

        if face is not None:
            for idx in FACE_IRIS_POINTS:
                cv2.circle(small, px(face.pts[idx]), 3, (80, 255, 80), -1)
            x0, y0 = px((face.pts[:, 0].max(), face.pts[:, 1].min()))
            x1, y1 = px((face.pts[:, 0].min(), face.pts[:, 1].max()))
            cv2.rectangle(small, (x0, y0), (x1, y1), (255, 180, 60), 1)
        color = {"left": (80, 220, 80), "right": (60, 160, 255)}.get(pinch, (255, 255, 255))
        if scrolling:
            color = (247, 129, 163)          # purple (BGR): closed hand = scroll mode
        for hand in hands:
            for a, b in HAND_LINKS:
                cv2.line(small, px(hand.pts[a]), px(hand.pts[b]), color, 2)
            for i in (4, 8, 12):
                cv2.circle(small, px(hand.pts[i]), 6, color, -1)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        return b"P6 %d %d 255\n" % (pw, ph) + rgb.tobytes()

    def _make_eye_views(self, proc: np.ndarray, raw: np.ndarray, face) -> tuple[bytes, tuple[float, float, float]]:
        """Zoomed strip of both eyes (mirrored like the preview) and luminance stats of the adjusted image."""
        h, w = proc.shape[:2]
        x0, y0, x1, y1 = eye_region_box(face.pts, w, h)
        self._eye_roi_raw = cv2.cvtColor(raw[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        shown = raw if self.show_original else proc
        crop = shown[y0:y1, x0:x1]
        ew = EYE_PREVIEW_WIDTH
        eh = max(2, int(ew * crop.shape[0] / crop.shape[1]))
        strip = cv2.flip(cv2.resize(crop, (ew, eh), interpolation=cv2.INTER_CUBIC), 1)
        rgb = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)
        stats = imaging.eye_stats(cv2.cvtColor(proc[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY))
        return b"P6 %d %d 255\n" % (ew, eh) + rgb.tobytes(), stats

    @staticmethod
    def _head_vis(pts: np.ndarray) -> tuple[float, float]:
        """Nose offset from the face-box centre (mirrored horizontally), a sign-clear proxy of head turn/tilt."""
        x0, x1, y0, y1 = pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()
        nose = pts[1]
        return -float((nose[0] - (x0 + x1) / 2) / max(x1 - x0, 1e-6)), float((nose[1] - (y0 + y1) / 2) / max(y1 - y0, 1e-6))
