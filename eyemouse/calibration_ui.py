"""Fullscreen calibration: setup + gesture test -> guided targets -> live refinement.

Phases: intro -> countdown -> run -> (fit) -> refine [-> test -> refine] -> close.
Every phase draws on a single fullscreen canvas.
"""
from __future__ import annotations

import time
import tkinter as tk
from typing import Callable

import numpy as np

from .config import CALIBRATION_PATH, Config
from . import head3d, mouse
from .gaze_model import GazeModel, reject_outliers
from .hands import derive_guard_curl, derive_thresholds, other_fingers_curl, pinch_metrics

BG, FG, DIM = "#0b0f14", "#e6edf3", "#8b949e"
ACCENT, GOOD, WARN, BAD, ORANGE = "#2f81f7", "#3fb950", "#d29922", "#f85149", "#f0883e"
FONT = "Segoe UI"

PRESETS = {"1": 9, "2": 16, "3": 25}
PRESET_NAMES = {9: "Rápida", 16: "Normal", 25: "Precisa"}
GRID_SIDE = {9: 3, 16: 4, 25: 5}
SECONDS_HINT = {9: 25, 16: 40, 25: 65}
HEAD_SECONDS = 40

SETTLE_S = 0.9        # time for the eyes to reach the new target
CAPTURE_MAX_S = 3.5
FRAMES_NEEDED = 18
FRAMES_MIN = 8
MAX_ATTEMPTS = 3

# Phase 2: look at the targets while moving the head, so the model can learn to compensate head pose.
HEAD_FRAMES_NEEDED = 20
HEAD_FRAMES_MIN = 8
HEAD_CAPTURE_MAX_S = 5.5
HEAD_MIN_DIST = 0.05        # in model pose units (yaw/pitch rad*3, translation cm*0.1): forces pose variety
HEAD_POINTS = 9
HEAD_SPREAD_MIN = 0.08      # below this the stored samples contain (almost) no head movement
POSE_IDX = (5, 6, 8, 9, 10)  # yaw, pitch, tx, ty, tz inside the feature vector
POSE_SCALE = np.array([3.0, 3.0, 0.1, 0.1, 0.1])
MODEL_COLOR, GHOST_COLOR, VIEW_BG = "#39c5cf", "#f0f6fc", "#0e131a"
PROMPT_SECONDS = 1.6
INTRO_BASE = (0.0, 0.08, 0.0)   # the webcam sits above the screen, so looking at the screen tilts the head slightly down
# Pinch calibration: (key, instruction). "open" gives the baseline, the others the pinched values per finger.
PINCH_STEPS = (("open", "Mão ABERTA, com os dedos bem separados"),
               ("index", "Pinça: ponta do POLEGAR na ponta do INDICADOR — segure, com os outros dedos ESTICADOS"),
               ("middle", "Pinça: ponta do POLEGAR na ponta do MÉDIO — segure, com os outros dedos ESTICADOS"))
PINCH_PREP_S, PINCH_REC_S, PINCH_MIN_FRAMES = 1.8, 3.0, 8
HEAD_PROMPTS = ("Vire a cabeça devagar para os lados", "Incline a cabeça para cima e para baixo",
                "Aproxime e afaste um pouco o rosto", "Faça um pequeno círculo com a cabeça")


def pose_is_new(pose: np.ndarray, kept: list[np.ndarray], min_dist: float = HEAD_MIN_DIST) -> bool:
    """A head pose is worth keeping only if it differs enough from every pose already kept."""
    return not kept or min(float(np.linalg.norm(pose - q)) for q in kept) >= min_dist


def grid_points(n_points: int, margin: float = 0.06) -> list[tuple[float, float]]:
    """Serpentine grid of normalised points (short saccades between consecutive targets)."""
    side = GRID_SIDE.get(n_points, 4)
    axis = np.linspace(margin, 1.0 - margin, side)
    pts: list[tuple[float, float]] = []
    for r, y in enumerate(axis):
        row = axis if r % 2 == 0 else axis[::-1]
        pts += [(float(x), float(y)) for x in row]
    return pts


def random_points(n: int, rng: np.random.Generator, margin: float = 0.1) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    while len(pts) < n:
        p = rng.uniform(margin, 1.0 - margin, 2)
        if all(np.hypot(*(p - q)) > 0.25 for q in pts):
            pts.append(p)
    return [(float(x), float(y)) for x, y in pts]


class CalibrationScreen:
    def __init__(self, root: tk.Tk, tracker, model: GazeModel, cfg: Config, on_close: Callable[[bool], None]):
        self.root, self.tracker, self.model, self.cfg, self.on_close = root, tracker, model, cfg, on_close
        self.top = tk.Toplevel(root)
        self.top.configure(bg=BG)
        self.top.attributes("-fullscreen", True)
        self.top.attributes("-topmost", True)
        self.W, self.H = self.top.winfo_screenwidth(), self.top.winfo_screenheight()
        self.cv = tk.Canvas(self.top, width=self.W, height=self.H, bg=BG, highlightthickness=0)
        self.cv.pack(fill="both", expand=True)
        self.top.bind("<Key>", self._on_key)
        self.cv.bind("<Button-1>", self._on_click)
        self.top.protocol("WM_DELETE_WINDOW", lambda: self._close(False))
        self._closed = False
        self.top.update_idletasks()
        self._focus()
        self.top.after(200, self._focus)   # once more after the window is really mapped

        self.snapshot = model.snapshot()
        self.rng = np.random.default_rng()
        self.append_mode = model.ready
        self.phase = "intro"
        self.msg = ""
        self.msg_color = FG
        self.mode = "calib"          # "calib" | "test"
        self.targets: list[tuple[float, float]] = []
        self.idx = 0
        self.sub = "settle"
        self.t0 = 0.0
        self.attempt = 0
        self.buf: list[np.ndarray] = []
        self.last_frame_t = 0.0
        self.test_results: list[tuple[tuple[float, float], tuple[float, float], float]] = []
        self.markers: list[tuple[int, int, float]] = []
        self.pending_fit = False
        self.stage = "still"         # "still" | "head" | "test"
        self.buf_poses: list[np.ndarray] = []
        self.buf_vis: list[tuple[float, float]] = []
        self.note = ""
        self.pinch_step, self.pinch_t0 = 0, 0.0
        self.pinch_buf: list[tuple[float, float, float, float]] = []
        self.pinch_vals: dict[str, np.ndarray] = {}
        self.ref = np.array([0.0, 0.08, 0.0, 0.0, 0.0, -45.0])   # yaw, pitch, roll, tx, ty, tz of the user's start pose
        self.ref_frozen = False
        self.head_t0 = 0.0
        self._photo: tk.PhotoImage | None = None
        self._photo_src: bytes | None = None
        self._closed = False
        self._image = None

        tracker.control_suspended = True
        tracker.want_preview = True
        self._tick()

    # ------------------------------------------------------------- lifecycle
    def _focus(self) -> None:
        if self._closed:
            return
        mouse.focus_window(self.top)
        self.top.focus_force()
        self.cv.focus_set()

    def _close(self, saved: bool) -> None:
        if self._closed:
            return
        self._closed = True
        self.tracker.control_suspended = False
        self.tracker.want_preview = False
        if saved and self.model.ready:
            self.model.save(CALIBRATION_PATH)
        self.top.destroy()
        self.on_close(saved)

    def _tick(self) -> None:
        if self._closed:
            return
        st = self.tracker.latest()
        if self.phase == "intro" and st is not None and st.feat is not None:
            self.ref = 0.8 * self.ref + 0.2 * st.feat[5:11]
        self.cv.delete("all")
        getattr(self, f"_phase_{self.phase}")(time.perf_counter(), st)
        self.top.after(16, self._tick)

    # --------------------------------------------------------------- helpers
    def _text(self, x, y, text, size=14, color=FG, anchor="center", bold=False, width=0):
        self.cv.create_text(x, y, text=text, fill=color, anchor=anchor, width=width, justify="center",
                            font=(FONT, size, "bold" if bold else "normal"))

    def _px(self, p: tuple[float, float]) -> tuple[float, float]:
        return p[0] * self.W, p[1] * self.H

    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        self.top.config(cursor="none" if phase in ("countdown", "run") else "")
        self.tracker.want_preview = phase == "intro"

    # ------------------------------------------------------------------ keys
    def _on_key(self, e: tk.Event) -> None:
        key = e.keysym.lower()
        if key == "escape":
            if self.phase == "pinch_cal":
                self._set_phase("intro")
                return
            if self.phase in ("countdown", "run", "head_intro"):
                self.model.restore(self.snapshot)
                self.msg = ""
                self._set_phase("intro")
            elif self.phase == "intro":
                self._close(False)
            else:
                self._close(True)
            return
        if self.phase == "intro":
            if key in PRESETS:
                self.cfg.calibration_points = PRESETS[key]
            elif key == "a":
                self.append_mode = not self.append_mode
            elif key == "h":
                self.cfg.calibration_head = not self.cfg.calibration_head
            elif key == "r":
                self.model.clear()
                self.append_mode = False
            elif key == "i":
                self.open_image_screen()
            elif key == "p":
                self._start_pinch_cal()
            elif key == "v" and self.model.ready:
                self.msg, self.msg_color = "Olhe ao redor: a bolha mostra para onde você está olhando.", FG
                self._set_phase("refine")
            elif key in ("return", "kp_enter", "space"):
                self._start_calibration()
        elif self.phase == "head_intro":
            if key in ("return", "kp_enter", "space"):
                self._begin_head()
        elif self.phase == "refine":
            if key in ("return", "kp_enter"):
                self.model.fit(cv=True)
                self._close(True)
            elif key == "t":
                self._start_test()
            elif key == "c":
                self.test_results = []
                self._set_phase("intro")

    def open_image_screen(self) -> None:
        from .image_ui import ImageScreen  # local import: image_ui imports colours from this module

        if self._image is None:
            self._image = ImageScreen(self.root, self.tracker, self.cfg, self._image_closed)

    def _image_closed(self) -> None:
        self._image = None
        if not self._closed:
            self._focus()

    def _on_click(self, e: tk.Event) -> None:
        """Refinement: the user looks at a spot and clicks on it; that click becomes a training sample."""
        if self.phase != "refine":
            return
        feats = self.tracker.recent_features(0.3)
        if len(feats) < 3:
            self.msg, self.msg_color = "Rosto não detectado — nada foi aprendido.", WARN
            return
        self.model.add_samples(np.median(np.stack(feats), axis=0), (e.x / self.W, e.y / self.H))
        self.model.fit(cv=False)
        self.markers.append((e.x, e.y, time.perf_counter()))
        self.msg, self.msg_color = f"Ponto aprendido ({self.model.n_groups} alvos no total).", GOOD

    # ------------------------------------------------------------ 3D head guide
    def _ref_dist(self) -> float:
        return float(np.clip(-self.ref[5], 25.0, 90.0))

    def _base(self) -> tuple[float, float, float]:
        return INTRO_BASE if self.phase in ("intro", "head_intro") else (float(self.ref[0]), float(self.ref[1]), float(self.ref[2]))

    def _head_viewport(self, cx: float, cy: float, size: float, key: str, phase: float, st, ghost: bool = True,
                       caption: str | None = None, caption_size: int = 11) -> None:
        """A small 3D view: the target head (cyan) animated for `key`, and optionally the user's live head (white)."""
        half = size / 2
        self.cv.create_rectangle(cx - half, cy - half, cx + half, cy + half, outline="#30363d", fill=VIEW_BG)
        d0 = self._ref_dist()
        target = head3d.project(head3d.target_pose(key, phase, d0, self._base()), size, d0)
        head3d.draw_on_canvas(self.cv, cx, cy, target, MODEL_COLOR, VIEW_BG, 2)
        if ghost and st is not None and st.feat is not None:
            live = head3d.project(head3d.live_pose(st.feat, self.ref, d0), size, d0)
            head3d.draw_on_canvas(self.cv, cx, cy, live, GHOST_COLOR, VIEW_BG, 1)
        if caption:
            self._text(cx, cy + half + 16, caption, caption_size, DIM)

    def _prompt_state(self, now: float) -> tuple[str, float]:
        """Which head-movement prompt is active and how far through its animation cycle we are."""
        el = max(now - self.head_t0, 0.0) / PROMPT_SECONDS
        return head3d.HEAD_PROMPT_KEYS[int(el) % len(head3d.HEAD_PROMPT_KEYS)], el % 1.0

    # ------------------------------------------------------- pinch calibration
    def _start_pinch_cal(self) -> None:
        self.pinch_step, self.pinch_t0, self.pinch_buf, self.pinch_vals = 0, time.perf_counter(), [], {}
        self.msg = ""
        self._set_phase("pinch_cal")

    def _phase_pinch_cal(self, now: float, st) -> None:
        W = self.W
        key, instruction = PINCH_STEPS[self.pinch_step]
        el = now - self.pinch_t0
        self._text(W / 2, 70, "Calibração da pinça", 28, bold=True)
        self._text(W / 2, 112, f"Passo {self.pinch_step + 1} de {len(PINCH_STEPS)} • ESC volta", 13, DIM)
        self._text(W / 2, 230, instruction, 22, FG if key == "open" else WARN, bold=True, width=W - 200)
        hand_ok = st is not None and st.hand_ok and st.ratios is not None
        live = (f"indicador {st.ratios[0]:.2f}     médio {st.ratios[1]:.2f}   (0 = dedos encostados, 1 = bem separados)"
                if hand_ok else "mão não detectada — mostre a mão para a câmera")
        self._text(W / 2, 320, live, 16, GOOD if hand_ok else BAD)
        if el < PINCH_PREP_S:
            self._text(W / 2, 470, str(int(np.ceil(PINCH_PREP_S - el))), 90, ACCENT, bold=True)
            self._text(W / 2, 560, "prepare-se…", 14, DIM)
            return
        rec = el - PINCH_PREP_S
        if hand_ok and st.t != self.last_frame_t:
            self.last_frame_t = st.t
            self.pinch_buf.append(self._pinch_sample(st))
        self._text(W / 2, 470, "gravando…", 22, GOOD, bold=True)
        bx0, bx1, by = W / 2 - 250, W / 2 + 250, 530
        self.cv.create_rectangle(bx0, by - 8, bx1, by + 8, outline="#30363d")
        self.cv.create_rectangle(bx0, by - 8, bx0 + 500 * min(rec / PINCH_REC_S, 1.0), by + 8, fill=GOOD, outline="")
        if rec >= PINCH_REC_S:
            self._finish_pinch_step(key)

    def _pinch_sample(self, st) -> tuple[float, float, float, float]:
        """(index metric, middle metric, how extended the other fingers are for an index / a middle pinch)."""
        hands, size = self.tracker.last_hands
        curls = (9.9, 9.9)
        if hands:
            hand = min(hands, key=lambda h: min(pinch_metrics(h, *size)))
            curls = (other_fingers_curl(hand, "index"), other_fingers_curl(hand, "middle"))
        return float(st.ratios[0]), float(st.ratios[1]), float(curls[0]), float(curls[1])

    def _finish_pinch_step(self, key: str) -> None:
        if len(self.pinch_buf) < PINCH_MIN_FRAMES:
            self.msg, self.msg_color = "A mão não foi detectada durante esse passo — repita com a mão bem visível.", WARN
            self.pinch_t0, self.pinch_buf = time.perf_counter(), []
            return
        self.pinch_vals[key] = np.array(self.pinch_buf)
        self.pinch_buf = []
        self.pinch_step += 1
        self.msg = ""
        if self.pinch_step < len(PINCH_STEPS):
            self.pinch_t0 = time.perf_counter()
            return
        self._apply_pinch_calibration()
        self._set_phase("intro")

    def _apply_pinch_calibration(self) -> None:
        open_v, index_v, middle_v = (self.pinch_vals[k] for k in ("open", "index", "middle"))
        parts, ok_all = [], True
        for finger, col, pinched in (("index", 0, index_v), ("middle", 1, middle_v)):
            # contact level = a low percentile: the recording also contains the finger closing in, not only the hold
            o, p = float(np.median(open_v[:, col])), float(np.percentile(pinched[:, col], 40))
            name = "indicador" if finger == "index" else "médio"
            th = derive_thresholds(o, p)
            if th is None:
                ok_all = False
                parts.append(f"{name}: não separou (aberta {o:.2f}, pinça {p:.2f})")
                continue
            setattr(self.cfg, f"pinch_on_{finger}", th[0])
            setattr(self.cfg, f"pinch_off_{finger}", th[1])
            parts.append(f"{name}: aberta {o:.2f}, pinça {p:.2f} → dispara < {th[0]:.2f}")
        # fist guard: learn how extended the other fingers are during *this* user's pinches
        if index_v.shape[1] >= 4:
            curls = np.concatenate([index_v[:, 2], middle_v[:, 3]])
            limit = derive_guard_curl(curls)
            self.cfg.pinch_fist_guard = limit is not None
            if limit is not None:
                self.cfg.pinch_guard_curl = limit
            parts.append(f"guarda de punho: ligada (limite {limit:.2f})" if limit is not None
                         else "guarda de punho: desligada (você pinça com os outros dedos dobrados)")
        self.cfg.save()
        self.msg = ("Pinça calibrada — " if ok_all else "Pinça calibrada parcialmente — ") + " | ".join(parts)
        if not ok_all:
            self.msg += ". Refaça (P) com os dedos realmente encostando e a mão bem visível."
        self.msg_color = GOOD if ok_all else WARN

    # ----------------------------------------------------------------- intro
    def _phase_intro(self, now: float, st) -> None:
        W, H = self.W, self.H
        self._text(W / 2, 52, "Calibração do EyeMouse", 26, bold=True)
        self._text(W / 2, 88, "Posicione-se, ajuste a imagem (I), teste a pinça e escolha a precisão.", 13, DIM)

        # camera preview with guidance
        x0, y0 = 60, 130
        if st is not None and st.preview and st.preview is not self._photo_src:
            self._photo = tk.PhotoImage(data=st.preview, format="PPM")
            self._photo_src = st.preview
        if self._photo is not None:
            self.cv.create_image(x0, y0, image=self._photo, anchor="nw")
        else:
            self.cv.create_rectangle(x0, y0, x0 + 480, y0 + 360, outline=DIM)
            self._text(x0 + 240, y0 + 180, "Iniciando câmera…", 13, DIM)

        msg, color = self._face_guidance(st)
        self._text(x0, y0 + 378, msg, 14, color, "w", True)
        hand = "detectada" if st and st.hand_ok else "não detectada (mostre a mão para a câmera)"
        self._text(x0, y0 + 408, f"Mão: {hand}", 12, GOOD if st and st.hand_ok else DIM, "w")
        pinch = {"left": ("indicador + polegar = CLIQUE ESQUERDO ✓", GOOD),
                 "right": ("médio + polegar = CLIQUE DIREITO ✓", ORANGE)}.get(st.pinch if st else None,
                                                                          ("faça a pinça para testar", DIM))
        self._text(x0, y0 + 434, f"Pinça: {pinch[0]}", 12, pinch[1], "w")
        if st and st.ratios:
            on_i, on_m = self.cfg.pinch_thresholds("index")[0], self.cfg.pinch_thresholds("middle")[0]
            self._text(x0, y0 + 460, f"indicador {st.ratios[0]:.2f} (dispara < {on_i:.2f})   "
                                     f"médio {st.ratios[1]:.2f} (dispara < {on_m:.2f})", 11, DIM, "w")
        self._text(x0, y0 + 486, "Não dispara? Aperte P para calibrar a pinça com a sua mão.", 11, DIM, "w")

        # options
        cx = 620
        self._text(cx, 136, "Precisão da calibração", 15, bold=True, anchor="w")
        for i, (key, n) in enumerate(PRESETS.items()):
            sel = n == self.cfg.calibration_points
            y = 176 + i * 40
            secs = SECONDS_HINT[n] + (HEAD_SECONDS if self.cfg.calibration_head else 0)
            self._text(cx, y, f"[{key}]  {PRESET_NAMES[n]} — {n} pontos (~{secs} s)", 15,
                       ACCENT if sel else FG, "w", sel)
        mode = "adicionar às amostras existentes" if self.append_mode else "substituir a calibração anterior"
        self._text(cx, 306, f"[A]  Modo: {mode}", 14, FG, "w")
        head = "LIGADO (recomendado)" if self.cfg.calibration_head else "desligado — mantenha a cabeça parada no uso"
        self._text(cx, 336, f"[H]  Fase de movimento da cabeça: {head}", 14,
                   GOOD if self.cfg.calibration_head else WARN, "w")
        info = (f"Calibração atual: {self.model.n_samples} amostras, {self.model.n_groups} alvos"
                + (f", erro estimado ≈ {self.model.cv_px:.0f} px" if self.model.cv_px else "")
                if self.model.ready else "Nenhuma calibração salva ainda.")
        self._text(cx, 368, info, 13, DIM, "w")

        tips = ["Dicas para uma boa precisão:",
                "•  Luz na frente do rosto (não atrás)",
                "•  Câmera na altura dos olhos",
                "•  Fase 1: cabeça parada, como no modelo",
                "•  Fase 2: mexa a cabeça devagar,",
                "    com os olhos sempre no ponto",
                "•  Olhe o centro de cada ponto até ficar verde",
                "•  A câmera precisa ver sua mão (pinça)"]
        for i, line in enumerate(tips):
            self._text(cx, 414 + i * 25, line, 13, FG if i == 0 else DIM, "w", i == 0)
        vs = 220
        self._head_viewport(1140, 500, vs, "still", 0.0, st, caption="ciano = posição ideal • branco = sua cabeça", caption_size=10)
        self._text(1140, 500 - vs / 2 - 16, "Posição inicial da cabeça", 12, FG, bold=True)

        keys = "ENTER iniciar  •  1/2/3 precisão  •  A modo  •  H cabeça  •  I imagem  •  P pinça  •  R apagar  •  ESC sair"
        if self.model.ready:
            keys += "   •   V refinar"
        self._text(W / 2, H - 40, keys, 13, DIM)
        if self.msg:
            self._text(W / 2, H - 78, self.msg, 13, self.msg_color, width=W - 160)

    def _face_guidance(self, st) -> tuple[str, str]:
        if st is None or not st.face_ok or st.face_box is None:
            return "Nenhum rosto detectado — fique de frente para a câmera", BAD
        cx, cy, wf = st.face_box
        if st.brightness < 30:
            return "Imagem quase preta — aperte I e depois R para restaurar os ajustes da câmera", BAD
        if st.brightness < 55:
            return "Pouca luz — acenda uma luz na sua frente", WARN
        if wf < 0.20:
            return "Aproxime-se da câmera", WARN
        if wf > 0.45:
            return "Afaste-se um pouco da câmera", WARN
        if abs(cx - 0.5) > 0.18 or abs(cy - 0.45) > 0.2:
            return "Centralize o rosto na imagem", WARN
        return "Posição ótima ✓", GOOD

    # ------------------------------------------------------------ calibration
    def _start_calibration(self) -> None:
        st = self.tracker.latest()
        if st is None or not st.face_ok:
            self.msg, self.msg_color = "Rosto não detectado.", BAD
            return
        self.snapshot = self.model.snapshot()
        if not self.append_mode:
            self.model.clear()
        self.mode, self.stage, self.note = "calib", "still", ""
        feats = self.tracker.recent_features(0.6)
        if feats:
            self.ref = np.median(np.stack(feats), axis=0)[5:11]
        self.targets = grid_points(self.cfg.calibration_points)
        self.t0 = time.perf_counter()
        self._set_phase("countdown")

    def _phase_countdown(self, now: float, st) -> None:
        left = 3.0 - (now - self.t0)
        if left <= 0:
            self._begin_targets()
            return
        self._text(self.W / 2, self.H / 2 - 20, str(int(np.ceil(left))), 96, ACCENT, bold=True)
        self._text(self.W / 2, self.H / 2 + 70, "Fase 1: olhe para o centro de cada círculo e mantenha a cabeça parada", 16)
        self._text(self.W / 2, self.H / 2 + 104, "ESC cancela", 12, DIM)

    def _begin_targets(self) -> None:
        self.idx = 0
        self._begin_point()
        self._set_phase("run")

    def _begin_point(self) -> None:
        self.sub, self.t0, self.attempt, self.buf = "settle", time.perf_counter(), 0, []
        self.buf_poses, self.buf_vis = [], []

    def _phase_run(self, now: float, st) -> None:
        tx, ty = self._px(self.targets[self.idx])
        n = len(self.targets)
        el = now - self.t0
        head = self.stage == "head"
        label = "Teste" if self.mode == "test" else ("Fase 2 · Ponto" if head else "Ponto")
        self._text(30, 28, f"{label} {self.idx + 1}/{n}", 14, DIM, "w")
        if st is None or not st.face_ok:
            self._text(self.W / 2, 28, "⚠ rosto não detectado", 14, BAD)
        self.cv.create_rectangle(0, self.H - 6, self.W * (self.idx / n), self.H, fill=ACCENT, outline="")
        if head:
            self._draw_head_guidance(now, st)
        elif self.mode == "calib":
            self._head_viewport(self.W - 120, self.H - 150, 170, "still", 0.0, st,
                                caption="cabeça parada (ciano = ideal)", caption_size=10)

        needed = HEAD_FRAMES_NEEDED if head else FRAMES_NEEDED
        minimum = HEAD_FRAMES_MIN if head else FRAMES_MIN
        max_s = HEAD_CAPTURE_MAX_S if head else CAPTURE_MAX_S
        if self.sub == "settle":
            r = 40 - 26 * min(el / SETTLE_S, 1.0)
            self.cv.create_oval(tx - r, ty - r, tx + r, ty + r, outline=FG, width=3)
            self.cv.create_oval(tx - 5, ty - 5, tx + 5, ty + 5, fill=BAD, outline="")
            if el >= SETTLE_S:
                self.sub, self.t0, self.buf = "capture", now, []
                self.buf_poses, self.buf_vis = [], []
        elif self.sub == "capture":
            (self._collect_head if head else self._collect)(st)
            if head and el > 2.0 and len(self.buf) < 5:
                self._text(self.W / 2, self.H - 90, "Mova a cabeça um pouco mais (devagar, olhos no círculo)", 16, BAD, bold=True)
            p = min(len(self.buf) / needed, 1.0)
            self.cv.create_oval(tx - 14, ty - 14, tx + 14, ty + 14, outline=DIM, width=2)
            if p > 0:
                self.cv.create_arc(tx - 14, ty - 14, tx + 14, ty + 14, start=90, extent=-359.9 * p,
                                   style="arc", outline=GOOD, width=5)
            self.cv.create_oval(tx - 5, ty - 5, tx + 5, ty + 5, fill=BAD if p < 1 else GOOD, outline="")
            if len(self.buf) >= needed:
                self._finish_point(tx, ty)
            elif el > max_s:
                if len(self.buf) >= minimum:
                    self._finish_point(tx, ty)
                elif self.attempt + 1 < (2 if head else MAX_ATTEMPTS):
                    self.sub, self.t0, self.buf, self.attempt = "settle", now, [], self.attempt + 1
                    self.buf_poses, self.buf_vis = [], []
                else:
                    self._next_point()
        else:  # flash
            self.cv.create_oval(tx - 18, ty - 18, tx + 18, ty + 18, fill=GOOD, outline="")
            if el > 0.2:
                self._next_point()

    def _draw_head_guidance(self, now: float, st) -> None:
        """Rotating instruction + a small pad that shows which head poses have already been captured."""
        W, H = self.W, self.H
        key, phase = self._prompt_state(now)
        prompt = head3d.TARGET_POSES[key][0]
        self._text(W / 2, H - 56, prompt, 18, WARN, bold=True)
        self._text(W / 2, H - 30, "mantenha os olhos no círculo", 12, DIM)
        size, cx, cy = 130, W - 90, H - 120
        self._head_viewport(W - 300, H - 130, 170, key, phase, st, caption="ciano = movimento a imitar", caption_size=10)
        self.cv.create_rectangle(cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2, outline="#30363d", width=2)
        self.cv.create_line(cx - size / 2, cy, cx + size / 2, cy, fill="#21262d")
        self.cv.create_line(cx, cy - size / 2, cx, cy + size / 2, fill="#21262d")
        self._text(cx, cy - size / 2 - 14, "poses da cabeça", 10, DIM)

        def to_pad(v):
            return (cx + max(-1.0, min(1.0, v[0] / 0.22)) * size / 2, cy + max(-1.0, min(1.0, v[1] / 0.22)) * size / 2)

        for v in self.buf_vis:
            x, y = to_pad(v)
            self.cv.create_oval(x - 3, y - 3, x + 3, y + 3, fill=GOOD, outline="")
        if st is not None and st.head_vis is not None:
            x, y = to_pad(st.head_vis)
            self.cv.create_oval(x - 6, y - 6, x + 6, y + 6, outline=FG, width=2)

    def _collect_head(self, st) -> None:
        """Keep frames whose head pose differs enough from those already kept (pose variety, not just quantity)."""
        if st is None or st.feat is None or st.eyes_closed or st.blink_score > 0.35 or st.t == self.last_frame_t:
            return
        self.last_frame_t = st.t
        pose = st.feat[list(POSE_IDX)] * POSE_SCALE
        if abs(st.feat[5]) > 0.7 or abs(st.feat[6]) > 0.6:
            return  # head turned too far for the eyes to stay visible
        if not pose_is_new(pose, self.buf_poses):
            return
        self.buf.append(st.feat)
        self.buf_poses.append(pose)
        self.buf_vis.append(st.head_vis or (0.0, 0.0))

    def _collect(self, st) -> None:
        if st is None or st.feat is None or st.eyes_closed or st.blink_score > 0.35 or st.t == self.last_frame_t:
            return
        self.last_frame_t = st.t
        self.buf.append(st.feat)

    def _finish_point(self, tx: float, ty: float) -> None:
        feats = np.stack(self.buf) if self.stage == "head" else reject_outliers(np.stack(self.buf))
        target = (tx / self.W, ty / self.H)
        if self.mode == "test":
            preds = [self.model.predict_px(f) for f in feats]
            pred = np.median(np.stack(preds), axis=0)
            self.test_results.append(((tx, ty), (float(pred[0]), float(pred[1])), float(np.hypot(pred[0] - tx, pred[1] - ty))))
        self.model.add_samples(feats, target)
        if self.mode == "test":
            self.model.fit(cv=False)  # learn immediately: the next target benefits from this one
        self.sub, self.t0 = "flash", time.perf_counter()

    def _next_point(self) -> None:
        self.idx += 1
        if self.idx >= len(self.targets):
            if self.mode == "calib" and self.stage == "still" and self.cfg.calibration_head:
                self.t0 = time.perf_counter()
                self._set_phase("head_intro")
            else:
                self._set_phase("fit")
                self.pending_fit = True
        else:
            self._begin_point()

    # -------------------------------------------------------- head-movement phase
    def _phase_head_intro(self, now: float, st) -> None:
        W, H = self.W, self.H
        left = 8.0 - (now - self.t0)
        if left <= 0:
            self._begin_head()
            return
        self._text(W / 2, 64, "Fase 2 de 2 — movimento da cabeça", 28, bold=True)
        self._text(W / 2, 108, "Fase 1 concluída. Agora, em cada círculo, mantenha os olhos no centro e mova a cabeça devagar.", 15)
        keys = head3d.HEAD_PROMPT_KEYS
        size, gap = 230, 40
        x0 = W / 2 - (len(keys) * size + (len(keys) - 1) * gap) / 2 + size / 2
        cyv = 300
        phase = (now - self.t0) / PROMPT_SECONDS % 1.0
        for i, key in enumerate(keys):
            cxv = x0 + i * (size + gap)
            self._head_viewport(cxv, cyv, size, key, phase, st, ghost=False)   # pure demonstration
            self._text(cxv, cyv + size / 2 + 34, head3d.TARGET_POSES[key][0], 12, WARN, bold=True, width=size)
        self._text(W / 2, 486, "Durante a captura, o modelo em ciano mostra o movimento a imitar e a linha branca mostra a sua cabeça ao vivo.", 13, DIM)
        self._text(W / 2, 520, "Assim o programa aprende a compensar a cabeça e continua acertando quando você se mexer.", 14, DIM)
        self._text(W / 2, 552, "Movimentos pequenos e suaves (alguns graus), sem tirar os olhos do círculo.", 14, DIM)
        self._text(W / 2, 640, str(int(np.ceil(left))), 56, ACCENT, bold=True)
        self._text(W / 2, H - 40, "ENTER começa agora   •   ESC cancela", 13, DIM)

    def _begin_head(self) -> None:
        self.stage = "head"
        self.head_t0 = time.perf_counter()
        self.targets = grid_points(HEAD_POINTS, margin=0.08)
        self._begin_targets()

    # ---------------------------------------------------------------- fitting
    def _phase_fit(self, now: float, st) -> None:
        self._text(self.W / 2, self.H / 2, "Calculando modelo…", 22)
        if self.mode == "test":
            self._finish_test()
            return
        if self.pending_fit:
            self.pending_fit = False
            return  # draw one frame with the message before the (short) blocking fit
        report = self.model.fit(cv=True)
        if report is None:
            self.msg, self.msg_color = "Poucos pontos válidos — tente de novo com melhor iluminação.", BAD
            self.model.restore(self.snapshot)
            self._set_phase("intro")
            return
        self.model.save(CALIBRATION_PATH)
        self.msg = (f"Calibração concluída — erro estimado ≈ {report.cv_px:.0f} px ({report.cv_frac * 100:.1f}% da tela).")
        self.msg_color = GOOD
        if self.model.head_spread() < HEAD_SPREAD_MIN:
            self.note = ("Sem movimento da cabeça nas amostras: a precisão cai se você mexer a cabeça. "
                         "Refaça com a fase 2 (tecla H na tela inicial).")
        else:
            self.note = "Movimento da cabeça incluído na calibração ✓"
        self.tracker.control_suspended = True
        self._set_phase("refine")

    # ----------------------------------------------------------------- refine
    def _phase_refine(self, now: float, st) -> None:
        W, H = self.W, self.H
        self._text(W / 2, 40, self.msg or "Refinamento", 16, self.msg_color, bold=True)
        cv_txt = f"erro estimado ≈ {self.model.cv_px:.0f} px" if self.model.cv_px else ""
        self._text(W / 2, 70, f"{self.model.n_samples} amostras • {self.model.n_groups} alvos • {cv_txt}", 12, DIM)
        if self.note:
            self._text(W / 2, 96, self.note, 12, GOOD if self.note.endswith("✓") else WARN)
        self._text(W / 2, H / 2 - 30, "Olhe para qualquer ponto: a bolha mostra para onde você está olhando.", 16)
        self._text(W / 2, H / 2 + 34, "Mova a cabeça um pouco: a bolha deve continuar onde você olha.", 13, DIM)
        self._text(W / 2, H / 2 + 4, "Se estiver errada, olhe fixamente para o ponto certo e CLIQUE nele com o mouse — o modelo aprende na hora.", 13, DIM)

        for (tx, ty), (px, py), err in self.test_results:
            self.cv.create_line(tx, ty, px, py, fill=WARN, width=2)
            self.cv.create_oval(tx - 5, ty - 5, tx + 5, ty + 5, fill=GOOD, outline="")
            self.cv.create_oval(px - 5, py - 5, px + 5, py + 5, fill=BAD, outline="")
            self._text(tx + 10, ty - 14, f"{err:.0f}px", 10, WARN, "w")
        self.markers = [m for m in self.markers if now - m[2] < 1.2]
        for x, y, t in self.markers:
            r = 10 + 30 * (now - t)
            self.cv.create_oval(x - r, y - r, x + r, y + r, outline=GOOD, width=2)

        if st is not None and st.gaze is not None:
            gx, gy = st.gaze
            color = {"left": GOOD, "right": ORANGE}.get(st.pinch, ACCENT)
            self.cv.create_oval(gx - 36, gy - 36, gx + 36, gy + 36, fill=color, outline="#ffffff", width=3, stipple="gray50")
            self.cv.create_oval(gx - 4, gy - 4, gx + 4, gy + 4, fill="#ffffff", outline="")
        elif st is None or not st.face_ok:
            self._text(W / 2, H / 2 + 60, "⚠ rosto não detectado", 14, BAD)

        keys = "ENTER salvar e sair   •   T teste guiado (aprende a cada ponto)   •   C nova calibração   •   ESC salvar e sair"
        self._text(W / 2, H - 40, keys, 13, DIM)

    def _start_test(self) -> None:
        if not self.model.ready:
            return
        self.mode, self.stage = "test", "test"
        self.test_results = []
        self.targets = random_points(8, self.rng)
        self._begin_targets()

    def _finish_test(self) -> None:
        errs = [e for _, _, e in self.test_results]
        if errs:
            self.msg = (f"Teste: erro médio {np.mean(errs):.0f} px (mediana {np.median(errs):.0f} px) — "
                        "o modelo já foi refinado com esses pontos.")
            self.msg_color = GOOD if np.mean(errs) < 0.08 * self.W else WARN
        self.model.fit(cv=True)
        self.model.save(CALIBRATION_PATH)
        self._set_phase("refine")
