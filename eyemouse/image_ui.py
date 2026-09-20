"""Fullscreen image-adjustment screen: make the eyes as contrasted as possible for the tracker.

Left: live camera preview (after adjustments) and a zoomed strip of both eyes.
Right, two tabs:
  * Software - brightness / contrast / gamma / local contrast (CLAHE) / sharpness applied before detection.
  * Câmera   - the driver's own settings (exposure, gain, saturation...), read from and written to the device.
Plus auto-adjust from the eye region and a button for the driver's native dialog ("Video Proc Amp").
"""
from __future__ import annotations

import tkinter as tk
from typing import Callable

from . import imaging, mouse
from .calibration_ui import ACCENT, BAD, BG, DIM, FG, FONT, GOOD, WARN
from .camera_props import CAMERA_DEFAULTS, CAMERA_PROPS
from .config import Config

PANEL = "#161b22"
# (label, config attribute, min, max, resolution, hint)
SOFTWARE_SLIDERS = [
    ("Brilho", "img_brightness", -100, 100, 1, "clareia/escurece tudo"),
    ("Contraste", "img_contrast", 0.5, 3.0, 0.05, "separa íris e esclera"),
    ("Gama", "img_gamma", 0.5, 2.5, 0.05, ">1 clareia só os tons médios (olhos escuros)"),
    ("Contraste local (CLAHE)", "img_clahe", 0.0, 1.0, 0.05, "realça detalhes do olho sem estourar o resto"),
    ("Nitidez", "img_sharpen", 0.0, 2.0, 0.1, "borda da íris mais definida"),
]
DEFAULTS = {"img_brightness": 0, "img_contrast": 1.0, "img_gamma": 1.0, "img_clahe": 0.0, "img_sharpen": 0.0}


class ImageScreen:
    def __init__(self, root: tk.Tk, tracker, cfg: Config, on_close: Callable[[], None]):
        self.tracker, self.cfg, self.on_close = tracker, cfg, on_close
        self.top = tk.Toplevel(root)
        self.top.configure(bg=BG)
        self.top.attributes("-fullscreen", True)
        self.top.attributes("-topmost", True)
        self.W, self.H = self.top.winfo_screenwidth(), self.top.winfo_screenheight()
        self.cv = tk.Canvas(self.top, width=self.W, height=self.H, bg=BG, highlightthickness=0)
        self.cv.pack(fill="both", expand=True)
        self.top.bind("<Key>", self._on_key)
        self.top.protocol("WM_DELETE_WINDOW", self.close)

        self._prev_want, self._prev_mode = tracker.want_preview, tracker.preview_mode
        tracker.want_preview, tracker.preview_mode = True, "image"
        self._closed = False
        self._photo = self._eye_photo = None
        self._photo_src = self._eye_src = None
        self._soft_vars: dict[str, tk.DoubleVar] = {}
        self._cam_vars: dict[str, tk.DoubleVar] = {}
        self._cam_scales: dict[str, tk.Scale] = {}
        self._cam_loaded: set[str] = set()
        self._loading = False       # True while sliders are filled from the device (must not count as user edits)

        self._build_static()
        self.top.update_idletasks()
        self._focus()
        self.top.after(200, self._focus)
        self._tick()

    def _focus(self) -> None:
        if not self._closed:
            mouse.focus_window(self.top)
            self.top.focus_force()

    # ------------------------------------------------------------------ build
    def _build_static(self) -> None:
        W, H = self.W, self.H
        self.cv.create_text(W / 2, 40, text="Imagem e contraste dos olhos", fill=FG, font=(FONT, 24, "bold"))
        self.cv.create_text(W / 2, 74, text="Ajuste até os olhos aparecerem nítidos e com bom contraste na faixa ampliada.",
                            fill=DIM, font=(FONT, 12))
        self.cv.create_text(60, 388 + 40, text="Olhos (ampliado)", fill=DIM, anchor="sw", font=(FONT, 10))

        panel = tk.Frame(self.top, bg=PANEL, padx=18, pady=12)
        self.cv.create_window(600, 110, window=panel, anchor="nw")

        style = dict(bg="#21262d", fg=FG, activebackground=ACCENT, activeforeground="#fff", bd=0, padx=12, pady=7,
                     font=(FONT, 10), cursor="hand2")
        tabs = tk.Frame(panel, bg=PANEL)
        tabs.pack(fill="x", pady=(0, 8))
        self.tab_btns = {
            "soft": tk.Button(tabs, text="Software (pós-processamento)  [1]", command=lambda: self.show_tab("soft"), **style),
            "cam": tk.Button(tabs, text="Câmera (driver)  [2]", command=lambda: self.show_tab("cam"), **style),
        }
        for b in self.tab_btns.values():
            b.pack(side="left", padx=(0, 6))

        self.hint = tk.StringVar(value=" ")
        rows = max(len(SOFTWARE_SLIDERS), len(CAMERA_PROPS))
        # both tabs share the same slot; size it for the taller one
        self.slot = tk.Frame(panel, bg=PANEL, height=rows * 32 + 6)
        self.slot.pack(fill="x")
        self.slot.pack_propagate(False)
        self.frames = {"soft": tk.Frame(self.slot, bg=PANEL), "cam": tk.Frame(self.slot, bg=PANEL)}
        for i, (label, attr, lo, hi, res, hint) in enumerate(SOFTWARE_SLIDERS):
            var = tk.DoubleVar(value=getattr(self.cfg, attr))
            self._soft_vars[attr] = var
            self._slider_row(self.frames["soft"], i, label, hint, lo, hi, res, var,
                             lambda v, a=attr: self._set_soft(a, float(v)))
        for i, (key, (_, label, hint, lo, hi, res)) in enumerate(CAMERA_PROPS.items()):
            var = tk.DoubleVar(value=self.cfg.camera_props.get(key, lo))
            self._cam_vars[key] = var
            self._cam_scales[key] = self._slider_row(self.frames["cam"], i, label, hint, lo, hi, res, var,
                                                     lambda v, k=key: self._set_cam(k, float(v)))

        tk.Label(panel, textvariable=self.hint, bg=PANEL, fg=DIM, font=(FONT, 9), anchor="w").pack(fill="x", pady=(2, 6))
        btns = tk.Frame(panel, bg=PANEL)
        btns.pack(fill="x")
        tk.Button(btns, text="Auto-ajustar pelos olhos  [A]", command=self.auto, **style).pack(side="left", padx=(0, 6))
        tk.Button(btns, text="Restaurar tudo  [R]", command=self.reset, **style).pack(side="left", padx=6)
        tk.Button(btns, text="Painel nativo…  [P]", command=self.tracker.open_camera_dialog, **style).pack(side="left", padx=6)
        self.orig_var = tk.BooleanVar(value=False)
        tk.Checkbutton(panel, text="Mostrar imagem original (comparar)  [O]", variable=self.orig_var, bg=PANEL, fg=FG,
                       selectcolor="#0b0f14", activebackground=PANEL, activeforeground=FG, font=(FONT, 10),
                       command=self._toggle_original).pack(anchor="w", pady=(8, 0))

        panel.update_idletasks()
        self._qy = 110 + panel.winfo_reqheight() + 26
        tk.Button(self.top, text="Concluir  [ENTER]", command=self.close, **style).place(x=W - 190, y=H - 62)
        self.cv.create_text(60, H - 14, anchor="w", fill=DIM, font=(FONT, 10),
                            text="Dica: de costas para a janela? Use 'Comp. luz de fundo' e o Ganho; com pouca luz, "
                                 "suba Gama/CLAHE em vez de aumentar a exposição.")
        self.show_tab("soft")

    def _slider_row(self, parent: tk.Frame, row: int, label: str, hint: str, lo, hi, res, var, command) -> tk.Scale:
        name = tk.Label(parent, text=label, bg=PANEL, fg=FG, font=(FONT, 11, "bold"), width=22, anchor="w")
        name.grid(row=row, column=0, sticky="w", pady=1)
        scale = tk.Scale(parent, from_=lo, to=hi, resolution=res, orient="horizontal", length=400, variable=var,
                         showvalue=True, bg=PANEL, fg=FG, troughcolor="#0b0f14", highlightthickness=0, bd=0,
                         activebackground=ACCENT, command=command, sliderlength=22, width=12)
        scale.grid(row=row, column=1, sticky="e", padx=(10, 0))
        for w in (name, scale):
            w.bind("<Enter>", lambda e, t=f"{label}: {hint}": self.hint.set(t))
            w.bind("<Leave>", lambda e: self.hint.set(" "))
        return scale

    def show_tab(self, tab: str) -> None:
        self.tab = tab
        for frame in self.frames.values():
            frame.place_forget()
        self.frames[tab].place(x=0, y=0, relwidth=1.0)
        for name, btn in self.tab_btns.items():
            btn.configure(bg=ACCENT if name == tab else "#21262d")

    # ---------------------------------------------------------------- actions
    def _set_soft(self, attr: str, value: float) -> None:
        is_int = isinstance(getattr(self.cfg, attr), int)
        setattr(self.cfg, attr, int(round(value)) if is_int else round(value, 3))

    def _set_cam(self, key: str, value: float) -> None:
        if self._loading:
            return
        self.cfg.camera_props[key] = value
        self.tracker.set_camera_prop(CAMERA_PROPS[key][0], value)

    def _load_camera_values(self) -> None:
        """Fill the driver sliders with the device's current values (and widen their range to fit)."""
        for key, value in dict(self.tracker.camera_props_now).items():
            if key in self._cam_loaded or key not in self._cam_vars or value != value or value <= -1e6:
                continue
            shown = self.cfg.camera_props.get(key, value)
            _, _, _, lo, hi, _ = CAMERA_PROPS[key]
            self._loading = True
            try:
                self._cam_scales[key].configure(from_=min(lo, value, shown), to=max(hi, value, shown))
                self._cam_vars[key].set(shown)
            finally:
                self._loading = False
            self._cam_loaded.add(key)

    def _toggle_original(self) -> None:
        self.tracker.show_original = self.orig_var.get()

    def auto(self) -> None:
        params = self.tracker.auto_adjust()
        if params is None:
            return
        brightness, contrast, gamma = params
        for attr, value in (("img_brightness", brightness), ("img_contrast", contrast), ("img_gamma", gamma)):
            self._soft_vars[attr].set(value)
            self._set_soft(attr, value)
        self.show_tab("soft")

    def reset(self) -> None:
        """Back to the untouched state: no software adjustments and the camera driver's original settings."""
        for attr, value in DEFAULTS.items():
            self._soft_vars[attr].set(value)
            self._set_soft(attr, value)
        self.tracker.restore_camera_defaults()
        self._loading = True
        try:
            for key, value in (self.cfg.camera_original or CAMERA_DEFAULTS).items():
                if key in self._cam_vars:
                    _, _, _, lo, hi, _ = CAMERA_PROPS[key]
                    self._cam_scales[key].configure(from_=min(lo, value), to=max(hi, value))
                    self._cam_vars[key].set(value)
        finally:
            self._loading = False

    def _on_key(self, e: tk.Event) -> None:
        key = e.keysym.lower()
        if key in ("escape", "return", "kp_enter"):
            self.close()
        elif key == "a":
            self.auto()
        elif key == "r":
            self.reset()
        elif key == "p":
            self.tracker.open_camera_dialog()
        elif key == "1":
            self.show_tab("soft")
        elif key == "2":
            self.show_tab("cam")
        elif key == "o":
            self.orig_var.set(not self.orig_var.get())
            self._toggle_original()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.tracker.want_preview, self.tracker.preview_mode = self._prev_want, self._prev_mode
        self.tracker.show_original = False
        self.cfg.save()
        self.top.destroy()
        self.on_close()

    # ------------------------------------------------------------------- loop
    def _tick(self) -> None:
        if self._closed:
            return
        self._load_camera_values()
        self._draw(self.tracker.latest())
        self.top.after(33, self._tick)

    def _draw(self, st) -> None:
        cv = self.cv
        cv.delete("dyn")
        x0, y0 = 60, 110
        if st is not None and st.preview and st.preview is not self._photo_src:
            self._photo, self._photo_src = tk.PhotoImage(data=st.preview, format="PPM"), st.preview
        if self._photo is not None:
            cv.create_image(x0, y0, image=self._photo, anchor="nw", tags="dyn")
        else:
            cv.create_text(x0 + 240, y0 + 180, text="Iniciando câmera…", fill=DIM, font=(FONT, 12), tags="dyn")
        label = "ORIGINAL" if self.orig_var.get() else "AJUSTADA (é isto que o rastreador vê)"
        cv.create_text(x0, y0 - 14, text=label, fill=WARN if self.orig_var.get() else GOOD, anchor="sw",
                       font=(FONT, 10, "bold"), tags="dyn")

        ey = y0 + 360 + 46
        if st is not None and st.eye_preview and st.eye_preview is not self._eye_src:
            self._eye_photo, self._eye_src = tk.PhotoImage(data=st.eye_preview, format="PPM"), st.eye_preview
        if st is not None and st.face_ok and self._eye_photo is not None:
            cv.create_image(x0, ey, image=self._eye_photo, anchor="nw", tags="dyn")
        else:
            cv.create_rectangle(x0, ey, x0 + 480, ey + 130, outline=DIM, tags="dyn")
            cv.create_text(x0 + 240, ey + 65, text="Rosto não detectado", fill=BAD, font=(FONT, 12), tags="dyn")

        qx, qy = 600, self._qy
        if st is not None and st.eye_stats and st.face_ok:
            mean, spread, clipped = st.eye_stats
            msg, level = imaging.describe_eye_quality(mean, spread, clipped)
            color = {"good": GOOD, "warn": WARN, "bad": BAD}[level]
            cv.create_text(qx, qy, text=msg, fill=color, anchor="w", font=(FONT, 15, "bold"), tags="dyn")
            self._bar(qx, qy + 32, "Luminância nos olhos", mean / 255, (70 / 255, 190 / 255), f"{mean:.0f}")
            self._bar(qx, qy + 58, "Contraste (p5–p95)", spread / 255, (60 / 255, 1.0), f"{spread:.0f}")
            self._bar(qx, qy + 84, "Pixels saturados", clipped / 0.2, (0.0, 0.4), f"{clipped * 100:.1f}%")
        else:
            cv.create_text(qx, qy, text="Olhe para a câmera para medir o contraste dos olhos", fill=DIM,
                           anchor="w", font=(FONT, 13), tags="dyn")
        if st is not None:
            hand = "mão ✓" if st.hand_ok else "mão –"
            low_fps = st.fps < 20
            cv.create_text(qx, qy + 116, anchor="w", fill=WARN if low_fps else DIM,
                           font=(FONT, 10, "bold" if low_fps else "normal"), tags="dyn",
                           text=f"{st.fps:.0f} fps • rosto {'✓' if st.face_ok else '✗'} • {hand}")
            if low_fps:
                cv.create_text(qx, qy + 134, anchor="nw", fill=WARN, font=(FONT, 10), width=540, tags="dyn",
                               text="FPS baixo: com pouca luz a câmera alonga a exposição. Acenda uma luz na frente do rosto "
                                    "ou diminua a Exposição (aba Câmera) e compense com Ganho, Gama e CLAHE.")

    def _bar(self, x: int, y: int, name: str, value: float, good: tuple[float, float], text: str) -> None:
        cv, w = self.cv, 400
        cv.create_text(x, y, text=name, fill=DIM, anchor="w", font=(FONT, 10), tags="dyn")
        bx = x + 170
        cv.create_rectangle(bx, y - 7, bx + w, y + 7, outline="#30363d", tags="dyn")
        cv.create_rectangle(bx + w * good[0], y - 7, bx + w * good[1], y + 7, fill="#12261a", outline="", tags="dyn")
        v = min(max(value, 0.0), 1.0)
        ok = good[0] <= v <= good[1]
        cv.create_rectangle(bx, y - 4, bx + w * v, y + 4, fill=GOOD if ok else WARN, outline="", tags="dyn")
        cv.create_text(bx + w + 10, y, text=text, fill=FG, anchor="w", font=(FONT, 10), tags="dyn")
