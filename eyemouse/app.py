"""Control panel + orchestration of tracker, bubble, calibration, hotkeys."""
from __future__ import annotations

import queue
import tkinter as tk
from tkinter import messagebox, ttk

from . import mouse
from .bubble import Bubble
from .calibration_ui import CalibrationScreen
from .config import CALIBRATION_PATH, HAND_MODES, HEAD_MODES, ICON_PATH, Config
from .gaze_model import GazeModel
from .image_ui import ImageScreen
from .preview_ui import CameraView
from .settings_ui import SettingsWindow
from .tracker import Tracker

HOTKEYS = {"<ctrl>+<alt>+e": "mouse", "<ctrl>+<alt>+b": "bubble", "<ctrl>+<alt>+c": "calibrate", "<ctrl>+<alt>+q": "quit",
           "<ctrl>+<alt>+h": "head_mode", "<ctrl>+<alt>+m": "hand_mode", "<ctrl>+<alt>+r": "recenter"}

HEAD_HELP = {
    "eye": "Olho: o cursor segue o olhar. Mexer a cabeça NÃO move o mouse (a cabeça só ajuda a compensar o erro).",
    "head_eye": "Cabeça + olho: o olhar posiciona o cursor e virar a cabeça o desloca também.",
    "head": "Cabeça: o cursor segue para onde o nariz aponta (Recentralizar define o centro).",
    "off": "Desativado: olho e cabeça não movem o mouse (use o mouse físico ou o modo de mão); os gestos da mão continuam ativos.",
}
HAND_HELP = {
    "pinch": "Mão: pinça polegar+indicador = clique esquerdo, polegar+médio = direito, polegar+anelar = rolagem (mova a mão).",
    "hand": "Mão: a ponta do dedo indicador move o cursor (também durante a pinça, para arrastar); pinça = bolas do polegar e do indicador/médio se tocam; polegar+anelar rola (mova a mão). As duas mãos valem. Sem mão, vale o modo de cabeça.",
}
SOURCE_NAMES = {"eye": "olho", "head_eye": "cabeça + olho", "head": "cabeça", "hand": "mão", "off": "gestos da mão (olho/cabeça desativados)", "none": "—"}


class App:
    def __init__(self, cfg: Config, calibrate_on_start: bool = False):
        mouse.enable_dpi_awareness()
        self.cfg = cfg
        self.model = GazeModel(mouse.screen_size())
        self.model.load(CALIBRATION_PATH)
        self.events: queue.SimpleQueue = queue.SimpleQueue()
        self.actions: queue.SimpleQueue = queue.SimpleQueue()
        self.tracker = Tracker(cfg, self.model, self.events)
        self.tracker.mouse_enabled = cfg.start_with_mouse and self.tracker.mode_ready()
        self.tracker.start()

        self.root = tk.Tk()
        self.root.title("EyeMouse")
        if ICON_PATH.exists():
            self.root.iconbitmap(default=str(ICON_PATH))
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.calibration: CalibrationScreen | None = None
        self._image: ImageScreen | None = None
        self._view: CameraView | None = None
        self._settings: SettingsWindow | None = None
        self._error_shown = False
        self._build_panel()
        self.bubble = Bubble(self.root, cfg, self.tracker)

        self._listeners = [
            mouse.start_hotkeys({k: (lambda a=v: self.actions.put(a)) for k, v in HOTKEYS.items()}),
            mouse.start_click_listener(self.tracker.on_physical_click),
        ]
        self.root.after(50, self._tick)
        if calibrate_on_start or (not self.model.ready and cfg.head_mode not in ("head", "off") and cfg.hand_mode != "hand"):
            self.root.after(800, self.open_calibration)

    # ------------------------------------------------------------------ panel
    def _build_panel(self) -> None:
        f = ttk.Frame(self.root, padding=14)
        f.pack()
        ttk.Label(f, text="EyeMouse", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, columnspan=2, sticky="w")
        self.status = {k: tk.StringVar() for k in ("cam", "calib", "mouse")}
        for i, k in enumerate(self.status, start=1):
            ttk.Label(f, textvariable=self.status[k], wraplength=380, justify="left").grid(row=i, column=0, columnspan=2, sticky="w")

        row = 4
        self.btn_calib = ttk.Button(f, text="Calibrar (tela cheia)", command=self.open_calibration)
        self.btn_image = ttk.Button(f, text="Imagem e contraste dos olhos (tela cheia)", command=self.open_image)
        self.btn_view = ttk.Button(f, text="Ver câmera (visão computacional)", command=self.toggle_view)
        self.btn_mouse = ttk.Button(f, text="", command=self.toggle_mouse)
        for b in (self.btn_calib, self.btn_image, self.btn_view, self.btn_mouse):
            b.grid(row=row, column=0, columnspan=2, sticky="ew", pady=3)
            row += 1

        # ---- control modes: one menu for the head, one for the hand
        modes = ttk.LabelFrame(f, text="Modos de controle", padding=8)
        modes.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 3))
        row += 1
        self.head_var, self.hand_var = tk.StringVar(value=self.cfg.head_mode), tk.StringVar(value=self.cfg.hand_mode)
        self.mb_head = ttk.Menubutton(modes, direction="below")
        self.mb_hand = ttk.Menubutton(modes, direction="below")
        for mb, var, names, cmd in ((self.mb_head, self.head_var, HEAD_MODES, self._on_head_mode),
                                    (self.mb_hand, self.hand_var, HAND_MODES, self._on_hand_mode)):
            menu = tk.Menu(mb, tearoff=0)
            for key, name in names.items():
                menu.add_radiobutton(label=name, variable=var, value=key, command=cmd)
            mb["menu"] = menu
        self.mb_head.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.mb_hand.grid(row=0, column=1, sticky="ew")
        modes.columnconfigure((0, 1), weight=1)
        self.mode_help = tk.StringVar()
        ttk.Label(modes, textvariable=self.mode_help, wraplength=370, justify="left", foreground="#555").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Button(modes, text="Recentralizar cabeça  (Ctrl+Alt+R)", command=self.recenter).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        # ---- sensitivity (eye mode has none: its accuracy comes from the calibration)
        sens = ttk.LabelFrame(f, text="Sensibilidade do movimento", padding=8)
        sens.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 3))
        row += 1
        self._sens_labels: dict[str, ttk.Label] = {}
        for i, (label, attr) in enumerate((("Cabeça", "head_gain"), ("Mão", "hand_gain"), ("Rolagem", "scroll_gain"))):
            ttk.Label(sens, text=label, width=8).grid(row=i, column=0, sticky="w")
            val = ttk.Label(sens, width=5, text=f"{getattr(self.cfg, attr):.2f}")
            scale = ttk.Scale(sens, from_=0.3, to=3.0, length=220, value=getattr(self.cfg, attr),
                              command=lambda v, a=attr, lab=val: self._set_gain(a, float(v), lab))
            scale.grid(row=i, column=1, padx=6)
            val.grid(row=i, column=2)
        self.scroll_var = tk.BooleanVar(value=self.cfg.hand_scroll)
        self.natural_var = tk.BooleanVar(value=self.cfg.scroll_natural)
        ttk.Checkbutton(sens, text="Rolar com polegar+anelar", variable=self.scroll_var,
                        command=lambda: self._set_flag("hand_scroll", self.scroll_var.get())).grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Checkbutton(sens, text="Inverter a direção da rolagem (natural)", variable=self.natural_var,
                        command=lambda: self._set_flag("scroll_natural", self.natural_var.get())).grid(row=4, column=0, columnspan=3, sticky="w")

        # ---- pinch ball size: typed value, in % of the hand size (radius). The click fires when two balls touch.
        ttk.Label(sens, text="Bola da pinça", width=12).grid(row=5, column=0, sticky="w", pady=(8, 0))
        self.ball_var = tk.StringVar(value=f"{self.cfg.pinch_ball_size * 100:.1f}")
        self.ball_spin = ttk.Spinbox(sens, from_=3.0, to=30.0, increment=0.5, width=7, textvariable=self.ball_var,
                                     command=self._apply_ball)
        self.ball_spin.grid(row=5, column=1, sticky="w", padx=6, pady=(8, 0))
        ttk.Label(sens, text="% da mão").grid(row=5, column=2, sticky="w", pady=(8, 0))
        for seq in ("<Return>", "<KP_Enter>", "<FocusOut>"):
            self.ball_spin.bind(seq, self._apply_ball)
        ttk.Label(sens, text="Digite o tamanho (3–30). A pinça só vale quando as duas bolas (pontas dos dedos, veja em "
                             "'Ver câmera') se tocam: bola maior = dispara com os dedos mais afastados.",
                  wraplength=360, justify="left", foreground="#555").grid(row=6, column=0, columnspan=3, sticky="w")

        self.btn_bubble = ttk.Button(f, text="", command=self.toggle_bubble)
        self.btn_learn = ttk.Button(f, text="", command=self.toggle_learn)
        for b in (self.btn_bubble, self.btn_learn):
            b.grid(row=row, column=0, columnspan=2, sticky="ew", pady=3)
            row += 1
        ttk.Button(f, text="Configurações…", command=self.open_settings).grid(row=row, column=0, sticky="ew", pady=3, padx=(0, 3))
        ttk.Button(f, text="Sair", command=self.quit).grid(row=row, column=1, sticky="ew", pady=3)
        row += 1
        hints = ("Atalhos globais: Ctrl+Alt+E mouse • B bolha • C calibrar • H modo da cabeça • M modo da mão\n"
                 "R recentralizar • Q sair.  Clique: pinça polegar+indicador (esq.) / polegar+médio (dir.); segure para arrastar.")
        ttk.Label(f, text=hints, foreground="#666", wraplength=380, justify="left").grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 0))
        self._refresh_buttons()

    def _refresh_buttons(self) -> None:
        cfg = self.cfg
        self.btn_mouse.configure(text=f"Mouse: {'LIGADO' if self.tracker.mouse_enabled else 'DESLIGADO'}")
        self.btn_bubble.configure(text=f"Bolha: {'VISÍVEL' if cfg.show_bubble else 'OCULTA'}")
        self.btn_learn.configure(text=f"Aprender com cliques do mouse: {'SIM' if cfg.learn_from_clicks else 'NÃO'}")
        self.head_var.set(cfg.head_mode)
        self.hand_var.set(cfg.hand_mode)
        self.mb_head.configure(text=f"Cabeça: {HEAD_MODES[cfg.head_mode]}  ▾")
        self.mb_hand.configure(text=f"Mão: {HAND_MODES[cfg.hand_mode]}  ▾")
        self.mode_help.set(HEAD_HELP[cfg.head_mode] + "\n" + HAND_HELP[cfg.hand_mode])
        try:                                        # do not overwrite what the user is typing; follow changes made elsewhere (wizard)
            if self.root.focus_get() is not self.ball_spin:
                self.ball_var.set(f"{cfg.pinch_ball_size * 100:.1f}")
        except KeyError:
            pass

    # ---------------------------------------------------------------- actions
    def toggle_mouse(self) -> None:
        if not self.tracker.mouse_enabled and not self.tracker.mode_ready():
            messagebox.showinfo("EyeMouse", "O modo Olho precisa de calibração. Calibre, ou escolha o modo de cabeça "
                                            "'Cabeça' / de mão 'Ponta do indicador move o cursor'.")
            return
        if not self.tracker.mouse_enabled:
            self.tracker.recenter_head()          # head pointing is relative to the pose at the moment you turn it on
        self.tracker.mouse_enabled = not self.tracker.mouse_enabled
        self._refresh_buttons()

    def toggle_view(self) -> None:
        """Open/close the 'Visão da câmera' window (annotated camera image with the gesture readings)."""
        if self._view is not None:
            self._view.close()
            return
        self._view = CameraView(self.root, self.tracker, self.cfg, on_close=self._view_closed)

    def _view_closed(self) -> None:
        self._view = None

    def toggle_bubble(self) -> None:
        self.cfg.show_bubble = not self.cfg.show_bubble
        self._refresh_buttons()

    def toggle_learn(self) -> None:
        self.cfg.learn_from_clicks = not self.cfg.learn_from_clicks
        self._refresh_buttons()

    def _on_head_mode(self) -> None:
        self.set_head_mode(self.head_var.get())

    def _on_hand_mode(self) -> None:
        self.set_hand_mode(self.hand_var.get())

    def set_head_mode(self, mode: str) -> None:
        self.cfg.head_mode = mode
        if mode in ("head", "head_eye"):
            self.tracker.recenter_head()
        self.cfg.save()
        self._refresh_buttons()

    def set_hand_mode(self, mode: str) -> None:
        self.cfg.hand_mode = mode
        self.cfg.save()
        self._refresh_buttons()

    def cycle_head_mode(self) -> None:
        keys = list(HEAD_MODES)
        self.set_head_mode(keys[(keys.index(self.cfg.head_mode) + 1) % len(keys)])

    def cycle_hand_mode(self) -> None:
        keys = list(HAND_MODES)
        self.set_hand_mode(keys[(keys.index(self.cfg.hand_mode) + 1) % len(keys)])

    def recenter(self) -> None:
        self.tracker.recenter_head()

    def _set_gain(self, attr: str, value: float, label: ttk.Label) -> None:
        setattr(self.cfg, attr, round(value, 2))
        label.configure(text=f"{value:.2f}")

    def _apply_ball(self, event=None) -> None:
        """The ball size typed in the menu (percent of the hand size). Invalid text falls back to the current value."""
        try:
            value = float(self.ball_var.get().replace(",", "."))
        except ValueError:
            value = None
        if value is not None:
            self.cfg.pinch_ball_size = round(min(max(value, 3.0), 30.0) / 100.0, 4)
            self.cfg.save()
        self.ball_var.set(f"{self.cfg.pinch_ball_size * 100:.1f}")

    def _set_flag(self, attr: str, value: bool) -> None:
        setattr(self.cfg, attr, bool(value))

    def open_settings(self) -> None:
        if self._settings is None or not self._settings.win.winfo_exists():
            self._settings = SettingsWindow(self.root, self.cfg, self.tracker)
        else:
            self._settings.win.lift()

    def open_image(self) -> None:
        if self.calibration is not None:
            self.calibration.open_image_screen()
        elif self._image is None:
            self.bubble.suspended = True
            self._image = ImageScreen(self.root, self.tracker, self.cfg, self._image_closed)

    def _image_closed(self) -> None:
        self._image = None
        self.bubble.suspended = self.calibration is not None

    def open_calibration(self) -> None:
        if self.calibration is not None or self._image is not None or (self.tracker.error and self.tracker.latest() is None):
            return
        self.tracker.mouse_enabled = False
        self.bubble.suspended = True
        self.calibration = CalibrationScreen(self.root, self.tracker, self.model, self.cfg, self._calibration_closed)
        self._refresh_buttons()

    def _calibration_closed(self, saved: bool) -> None:
        self.calibration = None
        self.bubble.suspended = False
        self.cfg.save()
        self._refresh_buttons()

    def quit(self) -> None:
        self.tracker.mouse_enabled = False
        self.tracker.stop()
        self.tracker.join(2)
        for listener in self._listeners:
            listener.stop()
        if self.model.ready:
            self.model.save(CALIBRATION_PATH)
        self.cfg.save()
        self.root.destroy()

    # ------------------------------------------------------------------- loop
    def _tick(self) -> None:
        while True:
            try:
                action = self.actions.get_nowait()
            except queue.Empty:
                break
            {"mouse": self.toggle_mouse, "bubble": self.toggle_bubble, "calibrate": self.open_calibration,
             "head_mode": self.cycle_head_mode, "hand_mode": self.cycle_hand_mode, "recenter": self.recenter,
             "quit": self.quit}[action]()
            if action == "quit":
                return
        while not self.events.empty():
            self.events.get_nowait()  # pinch events are already reflected by the bubble colour

        tr = self.tracker
        st = tr.latest()
        if tr.error and st is None:  # camera / model failed to start
            if not self._error_shown:
                self._error_shown = True
                messagebox.showerror("EyeMouse", f"Erro na câmera/modelo:\n{tr.error}")
        if st is None:
            self.status["cam"].set("Câmera: iniciando…")
        else:
            hands = "mão ✓" if st.hand_ok else "mão –"
            dark = " • ⚠ imagem muito escura (abra Imagem e use Restaurar tudo)" if st.brightness < 30 else ""
            self.status["cam"].set(f"Câmera: {st.fps:.0f} fps • rosto {'✓' if st.face_ok else '✗'} • {hands}{dark}")
        if self.model.ready:
            err = f" • erro ≈ {self.model.cv_px:.0f} px" if self.model.cv_px else ""
            self.status["calib"].set(f"Calibração: {self.model.n_groups} alvos, {self.model.n_samples} amostras{err}")
        else:
            self.status["calib"].set("Calibração: nenhuma (necessária só para o modo Olho)")
        if tr.mouse_active:
            src = SOURCE_NAMES.get(st.source if st else "none", "—")
            scrolling = " • rolando" if st and st.scrolling else ""
            self.status["mouse"].set(f"Mouse: controlado por {src}{scrolling}")
        else:
            self.status["mouse"].set("Mouse: livre" + ("" if tr.mode_ready() else " (modo atual precisa de calibração)"))
        self._refresh_buttons()
        self.root.after(200 if self.calibration is None else 500, self._tick)

    def run(self) -> None:
        self.root.mainloop()
