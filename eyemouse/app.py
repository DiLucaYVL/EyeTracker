"""Control panel + orchestration of tracker, bubble, calibration, hotkeys."""
from __future__ import annotations

import queue
import tkinter as tk
from tkinter import messagebox, ttk

from . import mouse
from .bubble import Bubble
from .calibration_ui import CalibrationScreen
from .config import CALIBRATION_PATH, ICON_PATH, Config
from .gaze_model import GazeModel
from .image_ui import ImageScreen
from .settings_ui import SettingsWindow
from .tracker import Tracker

HOTKEYS = {"<ctrl>+<alt>+e": "mouse", "<ctrl>+<alt>+b": "bubble", "<ctrl>+<alt>+c": "calibrate", "<ctrl>+<alt>+q": "quit"}


class App:
    def __init__(self, cfg: Config, calibrate_on_start: bool = False):
        mouse.enable_dpi_awareness()
        self.cfg = cfg
        self.model = GazeModel(mouse.screen_size())
        self.model.load(CALIBRATION_PATH)
        self.events: queue.SimpleQueue = queue.SimpleQueue()
        self.actions: queue.SimpleQueue = queue.SimpleQueue()
        self.tracker = Tracker(cfg, self.model, self.events)
        self.tracker.mouse_enabled = cfg.start_with_mouse and self.model.ready
        self.tracker.start()

        self.root = tk.Tk()
        self.root.title("EyeMouse")
        if ICON_PATH.exists():
            self.root.iconbitmap(default=str(ICON_PATH))
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.calibration: CalibrationScreen | None = None
        self._image: ImageScreen | None = None
        self._settings: SettingsWindow | None = None
        self._error_shown = False
        self._build_panel()
        self.bubble = Bubble(self.root, cfg, self.tracker)

        self._listeners = [
            mouse.start_hotkeys({k: (lambda a=v: self.actions.put(a)) for k, v in HOTKEYS.items()}),
            mouse.start_click_listener(self.tracker.on_physical_click),
        ]
        self.root.after(50, self._tick)
        if calibrate_on_start or not self.model.ready:
            self.root.after(800, self.open_calibration)

    # ------------------------------------------------------------------ panel
    def _build_panel(self) -> None:
        f = ttk.Frame(self.root, padding=14)
        f.pack()
        ttk.Label(f, text="EyeMouse", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, columnspan=2, sticky="w")
        self.status = {k: tk.StringVar() for k in ("cam", "calib", "mouse")}
        for i, k in enumerate(self.status, start=1):
            ttk.Label(f, textvariable=self.status[k]).grid(row=i, column=0, columnspan=2, sticky="w")

        self.btn_calib = ttk.Button(f, text="Calibrar (tela cheia)", command=self.open_calibration)
        self.btn_image = ttk.Button(f, text="Imagem e contraste dos olhos (tela cheia)", command=self.open_image)
        self.btn_mouse = ttk.Button(f, text="", command=self.toggle_mouse)
        self.btn_bubble = ttk.Button(f, text="", command=self.toggle_bubble)
        self.btn_learn = ttk.Button(f, text="", command=self.toggle_learn)
        for i, b in enumerate((self.btn_calib, self.btn_image, self.btn_mouse, self.btn_bubble, self.btn_learn)):
            b.grid(row=4 + i, column=0, columnspan=2, sticky="ew", pady=3)
        ttk.Button(f, text="Configurações…", command=self.open_settings).grid(row=9, column=0, sticky="ew", pady=3, padx=(0, 3))
        ttk.Button(f, text="Sair", command=self.quit).grid(row=9, column=1, sticky="ew", pady=3)
        hints = ("Atalhos globais:\nCtrl+Alt+E  liga/desliga o mouse\nCtrl+Alt+B  mostra/oculta a bolha\n"
                 "Ctrl+Alt+C  calibrar     Ctrl+Alt+Q  sair\n\nClique: pinça polegar+indicador (esquerdo)\n"
                 "polegar+médio (direito). Segure para arrastar.")
        ttk.Label(f, text=hints, foreground="#666").grid(row=10, column=0, columnspan=2, sticky="w", pady=(10, 0))
        self._refresh_buttons()

    def _refresh_buttons(self) -> None:
        self.btn_mouse.configure(text=f"Mouse com os olhos: {'LIGADO' if self.tracker.mouse_enabled else 'DESLIGADO'}")
        self.btn_bubble.configure(text=f"Bolha: {'VISÍVEL' if self.cfg.show_bubble else 'OCULTA'}")
        self.btn_learn.configure(text=f"Aprender com cliques do mouse: {'SIM' if self.cfg.learn_from_clicks else 'NÃO'}")

    # ---------------------------------------------------------------- actions
    def toggle_mouse(self) -> None:
        if not self.tracker.mouse_enabled and not self.model.ready:
            messagebox.showinfo("EyeMouse", "Calibre primeiro para controlar o mouse com os olhos.")
            return
        self.tracker.mouse_enabled = not self.tracker.mouse_enabled
        self._refresh_buttons()

    def toggle_bubble(self) -> None:
        self.cfg.show_bubble = not self.cfg.show_bubble
        self._refresh_buttons()

    def toggle_learn(self) -> None:
        self.cfg.learn_from_clicks = not self.cfg.learn_from_clicks
        self._refresh_buttons()

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
            {"mouse": self.toggle_mouse, "bubble": self.toggle_bubble,
             "calibrate": self.open_calibration, "quit": self.quit}[action]()
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
            self.status["calib"].set("Calibração: nenhuma")
        self.status["mouse"].set("Mouse: " + ("controlado pelos olhos" if tr.mouse_active else "livre"))
        self._refresh_buttons()
        self.root.after(200 if self.calibration is None else 500, self._tick)

    def run(self) -> None:
        self.root.mainloop()
