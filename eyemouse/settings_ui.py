"""Settings window: every value is applied live; "Salvar" persists it to data/config.json."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .config import Config

# (label, attribute, min, max, step)
ROWS = [
    ("Suavização do cursor (menor = mais suave)", "smoothing_min_cutoff", 0.1, 3.0, 0.05),
    ("Reatividade em movimentos rápidos", "smoothing_beta", 0.0, 0.1, 0.005),
    ("Zona morta do cursor (px)", "deadzone_px", 0, 80, 1),
    ("Tamanho da bolha (px)", "bubble_size", 40, 200, 2),
    ("Limiar de olho fechado (congela o cursor)", "blink_close_thr", 0.2, 0.9, 0.01),
    ("Limiar de olho aberto", "blink_open_thr", 0.1, 0.8, 0.01),
    ("Pinça: dispara abaixo de (razão)", "pinch_on_ratio", 0.1, 0.6, 0.01),
    ("Pinça: solta acima de (razão)", "pinch_off_ratio", 0.2, 0.9, 0.01),
    ("Segurar a pinça por (ms) para arrastar", "drag_hold_ms", 100, 1000, 10),
]


class SettingsWindow:
    def __init__(self, root: tk.Tk, cfg: Config, tracker):
        self.cfg, self.tracker = cfg, tracker
        self.win = tk.Toplevel(root)
        self.win.title("Configurações — EyeMouse")
        self.win.resizable(False, False)
        body = ttk.Frame(self.win, padding=14)
        body.pack()

        for i, (label, attr, lo, hi, step) in enumerate(ROWS):
            ttk.Label(body, text=label).grid(row=i, column=0, sticky="w", pady=4)
            is_int = isinstance(getattr(cfg, attr), int)
            var = tk.DoubleVar(value=getattr(cfg, attr))
            val = ttk.Label(body, width=6, text=self._fmt(getattr(cfg, attr), is_int))
            spin = ttk.Scale(body, from_=lo, to=hi, variable=var, length=240,
                             command=lambda v, a=attr, lab=val, ii=is_int, s=step: self._set(a, float(v), lab, ii, s))
            spin.grid(row=i, column=1, padx=10)
            val.grid(row=i, column=2)

        r = len(ROWS)
        self.hand_var = tk.BooleanVar(value=cfg.hand_clicks)
        ttk.Checkbutton(body, text="Cliques por pinça de mão (reinicie para aplicar ao ligar/desligar)",
                        variable=self.hand_var, command=lambda: setattr(cfg, "hand_clicks", self.hand_var.get())
                        ).grid(row=r, column=0, columnspan=3, sticky="w", pady=(10, 0))

        ttk.Label(body, text="Medidores em tempo real").grid(row=r + 1, column=0, sticky="w", pady=(12, 2))
        self.meter = tk.Canvas(body, width=460, height=64, highlightthickness=0, bg="#1c2128")
        self.meter.grid(row=r + 2, column=0, columnspan=3)

        btns = ttk.Frame(body)
        btns.grid(row=r + 3, column=0, columnspan=3, pady=(12, 0), sticky="e")
        ttk.Button(btns, text="Zerar calibração da pinça", command=self._reset_pinch).pack(side="left", padx=4)
        ttk.Button(btns, text="Salvar", command=self._save).pack(side="left", padx=4)
        ttk.Button(btns, text="Fechar", command=self.win.destroy).pack(side="left")
        self._closed = False
        self.win.bind("<Destroy>", lambda e: setattr(self, "_closed", True) if e.widget is self.win else None)
        self._tick()

    @staticmethod
    def _fmt(v: float, is_int: bool) -> str:
        return str(int(round(v))) if is_int else f"{v:.2f}"

    def _set(self, attr: str, value: float, label: ttk.Label, is_int: bool, step: float) -> None:
        value = round(value / step) * step
        setattr(self.cfg, attr, int(round(value)) if is_int else round(value, 4))
        label.configure(text=self._fmt(value, is_int))

    def _reset_pinch(self) -> None:
        """Forget the per-finger thresholds measured by the pinch calibration; the sliders above apply again."""
        for name in ("pinch_on_index", "pinch_off_index", "pinch_on_middle", "pinch_off_middle"):
            setattr(self.cfg, name, 0.0)

    def _save(self) -> None:
        self.cfg.save()

    def _tick(self) -> None:
        if self._closed:
            return
        st = self.tracker.latest()
        c = self.meter
        c.delete("all")
        # bar 1: eye-closure score with thresholds
        score = st.blink_score if st else 0.0
        c.create_text(4, 10, text="Olho fechado", fill="#8b949e", anchor="w", font=("Segoe UI", 8))
        c.create_rectangle(110, 3, 450, 17, outline="#30363d")
        c.create_rectangle(110, 3, 110 + 340 * min(score, 1.0), 17, fill="#f85149" if st and st.eyes_closed else "#2f81f7", outline="")
        for thr, col in ((self.cfg.blink_close_thr, "#f0883e"), (self.cfg.blink_open_thr, "#3fb950")):
            c.create_line(110 + 340 * thr, 1, 110 + 340 * thr, 19, fill=col, width=2)
        # bars 2-3: pinch ratios (smaller = fingers closer)
        rat = st.ratios if st else None
        for i, (name, key) in enumerate((("Polegar+indicador", 0), ("Polegar+médio", 1))):
            y = 26 + i * 18
            on_thr, off_thr = self.cfg.pinch_thresholds("index" if key == 0 else "middle")
            c.create_text(4, y + 7, text=name, fill="#8b949e", anchor="w", font=("Segoe UI", 8))
            c.create_rectangle(110, y, 450, y + 14, outline="#30363d")
            if rat:
                v = min(rat[key] / 1.0, 1.0)
                on = rat[key] < on_thr
                c.create_rectangle(110, y, 110 + 340 * v, y + 14, fill="#3fb950" if on else "#2f81f7", outline="")
            c.create_line(110 + 340 * on_thr, y - 1, 110 + 340 * on_thr, y + 15, fill="#f0883e", width=2)
            c.create_line(110 + 340 * off_thr, y - 1, 110 + 340 * off_thr, y + 15, fill="#3fb950", width=2)
        self.win.after(50, self._tick)
