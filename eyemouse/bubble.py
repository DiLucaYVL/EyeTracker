"""Click-through translucent bubble that follows the gaze point."""
from __future__ import annotations

import ctypes
import tkinter as tk
from ctypes import wintypes

import numpy as np

from .config import Config

KEY = "#010203"  # colour made transparent
COLORS = {"normal": "#2f81f7", "closed": "#8b949e", "lost": "#484f58", "left": "#3fb950", "right": "#f0883e", "scroll": "#a371f7"}

_user32 = ctypes.windll.user32
_user32.GetParent.argtypes = [wintypes.HWND]
_user32.GetParent.restype = wintypes.HWND
_user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
_user32.GetWindowLongW.restype = ctypes.c_long
_user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]


def make_click_through(win: tk.Toplevel) -> None:
    """Layered + transparent + tool window + no-activate: mouse clicks pass through the bubble."""
    hwnd = _user32.GetParent(win.winfo_id()) or win.winfo_id()
    style = _user32.GetWindowLongW(hwnd, -20)
    _user32.SetWindowLongW(hwnd, -20, style | 0x80000 | 0x20 | 0x80 | 0x08000000)


class Bubble:
    def __init__(self, root: tk.Tk, cfg: Config, tracker):
        self.cfg = cfg
        self.tracker = tracker
        self.suspended = False
        self._shown = True
        self._size = 0
        self._pos: np.ndarray | None = None
        self._last_geom = ""
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.attributes("-transparentcolor", KEY)
        self.win.attributes("-alpha", 0.6)
        self._canvas: tk.Canvas | None = None
        self._build(cfg.bubble_size)
        self.win.update_idletasks()
        make_click_through(self.win)
        self.root = root
        self.root.after(15, self._tick)

    def _build(self, size: int) -> None:
        if self._canvas is not None:
            self._canvas.destroy()
        self._size = size
        c = tk.Canvas(self.win, width=size, height=size, bg=KEY, highlightthickness=0)
        c.pack()
        pad = 3
        self._body = c.create_oval(pad, pad, size - pad, size - pad, fill=COLORS["normal"], outline="#ffffff", width=3)
        r = max(3, size // 14)
        self._dot = c.create_oval(size / 2 - r, size / 2 - r, size / 2 + r, size / 2 + r, fill="#ffffff", outline="")
        self._canvas = c
        self._last_geom = ""

    def _show(self, visible: bool) -> None:
        if visible != self._shown:
            self._shown = visible
            (self.win.deiconify if visible else self.win.withdraw)()

    def _tick(self) -> None:
        try:
            self._update()
        finally:
            self.root.after(15, self._tick)

    def _update(self) -> None:
        if self.suspended or not self.cfg.show_bubble:
            self._show(False)
            return
        self._show(True)
        if self.cfg.bubble_size != self._size:
            self._build(self.cfg.bubble_size)

        st = self.tracker.latest()
        if st is None:
            return
        if not st.face_ok:
            color = COLORS["lost"]
        elif st.scrolling:
            color = COLORS["scroll"]
        elif st.pinch in COLORS:
            color = COLORS[st.pinch]
        elif st.eyes_closed:
            color = COLORS["closed"]
        else:
            color = COLORS["normal"]
        self._canvas.itemconfigure(self._body, fill=color)

        if st.gaze is None:
            return
        target = np.asarray(st.gaze, dtype=np.float64)
        self._pos = target if self._pos is None else self._pos + (target - self._pos) * 0.45
        half = self._size / 2
        geom = f"+{int(self._pos[0] - half)}+{int(self._pos[1] - half)}"
        if geom != self._last_geom:
            self.win.geometry(geom)
            self._last_geom = geom
