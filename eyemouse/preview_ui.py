"""'Visão da câmera': a live window that shows what the computer vision sees (face, hand skeleton, gesture state)."""
from __future__ import annotations

import tkinter as tk

from .config import Config
from .tracker import PREVIEW_SIZE

BG, FG, DIM = "#0b0f14", "#e6edf3", "#8b949e"
GOOD, WARN, ORANGE, PURPLE, BAD = "#3fb950", "#d29922", "#f0883e", "#a371f7", "#f85149"
INFO_HEIGHT = 142


class CameraView:
    """Small always-on-top window with the annotated camera image and the current gesture readings."""

    def __init__(self, root: tk.Misc, tracker, cfg: Config, geometry: str = "+30+260", topmost: bool = True, on_close=None):
        self.tracker, self.cfg, self.on_close = tracker, cfg, on_close
        self.win = tk.Toplevel(root)
        self.win.title("Visão da câmera — EyeMouse")
        self.win.attributes("-topmost", topmost)
        self.win.geometry(geometry)
        self.win.configure(bg=BG)
        self.win.resizable(False, False)
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        pw, ph = PREVIEW_SIZE
        self.cv = tk.Canvas(self.win, width=pw, height=ph + INFO_HEIGHT, bg=BG, highlightthickness=0)
        self.cv.pack()
        self._photo: tk.PhotoImage | None = None
        self._src: bytes | None = None
        self._closed = False
        self._prev = (tracker.want_preview, tracker.preview_mode)
        tracker.want_preview, tracker.preview_mode = True, "small"
        self._tick()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.tracker.want_preview, self.tracker.preview_mode = self._prev
        self.win.destroy()
        if self.on_close:
            self.on_close()

    def _text(self, y: int, text: str, color: str = FG, size: int = 11, bold: bool = False) -> None:
        self.cv.create_text(10, PREVIEW_SIZE[1] + y, text=text, fill=color, anchor="w", tags="dyn",
                            font=("Segoe UI", size, "bold" if bold else "normal"))

    def _tick(self) -> None:
        if self._closed:
            return
        try:
            self._draw(self.tracker.latest())
        finally:
            self.win.after(33, self._tick)

    def _draw(self, st) -> None:
        cv = self.cv
        cv.delete("dyn")
        if st is not None and st.preview and st.preview is not self._src:
            self._photo, self._src = tk.PhotoImage(data=st.preview, format="PPM"), st.preview
        if self._photo is not None:
            cv.create_image(0, 0, image=self._photo, anchor="nw", tags="dyn")
        else:
            cv.create_text(PREVIEW_SIZE[0] / 2, PREVIEW_SIZE[1] / 2, text="Iniciando câmera…", fill=DIM, tags="dyn")
        if st is None:
            return
        cfg = self.cfg
        self._text(14, f"{st.fps:.0f} fps   rosto {'✓' if st.face_ok else '✗'}   mão {'✓' if st.hand_ok else '✗ (não detectada)'}",
                   GOOD if st.hand_ok else BAD, 11, True)
        names = {"idle": "relaxada", "scroll": "ROLAGEM (dedos dobrados)", "pinch_left": "PINÇA esquerda", "pinch_right": "PINÇA direita"}
        hands = "   |   ".join(f"mão {k + 1}: {names.get(s, s)}" for k, s in enumerate(st.hand_states)) or "sem mão"
        color = PURPLE if any(s == "scroll" for s in st.hand_states) else GOOD if any(s == "pinch_left" for s in st.hand_states) \
            else ORANGE if any(s == "pinch_right" for s in st.hand_states) else (FG if st.hand_ok else DIM)
        self._text(38, hands, color, 12, True)
        if any(s == "scroll" for s in st.hand_states):
            self._text(60, f"velocidade da mão {st.scroll_speed:.2f} alturas/s (mova a mão para rolar; polegar livre)", PURPLE, 9)
        elif st.ratios:
            self._text(60, f"distância das pontas — indicador {st.ratios[0]:.2f}   médio {st.ratios[1]:.2f}   (dispara abaixo de "
                           f"{cfg.pinch_thresholds()[0]:.2f})", DIM, 9)
        self._text(80, f"bolas = pontas dos dedos ({cfg.pinch_ball_size * 100:.1f}% da mão): a pinça dispara quando duas se TOCAM (4 dedos dobrados = rolagem)", DIM, 9)
        self._text(100, f"cursor: origem {st.source}   |   roxo = rolagem, verde = clique esq., laranja = clique dir.", DIM, 9)
        if cfg.hand_mode == "hand":
            who = f"mão {st.pointer + 1}" if st.pointer is not None else "nenhuma"
            self._text(122, f"mão que move o cursor: {who}   |   toque o indicador de uma mão no da outra para trocar", "#58d6ff", 9)
