"""Entry point: python -m eyemouse [--calibrate] [--doctor] [--reset] [--camera N]."""
from __future__ import annotations

import argparse
import queue
import sys
import time

from . import __version__
from .config import CALIBRATION_PATH, FROZEN, LOG_PATH, Config


def setup_logging() -> None:
    """The installed (windowed) app has no console: send print() output and tracebacks to a log file."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = log
    if sys.stderr is None:
        sys.stderr = log


def show_fatal_error(text: str) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "EyeMouse", "O EyeMouse encontrou um erro e será fechado.\n\n" + text + "\n\nDetalhes: " + str(LOG_PATH))
        root.destroy()
    except Exception:  # noqa: BLE001 - nothing else we can do
        pass


def doctor(cfg: Config) -> int:
    """Headless self-test: camera, models, face/hand detection rate for a few seconds."""
    from .gaze_model import GazeModel
    from .landmarks import ensure_models
    from .tracker import Tracker

    ensure_models()
    tracker = Tracker(cfg, GazeModel((1920, 1080)), queue.SimpleQueue())
    tracker.start()
    tracker.started.wait(15)
    if tracker.error:
        print("ERRO:", tracker.error)
        return 1
    frames = faces = hands = 0
    last_t = 0.0
    fps = brightness = score = 0.0
    end = time.time() + 6
    while time.time() < end:
        st = tracker.latest()
        if st is not None and st.t != last_t:
            last_t = st.t
            frames += 1
            faces += st.face_ok
            hands += st.hand_ok
            fps, brightness, score = st.fps, st.brightness, max(score, st.blink_score)
        time.sleep(0.005)
    tracker.stop()
    tracker.join(2)
    if not frames and tracker.error:
        print("ERRO no loop:", tracker.error)
    print(f"frames analisados: {frames}  fps: {fps:.1f}  brilho: {brightness:.0f}/255")
    print(f"rosto detectado em {100 * faces / max(frames, 1):.0f}% dos frames; mão em {100 * hands / max(frames, 1):.0f}%")
    print(f"maior score de olho fechado visto: {score:.2f}")
    return 0 if frames and faces else 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eyemouse", description="Mouse controlado pelo olhar, com cliques por pinça de mão.")
    ap.add_argument("--calibrate", action="store_true", help="abre a calibração ao iniciar")
    ap.add_argument("--doctor", action="store_true", help="testa câmera e modelos sem abrir a interface")
    ap.add_argument("--reset", action="store_true", help="apaga a calibração salva")
    ap.add_argument("--camera", type=int, help="índice da câmera (padrão: config)")
    ap.add_argument("--version", action="store_true", help="mostra a versão")
    args = ap.parse_args(argv)
    setup_logging()
    if args.version:
        print(f"EyeMouse {__version__}")
        return 0

    cfg = Config.load()
    if args.camera is not None:
        cfg.camera_index = args.camera
    if args.reset:
        CALIBRATION_PATH.unlink(missing_ok=True)
        print("Calibração apagada.")
        return 0
    if args.doctor:
        return doctor(cfg)

    from .landmarks import ensure_models
    from .app import App

    try:
        ensure_models()
        App(cfg, calibrate_on_start=args.calibrate).run()
    except Exception as exc:  # noqa: BLE001 - report instead of dying silently in the windowed build
        import traceback

        traceback.print_exc()
        if FROZEN:
            show_fatal_error(f"{type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
