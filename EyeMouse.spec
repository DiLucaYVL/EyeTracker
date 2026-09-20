# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec. Build with:  python -m PyInstaller EyeMouse.spec --noconfirm --clean
# Produces build/dist/EyeMouse/ with EyeMouse.exe (windowed) and EyeMouse-console.exe (same app, prints to a console:
# use it for --doctor / --version / debugging).
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH)

datas, binaries, hiddenimports = [], [], []
for pkg in ("mediapipe",):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
hiddenimports += ["pynput.keyboard._win32", "pynput.mouse._win32", "PIL._tkinter_finder"]

# Models are bundled so the installed app works offline from the first run (see config.find_model).
for name in ("face_landmarker.task", "hand_landmarker.task"):
    datas.append((str(ROOT / "data" / name), "models"))
datas.append((str(ROOT / "assets" / "eyemouse.ico"), "assets"))

# Heavy libraries pulled in by dependencies but never used by EyeMouse.
# (matplotlib and unittest must stay: mediapipe / numpy import them at load time.)
excludes = ["scipy", "pandas", "IPython", "notebook", "pytest", "jax", "jaxlib", "torch", "tensorflow",
            "sklearn", "sympy", "PyQt5", "PyQt6", "PySide2", "PySide6", "tkinter.test"]

a = Analysis(
    [str(ROOT / "run_eyemouse.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

icon = str(ROOT / "assets" / "eyemouse.ico")
exe_gui = EXE(pyz, a.scripts, [], exclude_binaries=True, name="EyeMouse", console=False, icon=icon)
exe_console = EXE(pyz, a.scripts, [], exclude_binaries=True, name="EyeMouse-console", console=True, icon=icon)

coll = COLLECT(exe_gui, exe_console, a.binaries, a.datas, strip=False, upx=False, name="EyeMouse")
