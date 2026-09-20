"""Build the Windows installer.

    .venv\\Scripts\\python tools\\build_installer.py [--skip-installer] [--skip-app]

Steps: MediaPipe models -> icon -> PyInstaller (build/dist/EyeMouse) -> Inno Setup (dist/EyeMouse-Setup-<version>.exe).
Requires: pip install -r requirements-build.txt, and Inno Setup 6 (https://jrsoftware.org/isinfo.php) for the last step.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eyemouse import __version__  # noqa: E402

BUILD = ROOT / "build"
APP_DIR = BUILD / "dist" / "EyeMouse"
ISCC_CANDIDATES = [
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Inno Setup 6" / "ISCC.exe",
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Inno Setup 6" / "ISCC.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe",
]


def run(cmd: list[str], **kw) -> None:
    print("\n>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT, **kw)


def find_iscc() -> Path | None:
    if os.environ.get("ISCC"):
        return Path(os.environ["ISCC"])
    found = shutil.which("ISCC")
    if found:
        return Path(found)
    return next((p for p in ISCC_CANDIDATES if p.exists()), None)


def build_app() -> None:
    from eyemouse.landmarks import ensure_models

    ensure_models()                                   # data/*.task, bundled by the spec
    if not (ROOT / "assets" / "eyemouse.ico").exists():
        run([sys.executable, "tools/make_icon.py"])
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        sys.exit("PyInstaller não encontrado. Rode: pip install -r requirements-build.txt")
    run([sys.executable, "-m", "PyInstaller", "EyeMouse.spec", "--noconfirm", "--clean",
         "--distpath", str(BUILD / "dist"), "--workpath", str(BUILD / "work")])
    exe = APP_DIR / "EyeMouse.exe"
    if not exe.exists():
        sys.exit(f"Build falhou: {exe} não foi gerado.")
    size = sum(f.stat().st_size for f in APP_DIR.rglob("*") if f.is_file()) / 1e6
    print(f"\nAplicativo gerado em {APP_DIR} ({size:.0f} MB)")


def build_installer() -> None:
    iscc = find_iscc()
    if iscc is None:
        sys.exit("Inno Setup 6 não encontrado. Instale (winget install JRSoftware.InnoSetup) ou defina a variável ISCC.")
    if not APP_DIR.exists():
        sys.exit("Rode o build do aplicativo primeiro (sem --skip-app).")
    run([str(iscc), f"/DAppVersion={__version__}", "installer/eyemouse.iss"])
    out = ROOT / "dist" / f"EyeMouse-Setup-{__version__}.exe"
    print(f"\nInstalador gerado: {out} ({out.stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-app", action="store_true", help="não refaz o PyInstaller (usa build/dist existente)")
    ap.add_argument("--skip-installer", action="store_true", help="gera só a pasta do aplicativo")
    args = ap.parse_args()
    if not args.skip_app:
        build_app()
    if not args.skip_installer:
        build_installer()


if __name__ == "__main__":
    main()
