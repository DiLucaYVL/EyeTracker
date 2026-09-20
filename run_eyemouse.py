"""Entry point used by the PyInstaller build (the package itself is started with `python -m eyemouse`)."""
import sys

from eyemouse.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
