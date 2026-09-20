"""Generate assets/eyemouse.ico (a stylised eye inside the gaze bubble). Run from the project root."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "assets" / "eyemouse.ico"
SIZES = [16, 24, 32, 48, 64, 128, 256]


def render(size: int = 1024) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = size / 256
    d.rounded_rectangle([8 * s, 8 * s, 248 * s, 248 * s], radius=52 * s, fill="#0b0f14", outline="#30363d", width=int(4 * s))
    # gaze bubble
    d.ellipse([34 * s, 34 * s, 222 * s, 222 * s], outline="#2f81f7", width=int(10 * s))
    # eye (almond made from two arcs, filled)
    cx, cy = 128 * s, 128 * s
    eye = [(cx - 78 * s, cy), (cx - 40 * s, cy - 36 * s), (cx, cy - 46 * s), (cx + 40 * s, cy - 36 * s), (cx + 78 * s, cy),
           (cx + 40 * s, cy + 36 * s), (cx, cy + 46 * s), (cx - 40 * s, cy + 36 * s)]
    d.polygon(eye, fill="#e6edf3")
    d.ellipse([cx - 34 * s, cy - 34 * s, cx + 34 * s, cy + 34 * s], fill="#2f81f7")
    d.ellipse([cx - 16 * s, cy - 16 * s, cx + 16 * s, cy + 16 * s], fill="#0b0f14")
    d.ellipse([cx + 4 * s, cy - 22 * s, cx + 18 * s, cy - 8 * s], fill="#ffffff")
    return img


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    render().save(OUT, format="ICO", sizes=[(n, n) for n in SIZES])
    print("salvo em", OUT)


if __name__ == "__main__":
    main()
