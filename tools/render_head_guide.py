"""Render docs/guia_posicao_da_cabeca.png: the head position for every calibration step, from the 3D head model.

Run from the project root:  .venv\\Scripts\\python tools\\render_head_guide.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eyemouse import head3d  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "guia_posicao_da_cabeca.png"
SS = 2                                   # supersampling factor for smooth lines
W, H = 1500, 1080
BG, PANEL, BORDER = "#0b0f14", "#0e131a", "#30363d"
FG, DIM = "#e6edf3", "#8b949e"
CYAN, WARN, GOOD = "#39c5cf", "#d29922", "#3fb950"
REF_DIST = 45.0
FONTS = "C:/Windows/Fonts/"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for name in (("segoeuib.ttf" if bold else "segoeui.ttf"), "arial.ttf"):
        try:
            return ImageFont.truetype(FONTS + name, size * SS)
        except OSError:
            continue
    return ImageFont.load_default()


def px(v: float) -> float:
    return v * SS


def text(d: ImageDraw.ImageDraw, x, y, s, size=16, color=FG, bold=False, anchor="la"):
    d.text((px(x), px(y)), s, font=font(size, bold), fill=color, anchor=anchor)


def draw_head(d: ImageDraw.ImageDraw, cx, cy, pose, view=250, color=CYAN, bg=PANEL):
    for proj in head3d.project(pose, px(view), REF_DIST):
        for xy, front in head3d.runs(proj):
            if proj.kind == "face":
                col, w = color, 4
            elif front:
                col, w = head3d.mix(color, bg, 0.75), 2
            else:
                col, w = head3d.mix(color, bg, 0.22), 1
            pts = [(px(cx) + float(p[0]), px(cy) + float(p[1])) for p in xy]
            d.line(pts, fill=col, width=w * SS, joint="curve")


def arrow(d, x0, y0, x1, y1, color=WARN, width=3):
    import math
    d.line([(px(x0), px(y0)), (px(x1), px(y1))], fill=color, width=width * SS)
    ang = math.atan2(y1 - y0, x1 - x0)
    for s in (-1, 1):
        a = ang + math.pi + s * 0.45
        d.line([(px(x1), px(y1)), (px(x1 + 12 * math.cos(a)), px(y1 + 12 * math.sin(a)))], fill=color, width=width * SS)


def panel(d, x, y, w, h, title, subtitle):
    d.rounded_rectangle([px(x), px(y), px(x + w), px(y + h)], radius=px(14), fill=PANEL, outline=BORDER, width=SS * 2)
    text(d, x + 24, y + 18, title, 22, FG, True)
    text(d, x + 24, y + 50, subtitle, 14, DIM)


def main() -> None:
    img = Image.new("RGB", (W * SS, H * SS), BG)
    d = ImageDraw.Draw(img)
    text(d, W / 2, 26, "Posição da cabeça em cada etapa da calibração", 32, FG, True, "ma")
    text(d, W / 2, 74, "Modelo 3D em ciano. Durante a calibração, a linha branca mostra a sua cabeça ao vivo para você comparar.",
         15, DIM, False, "ma")

    pw, ph, gx, gy, x0, y0 = 700, 440, 40, 24, 40, 112
    ref = (0.0, 0.08, 0.0, 0.0, 0.0, -REF_DIST)          # rest pose: slightly tilted down towards the screen

    # ---- Phase 1: still
    x, y = x0, y0
    panel(d, x, y, pw, ph, "Fase 1 · cabeça parada", "olhe cada ponto até ficar verde, sem mexer a cabeça")
    draw_head(d, x + 190, y + 262, ref, 300)
    # monitor + webcam schematic
    mx, my, mw, mh = x + 400, y + 130, 250, 150
    d.rounded_rectangle([px(mx), px(my), px(mx + mw), px(my + mh)], radius=px(8), outline=DIM, width=3 * SS)
    d.rectangle([px(mx + mw / 2 - 30), px(my + mh), px(mx + mw / 2 + 30), px(my + mh + 14)], fill=DIM)
    d.ellipse([px(mx + mw / 2 - 5), px(my - 12), px(mx + mw / 2 + 5), px(my - 2)], fill=GOOD)
    text(d, mx + mw / 2, my - 32, "webcam", 12, GOOD, False, "ma")
    for fx, fy in ((0.1, 0.15), (0.5, 0.15), (0.9, 0.15), (0.1, 0.5), (0.5, 0.5), (0.9, 0.5), (0.1, 0.85), (0.5, 0.85), (0.9, 0.85)):
        d.ellipse([px(mx + mw * fx - 4), px(my + mh * fy - 4), px(mx + mw * fx + 4), px(my + mh * fy + 4)], fill="#f85149")
    arrow(d, x + 290, y + 235, mx - 8, my + mh / 2, CYAN, 2)
    for i, line in enumerate(("• sente-se a ~50 cm da tela", "• rosto centralizado na câmera", "• olhos no centro de cada círculo")):
        text(d, x + 400, y + 316 + i * 26, line, 14, FG)

    # ---- Phase 2: sides
    x, y = x0 + pw + gx, y0
    panel(d, x, y, pw, ph, "Fase 2 · virar para os lados", "devagar, poucos graus, olhos sempre no círculo")
    for i, (yaw, label) in enumerate(((0.35, "esquerda"), (0.0, "centro"), (-0.35, "direita"))):
        cx = x + 130 + i * 220
        draw_head(d, cx, y + 262, (ref[0] + yaw, ref[1], ref[2], 0, 0, -REF_DIST), 220)
        text(d, cx, y + 385, label, 15, WARN, True, "ma")
    arrow(d, x + 175, y + 150, x + 85, y + 150)
    arrow(d, x + 525, y + 150, x + 615, y + 150)

    # ---- Phase 2: up / down
    x, y = x0, y0 + ph + gy
    panel(d, x, y, pw, ph, "Fase 2 · inclinar para cima e para baixo", "como um 'sim' bem suave com a cabeça")
    for i, (pitch, label) in enumerate(((-0.25, "para cima"), (0.0, "centro"), (0.25, "para baixo"))):
        cx = x + 130 + i * 220
        draw_head(d, cx, y + 262, (ref[0], ref[1] + pitch, ref[2], 0, 0, -REF_DIST), 220)
        text(d, cx, y + 385, label, 15, WARN, True, "ma")
    arrow(d, x + 130, y + 175, x + 130, y + 110)
    arrow(d, x + 570, y + 110, x + 570, y + 175)

    # ---- Phase 2: closer / farther
    x, y = x0 + pw + gx, y0 + ph + gy
    panel(d, x, y, pw, ph, "Fase 2 · aproximar e afastar", "cerca de 6 cm para cada lado; depois faça um pequeno círculo")
    for i, (dz, label) in enumerate(((-6.0, "mais longe"), (0.0, "posição inicial"), (6.0, "mais perto"))):
        cx = x + 130 + i * 220
        draw_head(d, cx, y + 262, (ref[0], ref[1], ref[2], 0, 0, -REF_DIST + dz), 220)
        text(d, cx, y + 385, label, 15, WARN, True, "ma")

    text(d, W / 2, H - 46, "Dica: evite girar mais que ~25° ou se afastar demais: os olhos precisam continuar visíveis para a câmera.",
         14, DIM, False, "ma")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.resize((W, H), Image.LANCZOS).save(OUT)
    print("salvo em", OUT)


if __name__ == "__main__":
    main()
