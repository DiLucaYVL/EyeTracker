"""Camera driver properties (DirectShow "Video Proc Amp" / "Camera control") exposed through OpenCV."""
from __future__ import annotations

import cv2

# key: (cv2 property id, label, hint, default min, default max, step)
# Ranges are typical UVC values; the UI widens them to include whatever the driver reports.
CAMERA_PROPS: dict[str, tuple[int, str, str, float, float, float]] = {
    "exposure": (cv2.CAP_PROP_EXPOSURE, "Exposição", "menor = mais fps e imagem mais escura", -10, -1, 1),
    "gain": (cv2.CAP_PROP_GAIN, "Ganho", "clareia sem gastar fps (com mais ruído)", 0, 100, 1),
    "brightness": (cv2.CAP_PROP_BRIGHTNESS, "Brilho", "clareia/escurece tudo", -64, 64, 1),
    "contrast": (cv2.CAP_PROP_CONTRAST, "Contraste", "diferença entre claro e escuro", 0, 100, 1),
    "saturation": (cv2.CAP_PROP_SATURATION, "Saturação (cor)", "0 = preto e branco", 0, 100, 1),
    "gamma": (cv2.CAP_PROP_GAMMA, "Gama", "abre as sombras (tons médios)", 100, 500, 5),
    "sharpness": (cv2.CAP_PROP_SHARPNESS, "Nitidez", "borda da íris mais definida", 0, 15, 1),
    "backlight": (cv2.CAP_PROP_BACKLIGHT, "Comp. luz de fundo", "ajuda com janela/luz atrás de você", 0, 4, 1),
}
