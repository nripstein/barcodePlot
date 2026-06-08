from __future__ import annotations

from functools import lru_cache

import numpy as np

from barcodeplot._matplotlib import configure_matplotlib

configure_matplotlib()

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=1)
def _dejavu_sans_path() -> str:
    return font_manager.findfont("DejaVu Sans", fallback_to_default=True)


@lru_cache(maxsize=16)
def _font(size: int) -> ImageFont.FreeTypeFont:
    if size <= 0:
        raise ValueError("font size must be positive.")
    return ImageFont.truetype(_dejavu_sans_path(), size=size)


def draw_text_bgr(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    *,
    font_size: int,
    color_bgr: tuple[int, int, int],
) -> np.ndarray:
    if not text:
        return image

    font = _font(font_size)
    left, top, right, bottom = font.getbbox(text)
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return image

    text_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0], 255)
    draw.text((-left, -top), text, font=font, fill=color_rgb)

    x, y = int(xy[0]), int(xy[1])
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(image.shape[1], x + width)
    y1 = min(image.shape[0], y + height)
    if x0 >= x1 or y0 >= y1:
        return image

    crop = text_layer.crop((x0 - x, y0 - y, x1 - x, y1 - y))
    roi_rgb = np.ascontiguousarray(image[y0:y1, x0:x1, ::-1])
    roi = Image.fromarray(roi_rgb).convert("RGBA")
    roi.alpha_composite(crop)
    image[y0:y1, x0:x1] = np.asarray(roi.convert("RGB"))[:, :, ::-1]
    return image
