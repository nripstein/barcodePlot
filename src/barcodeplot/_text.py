from __future__ import annotations

import cv2
import numpy as np


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
    if font_size <= 0:
        raise ValueError("font size must be positive.")
    if image.size == 0:
        return image

    scale = max(0.3, font_size / 28.0)
    thickness = max(1, round(font_size / 18))
    baseline_y = int(xy[1] + font_size)
    cv2.putText(
        image,
        text,
        (int(xy[0]), baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color_bgr,
        thickness,
        lineType=cv2.LINE_AA,
    )
    return image
