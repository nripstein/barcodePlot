from __future__ import annotations

HOLDING_RGB = (38, 140, 47)
NOT_HOLDING_RGB = (200, 28, 52)


def rgb_to_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return (rgb[2], rgb[1], rgb[0])
