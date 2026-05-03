from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from barcodeplot.colors import HOLDING_RGB, NOT_HOLDING_RGB, rgb_to_bgr
from barcodeplot.types import BinaryTrack

COLOR_HOLDING = rgb_to_bgr(HOLDING_RGB)
COLOR_NOT_HOLDING = rgb_to_bgr(NOT_HOLDING_RGB)
COLOR_TEXT = (32, 32, 32)
COLOR_BORDER = (220, 220, 220)


def _coerce_tracks(tracks: Sequence[BinaryTrack]) -> list[BinaryTrack]:
    rows = list(tracks)
    if not rows:
        raise ValueError("At least one track is required.")
    length = rows[0].values.size
    for track in rows[1:]:
        if track.values.size != length:
            raise ValueError("All tracks must have the same length. Align them first if needed.")
    return rows


def save_barcode_plot(
    tracks: Sequence[BinaryTrack],
    output_path: str | Path,
    *,
    dpi: int = 200,
    width: float = 10.0,
    row_height: float = 1.2,
    show: bool = False,
) -> str:
    rows = _coerce_tracks(tracks)
    canvas_width = max(600, int(round(width * dpi)))
    left_label_width = 150
    top_pad = 12
    bottom_pad = 44
    row_gap = 8
    row_px = max(24, int(round(row_height * dpi * 0.33)))
    canvas_height = top_pad + bottom_pad + (len(rows) * row_px) + ((len(rows) - 1) * row_gap)
    timeline_width = max(1, canvas_width - left_label_width - 12)
    image = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)

    for row_idx, track in enumerate(rows):
        y0 = top_pad + row_idx * (row_px + row_gap)
        y1 = y0 + row_px
        x_edges = np.linspace(0, timeline_width, num=(track.values.size + 1), dtype=np.int32)
        for idx in range(track.values.size):
            x0 = left_label_width + int(x_edges[idx])
            x1 = left_label_width + int(x_edges[idx + 1])
            color = COLOR_HOLDING if int(track.values[idx]) == 1 else COLOR_NOT_HOLDING
            cv2.rectangle(image, (x0, y0), (max(x0, x1 - 1), y1 - 1), color, thickness=-1)
        cv2.rectangle(image, (left_label_width, y0), (canvas_width - 12, y1 - 1), COLOR_BORDER, thickness=1)
        cv2.putText(
            image,
            track.label,
            (10, y0 + int(row_px * 0.72)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            COLOR_TEXT,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        image,
        "Frame",
        (left_label_width + max(0, timeline_width // 2 - 28), canvas_height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        COLOR_TEXT,
        2,
        cv2.LINE_AA,
    )

    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Failed to write barcode image: {output_path}")
    if show:
        cv2.imshow("barcodeplot", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return str(output_path)
