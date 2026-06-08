from __future__ import annotations

from pathlib import Path
from typing import Sequence

from barcodeplot._matplotlib import configure_matplotlib

configure_matplotlib()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

from barcodeplot.colors import HOLDING_RGB, NOT_HOLDING_RGB
from barcodeplot.types import BinaryTrack

COLOR_TEXT = "#202020"
COLOR_BORDER = "#dcdcdc"


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
    n_frames = rows[0].values.size
    row_gap_units = 0.22
    row_units = 1.0
    cmap = ListedColormap(
        [
            np.array(NOT_HOLDING_RGB, dtype=float) / 255.0,
            np.array(HOLDING_RGB, dtype=float) / 255.0,
        ]
    )
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    fig = plt.figure(
        figsize=(canvas_width / dpi, canvas_height / dpi),
        dpi=dpi,
        facecolor="white",
    )
    ax = fig.add_axes(
        [
            left_label_width / canvas_width,
            bottom_pad / canvas_height,
            max(1, canvas_width - left_label_width - 12) / canvas_width,
            max(1, canvas_height - top_pad - bottom_pad) / canvas_height,
        ]
    )
    ax.set_facecolor("white")

    for row_idx, track in enumerate(rows):
        y0 = row_idx * (row_units + row_gap_units)
        y1 = y0 + row_units
        ax.imshow(
            track.values[np.newaxis, :],
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            extent=(0, n_frames, y1, y0),
        )
        ax.add_patch(
            Rectangle(
                (0, y0),
                n_frames,
                row_units,
                fill=False,
                edgecolor=COLOR_BORDER,
                linewidth=0.6,
            )
        )

    y_ticks = [
        row_idx * (row_units + row_gap_units) + (row_units / 2.0)
        for row_idx in range(len(rows))
    ]
    y_max = (len(rows) * row_units) + ((len(rows) - 1) * row_gap_units)
    ax.set_xlim(0, n_frames)
    ax.set_ylim(y_max, 0)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([track.label for track in rows], color=COLOR_TEXT)
    ax.set_xticks([])
    ax.set_xlabel("Frame", color=COLOR_TEXT, labelpad=4)
    ax.tick_params(axis="y", length=0, pad=6, colors=COLOR_TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)

    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output_path, dpi=dpi, facecolor="white")
    except OSError as exc:
        raise RuntimeError(f"Failed to write barcode image: {output_path}") from exc
    if show:
        plt.show()
    plt.close(fig)
    return str(output_path)
