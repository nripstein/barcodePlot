from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from barcodeplot import BinaryTrack, save_barcode_plot
from barcodeplot.colors import HOLDING_RGB, NOT_HOLDING_RGB, rgb_to_bgr


def test_save_barcode_plot_single_track(tmp_path: Path):
    out_path = tmp_path / "barcode.png"
    saved = save_barcode_plot([BinaryTrack(label="Pred", values=[0, 1, 0, 1])], out_path)
    assert Path(saved).exists()
    assert Path(saved).stat().st_size > 0


def test_save_barcode_plot_many_tracks(tmp_path: Path):
    tracks = [BinaryTrack(label=f"Track {idx}", values=[idx % 2, 1, 0, 1]) for idx in range(5)]
    out_path = tmp_path / "stacked.png"
    save_barcode_plot(tracks, out_path)
    assert out_path.exists()


def test_save_barcode_plot_custom_dimensions_are_sane(tmp_path: Path):
    tracks = [
        BinaryTrack(label="GT", values=[0, 1, 0, 1, 1]),
        BinaryTrack(label="Pred", values=[1, 1, 0, 0, 1]),
    ]
    out_path = tmp_path / "custom.png"
    save_barcode_plot(tracks, out_path, dpi=120, width=3.0, row_height=1.0)

    image = cv2.imread(str(out_path))
    assert image is not None
    assert image.shape[1] == 600
    assert image.shape[0] > 80


def test_save_barcode_plot_does_not_draw_grey_row_outline(tmp_path: Path):
    out_path = tmp_path / "no_outline.png"
    save_barcode_plot([BinaryTrack(label="Pred", values=[0, 1, 0, 1])], out_path)

    image = cv2.imread(str(out_path))
    assert image is not None
    holding_bgr = rgb_to_bgr(HOLDING_RGB)
    not_holding_bgr = rgb_to_bgr(NOT_HOLDING_RGB)
    barcode_mask = (image == holding_bgr).all(axis=2) | (image == not_holding_bgr).all(axis=2)
    y_coords, x_coords = np.where(barcode_mask)
    assert y_coords.size > 0

    y0 = max(int(y_coords.min()) - 3, 0)
    y1 = min(int(y_coords.max()) + 4, image.shape[0])
    x0 = max(int(x_coords.min()) - 3, 0)
    x1 = min(int(x_coords.max()) + 4, image.shape[1])
    barcode_region = image[y0:y1, x0:x1]
    grey_outline_bgr = (220, 220, 220)
    assert not (barcode_region == grey_outline_bgr).all(axis=2).any()


def test_save_barcode_plot_rejects_mismatched_lengths(tmp_path: Path):
    with pytest.raises(ValueError):
        save_barcode_plot(
            [
                BinaryTrack(label="A", values=[0, 1, 0]),
                BinaryTrack(label="B", values=[0, 1]),
            ],
            tmp_path / "bad.png",
        )
