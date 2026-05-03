from __future__ import annotations

from pathlib import Path

import pytest

from barcodeplot import BinaryTrack, save_barcode_plot


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


def test_save_barcode_plot_rejects_mismatched_lengths(tmp_path: Path):
    with pytest.raises(ValueError):
        save_barcode_plot(
            [
                BinaryTrack(label="A", values=[0, 1, 0]),
                BinaryTrack(label="B", values=[0, 1]),
            ],
            tmp_path / "bad.png",
        )
