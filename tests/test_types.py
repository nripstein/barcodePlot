from __future__ import annotations

import pytest

from barcodeplot import BinaryTrack


def test_binary_track_coerces_bool_values():
    track = BinaryTrack(label="Pred", values=[False, True, True, False])
    assert track.values.tolist() == [0, 1, 1, 0]


def test_binary_track_rejects_non_binary_values():
    with pytest.raises(ValueError):
        BinaryTrack(label="Pred", values=[0, 2, 1])


def test_binary_track_rejects_duplicate_frame_numbers():
    with pytest.raises(ValueError):
        BinaryTrack(label="Pred", values=[0, 1], frame_numbers=[1, 1])
