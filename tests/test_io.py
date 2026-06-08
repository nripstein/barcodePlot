from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from barcodeplot import (
    BinaryTrack,
    align_tracks,
    load_binary_csv_track,
    load_npz_track,
    load_repo_condensed_track,
    load_repo_gt_track,
)


def test_load_repo_condensed_track(tmp_path: Path):
    path = tmp_path / "detections_condensed.csv"
    path.write_text(
        "frame_number,contact_label\n3,No Contact\n1,Portable Object\n2,Stationary Object\n",
        encoding="utf-8",
    )
    track = load_repo_condensed_track(path, label="Pred")
    assert track.frame_numbers.tolist() == [1, 2, 3]
    assert track.values.tolist() == [1, 0, 0]


def test_load_repo_gt_track_frame_number_format(tmp_path: Path):
    path = tmp_path / "gt.csv"
    path.write_text("frame_number,gt_binary\n1,0\n2,1\n3,0\n", encoding="utf-8")
    track = load_repo_gt_track(path, label="GT")
    assert track.values.tolist() == [0, 1, 0]


def test_load_repo_gt_track_frame_id_format(tmp_path: Path):
    path = tmp_path / "gt_labels.csv"
    path.write_text("frame_id,label\n000001.png,not_holding\n000002.png,holding\n", encoding="utf-8")
    track = load_repo_gt_track(path, label="GT")
    assert track.frame_numbers.tolist() == [1, 2]
    assert track.values.tolist() == [0, 1]


def test_load_binary_csv_track(tmp_path: Path):
    path = tmp_path / "generic.csv"
    path.write_text("frame_number,value\n2,true\n1,false\n", encoding="utf-8")
    track = load_binary_csv_track(path, label="Pred")
    assert track.frame_numbers.tolist() == [1, 2]
    assert track.values.tolist() == [0, 1]


def test_load_npz_track_loads_value_and_frame_numbers(tmp_path: Path):
    path = tmp_path / "tracks.npz"
    np.savez(
        path,
        sr1__frame_number=np.array([1, 2, 3]),
        sr1__y_true=np.array([0, 1, 0]),
        sr1__y_pred=np.array([0, 1, 1]),
    )
    track = load_npz_track(path, dataset="sr1", value="y_true", label="GT")
    assert track.label == "GT"
    assert track.frame_numbers.tolist() == [1, 2, 3]
    assert track.values.tolist() == [0, 1, 0]


def test_load_npz_track_sorts_frames_with_values(tmp_path: Path):
    path = tmp_path / "tracks.npz"
    np.savez(
        path,
        sr1__frame_number=np.array([3, 1, 2]),
        sr1__y_pred=np.array([1, 0, 1]),
    )
    track = load_npz_track(path, dataset="sr1", value="y_pred", label="Pred")
    assert track.frame_numbers.tolist() == [1, 2, 3]
    assert track.values.tolist() == [0, 1, 1]


def test_load_npz_track_missing_key_raises_helpful_error(tmp_path: Path):
    path = tmp_path / "tracks.npz"
    np.savez(path, sr1__frame_number=np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="sr1__y_pred"):
        load_npz_track(path, dataset="sr1", value="y_pred", label="Pred")


def test_align_tracks_fills_missing_frames_with_zero():
    reference = BinaryTrack(label="GT", values=[0, 1, 0], frame_numbers=[1, 2, 3])
    pred = BinaryTrack(label="Pred", values=[1, 1], frame_numbers=[1, 3])
    aligned = align_tracks(reference, [pred])
    assert [track.label for track in aligned] == ["GT", "Pred"]
    assert aligned[1].frame_numbers.tolist() == [1, 2, 3]
    assert aligned[1].values.tolist() == [1, 0, 1]


def test_align_tracks_rejects_length_mismatch_without_frame_numbers():
    reference = BinaryTrack(label="GT", values=[0, 1, 0])
    pred = BinaryTrack(label="Pred", values=[1, 0])
    with pytest.raises(ValueError):
        align_tracks(reference, [pred])
