from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from barcodeplot import BinaryTrack, render_timeline_video


def _write_dummy_frames(frame_dir: Path, count: int) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(1, count + 1):
        image = np.zeros((48, 80, 3), dtype=np.uint8)
        image[:, :, 1] = idx * 20
        cv2.imwrite(str(frame_dir / f"{idx:06d}.png"), image)


def test_render_timeline_video_smoke(tmp_path: Path):
    frame_dir = tmp_path / "frames"
    _write_dummy_frames(frame_dir, 4)
    tracks = [
        BinaryTrack(label="GT", values=[0, 1, 0, 0], frame_numbers=[1, 2, 3, 4]),
        BinaryTrack(label="Pred", values=[0, 1, 1, 0], frame_numbers=[1, 2, 3, 4]),
        BinaryTrack(label="Pred 2", values=[1, 0, 1, 1], frame_numbers=[1, 2, 3, 4]),
        BinaryTrack(label="Pred 3", values=[0, 0, 1, 0], frame_numbers=[1, 2, 3, 4]),
    ]
    out_path = tmp_path / "timeline.mp4"
    saved = render_timeline_video(frame_dir, tracks, out_path, fps=10.0, title="Test Timeline")
    assert Path(saved).exists()
    assert Path(saved).stat().st_size > 0


def test_render_timeline_video_rejects_frame_count_mismatch(tmp_path: Path):
    frame_dir = tmp_path / "frames"
    _write_dummy_frames(frame_dir, 3)
    track = BinaryTrack(label="Pred", values=[0, 1, 0, 1])
    with pytest.raises(ValueError):
        render_timeline_video(frame_dir, [track], tmp_path / "timeline.mp4", fps=10.0)


def test_render_timeline_video_trim_by_frame_number(tmp_path: Path):
    frame_dir = tmp_path / "frames"
    _write_dummy_frames(frame_dir, 5)  # 5 images on disk, track only covers 4
    track = BinaryTrack(label="GT", values=[0, 1, 0, 1], frame_numbers=[1, 2, 3, 4])
    out_path = tmp_path / "timeline.mp4"
    saved = render_timeline_video(frame_dir, [track], out_path, fps=10.0, trim_to_track=True)
    assert Path(saved).exists()
    assert Path(saved).stat().st_size > 0
