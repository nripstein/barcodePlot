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
COLOR_PLAYHEAD = (0, 0, 0)
COLOR_BORDER = (220, 220, 220)


def _frame_key(path: Path) -> tuple[int, int | str, str]:
    name = path.stem
    current = ""
    number = None
    for ch in name:
        if ch.isdigit():
            current += ch
        elif current:
            number = int(current)
            break
    if current and number is None:
        number = int(current)
    if number is not None:
        return (0, number, name)
    return (1, name, name)


def get_sorted_image_list(frame_dir: str | Path) -> list[str]:
    frame_dir = Path(frame_dir).expanduser()
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")
    paths = [p for p in frame_dir.iterdir() if p.is_file() and not p.name.startswith(".")]
    if not paths:
        raise ValueError(f"No images found in frame directory: {frame_dir}")
    return [str(p) for p in sorted(paths, key=_frame_key)]


def _coerce_tracks(tracks: Sequence[BinaryTrack]) -> list[BinaryTrack]:
    rows = list(tracks)
    if not rows:
        raise ValueError("At least one track is required.")
    length = rows[0].values.size
    for track in rows[1:]:
        if track.values.size != length:
            raise ValueError("All tracks must have the same length. Align them first if needed.")
    return rows


def build_timeline_panel(
    frame_width: int,
    frame_index: int,
    tracks: Sequence[BinaryTrack],
    *,
    left_label_width: int = 150,
    row_height: int = 28,
    row_gap: int = 6,
    top_pad: int = 10,
    bottom_pad: int = 10,
) -> np.ndarray:
    rows = _coerce_tracks(tracks)
    timeline_width = max(1, frame_width - left_label_width)
    panel_h = top_pad + bottom_pad + (len(rows) * row_height) + ((len(rows) - 1) * row_gap)
    panel = np.full((panel_h, frame_width, 3), 255, dtype=np.uint8)
    for row_idx, track in enumerate(rows):
        y0 = top_pad + row_idx * (row_height + row_gap)
        y1 = y0 + row_height
        x_edges = np.linspace(0, timeline_width, num=(track.values.size + 1), dtype=np.int32)
        for idx in range(track.values.size):
            x0 = left_label_width + int(x_edges[idx])
            x1 = left_label_width + int(x_edges[idx + 1])
            color = COLOR_HOLDING if int(track.values[idx]) == 1 else COLOR_NOT_HOLDING
            cv2.rectangle(panel, (x0, y0), (max(x0, x1 - 1), y1 - 1), color, thickness=-1)
        cv2.putText(
            panel,
            track.label,
            (10, y0 + int(row_height * 0.70)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            COLOR_TEXT,
            2,
            cv2.LINE_AA,
        )
    n_frames = rows[0].values.size
    ratio = min(max(frame_index / float(max(1, n_frames - 1)), 0.0), 1.0)
    x = left_label_width + int(round(ratio * (timeline_width - 1)))
    cv2.line(panel, (x, 0), (x, panel_h - 1), COLOR_PLAYHEAD, thickness=2)
    cv2.rectangle(panel, (0, 0), (frame_width - 1, panel_h - 1), COLOR_BORDER, thickness=1)
    return panel


def compose_frame(
    frame: np.ndarray,
    timeline_panel: np.ndarray,
    *,
    title: str,
    frame_number: int,
    frame_index: int,
    n_frames: int,
) -> np.ndarray:
    header_h = 56
    frame_width = frame.shape[1]
    header = np.full((header_h, frame_width, 3), 255, dtype=np.uint8)
    text = f"{title} | frame={frame_number} ({frame_index + 1}/{n_frames})"
    cv2.putText(header, text, (12, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_TEXT, 2, cv2.LINE_AA)
    return cv2.vconcat([frame, header, timeline_panel])


def render_timeline_video(
    frame_dir: str | Path,
    tracks: Sequence[BinaryTrack],
    output_path: str | Path,
    *,
    fps: float,
    title: str = "Barcode Timeline",
) -> str:
    if fps <= 0:
        raise ValueError("fps must be positive.")
    rows = _coerce_tracks(tracks)
    image_paths = get_sorted_image_list(frame_dir)
    n_frames = rows[0].values.size
    if len(image_paths) != n_frames:
        raise ValueError(
            f"Frame count mismatch: images={len(image_paths)} track_length={n_frames}. "
            "Provide a frame directory whose image count exactly matches the reference track."
        )
    first = cv2.imread(image_paths[0])
    if first is None:
        raise ValueError(f"Failed to read frame image: {image_paths[0]}")
    frame_h, frame_w = first.shape[:2]
    panel = build_timeline_panel(frame_w, 0, rows)
    expected_size = (frame_w, frame_h + 56 + panel.shape[0])
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        expected_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to initialize VideoWriter for: {output_path}")
    ref_numbers = rows[0].frame_numbers
    try:
        for idx, image_path in enumerate(image_paths):
            frame = cv2.imread(image_path)
            if frame is None:
                raise ValueError(f"Failed to read frame image: {image_path}")
            timeline_panel = build_timeline_panel(frame_w, idx, rows)
            frame_number = int(ref_numbers[idx]) if ref_numbers is not None else idx + 1
            composed = compose_frame(
                frame,
                timeline_panel,
                title=title,
                frame_number=frame_number,
                frame_index=idx,
                n_frames=n_frames,
            )
            writer.write(composed)
    finally:
        writer.release()
    return str(output_path)
