from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from barcodeplot.io import align_tracks, load_npz_track, load_track_auto
from barcodeplot.plotting import save_barcode_plot
from barcodeplot.types import BinaryTrack
from barcodeplot.video import render_timeline_video


def _parse_track_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"Track spec must be PATH:LABEL, got {spec!r}")
    path, label = spec.rsplit(":", 1)
    path = path.strip()
    label = label.strip()
    if not path or not label:
        raise ValueError(f"Track spec must be PATH:LABEL, got {spec!r}")
    return path, label


def _parse_npz_track_spec(spec: str) -> tuple[str, str, str, str] | None:
    marker = ".npz:"
    idx = spec.lower().find(marker)
    if idx == -1:
        return None
    path = spec[: idx + len(".npz")].strip()
    parts = [part.strip() for part in spec[idx + len(marker) :].split(":")]
    if len(parts) != 3:
        raise ValueError(
            f"NPZ track spec must be PATH.npz:DATASET:VALUE:LABEL, got {spec!r}"
        )
    dataset, value, label = parts
    if not path or not dataset or not value or not label:
        raise ValueError(
            f"NPZ track spec must be PATH.npz:DATASET:VALUE:LABEL, got {spec!r}"
        )
    return path, dataset, value, label


def _load_track_spec(spec: str) -> BinaryTrack:
    npz_spec = _parse_npz_track_spec(spec)
    if npz_spec is not None:
        path, dataset, value, label = npz_spec
        return load_npz_track(path, dataset=dataset, value=value, label=label)

    path, label = _parse_track_spec(spec)
    return load_track_auto(path, label=label)


def _load_tracks(track_specs: Sequence[str]) -> list[BinaryTrack]:
    loaded = []
    for spec in track_specs:
        loaded.append(_load_track_spec(spec))
    if not loaded:
        raise ValueError("At least one --track is required.")
    return align_tracks(loaded[0], loaded[1:])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Binary barcode plots and timeline videos.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plot_parser = subparsers.add_parser("plot", help="Render a single or stacked barcode plot.")
    plot_parser.add_argument("--track", action="append", required=True, help="Track spec PATH:LABEL.")
    plot_parser.add_argument("--out", required=True, help="Output PNG path.")
    plot_parser.add_argument("--dpi", type=int, default=200, help="Output figure DPI.")
    plot_parser.add_argument("--width", type=float, default=10.0, help="Figure width in inches.")
    plot_parser.add_argument("--row-height", type=float, default=1.2, help="Height per barcode row in inches.")

    video_parser = subparsers.add_parser("video", help="Render a timeline video with barcode rows.")
    video_parser.add_argument("--frame-dir", required=True, help="Directory of frame images.")
    video_parser.add_argument("--track", action="append", required=True, help="Track spec PATH:LABEL.")
    video_parser.add_argument("--fps", type=float, required=True, help="Output video FPS.")
    video_parser.add_argument("--title", default="Barcode Timeline", help="Header title.")
    video_parser.add_argument("--trim", action="store_true", help="Select images by frame number to match track length; ignores extra images in the frame directory.")
    video_parser.add_argument("--out", required=True, help="Output MP4 path.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    tracks = _load_tracks(args.track)
    if args.command == "plot":
        save_barcode_plot(
            tracks,
            Path(args.out),
            dpi=args.dpi,
            width=args.width,
            row_height=args.row_height,
        )
        return 0
    render_timeline_video(
        args.frame_dir,
        tracks,
        Path(args.out),
        fps=args.fps,
        title=args.title,
        trim_to_track=args.trim,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
