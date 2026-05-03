# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_plotting.py

# Run a single test by name
pytest tests/test_io.py::test_align_tracks_fills_missing_frames_with_zero

# CLI
barcodeplot plot --track path/to/gt.csv:GT --track path/to/pred.csv:Pred --out barcode.png
barcodeplot video --frame-dir frames/ --track gt.csv:GT --fps 30 --out timeline.mp4
```

## Architecture

The package has one core data type (`BinaryTrack`) and two output functions (`save_barcode_plot`, `render_timeline_video`). Everything else is in service of those.

**Data flow:**
1. Load CSV(s) → `BinaryTrack` objects (via `io.py`)
2. Optionally align sparse tracks with `align_tracks()` (also `io.py`)
3. Pass tracks to `save_barcode_plot()` or `render_timeline_video()`

**`BinaryTrack`** (`types.py`) is a frozen dataclass: `label`, `values` (uint8 ndarray, 0/1 only), and optional `frame_numbers` (int64 ndarray). Validation runs in `__post_init__`. `frame_numbers` enables sparse/non-sequential data; without it, tracks are assumed to be dense and same-length.

**`align_tracks()`** (`io.py`) is needed when tracks have `frame_numbers` that don't all cover the same set of frames — it aligns everything to the reference track's frame numbers, filling gaps with `fill_value=0`.

**Rendering** (`plotting.py`, `video.py`) uses OpenCV exclusively — no matplotlib. `save_barcode_plot` draws color bars directly onto an image array. `render_timeline_video` builds a per-frame composite: the original frame image on top, a fixed-height header with title/frame info, and a bottom timeline panel with moving playhead.

**CSV auto-detection** (`load_track_auto` in `io.py`) inspects column names to dispatch between the generic format (`frame_number`, `value`) and the two thesis-repo formats (`contact_label` column or `frame_id`/`label` columns).

**Colors:** green `(38, 140, 47)` = holding/1, red `(200, 28, 52)` = not-holding/0. Stored as RGB in `colors.py`; converted to BGR for OpenCV at render time.
