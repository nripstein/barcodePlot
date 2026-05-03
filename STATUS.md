# barcodeplot Status

## Done

- Created a standalone Python package in this directory with a `src/` layout.
- Added a public API for:
  - `BinaryTrack`
  - `save_barcode_plot(...)`
  - `render_timeline_video(...)`
  - thesis-repo CSV loaders and track alignment helpers
- Added a CLI entrypoint:
  - `barcodeplot plot`
  - `barcodeplot video`
- Implemented binary barcode plots with the thesis red/green palette.
- Implemented stacked barcode plots with any number of rows.
- Implemented timeline video rendering with:
  - frame images on top
  - header text
  - 1..N barcode tracks on the bottom
  - moving playhead/ticker line
- Added support for these CSV inputs:
  - thesis `detections_condensed.csv`
  - thesis GT CSV with `frame_number,gt_binary`
  - thesis GT CSV with `frame_id,label`
  - generic CSV with `frame_number,value`
- Added tests for API behavior, CSV loading, alignment, plotting, video rendering, and CLI smoke paths.
- Verified the package locally with:
  - `PYTHONPATH=src pytest -q`
  - Result: `16 passed`

## Important Note

- The static barcode plot renderer currently uses OpenCV instead of Matplotlib.
- This was done because the local environment had a broken Matplotlib dependency chain during verification.
- The current implementation still preserves the intended binary barcode behavior and palette.

## Suggested Next Steps

- Initialize a git repository in this directory if you want this package tracked separately.
- Add a real GitHub remote and update the install example in `README.md`.
- Install locally with `pip install -e .` and try the CLI on real data.
- Decide whether you want to keep OpenCV-only plotting or switch static plots back to Matplotlib later.
- Add example assets or screenshots to the README.
- Add versioning and release notes before publishing.
- If you want PyPI later:
  - add build/publish workflow files
  - build a source dist and wheel
  - verify package metadata and long README rendering
- Optional future feature work:
  - raw video input instead of frame-directory-only input
  - per-track custom colors
  - richer generic CSV formats
  - multiclass barcodes and legends
  - configurable layout/theme options

## Current Key Files

- `pyproject.toml`
- `README.md`
- `src/barcodeplot/`
- `tests/`
