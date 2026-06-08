# Repository Guidelines

## Project Structure & Module Organization

This is a small Python package using a `src/` layout. Package code lives in `src/barcodeplot/`:

- `types.py`: `BinaryTrack` validation and coercion.
- `io.py`: CSV/NPZ loaders and track alignment.
- `plotting.py`: static barcode PNG rendering.
- `video.py`: timeline MP4 rendering.
- `cli.py`: `barcodeplot plot` and `barcodeplot video` entrypoints.

Tests live in `tests/`, with one file per main module. Sample prediction data is in `data/`. Project docs and status notes are in `README.md`, `STATUS.md`, and `CLAUDE.md`.

## Build, Test, and Development Commands

Install locally with development dependencies:

```bash
pip install -e ".[dev]"
```

Run the full test suite:

```bash
pytest
```

Run a focused test file or test case:

```bash
pytest tests/test_io.py
pytest tests/test_io.py::test_align_tracks_fills_missing_frames_with_zero
```

Try the CLI after installation:

```bash
barcodeplot plot --track path/to/gt.csv:GT --track path/to/pred.csv:Pred --out barcode.png
barcodeplot video --frame-dir frames/ --track path/to/gt.csv:GT --fps 30 --out timeline.mp4
```

## Coding Style & Naming Conventions

Use Python 3.9+ with 4-space indentation and type hints for public functions. Keep module responsibilities narrow and follow existing patterns: loaders return `BinaryTrack`, renderers accept aligned `Sequence[BinaryTrack]`, and path arguments accept `str | Path`. Use `snake_case` for functions and variables, `PascalCase` for classes, and clear test names beginning with `test_`.

No formatter or linter is currently configured. Keep edits consistent with the surrounding code and avoid introducing large abstractions unless they simplify shared behavior.

## Testing Guidelines

Tests use `pytest`. Add or update tests for any changed behavior, especially input validation, loader auto-detection, alignment, CLI parsing, and rendering smoke paths. Prefer small fixtures created with `tmp_path`; generated PNG/MP4 outputs should remain temporary or ignored.

## Commit & Pull Request Guidelines

Git history uses short imperative subjects, for example `Add NPZ track loading` and `Add trim option for timeline videos`. Follow that style: describe the user-visible change in one concise sentence.

Pull requests should include a short summary, test commands run, and any relevant notes about generated visual/video outputs. For rendering changes, include example screenshots or describe the visual difference. Do not commit local caches, virtual environments, or generated `.png`/`.mp4` artifacts.

## Architecture Notes

The main data flow is: load CSV/NPZ data, create `BinaryTrack` objects, align tracks if needed, then render either a PNG barcode plot or MP4 timeline video. Rendering uses OpenCV only; colors are defined as RGB constants in `colors.py` and converted to BGR for OpenCV.
