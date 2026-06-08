# barcodeplot

`barcodeplot` is a small standalone package for binary barcode plots and timeline videos.

It is extracted from the barcode/timeline functionality in the thesis pipeline, but it does not depend on that repository.

## Features

- Single barcode plots
- Stacked barcode plots with any number of rows
- Timeline videos with frame images on top and a moving playhead over barcode rows on the bottom
- Python API and CLI
- Helpers for thesis-repo CSVs, a simple generic CSV format, and prediction NPZ archives

## Install

From a local checkout:

```bash
pip install -e .
```

From GitHub later:

```bash
pip install "git+https://github.com/<user>/barcodePlot.git"
```

## Python API

```python
from barcodeplot import (
    BinaryTrack,
    render_timeline_video,
    save_barcode_plot,
)

pred = BinaryTrack(label="Pred", values=[0, 1, 1, 0])
gt = BinaryTrack(label="GT", values=[0, 1, 0, 0])

save_barcode_plot([gt, pred], "barcode.png")
render_timeline_video("frames", [gt, pred], "timeline.mp4", fps=30.0, title="Example")
```

## CLI

Plot from thesis-style CSVs:

```bash
barcodeplot plot \
  --track /path/to/gt.csv:GT \
  --track /path/to/detections_condensed.csv:Pred \
  --out barcode.png
```

Render a timeline video:

```bash
barcodeplot video \
  --frame-dir /path/to/frames \
  --track /path/to/gt.csv:GT \
  --track /path/to/detections_condensed.csv:Pred \
  --fps 30 \
  --title "Contact Timeline" \
  --out timeline.mp4
```

## Generic CSV format

For a package-native CSV, use:

```text
frame_number,value
1,0
2,1
3,1
4,0
```

Accepted values include `0/1`, `true/false`, and `holding/not_holding`.

## NPZ format

NPZ tracks use keys in this layout:

```text
{dataset}__frame_number
{dataset}__y_true
{dataset}__y_pred
```

For example, load ground truth from dataset `sr1`:

```python
from barcodeplot import load_npz_track

gt = load_npz_track("data/all_preds_binary.npz", dataset="sr1", value="y_true", label="GT")
pred = load_npz_track("data/all_preds_binary.npz", dataset="sr1", value="y_pred", label="Pred")
```

The CLI form is `PATH.npz:DATASET:VALUE:LABEL`:

```bash
barcodeplot plot \
  --track data/all_preds_binary.npz:sr1:y_true:GT \
  --track data/all_preds_binary.npz:sr1:y_pred:Pred \
  --out sr1_comparison.png
```

## Thesis-repo compatibility

The loaders support:

- `detections_condensed.csv` with `frame_number` and `contact_label`
- GT CSV with `frame_number,gt_binary`
- GT CSV with `frame_id,label`

The thesis mapping is preserved:

- `Portable Object` -> `1`
- everything else -> `0`
