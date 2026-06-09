from barcodeplot.io import (
    align_tracks,
    load_binary_csv_track,
    load_npz_track,
    load_repo_condensed_track,
    load_repo_gt_track,
)
from barcodeplot.types import BinaryTrack

__all__ = [
    "BinaryTrack",
    "align_tracks",
    "load_binary_csv_track",
    "load_npz_track",
    "load_repo_condensed_track",
    "load_repo_gt_track",
    "render_timeline_video",
    "save_barcode_plot",
]


def __getattr__(name: str):
    if name == "render_timeline_video":
        from barcodeplot.video import render_timeline_video

        return render_timeline_video
    if name == "save_barcode_plot":
        from barcodeplot.plotting import save_barcode_plot

        return save_barcode_plot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
