from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from barcodeplot.types import BinaryTrack

_TRUTHY = {"1", "true", "t", "yes", "y", "holding"}
_FALSY = {"0", "false", "f", "no", "n", "not_holding"}


def _read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path).expanduser(), sep=None, engine="python")


def _coerce_scalar_binary(value: object) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError("Binary values cannot be missing.")
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        if int(value) not in {0, 1}:
            raise ValueError(f"Expected binary value, got {value}.")
        return int(value)
    if isinstance(value, float):
        if not np.isfinite(value) or int(value) != value or int(value) not in {0, 1}:
            raise ValueError(f"Expected binary value, got {value}.")
        return int(value)
    text = str(value).strip().lower()
    if text in _TRUTHY:
        return 1
    if text in _FALSY:
        return 0
    raise ValueError(f"Unsupported binary value: {value!r}")


def _normalize_frame_table(frame_numbers: Iterable[object], values: Iterable[object], label: str) -> BinaryTrack:
    numbers = np.asarray(list(frame_numbers), dtype=np.int64)
    binary = np.asarray([_coerce_scalar_binary(v) for v in values], dtype=np.uint8)
    order = np.argsort(numbers, kind="mergesort")
    numbers = numbers[order]
    binary = binary[order]
    return BinaryTrack(label=label, values=binary, frame_numbers=numbers)


def load_repo_condensed_track(path: str | Path, label: str = "Pred") -> BinaryTrack:
    df = _read_csv(path)
    required = {"frame_number", "contact_label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Repo condensed CSV missing columns: {sorted(missing)}")
    numbers = df["frame_number"].astype(int).tolist()
    values = (df["contact_label"].fillna("") == "Portable Object").astype(int).tolist()
    return _normalize_frame_table(numbers, values, label)


def load_repo_gt_track(path: str | Path, label: str = "GT") -> BinaryTrack:
    df = _read_csv(path)
    if {"frame_number", "gt_binary"}.issubset(df.columns):
        return _normalize_frame_table(df["frame_number"], df["gt_binary"], label)
    if {"frame_id", "label"}.issubset(df.columns):
        extracted = df["frame_id"].astype(str).str.extract(r"(\d+)")[0]
        valid = extracted.notna()
        if not valid.any():
            raise ValueError("frame_id,label CSV did not contain any parseable frame numbers.")
        numbers = extracted.loc[valid].astype(int).tolist()
        values = df.loc[valid, "label"].tolist()
        return _normalize_frame_table(numbers, values, label)
    raise ValueError(
        "Repo GT CSV must contain either ('frame_number', 'gt_binary') or ('frame_id', 'label')."
    )


def load_binary_csv_track(
    path: str | Path,
    *,
    label: str,
    frame_column: str = "frame_number",
    value_column: str = "value",
) -> BinaryTrack:
    df = _read_csv(path)
    required = {frame_column, value_column}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Generic CSV missing columns: {sorted(missing)}")
    return _normalize_frame_table(df[frame_column], df[value_column], label)


def load_npz_track(path: str | Path, *, dataset: str, value: str, label: str) -> BinaryTrack:
    path = Path(path).expanduser()
    frame_key = f"{dataset}__frame_number"
    value_key = f"{dataset}__{value}"
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in (frame_key, value_key) if key not in archive]
        if missing:
            raise ValueError(f"NPZ track missing arrays: {missing}")
        frame_numbers = archive[frame_key]
        values = archive[value_key]

    if frame_numbers.ndim != 1:
        raise ValueError(f"NPZ array {frame_key!r} must be one-dimensional.")
    if values.ndim != 1:
        raise ValueError(f"NPZ array {value_key!r} must be one-dimensional.")
    if frame_numbers.shape[0] != values.shape[0]:
        raise ValueError(
            f"NPZ arrays {frame_key!r} and {value_key!r} must have equal length."
        )
    return _normalize_frame_table(frame_numbers, values, label)


def load_track_auto(path: str | Path, *, label: str) -> BinaryTrack:
    df = _read_csv(path)
    columns = set(df.columns)
    if {"frame_number", "contact_label"}.issubset(columns):
        return load_repo_condensed_track(path, label=label)
    if {"frame_number", "gt_binary"}.issubset(columns) or {"frame_id", "label"}.issubset(columns):
        return load_repo_gt_track(path, label=label)
    if {"frame_number", "value"}.issubset(columns):
        return load_binary_csv_track(path, label=label)
    raise ValueError(
        "Could not auto-detect track CSV format. Supported formats are repo condensed, repo GT, and generic frame_number,value."
    )


def align_tracks(
    reference_track: BinaryTrack,
    other_tracks: Iterable[BinaryTrack],
    *,
    fill_value: int = 0,
) -> list[BinaryTrack]:
    if fill_value not in {0, 1}:
        raise ValueError("fill_value must be 0 or 1.")
    aligned = [reference_track]
    ref_frames = reference_track.frame_numbers
    ref_len = reference_track.values.size
    for track in other_tracks:
        if ref_frames is None or track.frame_numbers is None:
            if track.values.size != ref_len:
                raise ValueError(
                    f"Track '{track.label}' has length {track.values.size}, expected {ref_len}."
                )
            aligned.append(track)
            continue
        mapping = dict(zip(track.frame_numbers.tolist(), track.values.tolist()))
        values = np.asarray([mapping.get(int(frame), fill_value) for frame in ref_frames], dtype=np.uint8)
        aligned.append(BinaryTrack(label=track.label, values=values, frame_numbers=ref_frames.copy()))
    return aligned
