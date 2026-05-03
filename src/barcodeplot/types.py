from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _coerce_binary_values(values: object) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        raise ValueError("BinaryTrack values must be non-empty.")
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8)
    if np.issubdtype(arr.dtype, np.integer):
        coerced = arr.astype(np.int64)
    elif np.issubdtype(arr.dtype, np.floating):
        if not np.isfinite(arr).all():
            raise ValueError("BinaryTrack values must be finite.")
        if not np.all(np.equal(arr, np.floor(arr))):
            raise ValueError("BinaryTrack values must be binary integers.")
        coerced = arr.astype(np.int64)
    else:
        raise ValueError("BinaryTrack values must be numeric or boolean.")
    if not np.isin(coerced, [0, 1]).all():
        raise ValueError("BinaryTrack values must contain only 0 or 1.")
    return coerced.astype(np.uint8)


def _coerce_frame_numbers(frame_numbers: object | None, expected_size: int) -> np.ndarray | None:
    if frame_numbers is None:
        return None
    arr = np.asarray(frame_numbers).reshape(-1)
    if arr.size != expected_size:
        raise ValueError("frame_numbers must match the length of values.")
    if not np.issubdtype(arr.dtype, np.integer):
        if np.issubdtype(arr.dtype, np.floating):
            if not np.isfinite(arr).all():
                raise ValueError("frame_numbers must be finite.")
            if not np.all(np.equal(arr, np.floor(arr))):
                raise ValueError("frame_numbers must be integers.")
            arr = arr.astype(np.int64)
        else:
            raise ValueError("frame_numbers must be integers.")
    else:
        arr = arr.astype(np.int64)
    if len(np.unique(arr)) != arr.size:
        raise ValueError("frame_numbers must be unique.")
    return arr


@dataclass(frozen=True)
class BinaryTrack:
    label: str
    values: np.ndarray
    frame_numbers: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("BinaryTrack label must be a non-empty string.")
        values = _coerce_binary_values(self.values)
        frame_numbers = _coerce_frame_numbers(self.frame_numbers, values.size)
        object.__setattr__(self, "label", self.label.strip())
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "frame_numbers", frame_numbers)
