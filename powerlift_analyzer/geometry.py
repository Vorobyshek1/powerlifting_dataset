from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np

Point = Tuple[float, float]


def _as_xy(point: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(point), dtype=float)
    if arr.size < 2:
        raise ValueError("Point must contain at least x and y coordinates")
    return arr[:2]


def angle_deg(a: Iterable[float], b: Iterable[float], c: Iterable[float]) -> float:
    """Return angle ABC in degrees.

    The points can be 2D or 3D. Only x and y are used because the current
    prototype works with a regular video frame.
    """
    a_xy = _as_xy(a)
    b_xy = _as_xy(b)
    c_xy = _as_xy(c)
    ba = a_xy - b_xy
    bc = c_xy - b_xy
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return float("nan")
    cos_value = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    cos_value = max(-1.0, min(1.0, cos_value))
    return float(math.degrees(math.acos(cos_value)))


def angle_to_vertical_deg(a: Iterable[float], b: Iterable[float]) -> float:
    """Return absolute angle between segment AB and the vertical axis."""
    a_xy = _as_xy(a)
    b_xy = _as_xy(b)
    vector = a_xy - b_xy
    norm = np.linalg.norm(vector)
    if norm == 0:
        return float("nan")
    vertical = np.array([0.0, 1.0])
    cos_value = abs(float(np.dot(vector, vertical) / norm))
    cos_value = max(-1.0, min(1.0, cos_value))
    return float(math.degrees(math.acos(cos_value)))


def rolling_median(values: Iterable[float], window: int = 5) -> np.ndarray:
    """Simple centered rolling median without external dependencies."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr
    if window <= 1:
        return arr.copy()
    half = window // 2
    result = np.empty_like(arr)
    for i in range(arr.size):
        lo = max(0, i - half)
        hi = min(arr.size, i + half + 1)
        segment = arr[lo:hi]
        segment = segment[~np.isnan(segment)]
        result[i] = np.nan if segment.size == 0 else np.median(segment)
    return result


def safe_mean(values: Iterable[float], default: float = float("nan")) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return default
    return float(np.mean(arr))


def safe_min(values: Iterable[float], default: float = float("nan")) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return default
    return float(np.min(arr))


def safe_max(values: Iterable[float], default: float = float("nan")) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return default
    return float(np.max(arr))
