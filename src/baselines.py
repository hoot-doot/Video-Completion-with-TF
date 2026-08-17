"""Simple, transparent completion baselines."""
from __future__ import annotations

import cv2
import numpy as np


def previous_frame_copy(observed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill unknown entries from the nearest earlier estimated frame."""
    result = observed.copy()
    for t in range(1, result.shape[2]):
        unknown = ~mask[:, :, t]
        result[:, :, t][unknown] = result[:, :, t - 1][unknown]
    # If the first frame contains unknowns, copy the first later known value.
    for t in range(result.shape[2] - 2, -1, -1):
        unknown = ~mask[:, :, t]
        result[:, :, t][unknown] = result[:, :, t + 1][unknown]
    return result


def next_frame_copy(observed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill unknown entries from the nearest later estimated frame."""
    return previous_frame_copy(observed[:, :, ::-1], mask[:, :, ::-1])[:, :, ::-1]


def temporal_linear(observed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-pixel linear interpolation along time; endpoints use nearest value."""
    h, w, t = observed.shape
    result = observed.copy()
    grid = np.arange(t)
    for i in range(h):
        for j in range(w):
            known = mask[i, j, :]
            if known.any():
                result[i, j, ~known] = np.interp(grid[~known], grid[known], observed[i, j, known])
    return result


def spatial_inpaint(observed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """OpenCV Telea inpainting independently in each frame (no temporal model)."""
    result = observed.copy()
    for t in range(observed.shape[2]):
        unknown = (~mask[:, :, t]).astype(np.uint8)
        if unknown.any():
            image = (np.clip(observed[:, :, t], 0, 1) * 255).astype(np.uint8)
            result[:, :, t] = cv2.inpaint(image, unknown, 3, cv2.INPAINT_TELEA) / 255.0
    return result


def run_baseline(method: str, observed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    dispatch = {
        "previous": previous_frame_copy,
        "next": next_frame_copy,
        "temporal_linear": temporal_linear,
        "spatial": spatial_inpaint,
    }
    try:
        return dispatch[method](observed, mask)
    except KeyError as exc:
        raise ValueError(f"Unknown baseline: {method}") from exc

