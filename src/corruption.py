"""Controlled missingness and corruption models with ground-truth masks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CorruptionResult:
    observed: np.ndarray
    observed_mask: np.ndarray  # True means fixed/observed in completion
    evaluation_mask: np.ndarray  # True means score against ground truth here
    name: str
    metadata: dict[str, Any]


def _result(x: np.ndarray, mask: np.ndarray, name: str, **metadata: Any) -> CorruptionResult:
    return CorruptionResult(x * mask, mask, ~mask, name, metadata)


def random_missing(x: np.ndarray, fraction: float, rng: np.random.Generator) -> CorruptionResult:
    """Remove independent entries with probability ``fraction``."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must lie in (0, 1)")
    mask = rng.random(x.shape) >= fraction
    return _result(x, mask, "random", missing_fraction=fraction)


def central_blocks(x: np.ndarray, fraction: float) -> CorruptionResult:
    """Remove a same-sized centered rectangle from every frame."""
    h, w, t = x.shape
    side_ratio = np.sqrt(fraction)
    block_h, block_w = max(1, round(h * side_ratio)), max(1, round(w * side_ratio))
    r0, c0 = (h - block_h) // 2, (w - block_w) // 2
    mask = np.ones_like(x, dtype=bool)
    mask[r0 : r0 + block_h, c0 : c0 + block_w, :] = False
    return _result(x, mask, "block", requested_fraction=fraction, block=(r0, c0, block_h, block_w), frames=t)


def missing_frames(
    x: np.ndarray, gap_length: int, start: int | None = None
) -> CorruptionResult:
    """Remove a contiguous temporal gap, avoiding endpoints when possible."""
    t = x.shape[2]
    if not 0 < gap_length < t:
        raise ValueError("gap_length must be between 1 and T-1")
    if start is None:
        start = max(1, (t - gap_length) // 2)
        start = min(start, t - gap_length - 1) if t - gap_length > 1 else 0
    if not 0 <= start <= t - gap_length:
        raise ValueError("invalid temporal gap start")
    mask = np.ones_like(x, dtype=bool)
    mask[:, :, start : start + gap_length] = False
    return _result(x, mask, "frames", gap_length=gap_length, start=start)


def salt_pepper_as_missing(
    x: np.ndarray, fraction: float, rng: np.random.Generator
) -> CorruptionResult:
    """Corrupt pixels, then flag their locations missing for low-rank recovery.

    This cleanly distinguishes an unreliable/corrupted measurement from a
    trusted observation: a robust method would estimate the corruption mask;
    this controlled experiment supplies it.
    """
    bad = rng.random(x.shape) < fraction
    noisy = x.copy()
    noisy[bad] = rng.integers(0, 2, size=int(bad.sum()))
    observed_mask = ~bad
    return CorruptionResult(noisy * observed_mask, observed_mask, bad, "salt_pepper", {"fraction": fraction})

