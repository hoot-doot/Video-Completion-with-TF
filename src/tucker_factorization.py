"""Tucker HOSVD: transparent orthogonal low-multilinear-rank approximation."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .cp_factorization import unfold


@dataclass
class TuckerModel:
    core: np.ndarray
    factors: list[np.ndarray]


def mode_product(x: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
    """Compute x ×_mode matrix, with matrix shape (new_dimension, old_dimension)."""
    product = np.tensordot(matrix, x, axes=(1, mode))
    return np.moveaxis(product, 0, mode)


def tucker_hosvd(x: np.ndarray, ranks: tuple[int, int, int]) -> TuckerModel:
    """Compute HOSVD via leading singular vectors of all tensor unfoldings."""
    if x.ndim != 3:
        raise ValueError("This implementation expects an H x W x T tensor")
    factors: list[np.ndarray] = []
    for mode, requested in enumerate(ranks):
        u, _, _ = np.linalg.svd(unfold(x, mode), full_matrices=False)
        factors.append(u[:, : min(requested, u.shape[1])])
    core = x
    for mode, factor in enumerate(factors):
        core = mode_product(core, factor.T, mode)
    return TuckerModel(core, factors)


def reconstruct_tucker(model: TuckerModel) -> np.ndarray:
    x = model.core
    for mode, factor in enumerate(model.factors):
        x = mode_product(x, factor, mode)
    return x

