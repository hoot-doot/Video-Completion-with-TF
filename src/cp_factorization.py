"""Explicit CP/PARAFAC alternating-least-squares implementation."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CPModel:
    weights: np.ndarray
    factors: list[np.ndarray]


def unfold(x: np.ndarray, mode: int) -> np.ndarray:
    return np.moveaxis(x, mode, 0).reshape(x.shape[mode], -1)


def khatri_rao(factors: list[np.ndarray]) -> np.ndarray:
    """Columnwise Kronecker product in the given (unfolding) mode order."""
    result = factors[0]
    for factor in factors[1:]:
        result = np.einsum("ir,jr->ijr", result, factor).reshape(-1, result.shape[1])
    return result


def cp_als(
    x: np.ndarray, rank: int, max_iterations: int = 20, ridge: float = 1e-6,
    seed: int = 31,
) -> CPModel:
    """Fit X ≈ sum_r λ_r a_r∘b_r∘c_r using regularized ALS.

    For each mode n, ALS solves A_n = X_(n) Z (ZᵀZ + ridge I)^(-1), where Z is
    the Khatri--Rao product of the other factor matrices. The caller performs
    missing-data updates; therefore this routine receives a fully filled tensor.
    """
    if x.ndim != 3:
        raise ValueError("This educational CP-ALS implementation is for order-3 tensors")
    rank = min(rank, *x.shape)
    rng = np.random.default_rng(seed)
    factors = [rng.standard_normal((dimension, rank)) for dimension in x.shape]
    identity = np.eye(rank)
    for _ in range(max_iterations):
        for mode in range(3):
            rest = [factors[i] for i in range(3) if i != mode]
            z = khatri_rao(rest)
            gram = z.T @ z + ridge * identity
            factors[mode] = (unfold(x, mode) @ z) @ np.linalg.pinv(gram)
    weights = np.ones(rank)
    for mode in range(3):
        norms = np.linalg.norm(factors[mode], axis=0)
        norms[norms < 1e-12] = 1.0
        factors[mode] /= norms
        weights *= norms
    return CPModel(weights, factors)


def reconstruct_cp(model: CPModel) -> np.ndarray:
    a, b, c = model.factors
    return np.einsum("ir,jr,kr,r->ijk", a, b, c, model.weights, optimize=True)

