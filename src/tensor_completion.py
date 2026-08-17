"""Projected iterative low-rank tensor completion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .cp_factorization import CPModel, cp_als, reconstruct_cp
from .tucker_factorization import TuckerModel, reconstruct_tucker, tucker_hosvd

Method = Literal["cp", "tucker"]


@dataclass
class CompletionResult:
    tensor: np.ndarray
    history: list[float]
    iterations: int
    model: CPModel | TuckerModel


def initialize(observed: np.ndarray, mask: np.ndarray, strategy: str) -> np.ndarray:
    """Create a filled starting point without changing trusted measurements."""
    result = observed.copy()
    if strategy == "zero":
        return result
    if mask.any():
        global_mean = float(observed[mask].mean())
    else:
        raise ValueError("At least one entry must be observed")
    if strategy == "global_mean":
        result[~mask] = global_mean
    elif strategy == "temporal_mean":
        # Per-pixel temporal mean is stronger than a global fill and is defined
        # wherever a pixel has one observed time point; otherwise use global mean.
        counts = mask.sum(axis=2, keepdims=True)
        sums = observed.sum(axis=2, keepdims=True)
        pixel_mean = np.divide(sums, counts, out=np.full_like(sums, global_mean), where=counts > 0)
        result[~mask] = np.broadcast_to(pixel_mean, result.shape)[~mask]
    else:
        raise ValueError(f"Unknown initialization strategy: {strategy}")
    return result


def complete_tensor(
    observed: np.ndarray,
    observed_mask: np.ndarray,
    method: Method,
    rank: int | tuple[int, int, int],
    max_iterations: int = 30,
    tolerance: float = 1e-4,
    initialization: str = "temporal_mean",
    ridge: float = 1e-6,
    seed: int = 31,
) -> CompletionResult:
    r"""Alternate low-rank projection and data-consistency projection.

    At iteration k, factorize the filled tensor, reconstruct \hat X_k, then set
    X_{k+1}=P_Ω(Y)+P_{Ω^c}(\hat X_k). Convergence is the relative change only
    on unknown entries. Observations are never overwritten by predictions.
    """
    if observed.shape != observed_mask.shape or observed.ndim != 3:
        raise ValueError("observed and observed_mask must be matching order-3 arrays")
    if observed_mask.all():
        raise ValueError("Nothing is missing; completion is not needed")
    current = initialize(observed, observed_mask, initialization)
    missing = ~observed_mask
    history: list[float] = []
    model: CPModel | TuckerModel | None = None
    for iteration in range(1, max_iterations + 1):
        if method == "cp":
            if not isinstance(rank, int):
                raise ValueError("CP requires an integer rank")
            model = cp_als(current, rank, max_iterations=10, ridge=ridge, seed=seed + iteration)
            prediction = reconstruct_cp(model)
        elif method == "tucker":
            if isinstance(rank, int):
                rank = (rank, rank, rank)
            model = tucker_hosvd(current, rank)
            prediction = reconstruct_tucker(model)
        else:
            raise ValueError(f"Unknown completion method: {method}")
        next_tensor = observed.copy()
        next_tensor[missing] = prediction[missing]
        denominator = np.linalg.norm(current[missing]) + 1e-12
        change = float(np.linalg.norm(next_tensor[missing] - current[missing]) / denominator)
        history.append(change)
        current = next_tensor
        if change < tolerance:
            break
    assert model is not None
    return CompletionResult(np.clip(current, 0.0, 1.0), history, iteration, model)
