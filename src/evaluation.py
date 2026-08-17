"""Ground-truth metrics computed exclusively over the intended test region."""
from __future__ import annotations

import numpy as np

try:
    from skimage.metrics import structural_similarity
except ImportError:  # Keep core metrics usable with only NumPy.
    structural_similarity = None


def metrics(truth: np.ndarray, estimate: np.ndarray, evaluation_mask: np.ndarray) -> dict[str, float]:
    """MSE, RMSE, MAE, PSNR, and optional mean frame SSIM on missing entries."""
    if not evaluation_mask.any():
        raise ValueError("evaluation_mask has no test entries")
    error = truth[evaluation_mask] - estimate[evaluation_mask]
    mse = float(np.mean(error**2))
    output = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
        "psnr": float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse)),
    }
    if structural_similarity is not None:
        values = []
        for t in range(truth.shape[2]):
            if evaluation_mask[:, :, t].any() and min(truth.shape[:2]) >= 7:
                values.append(structural_similarity(truth[:, :, t], estimate[:, :, t], data_range=1.0))
        output["ssim_full_frame_mean"] = float(np.mean(values)) if values else float("nan")
    return output


def temporal_difference_error(truth: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> float:
    """Mean absolute error of frame-to-frame differences near evaluated pixels."""
    if truth.shape[2] < 2:
        return float("nan")
    support = mask[:, :, 1:] | mask[:, :, :-1]
    delta_error = np.abs(np.diff(truth, axis=2) - np.diff(estimate, axis=2))
    return float(delta_error[support].mean()) if support.any() else float("nan")

