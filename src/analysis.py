"""Evidence-bound narrative summaries for completed web-demo runs."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd


TENSOR_METHODS = {"CP-ALS", "Tucker/HOSVD"}


def describe_results(
    results: pd.DataFrame, tensor_shape: tuple[int, int, int], corruption: str,
    settings: dict[str, Any],
) -> str:
    """Write an interpretation only from measured metrics in ``results``.

    The text deliberately distinguishes a result for this clip/configuration
    from a general performance claim. It does not infer perceptual quality from
    PSNR alone.
    """
    h, w, t = tensor_shape
    ordered = results.sort_values("rmse", na_position="last")
    best = ordered.iloc[0]
    lines = [
        "### Measured interpretation",
        f"The uploaded clip was evaluated as a grayscale tensor of shape **{h} × {w} × {t}** under **{corruption}**. "
        f"Scores use only the intentionally hidden or untrusted entries; observed pixels were not scored.",
        f"For this configuration, **{best['method']}** had the lowest hidden-region RMSE "
        f"({best['rmse']:.5f}) and PSNR ({best['psnr']:.2f} dB).",
    ]
    tensors = results[results["method"].isin(TENSOR_METHODS)]
    baselines = results[~results["method"].isin(TENSOR_METHODS)]
    if not tensors.empty and not baselines.empty:
        tensor_best = tensors.loc[tensors["rmse"].idxmin()]
        baseline_best = baselines.loc[baselines["rmse"].idxmin()]
        improvement = 100 * (baseline_best["rmse"] - tensor_best["rmse"]) / max(baseline_best["rmse"], 1e-12)
        if improvement > 0:
            lines.append(
                f"The strongest tensor method ({tensor_best['method']}) reduced RMSE by **{improvement:.1f}%** "
                f"relative to the strongest selected baseline ({baseline_best['method']}) for this run."
            )
        else:
            lines.append(
                f"The strongest selected baseline ({baseline_best['method']}) outperformed the best tensor method "
                f"({tensor_best['method']}) by **{abs(improvement):.1f}%** in RMSE. For smooth short gaps, this can "
                "be a useful result rather than a failure: simple temporal interpolation may match the motion better."
            )
    if corruption.startswith("Missing consecutive"):
        lines.append(
            "A consecutive-frame gap is harder than scattered pixel loss because no direct measurement anchors the hidden times. "
            "Inspect the reconstructed MP4 for ghosting, blur, and flicker in addition to the frame metrics."
        )
    elif corruption.startswith("Central"):
        lines.append(
            "A spatial block tests whether the model can borrow information from both neighbouring pixels and adjacent frames; "
            "the error map reveals whether it primarily smooths the hole or restores structure."
        )
    else:
        lines.append(
            "Randomly distributed loss usually leaves local temporal and spatial anchors. A low-rank advantage here supports, "
            "but does not prove universally, the assumed spatiotemporal redundancy."
        )
    temporal = results.dropna(subset=["temporal_difference_mae"])
    if not temporal.empty:
        smoothest = temporal.loc[temporal["temporal_difference_mae"].idxmin()]
        value = smoothest["temporal_difference_mae"]
        if math.isfinite(value):
            lines.append(
                f"**{smoothest['method']}** had the lowest frame-difference error ({value:.5f}) among the selected methods. "
                "This is a temporal-consistency proxy, not a complete perceptual assessment."
            )
    lines += [
        "The factorization was fitted directly to this video; no pretrained neural model or external training set was used. "
        f"Completion used at most {settings['max_iterations']} outer iterations and tolerance {settings['tolerance']}. "
        "Rank and resolution trade accuracy against runtime and memory.",
        "Do not generalize this single run to all content. Abrupt cuts, fast motion, occlusion, camera shake, and fine textures "
        "can violate the low-rank assumption even when RMSE/PSNR look favorable.",
    ]
    return "\n\n".join(lines)
