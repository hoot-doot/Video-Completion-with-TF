"""Publication-ready diagnostics for reconstruction experiments."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def frame_comparison(truth: np.ndarray, observed: np.ndarray, estimate: np.ndarray, frame: int, path: str | Path) -> None:
    """Save original, corrupted, reconstruction, and absolute-error panels."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    images = [truth[:, :, frame], observed[:, :, frame], estimate[:, :, frame], np.abs(truth[:, :, frame] - estimate[:, :, frame])]
    titles = ["Original", "Corrupted", "Reconstructed", "Absolute error"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, image, title in zip(axes, images, titles):
        plot = ax.imshow(image, cmap="magma" if title == "Absolute error" else "gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
        if title == "Absolute error":
            fig.colorbar(plot, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def convergence_plot(history: list[float], path: str | Path) -> None:
    if not history:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 3.5))
    plt.semilogy(range(1, len(history) + 1), history, marker="o")
    plt.xlabel("Outer completion iteration")
    plt.ylabel("Relative change on missing entries")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(destination, dpi=160)
    plt.close()


def summary_plots(results: pd.DataFrame, output_dir: str | Path) -> None:
    """Plot PSNR vs missingness, RMSE vs rank, and runtime vs rank when present."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for metric, x, filename, ylabel in [
        ("psnr", "missing_fraction", "psnr_vs_missing.png", "PSNR (dB)"),
        ("rmse", "rank", "rmse_vs_rank.png", "RMSE"),
        ("runtime_seconds", "rank", "runtime_vs_rank.png", "Runtime (s)"),
    ]:
        subset = results.dropna(subset=[x, metric])
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        for method, group in subset.groupby("method"):
            group = group.sort_values(x)
            ax.plot(group[x], group[metric], marker="o", label=method)
        ax.set_xlabel(x.replace("_", " "))
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=160)
        plt.close(fig)

