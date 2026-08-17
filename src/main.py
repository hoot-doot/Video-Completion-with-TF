"""Command-line experiment runner for tensor video completion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import tracemalloc

import pandas as pd

from .baselines import run_baseline
from .config import load_config, set_seed
from .corruption import CorruptionResult, central_blocks, missing_frames, random_missing, salt_pepper_as_missing
from .evaluation import metrics, temporal_difference_error
from .tensor_completion import complete_tensor
from .video_loader import load_video_tensor, write_grayscale_video
from .visualization import convergence_plot, frame_comparison, summary_plots


def make_corruptions(tensor, config: dict, rng) -> list[CorruptionResult]:
    results: list[CorruptionResult] = []
    for specification in config["experiments"]:
        kind = specification["corruption"]
        if kind in {"random", "block", "salt_pepper"}:
            for fraction in specification.get("missing_fractions", []):
                if kind == "random":
                    results.append(random_missing(tensor, fraction, rng))
                elif kind == "block":
                    results.append(central_blocks(tensor, fraction))
                else:
                    results.append(salt_pepper_as_missing(tensor, fraction, rng))
        elif kind == "frames":
            for gap in specification.get("gap_lengths", []):
                results.append(missing_frames(tensor, gap))
        else:
            raise ValueError(f"Unsupported corruption type: {kind}")
    return results


def describe(corruption: CorruptionResult) -> tuple[float | None, int | None]:
    fraction = corruption.metadata.get("missing_fraction", corruption.metadata.get("requested_fraction"))
    return fraction, corruption.metadata.get("gap_length")


def execute_one(method: str, rank, corruption: CorruptionResult, completion: dict, seed: int):
    tracemalloc.start()
    started = time.perf_counter()
    history: list[float] = []
    iterations = 0
    if method in {"cp", "tucker"}:
        fitted = complete_tensor(
            corruption.observed, corruption.observed_mask, method, rank,
            max_iterations=completion["max_iterations"], tolerance=completion["tolerance"],
            initialization=completion["initialization"], ridge=completion["ridge"], seed=seed,
        )
        estimate, history, iterations = fitted.tensor, fitted.history, fitted.iterations
    else:
        estimate = run_baseline(method, corruption.observed, corruption.observed_mask)
    runtime = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return estimate, history, iterations, runtime, peak / (1024 * 1024)


def experiment_name(corruption: CorruptionResult, method: str, rank) -> str:
    fraction, gap = describe(corruption)
    suffix = f"p{fraction:.2f}" if fraction is not None else f"gap{gap}"
    rank_text = "x".join(map(str, rank)) if isinstance(rank, tuple) else str(rank)
    return f"{corruption.name}_{suffix}_{method}_r{rank_text}".replace(".", "_")


def run(config: dict, video_path: str | None = None, output_root: str | Path = "outputs") -> pd.DataFrame:
    """Run all configured experiments and write traceable tables/figures."""
    seed = int(config["seed"])
    rng = set_seed(seed)
    source = video_path or config["video_path"]
    pre = config["preprocessing"]
    tensor = load_video_tensor(source, tuple(pre["size"]), pre["max_frames"], pre["frame_stride"], pre["grayscale"])
    if tensor.ndim != 3:
        raise NotImplementedError("The experiment runner currently uses grayscale H x W x T tensors.")
    out = Path(output_root)
    figure_dir = out / "figures"
    video_dir = out / "reconstructed_videos"
    results_dir = Path("experiments/results")
    figure_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    rows: list[dict] = []
    for corruption_number, corruption in enumerate(make_corruptions(tensor, config, rng)):
        fraction, gap = describe(corruption)
        schedules: list[tuple[str, int | tuple[int, int, int] | None]] = []
        for method in config["methods"]:
            if method == "cp":
                schedules += [(method, int(rank)) for rank in config["cp_ranks"]]
            elif method == "tucker":
                schedules += [(method, tuple(rank)) for rank in config["tucker_ranks"]]
            else:
                schedules.append((method, None))
        for method, rank in schedules:
            estimate, history, iterations, runtime, peak_mb = execute_one(
                method, rank, corruption, config["completion"], seed + corruption_number
            )
            score = metrics(tensor, estimate, corruption.evaluation_mask)
            score["temporal_difference_mae"] = temporal_difference_error(tensor, estimate, corruption.evaluation_mask)
            label = experiment_name(corruption, method, rank)
            missing_frames_idx = corruption.evaluation_mask.any(axis=(0, 1)).nonzero()[0]
            display_frame = int(missing_frames_idx[len(missing_frames_idx) // 2]) if len(missing_frames_idx) else tensor.shape[2] // 2
            frame_comparison(tensor, corruption.observed, estimate, display_frame, figure_dir / f"{label}_comparison.png")
            convergence_plot(history, figure_dir / f"{label}_convergence.png")
            if config.get("save_video", False):
                write_grayscale_video(estimate, video_dir / f"{label}.mp4")
            rank_value = "x".join(map(str, rank)) if isinstance(rank, tuple) else rank
            rows.append({
                "method": method, "corruption": corruption.name,
                "missing_fraction": float(corruption.evaluation_mask.mean()),
                "requested_fraction": fraction, "gap_length": gap, "rank": rank_value, "iterations": iterations,
                "runtime_seconds": runtime, "peak_python_memory_mb": peak_mb, **score,
            })
            print(f"completed {label}: RMSE={score['rmse']:.5f}, PSNR={score['psnr']:.2f} dB")
    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "results.csv", index=False)
    summary_plots(table, figure_dir)
    return table


def cli() -> None:
    parser = argparse.ArgumentParser(description="Video completion using CP and Tucker tensor factorization")
    parser.add_argument("--config", default="experiments/configs/default.yaml")
    parser.add_argument("--video", help="Override video_path in the YAML config")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    results = run(load_config(args.config), args.video, args.output)
    print("\nResults written to experiments/results/results.csv")
    print(results.to_string(index=False))


if __name__ == "__main__":
    cli()
