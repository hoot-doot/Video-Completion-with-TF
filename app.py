"""Streamlit interface for the tensor video-completion project.

Launch with: streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path
import base64
import html
import shutil
import time
import tracemalloc
import uuid

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.analysis import describe_results
from src.baselines import run_baseline
from src.config import set_seed
from src.corruption import central_blocks, missing_frames, random_missing, salt_pepper_as_missing
from src.evaluation import metrics, temporal_difference_error
from src.tensor_completion import complete_tensor
from src.video_loader import load_video_tensor, write_grayscale_video
from src.visualization import convergence_plot, frame_comparison


st.set_page_config(page_title="Tensor Video Completion", page_icon="🎞️", layout="wide")
st.title("Tensor Video Completion Lab")
st.caption("Upload a video, hide controlled data, reconstruct it with low-rank tensor methods, and compare measured results.")


def save_upload(uploaded) -> Path:
    """Persist only an allowed upload in a unique, local result directory."""
    suffix = Path(uploaded.name).suffix.lower()
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if suffix not in allowed:
        raise ValueError(f"Unsupported file type {suffix!r}. Use: {', '.join(sorted(allowed))}")
    job = Path("outputs/web_demo") / uuid.uuid4().hex
    job.mkdir(parents=True, exist_ok=False)
    target = job / f"input{suffix}"
    target.write_bytes(uploaded.getvalue())
    return target


def select_corruption(tensor, label: str, amount: float, gap: int, seed: int):
    rng = set_seed(seed)
    if label == "Random missing pixels":
        return random_missing(tensor, amount, rng)
    if label == "Central missing block":
        return central_blocks(tensor, amount)
    if label == "Missing consecutive frames":
        return missing_frames(tensor, min(gap, tensor.shape[2] - 1))
    return salt_pepper_as_missing(tensor, amount, rng)


def run_method(name: str, corruption, options: dict, seed: int):
    """Run a selected method and return reconstruction plus traceability fields."""
    if name == "CP-ALS":
        fitted = complete_tensor(
            corruption.observed, corruption.observed_mask, "cp", options["cp_rank"],
            options["max_iterations"], options["tolerance"], "temporal_mean", options["ridge"], seed,
        )
        return fitted.tensor, fitted.history, fitted.iterations, f"CP rank {options['cp_rank']}"
    if name == "Tucker/HOSVD":
        ranks = (options["tucker_spatial_rank"], options["tucker_spatial_rank"], options["tucker_temporal_rank"])
        fitted = complete_tensor(
            corruption.observed, corruption.observed_mask, "tucker", ranks,
            options["max_iterations"], options["tolerance"], "temporal_mean", options["ridge"], seed,
        )
        return fitted.tensor, fitted.history, fitted.iterations, f"Tucker rank {ranks}"
    baseline_names = {
        "Previous-frame copy": "previous", "Next-frame copy": "next",
        "Temporal interpolation": "temporal_linear", "Spatial inpainting": "spatial",
    }
    return run_baseline(baseline_names[name], corruption.observed, corruption.observed_mask), [], 0, "baseline"


def timed_method(name: str, corruption, options: dict, seed: int):
    """Measure a single reconstruction consistently for main and sweep tables."""
    tracemalloc.start()
    started = time.perf_counter()
    estimate, history, iterations, rank_text = run_method(name, corruption, options, seed)
    runtime_seconds = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return estimate, history, iterations, rank_text, runtime_seconds, peak_bytes / (1024 * 1024)


def score_method(name: str, tensor, corruption, options: dict, seed: int):
    estimate, history, iterations, rank_text, runtime_seconds, peak_memory_mb = timed_method(name, corruption, options, seed)
    score = metrics(tensor, estimate, corruption.evaluation_mask)
    score["temporal_difference_mae"] = temporal_difference_error(tensor, estimate, corruption.evaluation_mask)
    row = {
        "method": name, "rank / model": rank_text,
        "missing_fraction": float(corruption.evaluation_mask.mean()), "iterations": iterations,
        "runtime_seconds": runtime_seconds, "peak_python_memory_mb": peak_memory_mb, **score,
    }
    return row, estimate, history


def run_sensitivity_sweeps(tensor, corruption_label: str, amount: float, gap: int, options: dict, method: str,
                           ranks: list[int], amounts: list[float], gaps: list[int], seed: int):
    """Measure rank and corruption severity sensitivity without fabricating curves."""
    rank_rows, severity_rows = [], []
    for position, rank in enumerate(ranks):
        local = dict(options)
        if method == "CP-ALS":
            local["cp_rank"] = rank
        else:
            local["tucker_spatial_rank"] = rank
            local["tucker_temporal_rank"] = min(rank, tensor.shape[2])
        row, _, _ = score_method(method, tensor, corruption, local, seed + position + 101)
        row["rank"] = rank
        rank_rows.append(row)
    values = gaps if corruption_label == "Missing consecutive frames" else amounts
    for position, value in enumerate(values):
        varied = select_corruption(
            tensor, corruption_label, amount if corruption_label == "Missing consecutive frames" else value,
            value if corruption_label == "Missing consecutive frames" else gap, seed + position + 501,
        )
        row, _, _ = score_method(method, tensor, varied, options, seed + position + 701)
        row["sweep_value"] = value
        row["severity_label"] = "temporal gap (frames)" if corruption_label == "Missing consecutive frames" else "missing / corrupted fraction"
        severity_rows.append(row)
    return pd.DataFrame(rank_rows), pd.DataFrame(severity_rows)


def synchronized_video_player(videos: dict[str, Path]) -> None:
    """Embed original/corrupted/reconstruction videos with shared transport controls.

    H.264 MP4 bytes are embedded so the Streamlit component iframe has no
    dependency on a local filesystem URL or an unsupported source container.
    """
    tiles = []
    for name, path in videos.items():
        if not path.exists():
            continue
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        tiles.append(
            f"<figure><figcaption>{html.escape(name)}</figcaption>"
            f"<video muted playsinline preload='metadata' src='data:video/mp4;base64,{payload}'></video></figure>"
        )
    if not tiles:
        st.warning("No reconstructed videos are available for synchronized playback.")
        return
    markup = """
    <style>
      body { margin: 0; font-family: sans-serif; background: #10131a; color: #e8edf5; }
      #sync-root { padding: 12px; }
      #controls { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; flex-wrap: wrap; }
      button { background: #2b6de0; color: white; border: 0; border-radius: 4px; padding: 8px 12px; cursor: pointer; }
      input[type=range] { flex: 1; min-width: 180px; }
      #clock { min-width: 80px; font-variant-numeric: tabular-nums; }
      #grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }
      figure { margin: 0; } figcaption { margin: 0 0 5px; font-weight: 600; }
      video { width: 100%; background: #000; max-height: 250px; }
    </style>
    <div id='sync-root'>
      <div id='controls'>
        <button id='play' type='button'>Play all</button><button id='pause' type='button'>Pause all</button>
        <input id='seek' type='range' min='0' value='0' step='0.01' aria-label='Synchronized timeline'>
        <span id='clock'>0.00 s</span>
      </div>
      <div id='grid'>""" + "".join(tiles) + """</div>
    </div>
    <script>
      const root = document.getElementById('sync-root');
      const videos = Array.from(root.querySelectorAll('video'));
      const seek = root.querySelector('#seek'); const clock = root.querySelector('#clock');
      let syncing = false;
      const duration = () => Math.min(...videos.map(v => Number.isFinite(v.duration) ? v.duration : Infinity));
      const format = value => `${value.toFixed(2)} s`;
      function setAllTime(value) { syncing = true; videos.forEach(v => { v.currentTime = Math.min(value, v.duration || value); }); syncing = false; }
      function refreshDuration() { const d = duration(); if (Number.isFinite(d)) seek.max = d; }
      videos.forEach((video, index) => {
        video.addEventListener('loadedmetadata', refreshDuration);
        video.addEventListener('timeupdate', () => {
          if (!syncing && index === 0) { seek.value = video.currentTime; clock.textContent = format(video.currentTime); }
        });
        video.addEventListener('seeking', () => { if (!syncing) setAllTime(video.currentTime); });
      });
      root.querySelector('#play').addEventListener('click', () => videos.forEach(v => v.play().catch(() => {})));
      root.querySelector('#pause').addEventListener('click', () => videos.forEach(v => v.pause()));
      seek.addEventListener('input', () => { const value = Number(seek.value); setAllTime(value); clock.textContent = format(value); });
    </script>
    """
    components.html(markup, height=440, scrolling=True)


with st.sidebar:
    st.header("Experiment controls")
    uploaded = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv", "webm"])
    width = st.slider("Resize width / height", 64, 256, 128, 32)
    max_frames = st.slider("Maximum sampled frames", 20, 160, 60, 10)
    stride = st.slider("Frame stride", 1, 8, 1)
    corruption_label = st.selectbox("Controlled corruption", [
        "Random missing pixels", "Central missing block", "Missing consecutive frames", "Salt-and-pepper corruption",
    ])
    amount = st.slider("Missing/corrupted fraction", 0.05, 0.90, 0.30, 0.05)
    gap = st.slider("Consecutive missing frames", 1, 20, 3)
    methods = st.multiselect(
        "Methods to compare",
        ["CP-ALS", "Tucker/HOSVD", "Temporal interpolation", "Previous-frame copy", "Next-frame copy", "Spatial inpainting"],
        default=["CP-ALS", "Tucker/HOSVD", "Temporal interpolation"],
    )
    st.subheader("Tensor settings")
    cp_rank = st.slider("CP rank", 1, 30, 10)
    tucker_spatial_rank = st.slider("Tucker spatial rank", 1, 40, 15)
    tucker_temporal_rank = st.slider("Tucker temporal rank", 1, 30, 8)
    max_iterations = st.slider("Maximum completion iterations", 1, 60, 20)
    tolerance = st.select_slider("Convergence tolerance", options=[1e-2, 1e-3, 1e-4, 1e-5], value=1e-4)
    seed = st.number_input("Random seed", 0, 1_000_000, 31, 1)
    st.subheader("Empirical sensitivity (optional)")
    sensitivity_enabled = st.checkbox("Run rank and severity sensitivity sweeps", value=False)
    if sensitivity_enabled:
        sensitivity_method = st.selectbox("Sensitivity factorization", ["CP-ALS", "Tucker/HOSVD"])
        sensitivity_ranks = st.multiselect("Ranks to benchmark", [2, 5, 10, 15, 20, 30], default=[5, 10, 20])
        sensitivity_amounts = st.multiselect("Fractions to benchmark", [0.10, 0.30, 0.50, 0.70], default=[0.10, 0.30, 0.50])
        sensitivity_gaps = st.multiselect("Frame gaps to benchmark", [1, 3, 5, 10], default=[1, 3, 5])
        st.caption("Runs extra completion jobs; start with three values on a small video.")
    run_button = st.button("Run completion experiment", type="primary", use_container_width=True)

st.info(
    "This classical tensor-completion workflow fits factors to the uploaded clip itself. "
    "It does not require neural-network training or a separate training video collection."
)

if run_button:
    if uploaded is None:
        st.error("Choose a video file before running an experiment.")
        st.stop()
    if not methods:
        st.error("Select at least one reconstruction method.")
        st.stop()
    if uploaded.size > 300 * 1024 * 1024:
        st.error("For local CPU use, please upload a video smaller than 300 MB.")
        st.stop()
    source = save_upload(uploaded)
    job_dir = source.parent
    options = {
        "cp_rank": cp_rank, "tucker_spatial_rank": tucker_spatial_rank,
        "tucker_temporal_rank": tucker_temporal_rank, "max_iterations": max_iterations,
        "tolerance": tolerance, "ridge": 1e-6,
    }
    try:
        with st.spinner("Extracting frames and building the H × W × T grayscale tensor…"):
            tensor = load_video_tensor(source, (width, width), max_frames, stride, grayscale=True)
        if tensor.shape[2] < 2:
            raise ValueError("The selected settings produced fewer than two frames.")
        corruption = select_corruption(tensor, corruption_label, amount, gap, int(seed))
        actual_missing = float(corruption.evaluation_mask.mean())
        st.success(f"Tensor ready: shape {tensor.shape}; evaluated missing/untrusted support: {actual_missing:.1%}.")
        rows, reconstructions = [], {}
        progress = st.progress(0, text="Preparing methods…")
        for index, method in enumerate(methods, start=1):
            with st.spinner(f"Running {method}…"):
                row, estimate, history = score_method(method, tensor, corruption, options, int(seed) + index)
            rows.append(row)
            reconstructions[method] = (estimate, history)
            progress.progress(index / len(methods), text=f"Completed {method}")
        progress.empty()
        results = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
        sensitivity = None
        if sensitivity_enabled:
            if not sensitivity_ranks:
                raise ValueError("Select at least one sensitivity rank.")
            if corruption_label == "Missing consecutive frames" and not sensitivity_gaps:
                raise ValueError("Select at least one temporal gap for sensitivity analysis.")
            if corruption_label != "Missing consecutive frames" and not sensitivity_amounts:
                raise ValueError("Select at least one missing/corrupted fraction for sensitivity analysis.")
            with st.spinner("Running empirical rank and severity sensitivity benchmarks…"):
                sensitivity = run_sensitivity_sweeps(
                    tensor, corruption_label, amount, gap, options, sensitivity_method,
                    sensitivity_ranks, sensitivity_amounts, sensitivity_gaps, int(seed),
                )
        st.session_state["web_results"] = (tensor, corruption, results, reconstructions, options, job_dir, corruption_label, sensitivity)
    except Exception as error:
        # Remove an incomplete job directory; user-uploaded source is not retained on failure.
        shutil.rmtree(job_dir, ignore_errors=True)
        st.exception(error)

if "web_results" in st.session_state:
    stored_results = st.session_state["web_results"]
    # Keep a session created before this feature update usable after hot reload.
    if len(stored_results) == 7:
        tensor, corruption, results, reconstructions, options, job_dir, corruption_label = stored_results
        sensitivity = None
    else:
        tensor, corruption, results, reconstructions, options, job_dir, corruption_label, sensitivity = stored_results
    st.header("Measured results")
    st.dataframe(
        results.style.format({
            "missing_fraction": "{:.1%}", "mse": "{:.6f}", "rmse": "{:.5f}", "mae": "{:.5f}",
            "psnr": "{:.2f}", "ssim_full_frame_mean": "{:.4f}", "temporal_difference_mae": "{:.5f}",
            "runtime_seconds": "{:.2f}", "peak_python_memory_mb": "{:.2f}",
        }),
        use_container_width=True,
    )
    st.download_button("Download measured results CSV", results.to_csv(index=False).encode(), "tensor_completion_results.csv", "text/csv")
    st.markdown(describe_results(results, tensor.shape, corruption_label, options))

    st.subheader("Empirical benchmark charts")
    benchmark_columns = st.columns(3)
    with benchmark_columns[0]:
        st.caption("Hidden-region RMSE (lower is better)")
        st.bar_chart(results.set_index("method")["rmse"])
    with benchmark_columns[1]:
        st.caption("PSNR in dB (higher is better)")
        st.bar_chart(results.set_index("method")["psnr"])
    with benchmark_columns[2]:
        st.caption("Measured CPU runtime in seconds")
        st.bar_chart(results.set_index("method")["runtime_seconds"])

    if sensitivity is not None:
        rank_sensitivity, severity_sensitivity = sensitivity
        st.subheader("Sensitivity analysis")
        left, right = st.columns(2)
        with left:
            st.caption("Rank sensitivity: RMSE (lower is better)")
            st.line_chart(rank_sensitivity.set_index("rank")["rmse"])
            st.caption("Rank sensitivity: runtime in seconds")
            st.line_chart(rank_sensitivity.set_index("rank")["runtime_seconds"])
            st.dataframe(rank_sensitivity, use_container_width=True)
        with right:
            axis = "sweep_value"
            axis_caption = severity_sensitivity["severity_label"].iloc[0]
            st.caption(f"Severity sensitivity ({axis_caption}): RMSE")
            st.line_chart(severity_sensitivity.set_index(axis)["rmse"])
            st.caption(f"Severity sensitivity ({axis_caption}): runtime in seconds")
            st.line_chart(severity_sensitivity.set_index(axis)["runtime_seconds"])
            st.dataframe(severity_sensitivity, use_container_width=True)
        combined = pd.concat([rank_sensitivity.assign(sweep="rank"), severity_sensitivity.assign(sweep="severity")], ignore_index=True)
        st.download_button("Download sensitivity benchmark CSV", combined.to_csv(index=False).encode(), "tensor_sensitivity.csv", "text/csv")

    st.header("Visual comparison")
    missing_times = corruption.evaluation_mask.any(axis=(0, 1)).nonzero()[0]
    representative = int(missing_times[len(missing_times) // 2]) if len(missing_times) else tensor.shape[2] // 2
    tabs = st.tabs(list(reconstructions))
    for tab, (method, (estimate, history)) in zip(tabs, reconstructions.items()):
        with tab:
            image_path = job_dir / f"{method.replace('/', '_').replace(' ', '_')}_comparison.png"
            frame_comparison(tensor, corruption.observed, estimate, representative, image_path)
            st.image(str(image_path), caption=f"Frame {representative}: original, corrupted, reconstruction, absolute error")
            video_path = job_dir / f"{method.replace('/', '_').replace(' ', '_')}_reconstruction.mp4"
            write_grayscale_video(estimate, video_path)
            st.video(str(video_path))
            if history:
                convergence_path = job_dir / f"{method.replace('/', '_').replace(' ', '_')}_convergence.png"
                convergence_plot(history, convergence_path)
                st.image(str(convergence_path), caption="Relative missing-region update per outer iteration")
            else:
                st.caption("This baseline has no iterative factorization convergence trace.")
    st.subheader("Synchronized video comparison")
    original_video = job_dir / "original_h264.mp4"
    corrupted_video = job_dir / "corrupted_h264.mp4"
    write_grayscale_video(tensor, original_video)
    write_grayscale_video(corruption.observed, corrupted_video)
    synchronized_sources = {"Original": original_video, "Corrupted": corrupted_video}
    for method, (estimate, _) in reconstructions.items():
        synchronized_sources[method] = job_dir / f"{method.replace('/', '_').replace(' ', '_')}_reconstruction.mp4"
    synchronized_video_player(synchronized_sources)
    with st.expander("Method and interpretation notes"):
        st.markdown(
            "**CP-ALS** represents the tensor as a sum of rank-one row × column × time components. "
            "**Tucker/HOSVD** learns separate row, column, and time subspaces plus a small interaction core. "
            "The completion loop preserves every trusted observation and replaces only the missing/untrusted support.\n\n"
            "PSNR/RMSE/MAE quantify agreement with known ground truth that was deliberately hidden. SSIM and "
            "frame-difference error add context, but inspect the reconstructed video for temporal artifacts."
        )
