# Project 31 — Video Completion Using Tensor Factorization

This project reconstructs intentionally missing video samples by exploiting a low-rank, spatiotemporal tensor model. It is designed as a reproducible master's-level experiment, not as a claim that low-rank completion always wins: every tensor result is compared against simple interpolation/inpainting baselines and scored only where ground truth was hidden.

## 1. Mathematical formulation

After grayscale preprocessing, a video becomes the third-order tensor

\[
\mathcal X\in\mathbb R^{H\times W\times T},\qquad
\mathcal X_{ijt}=\text{intensity at row }i,\text{ column }j,\text{ time }t.
\]

`src.video_loader` scales intensities to `[0,1]` and stacks frames on the last axis, so an 80-frame 128 x 128 clip has shape `(128, 128, 80)`. An RGB video can instead be represented by \(\mathcal X\in\mathbb R^{H\times W\times T\times3}\), but the main runner deliberately uses grayscale to retain the central third-order model. Extending Tucker to RGB means adding a fourth factor; a CP extension only requires generalizing the outer product and Khatri–Rao updates.

Let \(\Omega\) be the trusted, observed indices and define the binary mask \(\mathcal M\) by \(M_{ijt}=1\) for \((i,j,t)\in\Omega\). The incomplete measurement is

\[
\mathcal Y=P_\Omega(\mathcal X)=\mathcal M\odot\mathcal X.
\]

Completion estimates \(\widehat{\mathcal X}\) with \(P_\Omega(\widehat{\mathcal X})\approx\mathcal Y\) while constraining its multilinear structure to be low rank. Unlike matrix completion of an `(H*W) x T` frame stack, this preserves row, column, and time as separate modes, allowing independent spatial factors along height and width and a temporal factor along frames.

## 2. Implemented models

### CP/PARAFAC with explicit ALS

The CP model is

\[
\mathcal X\approx\sum_{r=1}^{R}\lambda_r\,a_r\circ b_r\circ c_r.
\]

Here `a_r`, `b_r`, and `c_r` respectively encode vertical, horizontal, and temporal patterns. `src/cp_factorization.py` implements alternating least squares rather than calling a black-box decomposition. With the other factors fixed, the mode-`n` update solves the ridge-stabilized least-squares system

\[
A_n=X_{(n)} Z\,(Z^\top Z+\epsilon I)^{-1},
\]

where \(Z\) is the Khatri–Rao product of the two other factors. Columns are normalized at the end and their scale is absorbed into \(\lambda\). CP uses one rank `R`; it is compact but can be ill-conditioned and may need more iterations for complex motion.

### Tucker/HOSVD

The Tucker model is

\[
\mathcal X\approx\mathcal G\times_1 A\times_2 B\times_3 C.
\]

`src/tucker_factorization.py` computes a truncated HOSVD: the leading left singular vectors of each unfolding form `A`, `B`, and `C`, then projects the tensor to its core `G`. Its rank is multilinear `(r_H, r_W, r_T)`, not the single CP component count. Tucker is often more flexible because its core allows interactions between components, at the cost of storing that core.

### Projected iterative completion

Both decompositions are used inside the same completion loop (`src/tensor_completion.py`):

```text
X0 <- fill unobserved entries (per-pixel temporal mean)
for k = 0, ..., K-1:
    fit low-rank CP or Tucker model to Xk
    Zk <- reconstruct fitted model
    Xk+1 <- PΩ(Y) + PΩᶜ(Zk)
    stop if ||PΩᶜ(Xk+1 - Xk)|| / ||PΩᶜ(Xk)|| < tolerance
```

The data-consistency update is essential: trusted pixels are copied from `Y` on every iteration and are never replaced by model predictions. The default maximum is 30 outer iterations with relative-change tolerance `1e-4`; CP uses ten internal ALS sweeps per outer iteration. The `temporal_mean` start is a reasonable low-bias initialization for video; compare it with `global_mean` or `zero` as an ablation.

An advanced extension is tensor nuclear-norm regularization,
\(\min_\mathcal X L(\mathcal X,\mathcal Y)+\lambda R(\mathcal X)\), where the loss imposes observed-data agreement and \(R\) penalizes rank surrogates of tensor unfoldings. It is discussed here but not presented as an implemented result because the choice of tensor nuclear norm and solver materially changes the method.

## 3. Data and controlled corruptions

Use a real video. UCF101 is the suggested primary source ([dataset page](https://www.crcv.ucf.edu/data/UCF101.php)); DAVIS is an excellent cleaner alternative. Download a modest-motion clip legitimately under the dataset terms and place it at `data/original/example.mp4`, or pass its path with `--video`. A 128 x 128, 50–100 frame grayscale sample is a good CPU pilot: resizing reduces the cost of unfolding/SVD, and frame limiting keeps experiments repeatable.

`src/corruption.py` preserves the original tensor and returns an observation mask plus a distinct evaluation mask:

- `random`: independent missing pixels at 10%, 30%, 50%, or 70%.
- `block`: a central spatial hole in each frame, testing structured inpainting.
- `frames`: one contiguous interior temporal gap (1, 3, 5, or 10 frames).
- `salt_pepper`: corrupted pixels are explicitly marked untrusted/missing for controlled recovery. In a realistic unknown-noise setting, robust tensor PCA or joint noise/mask estimation would be required.

Missing data means no reliable measurement; noisy/corrupted data initially contains a wrong measurement. Supplying the corruption support makes this a completion experiment rather than silently treating wrong values as ground truth.

## 4. Baselines and evaluation

The runner includes previous-frame copy, next-frame copy, per-pixel temporal linear interpolation, and independent-frame OpenCV Telea spatial inpainting. These are intentionally simple but important: a tensor method must beat them to justify its compute. Temporal interpolation can be very strong for a short, smooth gap; it may win over low rank when motion is simple.

Metrics are computed only on `evaluation_mask = Ωᶜ`, never on known pixels:

\[
\mathrm{MSE}=N^{-1}\sum_{\Omega^c}(X-\hat X)^2,\quad
\mathrm{RMSE}=\sqrt{\mathrm{MSE}},\quad
\mathrm{MAE}=N^{-1}\sum_{\Omega^c}|X-\hat X|,
\]

\[
\mathrm{PSNR}=10\log_{10}(1/\mathrm{MSE}),
\]

since normalized intensity has maximum 1. The runner also records full-frame mean SSIM when `scikit-image` is installed, a frame-difference temporal-consistency error, runtime, Python traced peak memory, and convergence iterations. SSIM is shown for context, but pixel-level missing-region metrics are the primary completion score.

## 5. Run it

Requires Python 3.11+ (the pinned environment was selected for this range).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main --video data/original/example.mp4
```

### Local web demo

The browser interface accepts a local MP4/AVI/MOV/MKV/WebM file, converts it to the grayscale tensor internally, lets you choose a controlled corruption and factorization settings, then displays measured hidden-region metrics, reconstruction panels, convergence, reconstructed MP4 previews, CSV export, and an evidence-bound narrative analysis.

```bash
streamlit run app.py
```

Open the local address printed by Streamlit (normally `http://localhost:8501`). The app processes the uploaded clip locally and fits CP/Tucker factors to that clip; it does not upload the video to a training service and does not use a pretrained neural model. Keep the initial experiment modest (for example, 128 x 128 and 60 frames) because CPU tensor factorization scales quickly with dimensions and rank.

Each completed run also includes a **Synchronized video comparison** panel. Its shared play, pause, and timeline controls keep the H.264-encoded Original, Corrupted, and all selected reconstructions at the same frame time. Use this to spot flicker, delayed motion, blur, and ghosting that a frame-level score can hide.

Enable **Run rank and severity sensitivity sweeps** in the sidebar to produce empirical—rather than assumed—charts. The rank sweep holds the corruption fixed while varying factor rank; the severity sweep holds the selected model settings fixed while varying missing fraction (or consecutive frame-gap length). It records RMSE and runtime at each point and makes the full benchmark table downloadable. This option runs additional decompositions, so use a smaller resolution/frame count first.

To change ranks, corruption rates, temporal gaps, resolution, maximum frames, or the convergence rule, edit `experiments/configs/default.yaml`. A fixed seed is in the config. The resolved configuration is saved to `experiments/results/resolved_config.json` and the measured table to `experiments/results/results.csv`; do not substitute expected values for these measured outputs. After validating the pilot, use `--config experiments/configs/full_study.yaml` for the complete 10/30/50/70% and rank 5/10/20/30 ablation.

Outputs include per-experiment original/corrupted/reconstructed/error panels, convergence curves, aggregate PSNR/RMSE/runtime plots when applicable, and optional reconstructed MP4 files. Set `save_video: false` for faster parameter sweeps.

## 6. Experimental protocol and research questions

Begin with a pilot (128 x 128 x 80), then run the matrix below on two real clips: one relatively static and one with stronger motion/camera movement. Keep the same preprocessing and seed within a comparison.

| Study | Sweep | What it answers |
|---|---|---|
| Missingness | random 10/30/50/70% | Does quality degrade gracefully? |
| Rank | CP 5/10/20/30; Tucker `(10,10,5)` upward | Under/over-smoothing and rank trade-off |
| Gap | 1/3/5/10 entire frames | How far can temporal coherence bridge? |
| Method | CP, Tucker, all baselines | Is factorization actually better? |
| Structure | random pixels vs blocks vs frames | Which corruption geometry is difficult? |
| Initialization | temporal/global mean/zero | Is success initialization-sensitive? |

For each row in the final report, include the saved configuration, exact video source/clip range, mask seed, rank, iteration count, runtime, memory, and metrics. Plot a representative frame sequence as Original → Missing/Corrupted → Reconstruction, inspect the video for flicker/ghosting, and explain rather than merely list the numbers. No experimental numeric claim belongs in the report until `results.csv` exists.

## 7. Complexity, interpretation, and limitations

For tensor dimensions \(H\times W\times T\), a dense CP ALS sweep costs approximately \(O(RHWT+R^2(H+W+T))\) per mode update; the reconstruction also costs \(O(RHWT)\). HOSVD performs SVDs of the three unfoldings, with cost dominated by their dense matrix decompositions. Both become expensive with resolution, duration, or rank and keep dense tensors in memory. Practical mitigations are resizing, frame sampling, small pilot ranks, truncated/randomized SVD for a scalable extension, avoiding copies, and processing separate clips. The results table's memory field tracks Python allocations, not total OS/GPU memory.

Low rank works because nearby pixels, repeated backgrounds, and consecutive frames are correlated; a small number of spatial patterns and their temporal activations can explain much of a clip. It fails or becomes overly smooth for abrupt cuts, fast/nonlinear motion, occlusion, camera shake, fine texture, large missing temporal gaps, and poorly selected rank. Good RMSE or PSNR does not guarantee perceptual or temporal realism—check the rendered video for blurring, flicker, ghosting, and motion discontinuities.

## 8. Report structure

Use the following master’s-report outline: (1) Introduction and research questions; (2) tensor-completion formulation and matrix-versus-tensor rationale; (3) CP, Tucker, and projected-completion methodology; (4) dataset, preprocessing, corruption masks, baselines, and reproducibility; (5) quantitative tables and qualitative frames/videos; (6) ablations and runtime/scaling; (7) discussion of where baselines win and numerical versus perceptual quality; (8) limitations and ethical/dataset notes; (9) conclusion; (10) future work: robust tensor PCA, nonnegative and nuclear-norm methods, RGB/higher-order models, optical-flow/motion-aware priors, and neural video inpainting. A fill-in-ready scaffold is in `REPORT_TEMPLATE.md`.

## Project layout

```text
data/                         # real source clips and optional processed copies
experiments/configs/          # versionable YAML experiment specifications
experiments/results/          # generated CSV + resolved configuration
outputs/figures/              # comparisons, error maps, curves
outputs/reconstructed_videos/ # optional MP4 previews
src/video_loader.py           # frames -> normalized tensor -> video
src/corruption.py             # known masks and corruption scenarios
src/cp_factorization.py       # explicit CP-ALS
src/tucker_factorization.py   # explicit Tucker/HOSVD
src/tensor_completion.py      # PΩ data-consistent completion loop
src/baselines.py              # non-tensor comparisons
src/evaluation.py             # Ωᶜ-only metrics
src/visualization.py          # figures and result curves
src/main.py                   # reproducible runner
```
