# Robust Nonlinear Fitting with an Adaptive Smoothed Lf Loss and SVD-Based Rank Selection

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/msaran1923/lf-norm-robust-fitting/blob/main/LICENSE)

Replication code and results for the paper:

> **A Robust Nonlinear Fitting Framework with an Adaptive Smoothed $L_f$ Loss and SVD-Based Rank Selection**
> Murat Saran, submitted to *PeerJ Computer Science*, 2026.

This repository contains the complete implementation of the proposed solver, every script required to reproduce the tables and figures in the paper, and the precomputed result files the paper reports.

---

## Description

This project provides a Python implementation of a robust nonlinear regression framework based on a generalized smoothed **Lf loss** combined with **SVD-based rank selection**. For f ≥ 1 the objective coincides (up to a constant) with the f-th power of the ℓf norm; for f < 1 it is a quasi-norm-type robust loss rather than a true norm. The code is designed for two audiences:

1. **Reviewers and researchers** who want to reproduce the experimental results reported in the accompanying paper (all 27 NIST StRD benchmarks, contamination studies, synthetic-model studies, leverage-point studies, scaling tests, sensitivity and ablation analyses, and all 8 paper figures).
2. **Practitioners** who want to apply the Lf solver to their own nonlinear regression problems where outliers or leverage points make classical least-squares unreliable.

The solver minimizes

$$\mathcal{L}_f(\boldsymbol{\theta}) = \frac{1}{f}\sum_{i=1}^{n} \bigl(d_i^2 + \varepsilon\bigr)^{f/2}$$

via iteratively reweighted least squares (IRLS) embedded in a Levenberg–Marquardt outer loop. Each iteration's weighted Jacobian is factorized by SVD to enable low-rank step computation. Key features:

- **Adaptive robustness** — continuous interpolation from OLS (*f* = 2) through a smoothed LAD (*f* = 1) to aggressive outlier suppression (*f* < 1).
- **SVD-based rank selection** — dual criterion combining energy-based truncation with gradient alignment; the component ablation in the paper identifies rank truncation as the primary failure-prevention mechanism.
- **Convex-to-robust warm-start** — graduated annealing of *f* and *ε* that advances only on accepted steps.
- **No user-supplied scale parameter** — the joint *f*–*ε* annealing adapts the effective scale implicitly.
- **Dual-strategy polishing** — recovers near-OLS precision on clean data.
- **Residual-adaptive f-target selection** — automatically chooses the mildest exponent preserving robustness.

---

## Dataset Information

All experiments use the **27 NIST Statistical Reference Datasets (StRD) for Nonlinear Regression**, a publicly available, certified benchmark suite maintained by the U.S. National Institute of Standards and Technology.

- **Source:** https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml
- **Difficulty levels:** Lower (8 datasets), Average (11 datasets), Higher (8 datasets).
- **Content:** Each dataset provides observed (x, y) pairs, a parametric nonlinear model, two sets of starting values, and certified parameter estimates with standard deviations. The benchmarks in this repository use the NIST **Start 2** vectors, as recorded in the data loader.
- **License:** The NIST StRD benchmarks are U.S. Government work and are in the public domain.

### How the data is obtained

The datasets are **not redistributed** in this repository. They are fetched directly from NIST on first use:

- `nist_all_data.py` downloads the 27 datasets from NIST on first call and caches them locally in `nist_cache.json`. Subsequent runs read from the cache.
- `download_nist.py` is an alternative standalone downloader that produces a self-contained Python data module.

Contamination for the robustness experiments is generated **synthetically** on top of the clean NIST data at fixed random seeds (20 seeds per configuration). The contamination scheme (rates, magnitudes, one-sided vs symmetric, Gaussian vs uniform) is fully described in the paper and implemented in `run_benchmark.py` and `run_benchmark_expanded.py`.

Additional synthetic datasets are generated on the fly by model functions defined in `lf_norm.py` and controlled by the seeds documented in the corresponding scripts.

---

## Code Information

### Repository structure

```
lf-norm-robust-fitting/
├── lf_norm.py                    # Core solver library
├── nist_all_data.py              # NIST StRD dataset loader (auto-downloads & caches)
├── download_nist.py              # Standalone NIST data downloader
├── run_experiments.py            # Individual experiment runner (tables & figures)
├── run_benchmark.py              # Full 27-dataset benchmark + leverage experiment
├── run_benchmark_expanded.py     # Expanded analysis (contam. types, magnitudes, synthetic)
├── generate_figures.py           # Generate paper figures from CSV results
├── scale_sensitivity.py          # Baseline scale-parameter sensitivity (Sec. 4.1)
├── hyperparam_sensitivity.py     # f_target / tau / delta / annealing-rate sweeps (Sec. 4.4)
├── component_ablation.py         # Incremental C1–C7 component ablation (Sec. 4.3)
├── leverage_grid.py              # Expanded 3×3×3 leverage grid, two protocols (Sec. 4.6)
├── nelson_diagnostic.py          # Nelson failure-chain diagnostic (Sec. 4.2)
├── pruning_lasso_comparison.py   # Feature screener vs LassoCV/ElasticNetCV (Appendix B)
├── revision_analysis.py          # Summary statistics, bootstrap CIs, Wilcoxon tests (Table 2)
├── experiment_results/           # Distributed result CSVs and figure PDFs
│   └── deprecated/               # Superseded files kept for provenance (see its README)
├── requirements.txt
├── LICENSE
└── README.md
```

### File descriptions

| File | Purpose |
| --- | --- |
| `lf_norm.py` | Core solver implementation: `SolverConfig`, `solve_lf`, `solve_lf_adaptive`, IRLS weights, SVD rank selection, feature pruning, baseline solvers, synthetic model functions, and all 27 NIST model functions with analytical Jacobians |
| `nist_all_data.py` | Loads all 27 NIST StRD datasets. Downloads from NIST on first use and caches locally as `nist_cache.json` |
| `download_nist.py` | Alternative standalone downloader that generates a self-contained Python data module |
| `run_experiments.py` | Runs individual experiments (clean-data recovery, outlier robustness, ablation, feature pruning, leverage points) with inline figure generation |
| `run_benchmark.py` | Comprehensive 27-dataset benchmark: 9 methods × 27 datasets × 4 contamination levels × 20 seeds, plus the single-setting leverage experiment. In the leverage experiment, the scale-dependent baselines (Huber, Cauchy, Barron) are initialized at the OLS solution — standard practice for redescending M-estimators and the protocol stated in the paper |
| `run_benchmark_expanded.py` | Expanded analysis: contamination types, magnitudes, synthetic models, nonlinear leverage, scaling |
| `generate_figures.py` | Generates all 8 paper figures (PDF/PNG) from the benchmark CSV results |
| `scale_sensitivity.py` | Re-runs all scale-dependent baselines at {0.5, 1, 2, 4}× the two-phase MAD scale and at an oracle scale over all 1,620 contaminated NIST scenarios |
| `hyperparam_sensitivity.py` | One-at-a-time sensitivity of `f_target`, `tau`, `delta` (and annealing rates) on a representative NIST subset |
| `component_ablation.py` | Incremental ablation C1–C7 (direct solve → annealing → energy/dual truncation → polishing / adaptive-f / sign detection) over the full contaminated benchmark |
| `leverage_grid.py` | Leverage grid n ∈ {100, 200, 400} × p ∈ {5, 10, 20} × magnitude ∈ {4, 8, 12}× under the OLS-initialization protocol; a same-start variant is reported in the paper |
| `nelson_diagnostic.py` | Traces the Nelson failure chain: Jacobian conditioning, OLS drift, and the two-phase scale versus the certified residual scale |
| `pruning_lasso_comparison.py` | Compares the SVD feature screener against scikit-learn `LassoCV` / `ElasticNetCV` on the identical synthetic design (requires scikit-learn) |
| `revision_analysis.py` | Computes Table 2 of the paper from `benchmark_full.csv`: means with bootstrap CIs, medians, IQRs, failure rates, paired Wilcoxon signed-rank tests, per-dataset failure counts |

### Public API

The solver is exposed through a small public surface in `lf_norm.py`:

- `SolverConfig` — dataclass holding all solver hyperparameters.
- `solve_lf(x, y, theta0, model, jacobian, cfg)` — single-pass Lf solver at a fixed `f_target`.
- `solve_lf_adaptive(x, y, theta0, model, jacobian, cfg)` — residual-adaptive wrapper that selects `f_target` automatically.

Both functions return a dictionary containing the fitted parameters, residuals, weights, convergence flag, iteration count, selected `f`, and the rank history.

---

## Precomputed Results

Unlike a code-only release, this repository **distributes the result files the paper reports**, under `experiment_results/`: the main benchmark suite (`benchmark_full.csv` and its aggregates), the expanded-analysis files (`benchmark_B…F`), and the revision analyses (`scale_sensitivity.csv`, `hyperparam_sensitivity.csv`, `annealing_sensitivity.csv`, `component_ablation.csv`, `component_ablation_C1d.csv`, `leverage_grid.csv`, `leverage_grid_samestart.csv`, `leverage_original_regen.csv`, `ols_reproduction_check.csv`, `pruning_lasso_comparison.csv`), together with the regenerated figures 5 and 8.

`experiment_results/deprecated/` contains superseded files kept for provenance only — see the README inside that folder. Do not use them for analysis or figures.

### Reproduction protocol

Results are reproduced by running the pipeline scripts end-to-end (`run_benchmark.py`, `run_benchmark_expanded.py`, `leverage_grid.py`, `scale_sensitivity.py`, `component_ablation.py`, `hyperparam_sensitivity.py`), **not** by invoking individual solvers in isolation: the annealed Barron baseline is sensitive to floating-point context, so isolated single-solver calls on the same data can settle in a different local trajectory. Every pipeline script reproduces its distributed result CSVs exactly under the pinned environment **numpy 1.26.4 / scipy 1.12.0** (bit-exact for the deterministic solvers; a handful of already-diverged OLS trust-region runs may show harmless floating-point drift that alters no reported statistic).

---

## Requirements

- Python ≥ 3.11
- NumPy ≥ 1.26
- SciPy ≥ 1.12
- pandas ≥ 2.0
- matplotlib ≥ 3.8
- scikit-learn ≥ 1.3 (only for `pruning_lasso_comparison.py`)

All dependencies are listed in `requirements.txt`. For **bit-exact** reproduction of the distributed CSVs, use `numpy==1.26.4` and `scipy==1.12.0` (the versions used for the paper). The code is pure Python and does not require a GPU. It has been tested on Windows 11 and Ubuntu 22.04.

---

## Usage Instructions

### Installation

```bash
git clone https://github.com/msaran1923/lf-norm-robust-fitting.git
cd lf-norm-robust-fitting
pip install -r requirements.txt
```

### Quick verification (~5 minutes)

Run a sanity check with reduced trial counts:

```bash
python run_benchmark.py --quick --all27
```

This runs 5 trials (instead of 20) across all 27 datasets to verify the code works correctly.

### Full replication

Complete replication involves four stages. Total runtime depends on hardware; expect 2–5 hours on a modern CPU.

#### Stage 1 — Main benchmark (Tables 1–2, catastrophic failure rates, leverage table)

```bash
python run_benchmark.py --all27 --trials 20
```

**Outputs** in `experiment_results/`:

| File | Contents |
| --- | --- |
| `benchmark_full.csv` | Raw per-trial results (19,440 rows) |
| `benchmark_summary.csv` | Aggregated mean/std/median per dataset × method × contamination |
| `benchmark_wins.csv` | Strict win counts per method |
| `benchmark_catastrophic.csv` | Catastrophic (RE > 1) and severe (RE > 10) failure rates |
| `benchmark_ablation_rank.csv` | Lf\_dual vs Lf\_full comparison |
| `benchmark_leverage.csv` | Leverage-point robustness results (OLS-initialized baselines) |

#### Stage 2 — Expanded analysis

```bash
python run_benchmark_expanded.py --all27 --trials 20
```

Individual sections can also be run in isolation:

```bash
python run_benchmark_expanded.py --section B --all27   # Contamination types
python run_benchmark_expanded.py --section C --all27   # Contamination magnitudes
python run_benchmark_expanded.py --section D           # Synthetic models
python run_benchmark_expanded.py --section E           # Nonlinear leverage points
python run_benchmark_expanded.py --section F           # Scaling tests
```

**Outputs** in `experiment_results/`: `benchmark_B_contam_types.csv`, `benchmark_C_magnitudes.csv`, `benchmark_D_synthetic.csv`, `benchmark_E_nonlinear_leverage.csv`, `benchmark_F_scaling.csv`, `benchmark_grand_summary.csv`.

#### Stage 3 — Sensitivity, ablation, and diagnostic analyses

```bash
python revision_analysis.py experiment_results/benchmark_full.csv   # Table 2 statistics (~1 min)
python scale_sensitivity.py        # ~25 min; checkpointed, resumable
python component_ablation.py       # ~5 min; checkpointed, resumable
python hyperparam_sensitivity.py   # ~1 min
python leverage_grid.py            # ~3 min
python nelson_diagnostic.py        # seconds; prints the diagnostic
python pruning_lasso_comparison.py # ~1 min; requires scikit-learn
```

`scale_sensitivity.py` and `component_ablation.py` write per-chunk checkpoints and can be safely interrupted and restarted.

#### Stage 4 — Figures

```bash
python generate_figures.py          # PDF figures (default)
python generate_figures.py --png    # PNG figures
python generate_figures.py --only 1 8   # Generate specific figures only
```

**Outputs** in `experiment_results/`: `fig1_outlier_robustness.pdf` … `fig8_leverage.pdf` (figure 8 reads `benchmark_leverage.csv` and therefore reflects the OLS-initialization protocol).

### Alternative: single-script runner

For a self-contained run that produces both tables and figures inline:

```bash
python run_experiments.py            # Full run
python run_experiments.py --quick    # Quick mode (fewer trials)
```

### Using the solver in your own code

```python
import numpy as np
from lf_norm import SolverConfig, solve_lf_adaptive

# Define your model and Jacobian
def my_model(x, theta):
    return theta[0] * np.exp(-theta[1] * x) + theta[2]

def my_jacobian(x, theta):
    e = np.exp(-theta[1] * x)
    return np.column_stack([e, -theta[0] * x * e, np.ones_like(x)])

# Data (with 10% outliers)
np.random.seed(42)
x = np.linspace(0.1, 5, 100)
theta_true = np.array([5.0, 0.3, 0.5])
y = my_model(x, theta_true) + np.random.normal(0, 0.1, 100)
y[90:100] = 50.0  # inject outliers

# Solve
cfg = SolverConfig(f_target=0.5, mode="dual", tau=0.95)
theta0 = np.array([4.0, 0.2, 0.4])
result = solve_lf_adaptive(x, y, theta0, my_model, my_jacobian, cfg)

print(f"Parameters: {result['theta']}")
print(f"Converged:  {result['success']}")
print(f"Final f:    {result['f_selected']}")
```

### Solver configuration

Key parameters in `SolverConfig`:

| Parameter | Default | Description |
| --- | --- | --- |
| `f_target` | 0.5 | Target robustness exponent |
| `mode` | `"dual"` | Rank selection: `"full"`, `"energy"`, or `"dual"` |
| `tau` | 0.95 | SVD energy threshold |
| `delta` | 0.05 | Gradient capture threshold |
| `max_iter` | 120 | Maximum outer iterations (the NIST benchmarks use 200) |
| `polish` | `True` | Enable dual-strategy polishing |
| `adaptive_f` | `True` | Enable residual-adaptive f-target selection |

Practical guidance (Sec. 3.8 and 4.4 of the paper): keep the defaults; when the contamination level is unknown, rely on `adaptive_f` rather than raising `f_target` manually; the contaminated-data robustness is insensitive to all of these parameters within the tested ranges.

---

## Methodology

The solver is built around four interacting ideas; full derivations and proofs are in the paper.

1. **Smoothed Lf loss.** The power *f* interpolates continuously between squared loss (*f* = 2), a smoothed absolute loss (*f* = 1), and sub-linear robust losses (*f* < 1, a quasi-norm-type regime that is non-convex). The ε term smooths the objective at the origin, keeping it differentiable and making Newton-type updates well-defined.
2. **IRLS inside Levenberg–Marquardt.** At every outer LM iteration, the Lf gradient and Hessian are expressed as a weighted least-squares system whose weights are $w_i = (d_i^2 + \varepsilon)^{(f-2)/2}$. The gradient of this formulation is exact; the approximation enters only at the Hessian level. The LM damping term stabilizes steps when the Hessian is ill-conditioned.
3. **SVD-based rank selection (dual criterion).** The weighted Jacobian is factorized by SVD. Only the leading singular components that jointly satisfy an energy threshold (τ) and a gradient-capture threshold (δ) are retained. The paper's component ablation identifies this truncation — rather than the warm start — as the primary mechanism preventing catastrophic failures.
4. **Convex-to-robust annealing.** The solver starts from a convex surrogate (*f* near 2, large ε) and gradually pushes *f* toward the user-specified target and ε toward zero. Annealing advances only on accepted LM steps and contributes typical-trial and clean-data accuracy.

A polishing phase and a residual-adaptive *f*-selection wrapper sit on top of this core, providing near-OLS precision on clean data and automatic robustness calibration on contaminated data, respectively.

---

## Key Results

On all 27 NIST StRD benchmarks (1,620 contaminated trials per method), the solver's central strength is **reliability**:

| Metric | Lf\_dual | Best Baseline |
| --- | --- | --- |
| Catastrophic failures (RE > 1) | **4.1%** | Tukey/Cauchy: 11.0% |
| Severe failures (RE > 10) | **0.3%** | Tukey: 3.9% |
| Worst-case RE | **77** | GM: 1.4×10⁴ |
| IQR of RE | **0.133** | Barron: 0.29 |
| Mean RE | **0.334** | GM: 37.2 |
| Strict wins (of 81 scenarios) | **25** | Barron: 16 |

On typical (non-failing) trials the solver is statistically indistinguishable from the strongest robust baselines and slightly behind them by median RE (0.054 vs 0.031–0.047 for Barron, GM, and Cauchy); its advantage is concentrated in the failure rate, dispersion, and worst-case behavior. On nonlinear synthetic models: **zero** catastrophic failures across 240 contaminated trials. See the paper for the complete picture, including clean-data trade-offs and the settings where individual baselines are superior.

---

## Citation

If you use this code or results in academic work, please cite:

```bibtex
@article{saran2026lf,
  title   = {A Robust Nonlinear Fitting Framework with an Adaptive
             Smoothed $L_f$ Loss and SVD-Based Rank Selection},
  author  = {Saran, Murat},
  journal = {PeerJ Computer Science},
  year    = {2026}
}
```

Please also cite the underlying NIST StRD benchmark suite:

> National Institute of Standards and Technology (NIST). *Statistical Reference Datasets: Nonlinear Regression.* https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

---

## License and Contribution Guidelines

### License

This project is released under the **MIT License** — see the [LICENSE](https://github.com/msaran1923/lf-norm-robust-fitting/blob/main/LICENSE) file for the full text. You are free to use, modify, and redistribute the code for academic and commercial purposes, provided the original copyright notice is retained.

### Contributions

Contributions, bug reports, and feature requests are welcome:

- **Bug reports and feature requests:** please open an issue on the [GitHub issue tracker](https://github.com/msaran1923/lf-norm-robust-fitting/issues) with a minimal reproducing example (random seed, dataset, command line used).
- **Pull requests:** please fork the repository, create a feature branch, and open a pull request against `main`. Include a short description of the change and, for non-trivial changes, a test or experiment that demonstrates the effect.
- **Reproducibility issues:** if a specific table or figure in the paper fails to reproduce on your machine, please include the full command line, Python version, package versions (`pip freeze`), and operating system in the issue. Note the exact-reproduction environment above (`numpy==1.26.4`, `scipy==1.12.0`) and that results are reproduced via the pipeline scripts, not isolated solver calls.

### Contact

Correspondence regarding the paper or the code: **msaran \[at\] cankaya.edu.tr** (Department of Computer Engineering, Çankaya University, Ankara, Turkey).
