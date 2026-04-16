# Robust Nonlinear Fitting with Adaptive Lf-Norm and SVD-Based Rank Selection

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/msaran1923/lf-norm-robust-fitting/blob/main/LICENSE)

Replication code for the paper:

> **A Robust Nonlinear Fitting Framework with Adaptive Lf-Norm and SVD-Based Rank Selection**
> Murat Saran, submitted to *PeerJ Computer Science*, 2026.

This repository contains the complete implementation of the proposed solver and every script required to reproduce the tables and figures in the paper.

---

## Description

This project provides a Python implementation of a robust nonlinear regression framework based on a generalized **Lf-norm** objective combined with **SVD-based rank selection**. The code is designed for two audiences:

1. **Reviewers and researchers** who want to reproduce the experimental results reported in the accompanying paper (all 27 NIST StRD benchmarks, contamination studies, synthetic-model studies, leverage-point studies, scaling tests, and all 8 paper figures).
2. **Practitioners** who want to apply the Lf-norm solver to their own nonlinear regression problems where outliers or leverage points make classical least-squares unreliable.

The solver minimizes

$$\mathcal{L}_f(\boldsymbol{\theta}) = \frac{1}{f}\sum_{i=1}^{n} \bigl(d_i^2 + \varepsilon\bigr)^{f/2}$$

via iteratively reweighted least squares (IRLS) embedded in a Levenberg–Marquardt outer loop. Each iteration's weighted Jacobian is factorized by SVD to enable low-rank step computation. Key features:

- **Adaptive robustness** — continuous interpolation from OLS (*f* = 2) through LAD (*f* = 1) to aggressive outlier suppression (*f* < 1).
- **SVD-based rank selection** — dual criterion combining energy-based truncation with gradient alignment.
- **Convex-to-robust warm-start** — graduated annealing of *f* and *ε* that advances only on accepted steps.
- **No user-supplied scale parameter** — implicit scale adaptation through joint *f*–*ε* annealing.
- **Dual-strategy polishing** — recovers near-OLS precision on clean data.
- **Residual-adaptive f-target selection** — automatically chooses the mildest exponent preserving robustness.

---

## Dataset Information

All experiments use the **27 NIST Statistical Reference Datasets (StRD) for Nonlinear Regression**, a publicly available, certified benchmark suite maintained by the U.S. National Institute of Standards and Technology.

- **Source:** https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml
- **Difficulty levels:** Lower (8 datasets), Average (11 datasets), Higher (8 datasets).
- **Content:** Each dataset provides observed (x, y) pairs, a parametric nonlinear model, two sets of starting values, and certified parameter estimates with standard deviations.
- **License:** The NIST StRD benchmarks are U.S. Government work and are in the public domain.

### How the data is obtained

The datasets are **not redistributed** in this repository. They are fetched directly from NIST on first use:

- `nist_all_data.py` downloads the 27 datasets from NIST on first call and caches them locally in `nist_cache.json`. Subsequent runs read from the cache.
- `download_nist.py` is an alternative standalone downloader that produces a self-contained Python data module.

Contamination for the robustness experiments is generated **synthetically** on top of the clean NIST data at fixed random seeds (20 seeds per configuration). The contamination scheme (rates, magnitudes, one-sided vs symmetric, Gaussian vs uniform) is fully described in the paper and implemented in `run_benchmark.py` and `run_benchmark_expanded.py`.

Additional synthetic datasets (Section D of the paper) are generated on the fly by model functions defined in `lf_norm.py` and controlled by the seeds documented in `run_benchmark_expanded.py`.

---

## Code Information

### Repository structure

```
lf-norm-robust-fitting/
├── lf_norm.py                  # Core solver library (~1800 lines)
├── nist_all_data.py            # NIST StRD dataset loader (auto-downloads & caches)
├── download_nist.py            # Standalone NIST data downloader
├── run_experiments.py          # Individual experiment runner (tables & figures)
├── run_benchmark.py            # Full 27-dataset benchmark (Tables 1–2)
├── run_benchmark_expanded.py   # Expanded analysis (contam. types, magnitudes, synthetic)
├── generate_figures.py         # Generate paper figures from CSV results
├── experiment_results/         # Output directory (created automatically)
├── requirements.txt
├── LICENSE
└── README.md
```

### File descriptions

| File | Purpose |
| --- | --- |
| `lf_norm.py` | Core solver implementation: `SolverConfig`, `solve_lf`, `solve_lf_adaptive`, IRLS weights, SVD rank selection, feature pruning, synthetic model functions, and all 27 NIST model functions with analytical Jacobians |
| `nist_all_data.py` | Loads all 27 NIST StRD datasets. Downloads from NIST on first use and caches locally as `nist_cache.json` |
| `download_nist.py` | Alternative standalone downloader that generates a self-contained Python data module |
| `run_experiments.py` | Runs individual experiments (clean-data recovery, outlier robustness, ablation, feature pruning, leverage points) with inline figure generation |
| `run_benchmark.py` | Comprehensive 27-dataset benchmark: 9 methods × 27 datasets × 4 contamination levels × 20 seeds. Produces all CSV result files |
| `run_benchmark_expanded.py` | Expanded analysis: contamination types (Sec. B), magnitudes (Sec. C), synthetic models (Sec. D), nonlinear leverage (Sec. E), scaling (Sec. F) |
| `generate_figures.py` | Generates all 8 paper figures (PDF/PNG) from the benchmark CSV results |

### Public API

The solver is exposed through a small public surface in `lf_norm.py`:

- `SolverConfig` — dataclass holding all solver hyperparameters.
- `solve_lf(x, y, theta0, model, jacobian, cfg)` — single-pass Lf solver at a fixed `f_target`.
- `solve_lf_adaptive(x, y, theta0, model, jacobian, cfg)` — residual-adaptive wrapper that selects `f_target` automatically.

Both functions return a dictionary containing the fitted parameters, residuals, weights, convergence flag, iteration count, selected `f`, and the rank history.

---

## Requirements

- Python ≥ 3.11
- NumPy ≥ 1.26
- SciPy ≥ 1.12
- pandas ≥ 2.0
- matplotlib ≥ 3.8

All dependencies are listed in `requirements.txt` with pinned minimum versions. The code is pure Python and does not require a GPU or any compiled extensions beyond those shipped with the above packages. It has been tested on Windows 11 and Ubuntu 22.04.

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

Complete replication involves three stages. Total runtime depends on hardware; expect 2–4 hours on a modern CPU.

#### Stage 1 — Main benchmark (Tables 1–2, catastrophic failure rates)

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
| `benchmark_leverage.csv` | Leverage-point robustness results |

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

**Outputs** in `experiment_results/`:

| File | Contents |
| --- | --- |
| `benchmark_B_contam_types.csv` | Gaussian, uniform, one-sided contamination results |
| `benchmark_C_magnitudes.csv` | 25%, 50%, 100% outlier magnitude results |
| `benchmark_D_synthetic.csv` | Nonlinear synthetic model results |
| `benchmark_E_nonlinear_leverage.csv` | Nonlinear leverage-point results |
| `benchmark_F_scaling.csv` | Scaling tests (n=100–500, p=10–50) |
| `benchmark_grand_summary.csv` | Grand summary across all experiments |

#### Stage 3 — Figures

```bash
python generate_figures.py          # PDF figures (default)
python generate_figures.py --png    # PNG figures
python generate_figures.py --only 1 8   # Generate specific figures only
```

**Outputs** in `experiment_results/`:

| File | Description |
| --- | --- |
| `fig1_outlier_robustness.pdf` | RE vs contamination rate across NIST datasets |
| `fig2_convergence.pdf` | Lf objective convergence (dual vs full rank) |
| `fig3_irls_weights.pdf` | IRLS weight evolution showing outlier suppression |
| `fig4_ablation.pdf` | Ablation: sign detection & momentum |
| `fig5_feature_pruning.pdf` | SVD-based feature pruning |
| `fig6_landscape.pdf` | 2D objective contours at *f* = 2, 1, 0.5 |
| `fig7_hessian_condition.pdf` | Hessian condition number vs *f* |
| `fig8_leverage.pdf` | Leverage-point robustness comparison |

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
| `max_iter` | 120 | Maximum outer iterations |
| `polish` | `True` | Enable dual-strategy polishing |
| `adaptive_f` | `True` | Enable residual-adaptive f-target selection |

---

## Methodology

The solver is built around four interacting ideas; full derivations and proofs are in the paper.

1. **Lf-norm objective with ε-smoothing.** The power *f* interpolates continuously between squared loss (*f* = 2), absolute loss (*f* = 1), and sub-linear losses (*f* < 1). The ε term smooths the objective at the origin, keeping it differentiable and making Newton-type updates well-defined.
2. **IRLS inside Levenberg–Marquardt.** At every outer LM iteration, Lf's gradient and Hessian are expressed as a weighted least-squares system whose weights are $w_i = (d_i^2 + \varepsilon)^{(f-2)/2}$. The LM damping term stabilizes steps when the Hessian is ill-conditioned.
3. **SVD-based rank selection (dual criterion).** The weighted Jacobian is factorized by SVD. Only the leading singular components that jointly satisfy an energy threshold (τ) and a gradient-alignment threshold (δ) are retained. This yields large-scale robustness on near-rank-deficient problems without requiring the user to choose a regularization strength.
4. **Convex-to-robust annealing.** The solver starts from a convex surrogate (*f* near 2, large ε) and gradually pushes *f* toward the user-specified target and ε toward zero. Annealing advances only on accepted LM steps, which prevents the solver from jumping into a local minimum of an aggressively non-convex Lf landscape.

A polishing phase and a residual-adaptive *f*-selection wrapper sit on top of this core, providing near-OLS precision on clean data and automatic robustness calibration on contaminated data, respectively.

---

## Key Results

On all 27 NIST StRD benchmarks (1,620 contaminated trials per method):

| Metric | Lf\_dual | Best Baseline |
| --- | --- | --- |
| Mean RE | **0.334** | GM: 37.2 |
| Catastrophic failures (RE > 1) | **4.1%** | Tukey/Cauchy: 11.0% |
| Severe failures (RE > 10) | **0.3%** | Tukey: 3.9% |
| Strict wins (of 81 scenarios) | **25** | Barron: 16 |

On nonlinear synthetic models: **zero** catastrophic failures across 240 contaminated trials.

---

## Citation

If you use this code or results in academic work, please cite:

```bibtex
@article{saran2026lf,
  title   = {A Robust Nonlinear Fitting Framework with Adaptive
             $L_f$-Norm and SVD-Based Rank Selection},
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
- **Reproducibility issues:** if a specific table or figure in the paper fails to reproduce on your machine, please include the full command line, Python version, package versions (`pip freeze`), and operating system in the issue.

### Contact

Correspondence regarding the paper or the code: **msaran \[at\] cankaya.edu.tr** (Department of Computer Engineering, Çankaya University, Ankara, Turkey).
