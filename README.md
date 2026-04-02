# Robust Nonlinear Fitting with Adaptive L<sub>f</sub>-Norm and SVD-Based Rank Selection

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Replication code for the paper:

> **A Robust Nonlinear Fitting Framework with Adaptive L<sub>f</sub>-Norm and SVD-Based Rank Selection**

This repository contains the complete implementation of the proposed solver and all scripts needed to reproduce every table and figure in the paper.

---

## Overview

The solver minimizes a generalized L<sub>f</sub>-norm objective:

$$\mathcal{L}_f(\boldsymbol{\theta}) = \frac{1}{f}\sum_{i=1}^{n} \bigl(d_i^2 + \varepsilon\bigr)^{f/2}$$

via iteratively reweighted least squares (IRLS) embedded within a Levenberg–Marquardt loop, where each iteration's weighted Jacobian is factored by SVD to enable low-rank step computation. Key features include:

- **Adaptive robustness**: continuous interpolation from OLS (*f* = 2) through LAD (*f* = 1) to aggressive outlier suppression (*f* < 1)
- **SVD-based rank selection**: dual criterion combining energy-based truncation with gradient alignment
- **Convex-to-robust warm-start**: graduated annealing of *f* and *ε* that advances only on accepted steps
- **No scale parameter required**: implicit scale adaptation through joint *f*–*ε* annealing
- **Dual-strategy polishing**: recovers near-OLS precision on clean data
- **Residual-adaptive f-target selection**: automatically chooses the mildest exponent preserving robustness

## Repository Structure

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

### File Descriptions

| File | Purpose |
|------|---------|
| `lf_norm.py` | Core solver implementation: `SolverConfig`, `solve_lf`, `solve_lf_adaptive`, IRLS weights, SVD rank selection, feature pruning, all synthetic model functions, and all 27 NIST model functions with analytical Jacobians |
| `nist_all_data.py` | Loads all 27 NIST StRD datasets. Downloads from NIST on first use and caches locally as `nist_cache.json` |
| `download_nist.py` | Alternative standalone downloader that generates a self-contained Python data module |
| `run_experiments.py` | Runs individual experiments (clean-data recovery, outlier robustness, ablation, feature pruning, leverage points) with inline figure generation |
| `run_benchmark.py` | Comprehensive 27-dataset benchmark: 9 methods × 27 datasets × 4 contamination levels × 20 seeds. Produces all CSV result files |
| `run_benchmark_expanded.py` | Expanded analysis: contamination types (Sec. B), magnitudes (Sec. C), synthetic models (Sec. D), nonlinear leverage (Sec. E), scaling (Sec. F) |
| `generate_figures.py` | Generates all 8 paper figures (PDF/PNG) from the benchmark CSV results |

## Installation

### Requirements

- Python ≥ 3.11
- NumPy ≥ 1.26
- SciPy ≥ 1.12
- pandas ≥ 2.0
- matplotlib ≥ 3.8

### Setup

```bash
git clone https://github.com/msaran1923/lf-norm-robust-fitting.git
cd lf-norm-robust-fitting
pip install -r requirements.txt
```

## Reproducing Paper Results

### Quick Verification (~5 minutes)

Run a quick sanity check with reduced trial counts:

```bash
python run_benchmark.py --quick --all27
```

This runs 5 trials (instead of 20) across all 27 datasets to verify the code works correctly.

### Full Replication

The complete replication involves three stages. Total runtime depends on hardware; expect 2–4 hours on a modern CPU.

#### Stage 1: Main Benchmark (Tables 1–2, catastrophic failure rates)

```bash
python run_benchmark.py --all27 --trials 20
```

**Outputs** in `experiment_results/`:
| File | Contents |
|------|----------|
| `benchmark_full.csv` | Raw per-trial results (19,440 rows) |
| `benchmark_summary.csv` | Aggregated mean/std/median per dataset × method × contamination |
| `benchmark_wins.csv` | Strict win counts per method |
| `benchmark_catastrophic.csv` | Catastrophic (RE > 1) and severe (RE > 10) failure rates |
| `benchmark_ablation_rank.csv` | Lf_dual vs Lf_full comparison |
| `benchmark_leverage.csv` | Leverage-point robustness results |

#### Stage 2: Expanded Analysis (contamination types, magnitudes, synthetic models)

```bash
python run_benchmark_expanded.py --all27 --trials 20
```

Run individual sections if needed:

```bash
python run_benchmark_expanded.py --section B --all27   # Contamination types
python run_benchmark_expanded.py --section C --all27   # Contamination magnitudes
python run_benchmark_expanded.py --section D           # Synthetic models
python run_benchmark_expanded.py --section E           # Nonlinear leverage points
python run_benchmark_expanded.py --section F           # Scaling tests
```

**Outputs** in `experiment_results/`:
| File | Contents |
|------|----------|
| `benchmark_B_contam_types.csv` | Gaussian, uniform, one-sided contamination results |
| `benchmark_C_magnitudes.csv` | 25%, 50%, 100% outlier magnitude results |
| `benchmark_D_synthetic.csv` | Nonlinear synthetic model results |
| `benchmark_E_nonlinear_leverage.csv` | Nonlinear leverage-point results |
| `benchmark_F_scaling.csv` | Scaling tests (n=100–500, p=10–50) |
| `benchmark_grand_summary.csv` | Grand summary across all experiments |

#### Stage 3: Figures

```bash
python generate_figures.py          # PDF figures (default)
python generate_figures.py --png    # PNG figures
python generate_figures.py --only 1 8   # Generate specific figures only
```

**Outputs** in `experiment_results/`:
| File | Description |
|------|-------------|
| `fig1_outlier_robustness.pdf` | RE vs contamination rate across NIST datasets |
| `fig2_convergence.pdf` | L<sub>f</sub> objective convergence (dual vs full rank) |
| `fig3_irls_weights.pdf` | IRLS weight evolution showing outlier suppression |
| `fig4_ablation.pdf` | Ablation: sign detection & momentum |
| `fig5_feature_pruning.pdf` | SVD-based feature pruning |
| `fig6_landscape.pdf` | 2D objective contours at *f* = 2, 1, 0.5 |
| `fig7_hessian_condition.pdf` | Hessian condition number vs *f* |
| `fig8_leverage.pdf` | Leverage-point robustness comparison |

### Alternative: Single-Script Experiment Runner

For a self-contained run that produces both tables and figures inline:

```bash
python run_experiments.py            # Full run
python run_experiments.py --quick    # Quick mode (fewer trials)
```

## Using the Solver in Your Own Code

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

print(f"Parameters: {result['theta']}")       # recovered parameters
print(f"Converged:  {result['success']}")      # convergence flag
print(f"Final f:    {result['f_selected']}")   # selected f-target
```

### Solver Configuration

Key parameters in `SolverConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `f_target` | 0.5 | Target robustness exponent |
| `mode` | `"dual"` | Rank selection: `"full"`, `"energy"`, or `"dual"` |
| `tau` | 0.95 | SVD energy threshold |
| `delta` | 0.05 | Gradient capture threshold |
| `max_iter` | 120 | Maximum outer iterations |
| `polish` | `True` | Enable dual-strategy polishing |
| `adaptive_f` | `True` | Enable residual-adaptive f-target selection |

## Key Results

On all 27 NIST StRD benchmarks (1,620 contaminated trials per method):

| Metric | Lf_dual | Best Baseline |
|--------|---------|---------------|
| Mean RE | **0.334** | GM: 37.2 |
| Catastrophic failures (RE > 1) | **4.1%** | Tukey/Cauchy: 11.0% |
| Severe failures (RE > 10) | **0.3%** | Tukey: 3.9% |
| Strict wins (of 81 scenarios) | **25** | Barron: 16 |

On nonlinear synthetic models: **zero** catastrophic failures across 240 contaminated trials.

## Citation

```bibtex
@article{saran2026lf,
  title   = {A Robust Nonlinear Fitting Framework with Adaptive
             $L_f$-Norm and SVD-Based Rank Selection},
  author  = {Saran, Murat},
  journal = {PeerJ Computer Science},
  year    = {2026}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
