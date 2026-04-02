#!/usr/bin/env python3
"""
Expanded Benchmark: Comprehensive robustness evaluation
=========================================================
Addresses reviewer concerns about scenario coverage:

  Section A: NIST StRD (6 datasets × 4 contamination levels × 50 trials)
             — Same as before, now with Lf_full as ablation

  Section B: Contamination types (3 types × 3 datasets × 3 levels × 20 trials)
             — Gaussian additive, uniform additive, one-sided (positive only)

  Section C: Contamination magnitudes (3 magnitudes × 3 datasets × 20 trials)
             — 25%, 50%, 100% of response range

  Section D: Nonlinear synthetic models (4 models × 3 levels × 20 trials)
             — exp_decay, logistic, biexponential, gaussian_rbf

  Section E: Nonlinear leverage points (2 models × 4 levels × 20 trials)
             — Leverage outliers applied to nonlinear models

  Section F: Scaling test (n=500, p=20/50 linear × 20 trials)
             — Larger problems to test computational behavior

Usage:
    python run_benchmark_expanded.py                # full run
    python run_benchmark_expanded.py --quick        # quick check (5 trials)
    python run_benchmark_expanded.py --section B    # run only section B
    python run_benchmark_expanded.py --trials 50    # custom trial count
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

from lf_norm import (
    SolverConfig, solve_lf, solve_lf_adaptive,
    lf_objective, irls_weights, residuals,
    model_linear_nd, jac_linear_nd,
    model_exp_decay, jac_exp_decay,
    model_logistic, jac_logistic,
    model_biexp, jac_biexp,
    model_gaussian_rbf, jac_gaussian_rbf,
    SYNTHETIC_REGISTRY, TRUE_THETA, default_init,
    generate_1d_data, generate_nd_data, generate_nd_leverage,
    estimate_initial_scale, solve_barron_annealing,
)
from run_experiments import (
    NIST_DATASETS, _run_lf, _run_scipy, _run_irls, _run_barron,
    get_nist_datasets,
)

OUT = "experiment_results"
os.makedirs(OUT, exist_ok=True)

MAIN_METHODS = {
    "Lf_dual":  lambda x, y, t0, m, j: _run_lf(x, y, t0, m, j, "dual"),
    "OLS":      lambda x, y, t0, m, j: _run_scipy(x, y, t0, m, j, "linear"),
    "Huber":    lambda x, y, t0, m, j: _run_scipy(x, y, t0, m, j, "huber"),
    "Cauchy":   lambda x, y, t0, m, j: _run_scipy(x, y, t0, m, j, "cauchy"),
    "GM":       lambda x, y, t0, m, j: _run_irls(x, y, t0, m, j, "geman_mcclure"),
    "Welsch":   lambda x, y, t0, m, j: _run_irls(x, y, t0, m, j, "welsch"),
    "Tukey":    lambda x, y, t0, m, j: _run_irls(x, y, t0, m, j, "tukey"),
    "Barron":   lambda x, y, t0, m, j: _run_barron(x, y, t0, m, j, alpha_target=0.0),
}


def print_header(title):
    w = 70
    print("\n" + "=" * w)
    print(f"  {title}")
    print("=" * w)


def summarize_section(df, section_name):
    """Print summary for a benchmark section."""
    if df.empty:
        return

    contam = df[df["Outlier_Frac"] > 0] if "Outlier_Frac" in df.columns else df
    if contam.empty:
        return

    summary = contam.groupby("Method", as_index=False).agg(
        Grand_Mean_RE=("Rel_Error", "mean"),
        Median_RE=("Rel_Error", "median"),
        Worst_Case=("Rel_Error", "max"),
    ).sort_values("Grand_Mean_RE")

    print(f"\n  Overall ranking ({section_name}):")
    print(summary.to_string(index=False, float_format="%.4f"))

    # Catastrophic failures
    print(f"\n  Catastrophic failures ({section_name}):")
    for m in sorted(contam["Method"].unique()):
        m_data = contam[contam["Method"] == m]
        n_total = len(m_data)
        n_fail = int((m_data["Rel_Error"] > 1.0).sum())
        if n_fail > 0 or m == "Lf_dual":
            print(f"    {m:12s}: {n_fail}/{n_total} ({100*n_fail/max(n_total,1):.1f}%)")


# ═══════════════════════════════════════════════════════════════════
# SECTION B: Contamination Types
# ═══════════════════════════════════════════════════════════════════

def run_section_B(n_trials=20, dataset_names=None):
    """Test different contamination types on NIST datasets."""
    nist_ds = get_nist_datasets(use_all_27=(dataset_names is not None and
                                             len(dataset_names) > 6))
    if dataset_names is None:
        dataset_names = ["Misra1a", "MGH17", "Thurber"]
    # Filter to datasets available in the loaded NIST dict
    TEST_DATASETS = [n for n in dataset_names if n in nist_ds]
    print_header(f"SECTION B: Contamination Types ({len(TEST_DATASETS)} datasets × {n_trials} trials)")
    CONTAM_TYPES = ["gaussian", "uniform", "one_sided"]
    CONTAM_LEVELS = [0.10, 0.20, 0.30]

    rows = []
    total = len(TEST_DATASETS) * len(CONTAM_TYPES) * len(CONTAM_LEVELS)
    done = 0

    for ds_name in TEST_DATASETS:
        ds = nist_ds.get(ds_name, NIST_DATASETS.get(ds_name))
        x, y_clean = ds["x"], ds["y"]
        theta0 = ds["theta_start"]
        theta_cert = ds["theta_cert"]
        norm_cert = max(np.linalg.norm(theta_cert), 1e-30)
        y_range = y_clean.max() - y_clean.min()

        for ctype in CONTAM_TYPES:
            for frac in CONTAM_LEVELS:
                for trial in range(n_trials):
                    rng = np.random.default_rng(trial + 4000)
                    y = y_clean.copy()
                    n_out = max(1, int(round(frac * len(y))))
                    idx = rng.choice(len(y), n_out, replace=False)

                    if ctype == "gaussian":
                        y[idx] += rng.normal(0, y_range * 0.5, n_out)
                    elif ctype == "uniform":
                        y[idx] += rng.uniform(-y_range, y_range, n_out)
                    elif ctype == "one_sided":
                        y[idx] += np.abs(rng.normal(0, y_range * 0.5, n_out))

                    for mname, runner in MAIN_METHODS.items():
                        try:
                            theta_hat, _ = runner(x, y, theta0.copy(),
                                                  ds["model"], ds["jac"])
                            rel_err = np.linalg.norm(theta_hat - theta_cert) / norm_cert
                        except Exception:
                            rel_err = np.nan
                        rows.append({
                            "Dataset": ds_name, "Contam_Type": ctype,
                            "Outlier_Frac": frac, "Trial": trial,
                            "Method": mname, "Rel_Error": rel_err,
                        })

                done += 1
                sys.stdout.write(f"\r  [{done}/{total}] {ds_name} {ctype} {frac:.0%}")
                sys.stdout.flush()

    print()
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/benchmark_B_contam_types.csv", index=False)

    # Summary by contamination type
    for ctype in CONTAM_TYPES:
        sub = df[df["Contam_Type"] == ctype]
        summarize_section(sub, f"Section B — {ctype}")

    return df


# ═══════════════════════════════════════════════════════════════════
# SECTION C: Contamination Magnitudes
# ═══════════════════════════════════════════════════════════════════

def run_section_C(n_trials=20, dataset_names=None):
    """Test different contamination magnitudes."""
    nist_ds = get_nist_datasets(use_all_27=(dataset_names is not None and
                                             len(dataset_names) > 6))
    if dataset_names is None:
        dataset_names = ["Misra1a", "MGH17", "Thurber"]
    TEST_DATASETS = [n for n in dataset_names if n in nist_ds]
    print_header(f"SECTION C: Contamination Magnitudes ({len(TEST_DATASETS)} datasets × {n_trials} trials)")
    MAGNITUDES = [0.25, 0.50, 1.00]  # fraction of response range
    FRAC = 0.20  # fixed contamination fraction

    rows = []
    total = len(TEST_DATASETS) * len(MAGNITUDES)
    done = 0

    for ds_name in TEST_DATASETS:
        ds = nist_ds.get(ds_name, NIST_DATASETS.get(ds_name))
        x, y_clean = ds["x"], ds["y"]
        theta0 = ds["theta_start"]
        theta_cert = ds["theta_cert"]
        norm_cert = max(np.linalg.norm(theta_cert), 1e-30)
        y_range = y_clean.max() - y_clean.min()

        for mag in MAGNITUDES:
            for trial in range(n_trials):
                rng = np.random.default_rng(trial + 6000)
                y = y_clean.copy()
                n_out = max(1, int(round(FRAC * len(y))))
                idx = rng.choice(len(y), n_out, replace=False)
                y[idx] += rng.normal(0, y_range * mag, n_out)

                for mname, runner in MAIN_METHODS.items():
                    try:
                        theta_hat, _ = runner(x, y, theta0.copy(),
                                              ds["model"], ds["jac"])
                        rel_err = np.linalg.norm(theta_hat - theta_cert) / norm_cert
                    except Exception:
                        rel_err = np.nan
                    rows.append({
                        "Dataset": ds_name, "Magnitude": mag,
                        "Outlier_Frac": FRAC, "Trial": trial,
                        "Method": mname, "Rel_Error": rel_err,
                    })

            done += 1
            sys.stdout.write(f"\r  [{done}/{total}] {ds_name} mag={mag}")
            sys.stdout.flush()

    print()
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/benchmark_C_magnitudes.csv", index=False)

    # Summary by magnitude
    pivot = df.groupby(["Magnitude", "Method"], as_index=False).agg(
        Mean_RE=("Rel_Error", "mean")).pivot_table(
        index="Magnitude", columns="Method", values="Mean_RE")
    print("\n  Mean RE by contamination magnitude:")
    print(pivot.to_string(float_format="%.3e"))

    return df


# ═══════════════════════════════════════════════════════════════════
# SECTION D: Nonlinear Synthetic Models (main comparison)
# ═══════════════════════════════════════════════════════════════════

def run_section_D(n_trials=20):
    """Test all methods on nonlinear synthetic models with outliers."""
    print_header(f"SECTION D: Nonlinear Synthetic Models ({n_trials} trials)")

    MODELS = {
        "exp_decay":     (model_exp_decay, jac_exp_decay, TRUE_THETA["exp_decay"],
                          default_init("exp_decay")),
        "logistic":      (model_logistic, jac_logistic, TRUE_THETA["logistic"],
                          default_init("logistic")),
        "biexponential": (model_biexp, jac_biexp, TRUE_THETA["biexponential"],
                          default_init("biexponential")),
        "gaussian_rbf":  (model_gaussian_rbf, jac_gaussian_rbf, TRUE_THETA["gaussian_rbf"],
                          default_init("gaussian_rbf")),
    }
    CONTAM_LEVELS = [0.10, 0.20, 0.30]

    rows = []
    total = len(MODELS) * len(CONTAM_LEVELS)
    done = 0

    from lf_norm import SyntheticConfig, generate_1d_data

    for mn, (model_fn, jac_fn, theta_true, theta0) in MODELS.items():
        for frac in CONTAM_LEVELS:
            for trial in range(n_trials):
                cfg_d = SyntheticConfig(
                    model_name=mn, seed=trial,
                    n_samples=120, noise_std=0.03,
                    outlier_ratio=frac, outlier_std=1.5)
                x, y, _, _ = generate_1d_data(mn, cfg_d)

                for mname, runner in MAIN_METHODS.items():
                    try:
                        theta_hat, _ = runner(x, y, theta0.copy(),
                                              model_fn, jac_fn)
                        rel_err = np.linalg.norm(theta_hat - theta_true) / max(
                            np.linalg.norm(theta_true), 1e-15)
                    except Exception:
                        rel_err = np.nan
                    rows.append({
                        "Model": mn, "Outlier_Frac": frac,
                        "Trial": trial, "Method": mname,
                        "Rel_Error": rel_err,
                    })

            done += 1
            sys.stdout.write(f"\r  [{done}/{total}] {mn} {frac:.0%}")
            sys.stdout.flush()

    print()
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/benchmark_D_synthetic.csv", index=False)
    summarize_section(df, "Section D — Synthetic nonlinear")

    return df


# ═══════════════════════════════════════════════════════════════════
# SECTION E: Nonlinear Leverage Points
# ═══════════════════════════════════════════════════════════════════

def run_section_E(n_trials=20):
    """Leverage-point outliers on nonlinear synthetic models."""
    print_header(f"SECTION E: Nonlinear Leverage Points ({n_trials} trials)")

    MODELS = {
        "exp_decay":    (model_exp_decay, jac_exp_decay, TRUE_THETA["exp_decay"],
                         default_init("exp_decay"), 0.0, 5.0),
        "gaussian_rbf": (model_gaussian_rbf, jac_gaussian_rbf, TRUE_THETA["gaussian_rbf"],
                         default_init("gaussian_rbf"), 0.0, 5.0),
    }
    LEV_RATIOS = [0.0, 0.10, 0.20, 0.30]

    rows = []
    total = len(MODELS) * len(LEV_RATIOS)
    done = 0

    for mn, (model_fn, jac_fn, theta_true, theta0, x_min, x_max) in MODELS.items():
        n_samples = 120
        for lev_ratio in LEV_RATIOS:
            for trial in range(n_trials):
                rng = np.random.default_rng(trial + 8000)
                x = np.linspace(x_min, x_max, n_samples)
                y_clean = model_fn(x, theta_true)
                y = y_clean + rng.normal(0, 0.03, n_samples)

                if lev_ratio > 0:
                    n_lev = int(round(lev_ratio * n_samples))
                    lev_idx = rng.choice(n_samples, n_lev, replace=False)
                    # Leverage: shift x to extreme values and corrupt y
                    for i in lev_idx:
                        x[i] = rng.uniform(x_max * 1.5, x_max * 3.0)
                        y[i] = rng.normal(0, np.abs(y_clean).max() * 2)

                for mname, runner in MAIN_METHODS.items():
                    try:
                        theta_hat, _ = runner(x, y, theta0.copy(),
                                              model_fn, jac_fn)
                        param_err = np.linalg.norm(theta_hat - theta_true) / max(
                            np.linalg.norm(theta_true), 1e-15)
                    except Exception:
                        param_err = np.nan
                    rows.append({
                        "Model": mn, "Leverage_Ratio": lev_ratio,
                        "Trial": trial, "Method": mname,
                        "Param_Error": param_err,
                    })

            done += 1
            sys.stdout.write(f"\r  [{done}/{total}] {mn} lev={lev_ratio:.0%}")
            sys.stdout.flush()

    print()
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/benchmark_E_nonlinear_leverage.csv", index=False)

    # Summary
    pivot = df[df["Leverage_Ratio"] > 0].groupby(
        ["Model", "Leverage_Ratio", "Method"], as_index=False).agg(
        Mean_PE=("Param_Error", "mean")).pivot_table(
        index=["Model", "Leverage_Ratio"], columns="Method", values="Mean_PE")
    print("\n  Mean Parameter Error (nonlinear leverage):")
    print(pivot.to_string(float_format="%.3e"))

    return df


# ═══════════════════════════════════════════════════════════════════
# SECTION F: Scaling Test
# ═══════════════════════════════════════════════════════════════════

def run_section_F(n_trials=10):
    """Scaling behavior with larger problems."""
    print_header(f"SECTION F: Scaling Test ({n_trials} trials)")

    CONFIGS = [
        (100, 10),
        (200, 20),
        (500, 20),
        (500, 50),
    ]
    FRAC = 0.20

    rows = []
    total = len(CONFIGS)
    done = 0

    for n_samples, n_features in CONFIGS:
        for trial in range(n_trials):
            rng = np.random.default_rng(trial + 9000)
            theta_true = rng.standard_normal(n_features)
            X = rng.standard_normal((n_samples, n_features))
            y_clean = X @ theta_true
            y = y_clean + rng.normal(0, 0.1, n_samples)
            n_out = int(FRAC * n_samples)
            idx = rng.choice(n_samples, n_out, replace=False)
            y[idx] += rng.normal(0, 5.0, n_out)

            theta0 = np.zeros(n_features)
            for mname, runner in MAIN_METHODS.items():
                try:
                    t_start = time.perf_counter()
                    theta_hat, _ = runner(X, y, theta0.copy(),
                                          model_linear_nd, jac_linear_nd)
                    runtime_ms = (time.perf_counter() - t_start) * 1000
                    param_err = np.linalg.norm(theta_hat - theta_true)
                except Exception:
                    param_err = np.nan
                    runtime_ms = np.nan
                rows.append({
                    "n": n_samples, "p": n_features,
                    "Trial": trial, "Method": mname,
                    "Param_Error": param_err, "Runtime_ms": runtime_ms,
                })

        done += 1
        sys.stdout.write(f"\r  [{done}/{total}] n={n_samples}, p={n_features}")
        sys.stdout.flush()

    print()
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/benchmark_F_scaling.csv", index=False)

    pivot_pe = df.groupby(["n", "p", "Method"], as_index=False).agg(
        Mean_PE=("Param_Error", "mean")).pivot_table(
        index=["n", "p"], columns="Method", values="Mean_PE")
    print("\n  Mean Parameter Error (scaling):")
    print(pivot_pe.to_string(float_format="%.3f"))

    pivot_rt = df.groupby(["n", "p", "Method"], as_index=False).agg(
        Mean_RT=("Runtime_ms", "mean")).pivot_table(
        index=["n", "p"], columns="Method", values="Mean_RT")
    print("\n  Mean Runtime ms (scaling):")
    print(pivot_rt.to_string(float_format="%.1f"))

    return df


# ═══════════════════════════════════════════════════════════════════
# GRAND SUMMARY
# ═══════════════════════════════════════════════════════════════════

def grand_summary(results):
    """Aggregate across all sections."""
    print_header("GRAND SUMMARY ACROSS ALL SECTIONS")

    all_rows = []
    for section, df in results.items():
        re_col = "Rel_Error" if "Rel_Error" in df.columns else "Param_Error"
        for _, row in df.iterrows():
            all_rows.append({
                "Section": section,
                "Method": row["Method"],
                "Error": row[re_col],
            })

    df_all = pd.DataFrame(all_rows)
    df_all = df_all.dropna(subset=["Error"])

    overall = df_all.groupby("Method", as_index=False).agg(
        Grand_Mean=("Error", "mean"),
        Grand_Median=("Error", "median"),
        Worst_Case=("Error", "max"),
        Total_Trials=("Error", "count"),
    )
    overall["Failure_Rate_%"] = df_all.groupby("Method")["Error"].apply(
        lambda x: 100.0 * (x > 1.0).sum() / len(x)).values
    overall = overall.sort_values("Grand_Mean")

    print("\n  Grand ranking across all benchmark sections:")
    print(overall.to_string(index=False, float_format="%.4f"))

    overall.to_csv(f"{OUT}/benchmark_grand_summary.csv", index=False)

    return overall


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Expanded Benchmark")
    parser.add_argument("--trials", type=int, default=20,
                        help="Trials per scenario (default: 20)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (5 trials)")
    parser.add_argument("--section", type=str, default=None,
                        help="Run only this section (B, C, D, E, F)")
    parser.add_argument("--all27", action="store_true",
                        help="Use all 27 NIST StRD datasets (requires nist_all_data.py)")
    args = parser.parse_args()

    n_trials = 5 if args.quick else args.trials

    # Select datasets for sections B/C
    nist_ds = get_nist_datasets(use_all_27=args.all27)
    ds_names_bc = list(nist_ds.keys())
    print(f"  Using {len(ds_names_bc)} NIST datasets for sections B/C")

    t0 = time.time()

    results = {}

    sections = {
        "B": ("Contamination Types", lambda: run_section_B(n_trials, ds_names_bc)),
        "C": ("Contamination Magnitudes", lambda: run_section_C(n_trials, ds_names_bc)),
        "D": ("Nonlinear Synthetic", lambda: run_section_D(n_trials)),
        "E": ("Nonlinear Leverage", lambda: run_section_E(n_trials)),
        "F": ("Scaling", lambda: run_section_F(min(n_trials, 10))),
    }

    if args.section:
        key = args.section.upper()
        if key in sections:
            name, fn = sections[key]
            results[key] = fn()
        else:
            print(f"Unknown section: {key}. Choose from {list(sections.keys())}")
            return
    else:
        for key, (name, fn) in sections.items():
            results[key] = fn()

    if len(results) > 1:
        grand_summary(results)

    elapsed = time.time() - t0
    print_header("EXPANDED BENCHMARK COMPLETE")
    print(f"\n  Trials per scenario: {n_trials}")
    print(f"  Total runtime: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"\n  Output files:")
    for f in sorted(os.listdir(OUT)):
        if f.startswith("benchmark"):
            fpath = os.path.join(OUT, f)
            size = os.path.getsize(fpath)
            print(f"    {f:50s}  {size:>8,} bytes")


if __name__ == "__main__":
    main()
