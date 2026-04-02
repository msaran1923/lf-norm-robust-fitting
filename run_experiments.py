#!/usr/bin/env python3
"""
Lf-Norm Robust Nonlinear Fitting — Complete Experiment Runner
=================================================================
Produces all tables and figures for the paper (numbering matches paper):

  Table 2: NIST StRD benchmark — clean data (certified value recovery)
  Table 3: NIST StRD benchmark — outlier robustness across contamination rates
  Table 4: Ablation study — sign detection and momentum contributions
  Table 5: SVD-based feature pruning — precision/recall on synthetic data
  Table 6: Leverage-point robustness
  (Table 1 = hyperparameters, in paper only)

  Figure 1: Outlier robustness bar chart (relative error vs contamination %)
  Figure 2: Convergence curves for representative NIST problems
  Figure 3: IRLS weight evolution showing outlier down-weighting
  Figure 4: Ablation: parameter error across sign-det / momentum combos
  Figure 5: Feature pruning — importance scores and elimination path
  Figure 6: 2D objective landscape (f=2 vs f=0.5)
  Figure 7: Hessian condition number across f values
  Figure 8: Leverage vs response outlier comparison

Usage:
    python run_experiments.py                    # run all
    python run_experiments.py --quick            # quick mode (fewer trials)

Requires: lf_norm.py in the same directory.
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Import the solver
from lf_norm import (
    SolverConfig, SyntheticConfig,
    solve_lf, solve_lf_adaptive, lf_objective, irls_weights, residuals,
    surrogate_objective, feature_importance_svd,
    model_exp_decay, jac_exp_decay,
    model_logistic, jac_logistic,
    model_biexp, jac_biexp,
    model_gaussian_rbf, jac_gaussian_rbf,
    model_linear_nd, jac_linear_nd,
    SYNTHETIC_REGISTRY, TRUE_THETA, default_init,
    generate_1d_data, generate_nd_data, generate_nd_leverage,
    lf_feature_pruning, stability_selection_svd,
    analyze_hessian, solve_scipy_baseline,
    mad_scale, solve_method, estimate_initial_scale,
    solve_barron_annealing,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ═══════════════════════════════════════════════════════════════════
# OUTPUT DIRECTORY
# ═══════════════════════════════════════════════════════════════════
OUT = "experiment_results"
os.makedirs(OUT, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# NIST StRD DATASETS (hardcoded data + certified values)
# ═══════════════════════════════════════════════════════════════════

_misra1a_x = np.array([77.6, 114.9, 141.1, 190.8, 239.9, 289.0, 332.8, 378.4,
                        434.8, 477.3, 536.8, 593.1, 689.1, 760.0])
_misra1a_y = np.array([10.07, 14.73, 17.94, 23.93, 29.61, 35.18, 40.02, 44.82,
                        50.76, 55.05, 61.01, 66.40, 75.47, 81.78])

_chwirut2_x = np.array([0.5, 1.0, 1.75, 3.75, 5.75, 0.875, 2.25, 3.25, 5.25, 0.75,
                         1.75, 2.75, 4.75, 0.625, 1.25, 2.25, 4.25, 0.5, 3.0, 0.75,
                         2.5, 4.0, 0.75, 2.5, 4.0, 0.5, 2.5, 4.0, 0.5, 2.5, 4.0,
                         0.5, 2.5, 4.0, 0.625, 2.75, 4.25, 0.5, 1.75, 3.5, 0.5,
                         1.75, 3.5, 1.5, 3.0, 4.5, 0.5, 2.0, 4.0, 0.625, 2.25, 3.75,
                         3.25, 5.75])
_chwirut2_y = np.array([92.9, 57.1, 31.05, 11.5875, 8.025, 63.6, 21.4, 14.25, 8.475,
                         63.8, 26.8, 16.4625, 7.125, 67.3, 41.0, 21.15, 8.175, 81.5,
                         13.12, 59.9, 14.62, 9.05, 57.5, 14.62, 8.85, 74.1, 15.72, 8.84,
                         72.0, 14.97, 8.65, 71.3, 14.10, 8.86, 64.0, 12.04, 8.44, 72.0,
                         20.84, 10.0, 71.5, 21.5, 9.5, 22.93, 11.475, 7.67, 68.95,
                         18.55, 9.0, 67.0, 17.75, 9.65, 12.24, 7.93])

_danwood_x = np.array([1.309, 1.471, 1.490, 1.565, 1.611, 1.680])
_danwood_y = np.array([2.138, 3.421, 3.597, 4.340, 4.882, 5.660])

_eckerle4_x = np.array([400., 405., 410., 415., 420., 425., 430., 435., 436., 437.,
                         438., 439., 440., 441., 442., 443., 444., 445., 446., 447.,
                         448., 449., 450., 451., 452., 453., 454., 455., 460., 465.,
                         470., 475., 480., 485., 490.])
_eckerle4_y = np.array([0.0001575, 0.0001699, 0.0002350, 0.0003102, 0.0004917,
                         0.0008710, 0.0017418, 0.0046400, 0.0065895, 0.0097302,
                         0.0149002, 0.0237310, 0.0401683, 0.0712559, 0.1264458,
                         0.2073413, 0.2902366, 0.3445623, 0.3698049, 0.3668534,
                         0.3106727, 0.2078154, 0.1164354, 0.0616764, 0.0337200,
                         0.0194023, 0.0117831, 0.0074357, 0.0022732, 0.0008800,
                         0.0004579, 0.0002345, 0.0001586, 0.0001143, 0.0000710])

_mgh17_x = np.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100.,
                      110., 120., 130., 140., 150., 160., 170., 180., 190., 200.,
                      210., 220., 230., 240., 250., 260., 270., 280., 290., 300.,
                      310., 320.])
_mgh17_y = np.array([8.44e-01, 9.08e-01, 9.32e-01, 9.36e-01, 9.25e-01, 9.08e-01,
                      8.81e-01, 8.50e-01, 8.18e-01, 7.84e-01, 7.51e-01, 7.18e-01,
                      6.85e-01, 6.58e-01, 6.28e-01, 6.03e-01, 5.80e-01, 5.58e-01,
                      5.38e-01, 5.22e-01, 5.06e-01, 4.90e-01, 4.78e-01, 4.67e-01,
                      4.57e-01, 4.48e-01, 4.38e-01, 4.31e-01, 4.24e-01, 4.20e-01,
                      4.14e-01, 4.11e-01, 4.06e-01])

_thurber_x = np.array([-3.067, -2.981, -2.921, -2.912, -2.840, -2.797, -2.702, -2.699,
                        -2.633, -2.481, -2.363, -2.322, -1.501, -1.460, -1.274, -1.212,
                        -1.100, -1.046, -0.915, -0.714, -0.566, -0.545, -0.400, -0.309,
                         0.054,  0.430,  2.882,  2.858,  3.501,  3.585,  4.010,  4.280,
                         4.366,  4.533,  4.573,  4.644,  5.190])
_thurber_y = np.array([80.574, 84.248, 87.264, 87.195, 89.076, 89.608, 89.868, 90.101,
                        92.405, 95.854, 100.696, 101.060, 401.672, 390.724, 567.534,
                        635.316, 733.054, 759.087, 894.206, 990.785, 1090.109, 1080.914,
                        1122.643, 1178.351, 1260.531, 1273.514, 1288.339, 1327.543,
                        1353.863, 1414.509, 1425.208, 1421.384, 1442.962, 1464.350,
                        1468.705, 1447.894, 1457.628])


# --- NIST model/jacobian wrappers (local to avoid name clashes) ---
def _m_misra1a(x, t): return t[0] * (1 - np.exp(-t[1] * x))
def _j_misra1a(x, t):
    e = np.exp(-t[1] * x); J = np.empty((x.size, 2))
    J[:,0] = 1-e; J[:,1] = t[0]*x*e; return J

def _m_chwirut2(x, t): return np.exp(-t[0]*x) / (t[1]+t[2]*x)
def _j_chwirut2(x, t):
    e = np.exp(-t[0]*x); v = t[1]+t[2]*x; J = np.empty((x.size,3))
    J[:,0]=-x*e/v; J[:,1]=-e/v**2; J[:,2]=-x*e/v**2; return J

def _m_danwood(x, t): return t[0]*x**t[1]
def _j_danwood(x, t):
    J = np.empty((x.size,2))
    J[:,0] = x**t[1]; J[:,1] = t[0]*x**t[1]*np.log(np.maximum(x,1e-30)); return J

def _m_eckerle4(x, t):
    return (t[0]/t[1])*np.exp(-0.5*((x-t[2])/t[1])**2)
def _j_eckerle4(x, t):
    z = (x-t[2])/t[1]; g = np.exp(-0.5*z**2); v = (t[0]/t[1])*g
    J = np.empty((x.size,3))
    J[:,0] = g/t[1]; J[:,1] = v*(z**2-1)/t[1]; J[:,2] = v*z/t[1]; return J

def _m_mgh17(x, t): return t[0]+t[1]*np.exp(-x*t[3])+t[2]*np.exp(-x*t[4])
def _j_mgh17(x, t):
    e4=np.exp(-x*t[3]); e5=np.exp(-x*t[4]); J=np.empty((x.size,5))
    J[:,0]=1; J[:,1]=e4; J[:,2]=e5; J[:,3]=-t[1]*x*e4; J[:,4]=-t[2]*x*e5; return J

def _m_thurber(x, t):
    x2=x**2; x3=x**3
    return (t[0]+t[1]*x+t[2]*x2+t[3]*x3)/(1+t[4]*x+t[5]*x2+t[6]*x3)
def _j_thurber(x, t):
    x2=x**2; x3=x**3; n=t[0]+t[1]*x+t[2]*x2+t[3]*x3; d=1+t[4]*x+t[5]*x2+t[6]*x3
    J=np.empty((x.size,7))
    J[:,0]=1/d; J[:,1]=x/d; J[:,2]=x2/d; J[:,3]=x3/d
    J[:,4]=-n*x/d**2; J[:,5]=-n*x2/d**2; J[:,6]=-n*x3/d**2; return J


NIST_DATASETS = {
    "Misra1a": {
        "x": _misra1a_x, "y": _misra1a_y,
        "model": _m_misra1a, "jac": _j_misra1a,
        "theta_cert": np.array([2.3894212918e+02, 5.5015643181e-04]),
        "theta_start": np.array([250.0, 0.0005]),
        "p": 2, "difficulty": "Lower",
    },
    "Chwirut2": {
        "x": _chwirut2_x, "y": _chwirut2_y,
        "model": _m_chwirut2, "jac": _j_chwirut2,
        "theta_cert": np.array([1.6657666537e-01, 5.1653291286e-03, 1.2150007096e-02]),
        "theta_start": np.array([0.15, 0.008, 0.010]),
        "p": 3, "difficulty": "Lower",
    },
    "DanWood": {
        "x": _danwood_x, "y": _danwood_y,
        "model": _m_danwood, "jac": _j_danwood,
        "theta_cert": np.array([7.6886226176e-01, 3.8604055871e+00]),
        "theta_start": np.array([0.7, 4.0]),
        "p": 2, "difficulty": "Lower",
    },
    "Eckerle4": {
        "x": _eckerle4_x, "y": _eckerle4_y,
        "model": _m_eckerle4, "jac": _j_eckerle4,
        "theta_cert": np.array([1.5543827178e+00, 4.0888321754e+00, 4.5154121844e+02]),
        "theta_start": np.array([1.5, 5.0, 450.0]),
        "p": 3, "difficulty": "Higher",
    },
    "MGH17": {
        "x": _mgh17_x, "y": _mgh17_y,
        "model": _m_mgh17, "jac": _j_mgh17,
        "theta_cert": np.array([3.7541005211e-01, 1.9358469127e+00, -1.4646871366e+00,
                                 1.2867534640e-02, 2.2122699662e-02]),
        "theta_start": np.array([0.5, 1.5, -1.0, 0.01, 0.02]),
        "p": 5, "difficulty": "Higher",
    },
    "Thurber": {
        "x": _thurber_x, "y": _thurber_y,
        "model": _m_thurber, "jac": _j_thurber,
        "theta_cert": np.array([1.2881396800e+03, 1.4910792535e+03, 5.8323836877e+02,
                                 7.5416644291e+01, 9.6629502864e-01, 3.9797285797e-01,
                                 4.9727297349e-02]),
        "theta_start": np.array([1300., 1500., 500., 75., 1.0, 0.4, 0.05]),
        "p": 7, "difficulty": "Higher",
    },
}

# ── 27-dataset expansion ──
# Try to import full 27 datasets; fall back to 6 hardcoded if unavailable
try:
    from nist_all_data import NIST_ALL_DATASETS
    _HAVE_ALL_27 = True
except ImportError:
    _HAVE_ALL_27 = False
    NIST_ALL_DATASETS = NIST_DATASETS  # fallback to 6

def get_nist_datasets(use_all_27=False):
    """Return NIST dataset dict. If use_all_27=True and nist_all_data.py
    is available, returns all 27 datasets; otherwise returns the 6 hardcoded ones."""
    if use_all_27:
        if _HAVE_ALL_27:
            try:
                loaded = dict(NIST_ALL_DATASETS)  # force load via proxy
                if len(loaded) >= 20:  # sanity check: got most/all datasets
                    return loaded
                else:
                    print(f"WARNING: Only {len(loaded)} datasets loaded. "
                          f"Falling back to 6 hardcoded.")
            except Exception as e:
                print(f"WARNING: Failed to load 27 datasets: {e}")
        else:
            print("WARNING: nist_all_data.py not found. Using 6 hardcoded datasets.")
            print("  Place nist_all_data.py in the same directory and run:")
            print("    python nist_all_data.py")
            print("  to download and cache all 27 NIST StRD datasets.")
    return NIST_DATASETS


# Plotting style
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})
COLORS = {
    "Lf_dual": "#2196F3",
    "Lf_full": "#1565C0",
    "OLS": "#9E9E9E",
    "Huber": "#FF9800",
    "Cauchy": "#4CAF50",
}


def print_header(title):
    w = 70
    print("\n" + "=" * w)
    print(f"  {title}")
    print("=" * w)


# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 1: NIST StRD Benchmark (Clean + Outlier Contamination)
# ═══════════════════════════════════════════════════════════════════

def run_experiment_1(n_trials=5):
    """
    Table 1: Clean-data recovery (relative error vs certified values)
    Table 2: Outlier robustness (contamination 0%-30%)
    Figure 1: Bar chart of relative error vs contamination
    Figure 2: Convergence curves
    Figure 3: IRLS weight evolution
    """
    print_header("EXPERIMENT 1: NIST StRD Benchmark")

    METHODS = {
        "Lf_dual":  lambda x, y, t0, m, j: _run_lf(x, y, t0, m, j, "dual"),
        "Lf_full":  lambda x, y, t0, m, j: _run_lf(x, y, t0, m, j, "full"),
        "OLS":      lambda x, y, t0, m, j: _run_scipy(x, y, t0, m, j, "linear"),
        "Huber":    lambda x, y, t0, m, j: _run_scipy(x, y, t0, m, j, "huber"),
        "Cauchy":   lambda x, y, t0, m, j: _run_scipy(x, y, t0, m, j, "cauchy"),
        "GM":       lambda x, y, t0, m, j: _run_irls(x, y, t0, m, j, "geman_mcclure"),
        "Welsch":   lambda x, y, t0, m, j: _run_irls(x, y, t0, m, j, "welsch"),
        "Tukey":    lambda x, y, t0, m, j: _run_irls(x, y, t0, m, j, "tukey"),
        "Barron":   lambda x, y, t0, m, j: _run_barron(x, y, t0, m, j, alpha_target=0.0),
    }

    # --- Table 2 (paper): Clean data ---
    print("\n[1/5] Running clean-data benchmark...")
    clean_rows = []
    for ds_name, ds in NIST_DATASETS.items():
        x, y = ds["x"], ds["y"]
        theta0 = ds["theta_start"]
        theta_cert = ds["theta_cert"]
        norm_cert = max(np.linalg.norm(theta_cert), 1e-30)
        for mname, runner in METHODS.items():
            try:
                t_start = time.perf_counter()
                theta_hat, hist = runner(x, y, theta0.copy(), ds["model"], ds["jac"])
                runtime_ms = (time.perf_counter() - t_start) * 1000
                rel_err = np.linalg.norm(theta_hat - theta_cert) / norm_cert
                iters = len(hist["obj"]) if hist else 0
            except Exception:
                rel_err = np.nan; iters = 0; runtime_ms = np.nan
            clean_rows.append({
                "Dataset": ds_name, "Difficulty": ds["difficulty"],
                "Method": mname, "Rel_Error": rel_err,
                "Converged": rel_err < 1e-2 if not np.isnan(rel_err) else False,
                "Iterations": iters, "Runtime_ms": runtime_ms,
            })
    df_clean = pd.DataFrame(clean_rows)
    tbl1 = df_clean.pivot_table(index=["Dataset", "Difficulty"],
                                 columns="Method", values="Rel_Error")
    tbl1.to_csv(f"{OUT}/table2_nist_clean.csv")
    print("\n  TABLE 2: NIST Clean-Data Recovery (Relative Error)")
    print("  " + "-" * 66)
    print(df_clean.pivot_table(index="Dataset", columns="Method",
                                values="Rel_Error").to_string(float_format="%.2e"))

    # Runtime summary
    rt_pivot = df_clean.pivot_table(index="Dataset", columns="Method",
                                     values="Runtime_ms")
    rt_pivot.to_csv(f"{OUT}/table2_runtime_ms.csv")
    print("\n  Runtime per method (ms):")
    print("  " + "-" * 66)
    print(rt_pivot.to_string(float_format="%.1f"))

    # --- Table 2: Outlier robustness ---
    print("\n[2/5] Running outlier-contaminated benchmark...")
    OUTLIER_FRACS = [0.0, 0.10, 0.20, 0.30]
    outlier_rows = []
    for ds_name, ds in NIST_DATASETS.items():
        x, y_clean = ds["x"], ds["y"]
        theta0 = ds["theta_start"]
        theta_cert = ds["theta_cert"]
        norm_cert = max(np.linalg.norm(theta_cert), 1e-30)
        y_range = y_clean.max() - y_clean.min()

        for frac in OUTLIER_FRACS:
            for trial in range(n_trials):
                rng = np.random.default_rng(trial + 1000)
                y = y_clean.copy()
                if frac > 0:
                    n_out = max(1, int(round(frac * len(y))))
                    idx = rng.choice(len(y), n_out, replace=False)
                    y[idx] += rng.normal(0, y_range * 0.5, n_out)

                for mname, runner in METHODS.items():
                    try:
                        theta_hat, _ = runner(x, y, theta0.copy(),
                                              ds["model"], ds["jac"])
                        rel_err = np.linalg.norm(theta_hat - theta_cert) / norm_cert
                    except Exception:
                        rel_err = np.nan
                    outlier_rows.append({
                        "Dataset": ds_name, "Difficulty": ds["difficulty"],
                        "Method": mname, "Outlier_Frac": frac,
                        "Trial": trial, "Rel_Error": rel_err,
                    })
        sys.stdout.write(f"    {ds_name} done\n"); sys.stdout.flush()

    df_outlier = pd.DataFrame(outlier_rows)
    tbl2 = df_outlier.groupby(["Dataset", "Method", "Outlier_Frac"], as_index=False)\
                     .agg(Mean_RE=("Rel_Error", "mean"), Std_RE=("Rel_Error", "std"))
    tbl2.to_csv(f"{OUT}/table3_nist_outlier.csv", index=False)
    print("\n  TABLE 3: Mean Relative Error by Contamination Rate (averaged)")
    tbl2_pivot = tbl2.pivot_table(index=["Dataset", "Outlier_Frac"],
                                   columns="Method", values="Mean_RE")
    print(tbl2_pivot.to_string(float_format="%.3e"))

    # --- Figure 1: Outlier robustness bar chart ---
    print("\n[3/5] Generating Figure 1: Outlier robustness...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, ds_name in zip(axes.flat, NIST_DATASETS.keys()):
        sub = tbl2[tbl2["Dataset"] == ds_name]
        width = 0.15
        fracs = sorted(sub["Outlier_Frac"].unique())
        x_pos = np.arange(len(fracs))
        for i, mname in enumerate(["Lf_dual", "OLS", "Huber", "Cauchy"]):
            ms = sub[sub["Method"] == mname]
            vals = [ms[ms["Outlier_Frac"] == f]["Mean_RE"].values[0]
                    if len(ms[ms["Outlier_Frac"] == f]) > 0 else 0
                    for f in fracs]
            ax.bar(x_pos + i * width, vals, width,
                   label=mname, color=COLORS.get(mname, "#666"), alpha=0.85)
        ax.set_xticks(x_pos + 1.5 * width)
        ax.set_xticklabels([f"{int(f*100)}%" for f in fracs])
        ax.set_title(f"{ds_name} ({NIST_DATASETS[ds_name]['difficulty']})")
        ax.set_ylabel("Mean Relative Error")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
    plt.suptitle("Figure 1: Outlier Robustness — NIST StRD Benchmarks", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{OUT}/fig1_outlier_robustness.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Figure 2: Convergence curves ---
    print("[4/5] Generating Figure 2: Convergence curves...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, ds_name in zip(axes.flat, NIST_DATASETS.keys()):
        ds = NIST_DATASETS[ds_name]
        x, y = ds["x"], ds["y"]
        theta0 = ds["theta_start"]
        for mode, ls in [("dual", "-"), ("full", "--")]:
            cfg = SolverConfig(mode=mode, f_target=0.5, max_iter=200,
                               eps0=1.0, mu0=1e-4, capture_spectrum=False)
            try:
                res = solve_lf(x, y, theta0.copy(), ds["model"], ds["jac"], cfg)
                obj_hist = res["history"]["obj"]
                ax.semilogy(obj_hist, ls, label=f"Lf_{mode}", lw=1.5)
            except Exception:
                pass
        ax.set_title(ds_name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective")
        ax.legend(fontsize=7)
    plt.suptitle("Figure 2: Convergence Curves — NIST StRD Benchmarks", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{OUT}/fig2_convergence.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Figure 3: IRLS weight evolution ---
    print("[5/5] Generating Figure 3: IRLS weight evolution...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    cfg_d = SyntheticConfig(seed=0, outlier_ratio=0.25, outlier_std=1.5)
    x_w, y_w, _, out_idx = generate_1d_data("exp_decay", cfg_d)
    theta0_w = default_init("exp_decay")
    cfg_w = SolverConfig(mode="dual", f_target=0.5, max_iter=200)
    # Run solver and record weights at key iterations
    theta_trace = theta0_w.copy()
    f_k, eps_k = 2.0, 1.0
    steps_to_show = [0, 2, 5, 10, 30, 80]
    step_i = 0
    for k in range(100):
        d = residuals(x_w, y_w, theta_trace, model_exp_decay)
        w = irls_weights(d, f_k, eps_k)
        if k in steps_to_show and step_i < 6:
            ax = axes.flat[step_i]
            inlier_mask = np.ones(len(x_w), dtype=bool)
            inlier_mask[out_idx] = False
            ax.scatter(x_w[inlier_mask], w[inlier_mask], c="tab:blue", s=12,
                       alpha=0.6, label="Inlier")
            ax.scatter(x_w[~inlier_mask], w[~inlier_mask], c="tab:red", s=18,
                       alpha=0.8, label="Outlier", marker="x")
            ax.set_title(f"Iter {k} (f={f_k:.2f}, eps={eps_k:.3f})")
            ax.set_ylabel("IRLS weight")
            ax.set_xlabel("x")
            ax.legend(fontsize=7)
            step_i += 1
        # Take one solver step
        J = jac_exp_decay(x_w, theta_trace)
        WJ = w[:, None] * J
        g = WJ.T @ (w * d)
        U, s, Vt = np.linalg.svd(WJ, full_matrices=False)
        V = Vt.T
        step = -V @ ((1.0 / (s**2 + 1e-3)) * (V.T @ g))
        theta_trace = theta_trace + step
        f_k = max(0.5, 0.92 * f_k)
        eps_k = max(1e-8, 0.70 * eps_k)

    plt.suptitle("Figure 3: IRLS Weight Evolution — Outlier Down-weighting", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{OUT}/fig3_irls_weights.png", dpi=150, bbox_inches="tight")
    plt.close()

    return df_clean, df_outlier


# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Ablation — Sign Detection + Momentum
# ═══════════════════════════════════════════════════════════════════

def run_experiment_2(n_trials=20):
    """
    Table 3: Ablation of sign detection and momentum
    Figure 4: Box plot of parameter error across ablation configs
    """
    print_header("EXPERIMENT 2: Sign Detection & Momentum Ablation")

    SYNTHETIC_MODELS = ["exp_decay", "logistic", "biexponential", "gaussian_rbf"]
    CONFIGS = [
        ("Baseline",  False, False),
        ("+Momentum", False, True),
        ("+SignDet",   True,  False),
        ("+Both",      True,  True),
    ]
    F_TARGETS = [0.3, 0.5, 1.0]

    rows = []
    for mn in SYNTHETIC_MODELS:
        model_fn, jac_fn = SYNTHETIC_REGISTRY[mn]
        theta_true = TRUE_THETA[mn]
        theta0 = default_init(mn)
        for trial in range(n_trials):
            cfg_d = SyntheticConfig(
                model_name=mn, seed=trial,
                outlier_ratio=0.30, outlier_std=2.5)
            x, y, _, _ = generate_1d_data(mn, cfg_d)
            for f_target in F_TARGETS:
                for label, use_sd, use_mom in CONFIGS:
                    cfg = SolverConfig(
                        mode="dual", f_target=f_target,
                        use_sign_detection=use_sd, use_momentum=use_mom)
                    try:
                        res = solve_lf(x, y, theta0.copy(), model_fn, jac_fn, cfg)
                        pe = float(np.linalg.norm(res["theta"] - theta_true))
                        h = res["history"]
                        failed = False
                    except Exception:
                        pe = np.inf  # penalty: failed runs count as worst
                        h = {"sign_flips": 0, "momentum_applied": 0,
                             "obj": [], "rejections": 0}
                        failed = True
                    rows.append({
                        "Model": mn, "Trial": trial,
                        "f_target": f_target, "Config": label,
                        "Param_Error": pe,
                        "Sign_Flips": h["sign_flips"],
                        "Momentum_Steps": h["momentum_applied"],
                        "Iterations": len(h["obj"]),
                        "Rejections": h["rejections"],
                        "Failed": failed,
                    })
        sys.stdout.write(f"  {mn} done\n"); sys.stdout.flush()

    df_abl = pd.DataFrame(rows)

    # Report failures before aggregation
    n_failed = df_abl["Failed"].sum()
    if n_failed > 0:
        print(f"\n  WARNING: {n_failed} solver failures (counted as inf in means)")
        fail_summary = df_abl[df_abl["Failed"]].groupby(["Model", "Config"]).size()
        print(fail_summary.to_string())

    tbl3 = df_abl.groupby(["Model", "f_target", "Config"], as_index=False).agg(
        Mean_PE=("Param_Error", "mean"),
        Std_PE=("Param_Error", "std"),
        Median_PE=("Param_Error", "median"),
        Failures=("Failed", "sum"),
        Mean_SignFlips=("Sign_Flips", "mean"),
        Mean_MomSteps=("Momentum_Steps", "mean"),
        Mean_Iters=("Iterations", "mean"),
        Mean_Rej=("Rejections", "mean"),
    )
    tbl3.to_csv(f"{OUT}/table4_ablation.csv", index=False)
    print("\n  TABLE 4: Ablation Results (Mean ± Std Parameter Error)")
    print("  " + "-" * 66)
    tbl3_pivot = tbl3.pivot_table(
        index=["Model", "f_target"], columns="Config", values="Mean_PE")
    print(tbl3_pivot.to_string(float_format="%.4f"))

    # Sign flips summary
    print("\n  Sign flips summary:")
    sf = tbl3[tbl3["Config"].isin(["+SignDet", "+Both"])]
    print(sf[["Model", "f_target", "Config", "Mean_SignFlips"]].to_string(index=False))

    # --- Figure 4: Ablation box plot ---
    print("\n  Generating Figure 4: Ablation box plots...")
    fig, axes = plt.subplots(1, len(F_TARGETS), figsize=(5 * len(F_TARGETS), 5))
    if len(F_TARGETS) == 1:
        axes = [axes]
    config_labels = [c[0] for c in CONFIGS]
    for ax, ft in zip(axes, F_TARGETS):
        data = []
        labels = []
        for label in config_labels:
            sub = df_abl[(df_abl["f_target"] == ft) & (df_abl["Config"] == label)]
            data.append(sub["Param_Error"].dropna().values)
            labels.append(label)
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                        medianprops=dict(color="black"))
        box_colors = ["#BBDEFB", "#C8E6C9", "#FFE0B2", "#E1BEE7"]
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
        ax.set_title(f"f_target = {ft}")
        ax.set_ylabel("Parameter Error")
        ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=15)
    plt.suptitle("Figure 4: Ablation — Sign Detection & Momentum", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{OUT}/fig4_ablation.png", dpi=150, bbox_inches="tight")
    plt.close()

    return df_abl


# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 3: SVD-Based Feature Pruning
# ═══════════════════════════════════════════════════════════════════

def run_experiment_3(n_trials=10):
    """
    Table 4: Feature pruning precision/recall/F1
    Figure 5: Importance scores and elimination path
    Figure 6: 2D objective landscape
    Figure 7: Hessian condition number across f
    """
    print_header("EXPERIMENT 3: Feature Pruning & Analysis")

    N_FEATURES = 15
    N_RELEVANT = 5
    N_SAMPLES = 200
    OUTLIER_RATIO = 0.20
    OUTLIER_STD = 10.0

    # --- Table 4: Pruning accuracy ---
    print("\n[1/4] Running feature pruning trials...")
    pruning_rows = []
    for trial in range(n_trials):
        rng = np.random.default_rng(trial + 100)
        theta_true = np.zeros(N_FEATURES)
        theta_true[:N_RELEVANT] = rng.uniform(2.0, 6.0, N_RELEVANT)

        X = rng.standard_normal((N_SAMPLES, N_FEATURES))
        X[:, N_RELEVANT:N_RELEVANT+3] += 0.3 * X[:, :3]  # mild correlation
        y_clean = X @ theta_true
        y = y_clean + rng.normal(0, 0.5, N_SAMPLES)
        n_out = int(OUTLIER_RATIO * N_SAMPLES)
        out_idx = rng.choice(N_SAMPLES, n_out, replace=False)
        y[out_idx] += rng.normal(0, OUTLIER_STD, n_out)

        theta0 = np.zeros(N_FEATURES)
        true_relevant = set(range(N_RELEVANT))

        for f_val, label in [(0.5, "Lf (f=0.5)"), (2.0, "OLS (f=2)")]:
            result = lf_feature_pruning(
                X, y, theta0, f_target=f_val, tol=0.20,
                min_features=1, verbose=False)
            selected = set(result["selected_features"])
            tp = len(selected & true_relevant)
            fp = len(selected - true_relevant)
            fn = len(true_relevant - selected)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-10)
            pruning_rows.append({
                "Trial": trial, "Method": label,
                "Selected": len(selected), "TP": tp, "FP": fp, "FN": fn,
                "Precision": prec, "Recall": rec, "F1": f1,
            })
        sys.stdout.write(f"    Trial {trial+1}/{n_trials}\r"); sys.stdout.flush()

    print()
    df_prune = pd.DataFrame(pruning_rows)
    tbl4 = df_prune.groupby("Method", as_index=False).agg(
        Mean_Prec=("Precision", "mean"), Std_Prec=("Precision", "std"),
        Mean_Rec=("Recall", "mean"), Std_Rec=("Recall", "std"),
        Mean_F1=("F1", "mean"), Std_F1=("F1", "std"),
        Mean_Selected=("Selected", "mean"),
    )
    tbl4.to_csv(f"{OUT}/table5_pruning.csv", index=False)
    print("\n  TABLE 5: Feature Pruning Accuracy (15 features, 5 relevant)")
    print("  " + "-" * 66)
    print(tbl4.to_string(index=False, float_format="%.3f"))

    # --- Figure 5: Importance and elimination (single trial) ---
    print("\n[2/4] Generating Figure 5: Feature importance...")
    rng = np.random.default_rng(42)
    theta_true = np.zeros(N_FEATURES)
    theta_true[:N_RELEVANT] = rng.uniform(2.0, 6.0, N_RELEVANT)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    X[:, N_RELEVANT:N_RELEVANT+3] += 0.3 * X[:, :3]
    y_clean = X @ theta_true
    y = y_clean + rng.normal(0, 0.5, N_SAMPLES)
    n_out = int(0.20 * N_SAMPLES)
    y[rng.choice(N_SAMPLES, n_out, replace=False)] += rng.normal(0, 10.0, n_out)

    result_demo = lf_feature_pruning(
        X, y, np.zeros(N_FEATURES), f_target=0.5, tol=0.20,
        min_features=1, verbose=False)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # (a) Residual norm vs features
    err_df = pd.DataFrame(result_demo["error_history"])
    axes[0].plot(err_df["n_features"], err_df["resid_norm"], "o-",
                 color="tab:green", lw=2)
    axes[0].axvline(N_RELEVANT, color="red", ls="--", alpha=0.7,
                    label=f"True support = {N_RELEVANT}")
    axes[0].set_xlabel("Number of features")
    axes[0].set_ylabel("Residual norm")
    axes[0].set_title("(a) Model fit vs feature count")
    axes[0].legend()
    axes[0].invert_xaxis()

    # (b) Initial importance
    if result_demo["importance_history"]:
        imp0 = result_demo["importance_history"][0]
        features_initial = imp0["active_features"]
        imp_vals = [imp0["importance"].get(f, 0) for f in features_initial]
        colors = ["tab:green" if f < N_RELEVANT else "tab:red"
                  for f in features_initial]
        axes[1].bar(range(len(features_initial)), imp_vals, color=colors, alpha=0.8)
        axes[1].set_xlabel("Feature index")
        axes[1].set_ylabel("SVD importance (initial)")
        axes[1].set_title("(b) Initial importance (green=relevant)")
        axes[1].set_xticks(range(len(features_initial)))
        axes[1].set_xticklabels(features_initial, fontsize=7)

    # (c) Elimination order
    elim_feats, elim_imps = [], []
    for rec in result_demo["error_history"][1:]:
        if rec["eliminated"] is not None:
            elim_feats.append(str(rec["eliminated"]))
            elim_imps.append(rec["importance"])
    if elim_feats:
        bar_colors = ["tab:green" if int(f) >= N_RELEVANT else "tab:red"
                      for f in elim_feats]
        axes[2].barh(range(len(elim_feats)), elim_imps, color=bar_colors, alpha=0.8)
        axes[2].set_yticks(range(len(elim_feats)))
        axes[2].set_yticklabels(elim_feats, fontsize=8)
        axes[2].set_xlabel("Importance at removal")
        axes[2].set_title("(c) Elimination order (green=irrelevant=correct)")

    plt.suptitle("Figure 5: SVD-Based Feature Pruning", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{OUT}/fig5_feature_pruning.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Figure 6: 2D Objective landscape ---
    print("[3/4] Generating Figure 6: Objective landscape...")
    cfg_land = SyntheticConfig(seed=0, outlier_ratio=0.25, outlier_std=1.5)
    x_land, y_land, _, _ = generate_1d_data("exp_decay", cfg_land)
    theta_true_land = TRUE_THETA["exp_decay"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    f_vals_land = [2.0, 1.0, 0.5]
    for ax, f_val in zip(axes, f_vals_land):
        n_grid = 80
        a_range = np.linspace(theta_true_land[0] - 1.5, theta_true_land[0] + 1.5, n_grid)
        b_range = np.linspace(theta_true_land[1] - 0.8, theta_true_land[1] + 0.8, n_grid)
        Z = np.zeros((n_grid, n_grid))
        for i, a in enumerate(a_range):
            for j, b in enumerate(b_range):
                th = np.array([a, b, theta_true_land[2]])
                d = model_exp_decay(x_land, th) - y_land
                Z[j, i] = lf_objective(d, f_val, 1e-6)
        Z = np.log10(np.maximum(Z, 1e-15))
        cs = ax.contourf(a_range, b_range, Z, levels=30, cmap="viridis")
        ax.plot(theta_true_land[0], theta_true_land[1], "r*", ms=12,
                label="True θ", zorder=5)
        ax.set_xlabel("θ₀ (amplitude)")
        ax.set_ylabel("θ₁ (decay rate)")
        ax.set_title(f"f = {f_val}")
        ax.legend(fontsize=8)
        plt.colorbar(cs, ax=ax, label="log₁₀(Lf)")

    plt.suptitle("Figure 6: Lf Objective Landscape (exp_decay, 25% outliers)",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{OUT}/fig6_landscape.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Figure 7: Hessian condition number ---
    print("[4/4] Generating Figure 7: Hessian conditioning...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    f_test_vals = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    for ax, mn in zip(axes.flat, ["exp_decay", "logistic",
                                   "biexponential", "gaussian_rbf"]):
        model_fn, jac_fn = SYNTHETIC_REGISTRY[mn]
        theta_true = TRUE_THETA[mn]
        cfg_h = SyntheticConfig(model_name=mn, seed=0,
                                outlier_ratio=0.25, outlier_std=1.5)
        x_h, y_h, _, _ = generate_1d_data(mn, cfg_h)
        df_h = analyze_hessian(x_h, y_h, theta_true, model_fn, jac_fn,
                               f_test_vals, eps=1e-4)
        ax.semilogy(df_h["f"], df_h["condition"], "o-", color="tab:blue", lw=2)
        for _, row in df_h.iterrows():
            color = "tab:green" if row["is_pd"] else "tab:red"
            ax.plot(row["f"], row["condition"], "o", color=color, ms=8)
        ax.set_xlabel("f")
        ax.set_ylabel("Condition number")
        ax.set_title(f"{mn}")
        ax.axvline(1.0, color="gray", ls="--", alpha=0.5, label="f=1 boundary")
        ax.legend(fontsize=7)

    plt.suptitle("Figure 7: Hessian Condition Number vs f\n"
                 "(green=PD, red=not PD at true θ with outliers)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"{OUT}/fig7_hessian_condition.png", dpi=150, bbox_inches="tight")
    plt.close()

    return df_prune


# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Leverage-Point Outliers
# ═══════════════════════════════════════════════════════════════════

def run_experiment_4(n_trials=10):
    """
    Table 5: Leverage-point robustness (X-space outliers)
    Figure 8: Leverage vs response outlier comparison
    """
    print_header("EXPERIMENT 4: Leverage-Point Outliers")

    N_FEATURES = 10
    N_SAMPLES = 200
    LEV_RATIOS = [0.0, 0.05, 0.10, 0.15, 0.20]
    LEV_MAGNITUDE = 8.0

    rows = []
    for lev_ratio in LEV_RATIOS:
        for trial in range(n_trials):
            seed = trial + 2000
            if lev_ratio == 0.0:
                # Clean baseline
                rng = np.random.default_rng(seed)
                theta_true = rng.standard_normal(N_FEATURES)
                X = rng.standard_normal((N_SAMPLES, N_FEATURES))
                y = X @ theta_true + rng.normal(0, 0.1, N_SAMPLES)
                lev_idx = np.array([], dtype=int)
            else:
                X, y, _, theta_true, lev_idx = generate_nd_leverage(
                    n_samples=N_SAMPLES, n_features=N_FEATURES,
                    noise_std=0.1, leverage_ratio=lev_ratio,
                    leverage_magnitude=LEV_MAGNITUDE, seed=seed)

            theta0 = np.zeros(N_FEATURES)

            # Lf_dual
            cfg = SolverConfig(mode="dual", f_target=0.5, max_iter=200,
                               use_sign_detection=True, use_momentum=False)
            try:
                res = solve_lf(X, y, theta0.copy(), model_linear_nd,
                               jac_linear_nd, cfg)
                pe_lf = float(np.linalg.norm(res["theta"] - theta_true))
            except Exception:
                pe_lf = np.inf

            # OLS
            try:
                theta_ols, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                pe_ols = float(np.linalg.norm(theta_ols - theta_true))
            except Exception:
                pe_ols = np.inf

            # Huber via scipy
            try:
                res_h = least_squares(
                    fun=lambda th: X @ th - y, x0=theta0.copy(),
                    jac=lambda th: X, method="trf", loss="huber",
                    f_scale=1.0, max_nfev=5000)
                pe_huber = float(np.linalg.norm(res_h.x - theta_true))
            except Exception:
                pe_huber = np.inf

            # Cauchy via scipy
            try:
                res_c = least_squares(
                    fun=lambda th: X @ th - y, x0=theta0.copy(),
                    jac=lambda th: X, method="trf", loss="cauchy",
                    f_scale=1.0, max_nfev=5000)
                pe_cauchy = float(np.linalg.norm(res_c.x - theta_true))
            except Exception:
                pe_cauchy = np.inf

            for method, pe in [("Lf_dual", pe_lf), ("OLS", pe_ols),
                                ("Huber", pe_huber), ("Cauchy", pe_cauchy)]:
                rows.append({
                    "Leverage_Ratio": lev_ratio, "Trial": trial,
                    "Method": method, "Param_Error": pe,
                })
        sys.stdout.write(f"  leverage_ratio={lev_ratio:.0%} done\n")
        sys.stdout.flush()

    df_lev = pd.DataFrame(rows)
    tbl5 = df_lev.groupby(["Leverage_Ratio", "Method"], as_index=False).agg(
        Mean_PE=("Param_Error", "mean"),
        Std_PE=("Param_Error", "std"),
        Median_PE=("Param_Error", "median"),
    )
    tbl5.to_csv(f"{OUT}/table6_leverage.csv", index=False)
    print("\n  TABLE 6: Leverage-Point Robustness (Mean Param Error)")
    print("  " + "-" * 66)
    tbl5_pivot = tbl5.pivot_table(index="Leverage_Ratio",
                                   columns="Method", values="Mean_PE")
    print(tbl5_pivot.to_string(float_format="%.4f"))

    # --- Figure 8: Leverage vs response comparison ---
    print("\n  Generating Figure 8: Leverage-point robustness...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Leverage-point outliers
    ax = axes[0]
    methods_plot = ["Lf_dual", "OLS", "Huber", "Cauchy"]
    for m in methods_plot:
        sub = tbl5[tbl5["Method"] == m]
        ax.plot(sub["Leverage_Ratio"] * 100, sub["Mean_PE"], "o-",
                label=m, color=COLORS.get(m, "#666"), lw=2)
        ax.fill_between(sub["Leverage_Ratio"] * 100,
                         sub["Mean_PE"] - sub["Std_PE"],
                         sub["Mean_PE"] + sub["Std_PE"],
                         alpha=0.15, color=COLORS.get(m, "#666"))
    ax.set_xlabel("Leverage-point contamination (%)")
    ax.set_ylabel("Mean Parameter Error")
    ax.set_title("(a) Leverage-point outliers (extreme in X-space)")
    ax.legend(fontsize=9)
    ax.set_yscale("log")

    # (b) Response-only outliers for comparison
    resp_rows = []
    for out_ratio in [0.0, 0.05, 0.10, 0.15, 0.20]:
        for trial in range(n_trials):
            rng = np.random.default_rng(trial + 3000)
            theta_true = rng.standard_normal(N_FEATURES)
            X = rng.standard_normal((N_SAMPLES, N_FEATURES))
            y_clean = X @ theta_true
            y = y_clean + rng.normal(0, 0.1, N_SAMPLES)
            if out_ratio > 0:
                n_out = int(round(out_ratio * N_SAMPLES))
                idx = rng.choice(N_SAMPLES, n_out, replace=False)
                y[idx] += rng.normal(0, 5.0, n_out)

            theta0 = np.zeros(N_FEATURES)
            cfg = SolverConfig(mode="dual", f_target=0.5, max_iter=200,
                               use_sign_detection=True, use_momentum=False)
            try:
                res = solve_lf(X, y, theta0.copy(), model_linear_nd,
                               jac_linear_nd, cfg)
                pe_lf = float(np.linalg.norm(res["theta"] - theta_true))
            except Exception:
                pe_lf = np.inf
            try:
                theta_ols, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                pe_ols = float(np.linalg.norm(theta_ols - theta_true))
            except Exception:
                pe_ols = np.inf

            for method, pe in [("Lf_dual", pe_lf), ("OLS", pe_ols)]:
                resp_rows.append({
                    "Outlier_Ratio": out_ratio, "Method": method,
                    "Param_Error": pe,
                })

    df_resp = pd.DataFrame(resp_rows)
    resp_agg = df_resp.groupby(["Outlier_Ratio", "Method"], as_index=False).agg(
        Mean_PE=("Param_Error", "mean"), Std_PE=("Param_Error", "std"))

    ax = axes[1]
    for m in ["Lf_dual", "OLS"]:
        sub = resp_agg[resp_agg["Method"] == m]
        ax.plot(sub["Outlier_Ratio"] * 100, sub["Mean_PE"], "o-",
                label=m, color=COLORS.get(m, "#666"), lw=2)
    ax.set_xlabel("Response-only contamination (%)")
    ax.set_ylabel("Mean Parameter Error")
    ax.set_title("(b) Response-only outliers (additive in Y)")
    ax.legend(fontsize=9)
    ax.set_yscale("log")

    plt.suptitle("Figure 8: Leverage-Point vs Response-Only Outliers "
                 f"(n={N_SAMPLES}, p={N_FEATURES})", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{OUT}/fig8_leverage.png", dpi=150, bbox_inches="tight")
    plt.close()

    return df_lev


# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def _run_lf(x, y, theta0, model, jac, mode, f_target=0.5):
    cfg = SolverConfig(mode=mode, f_target=f_target, max_iter=200,
                       eps0=1.0, mu0=1e-4,
                       use_sign_detection=True, use_momentum=False,
                       adaptive_f=True)
    res = solve_lf_adaptive(x, y, theta0, model, jac, cfg)
    return res["theta"], res["history"]


def _run_scipy(x, y, theta0, model, jac, loss):
    """Run SciPy baseline with MAD-adaptive f_scale."""
    s = estimate_initial_scale(x, y, theta0, model, jac)
    res = least_squares(
        fun=lambda th: model(x, th) - y,
        x0=theta0, jac=lambda th: jac(x, th),
        method="trf", loss=loss, f_scale=s, max_nfev=5000)
    return res.x, None


def _run_irls(x, y, theta0, model, jac, method_name):
    """Run custom IRLS baseline (Geman-McClure, Welsch, Tukey) with MAD scale."""
    s = estimate_initial_scale(x, y, theta0, model, jac)
    cfg = SolverConfig()
    # Use MAD-based scale for the IRLS baselines
    res = solve_method(method_name, x, y, theta0, model, jac, cfg, noise_std=s)
    return res["theta"], None


def _run_barron(x, y, theta0, model, jac, alpha_target=0.0):
    """Run Barron adaptive robust loss with annealing and MAD scale."""
    s = estimate_initial_scale(x, y, theta0, model, jac)
    res = solve_barron_annealing(
        x, y, theta0, model, jac,
        c=s, alpha_target=alpha_target,
        alpha0=2.0, rho_alpha=0.92,
        beta=5.0, mu0=1e-3,
        max_iter=200, max_inner=15)
    return res["theta"], None


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Lf-Norm Experiment Runner")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (fewer trials)")
    args = parser.parse_args()

    if args.quick:
        n1, n2, n3, n4 = 2, 5, 3, 3
        print("*** QUICK MODE: reduced trial counts ***")
    else:
        n1, n2, n3, n4 = 5, 20, 10, 10

    t0 = time.time()

    df_clean, df_outlier = run_experiment_1(n_trials=n1)
    df_ablation = run_experiment_2(n_trials=n2)
    df_pruning = run_experiment_3(n_trials=n3)
    df_leverage = run_experiment_4(n_trials=n4)

    elapsed = time.time() - t0

    # Final summary
    print_header("ALL EXPERIMENTS COMPLETE")
    print(f"\n  Total runtime: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"\n  Output directory: {os.path.abspath(OUT)}/")
    print("  Files generated:")
    for f in sorted(os.listdir(OUT)):
        fpath = os.path.join(OUT, f)
        size = os.path.getsize(fpath)
        print(f"    {f:40s}  {size:>8,} bytes")

    print("\n  Tables (numbering matches paper):")
    print("    table2_nist_clean.csv      — Clean-data certified value recovery")
    print("    table3_nist_outlier.csv    — Outlier robustness across contamination")
    print("    table4_ablation.csv        — Sign detection & momentum ablation")
    print("    table5_pruning.csv         — Feature pruning precision/recall/F1")
    print("    table6_leverage.csv        — Leverage-point robustness")
    print("    (Table 1 = hyperparameters, in paper only)")
    print("\n  Figures:")
    print("    fig1_outlier_robustness.png — Bar chart: rel. error vs contamination")
    print("    fig2_convergence.png        — Convergence curves per NIST problem")
    print("    fig3_irls_weights.png       — IRLS weight evolution (outlier suppression)")
    print("    fig4_ablation.png           — Box plots: ablation configurations")
    print("    fig5_feature_pruning.png    — SVD importance & elimination path")
    print("    fig6_landscape.png          — 2D Lf objective landscapes (f=2,1,0.5)")
    print("    fig7_hessian_condition.png  — Hessian conditioning across f values")
    print("    fig8_leverage.png           — Leverage vs response outlier comparison")


if __name__ == "__main__":
    main()
