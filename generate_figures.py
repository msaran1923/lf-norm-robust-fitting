#!/usr/bin/env python3
"""
Generate all paper figures from benchmark results and solver runs.
=================================================================

Reads from:
  experiment_results/benchmark_full.csv      (from run_benchmark.py --all27)
  experiment_results/benchmark_leverage.csv  (from run_benchmark.py --all27)

Generates:
  fig1_outlier_robustness.pdf  — RE vs contamination for 8 representative datasets
  fig2_convergence.pdf         — Lf objective convergence (dual vs full rank)
  fig3_irls_weights.pdf        — IRLS weight evolution showing outlier suppression
  fig4_ablation.pdf            — Ablation: sign detection & momentum box plots
  fig5_feature_pruning.pdf     — SVD importance & backward elimination path
  fig6_landscape.pdf           — 2D Lf objective contours at f=2, 1, 0.5
  fig7_hessian_condition.pdf   — Hessian condition number vs f
  fig8_leverage.pdf            — Leverage-point robustness comparison

Usage:
    python generate_figures.py                # all figures
    python generate_figures.py --only 1 8     # specific figures
    python generate_figures.py --png          # PNG instead of PDF

Requires: benchmark CSVs from run_benchmark.py, lf_norm.py, nist_all_data.py
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=RuntimeWarning)

from lf_norm import (
    SolverConfig, solve_lf, solve_lf_adaptive,
    lf_objective, irls_weights, residuals,
    surrogate_objective, feature_importance_svd,
    model_exp_decay, jac_exp_decay,
    model_logistic, jac_logistic,
    model_biexp, jac_biexp,
    model_gaussian_rbf, jac_gaussian_rbf,
    model_linear_nd, jac_linear_nd,
    SYNTHETIC_REGISTRY, TRUE_THETA, default_init,
    generate_1d_data, generate_nd_data,
    lf_feature_pruning, analyze_hessian,
    estimate_initial_scale, solve_barron_annealing,
)

OUT = "experiment_results"
os.makedirs(OUT, exist_ok=True)

# ── Style ──
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "font.family": "serif",
})

COLORS = {
    "Lf_dual": "#2196F3",
    "Lf_full": "#1565C0",
    "OLS": "#9E9E9E",
    "Huber": "#FF9800",
    "Cauchy": "#4CAF50",
    "GM": "#9C27B0",
    "Welsch": "#795548",
    "Tukey": "#E91E63",
    "Barron": "#00BCD4",
}

METHOD_ORDER = ["Lf_dual", "OLS", "Huber", "Cauchy", "GM", "Welsch", "Tukey", "Barron"]


def get_ext(args):
    return "png" if args.png else "pdf"


# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: Outlier Robustness — 8 representative datasets
# ═══════════════════════════════════════════════════════════════════

def fig1(args):
    ext = get_ext(args)
    print("  Generating Figure 1: Outlier robustness...")

    csv_path = f"{OUT}/benchmark_full.csv"
    if not os.path.exists(csv_path):
        print(f"    ERROR: {csv_path} not found. Run: python run_benchmark.py --all27 --trials 20")
        return

    df = pd.read_csv(csv_path)

    # Select 8 representative datasets across difficulty levels
    SHOW = ["Misra1a", "Chwirut2", "DanWood", "MGH17",
            "Thurber", "Eckerle4", "Hahn1", "Kirby2"]
    available = [d for d in SHOW if d in df["Dataset"].unique()]
    n_show = len(available)
    ncols = 4
    nrows = (n_show + ncols - 1) // ncols

    # Methods to show (exclude Lf_full ablation)
    show_methods = ["Lf_dual", "Barron", "Cauchy", "GM", "Huber", "Tukey"]

    agg = df[~df.get("Is_Ablation", False)].groupby(
        ["Dataset", "Method", "Outlier_Frac"], as_index=False
    ).agg(Mean_RE=("Rel_Error", "mean"))

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, ds_name in enumerate(available):
        ax = axes[idx // ncols, idx % ncols]
        sub = agg[agg["Dataset"] == ds_name]
        fracs = sorted(sub["Outlier_Frac"].unique())
        x_pos = np.arange(len(fracs))
        n_m = len(show_methods)
        width = 0.8 / n_m

        for i, mname in enumerate(show_methods):
            ms = sub[sub["Method"] == mname]
            vals = []
            for f in fracs:
                v = ms[ms["Outlier_Frac"] == f]["Mean_RE"].values
                vals.append(v[0] if len(v) > 0 else np.nan)
            ax.bar(x_pos + i * width, vals, width,
                   label=mname, color=COLORS.get(mname, "#666"), alpha=0.85)

        ax.set_xticks(x_pos + (n_m / 2) * width)
        ax.set_xticklabels([f"{int(f*100)}%" for f in fracs])
        diff = df[df["Dataset"] == ds_name]["Difficulty"].iloc[0] if "Difficulty" in df.columns else ""
        ax.set_title(f"{ds_name} ({diff})" if diff else ds_name)
        ax.set_ylabel("Mean RE")
        ax.set_yscale("log")
        if idx == 0:
            ax.legend(fontsize=6, ncol=2)

    # Hide unused axes
    for idx in range(n_show, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    plt.suptitle("Outlier Robustness: Mean RE across contamination levels (27-dataset benchmark)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{OUT}/fig1_outlier_robustness.{ext}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved fig1_outlier_robustness.{ext}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2: Convergence curves (solver internal — no CSV needed)
# ═══════════════════════════════════════════════════════════════════

def fig2(args):
    ext = get_ext(args)
    print("  Generating Figure 2: Convergence curves...")

    try:
        from nist_all_data import NIST_ALL_DATASETS
        nist = dict(NIST_ALL_DATASETS)
        if len(nist) == 0:
            raise ImportError("empty")
    except Exception:
        from run_experiments import NIST_DATASETS
        nist = NIST_DATASETS

    SHOW = [n for n in ["Misra1a", "Chwirut2", "DanWood", "Eckerle4", "MGH17", "Thurber"]
            if n in nist]
    if not SHOW:
        print("    SKIPPED: no NIST datasets available")
        return
    ncols = 3
    nrows = (len(SHOW) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, ds_name in enumerate(SHOW):
        ax = axes[idx // ncols, idx % ncols]
        ds = nist[ds_name]
        x, y = ds["x"], ds["y"]
        theta0 = ds["theta_start"]

        for mode, ls, label in [("dual", "-", "Lf_dual"), ("full", "--", "Lf_full")]:
            cfg = SolverConfig(mode=mode, f_target=0.5, max_iter=200,
                               use_sign_detection=True, adaptive_f=True)
            res = solve_lf_adaptive(x, y, theta0.copy(), ds["model"], ds["jac"], cfg)
            h = res["history"]
            ax.plot(h["obj"], ls, label=label,
                    color=COLORS.get(label, "#666"), lw=1.5)

        ax.set_title(ds_name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("$\\mathcal{L}_f$ objective")
        ax.set_yscale("log")
        if idx == 0:
            ax.legend()

    for idx in range(len(SHOW), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    plt.suptitle("Convergence: $L_f$ objective vs iteration (dual-rank vs full-rank)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{OUT}/fig2_convergence.{ext}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved fig2_convergence.{ext}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3: IRLS weight evolution
# ═══════════════════════════════════════════════════════════════════

def fig3(args):
    ext = get_ext(args)
    print("  Generating Figure 3: IRLS weight evolution...")

    rng = np.random.default_rng(42)
    x = np.linspace(0.0, 5.0, 120)
    y_clean = model_exp_decay(x, TRUE_THETA["exp_decay"])
    y = y_clean + rng.normal(0, 0.03, 120)
    n_out = int(0.25 * 120)
    idx_out = rng.choice(120, n_out, replace=False)
    y[idx_out] += rng.normal(0, 1.0, n_out)

    theta0 = default_init("exp_decay")
    cfg = SolverConfig(mode="dual", f_target=0.5, max_iter=200,
                       use_sign_detection=True)

    # Run solver step by step, recording weights at key iterations
    theta = theta0.astype(float).copy()
    f_k, eps_k = cfg.f0, cfg.eps0
    snapshots = []
    snap_iters = [0, 3, 8, 15, 30, 60]

    for k in range(80):
        d = residuals(x, y, theta, model_exp_decay)
        w = irls_weights(d, f_k, eps_k)

        if k in snap_iters:
            snapshots.append((k, f_k, eps_k, w.copy()))

        W = np.diag(w)
        J = jac_exp_decay(x, theta)
        WJ = w[:, None] * J
        g = WJ.T @ (w * d)
        if np.linalg.norm(g) < 1e-8:
            break
        U, svals, Vt = np.linalg.svd(WJ, full_matrices=False)
        V = Vt.T
        mu = 1e-3
        S_inv = svals / (svals**2 + mu)
        step = -V @ (S_inv * (U.T @ (w * d)))
        obj = lf_objective(d, f_k, eps_k)
        d_trial = residuals(x, y, theta + step, model_exp_decay)
        if lf_objective(d_trial, f_k, eps_k) < obj:
            theta = theta + step
            f_k = max(cfg.f_target, cfg.rho_f * f_k)
            eps_k = max(cfg.eps_min, cfg.rho_eps * eps_k)

    n_snap = min(len(snapshots), 6)
    fig, axes = plt.subplots(1, n_snap, figsize=(3.2 * n_snap, 3))
    if n_snap == 1:
        axes = [axes]

    for i, (k, f, eps, w) in enumerate(snapshots[:n_snap]):
        ax = axes[i]
        is_outlier = np.zeros(120, dtype=bool)
        is_outlier[idx_out] = True
        ax.scatter(x[~is_outlier], w[~is_outlier], s=12, c="#2196F3",
                   alpha=0.7, label="Inlier")
        ax.scatter(x[is_outlier], w[is_outlier], s=18, c="#F44336",
                   marker="x", alpha=0.9, label="Outlier")
        ax.set_title(f"Iter {k}\n$f$={f:.2f}, $\\varepsilon$={eps:.2f}", fontsize=9)
        ax.set_xlabel("$x$")
        if i == 0:
            ax.set_ylabel("IRLS weight $w_i$")
            ax.legend(fontsize=7)
        ax.set_ylim(-0.05, max(w) * 1.15 + 0.01)

    plt.suptitle("IRLS Weight Evolution (exp_decay, 25% outliers)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"{OUT}/fig3_irls_weights.{ext}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved fig3_irls_weights.{ext}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 4: Ablation — sign detection & momentum
# ═══════════════════════════════════════════════════════════════════

def fig4(args):
    ext = get_ext(args)
    print("  Generating Figure 4: Ablation box plots...")
    n_trials = 20

    MODELS = ["exp_decay", "logistic", "biexponential", "gaussian_rbf"]
    X_RANGES = {
        "exp_decay": (0.0, 5.0), "logistic": (0.0, 5.0),
        "biexponential": (0.0, 5.0), "gaussian_rbf": (-3.0, 3.0),
    }
    CONFIGS = {
        "Baseline":  {"sign": False, "mom": False},
        "+SignDet":   {"sign": True,  "mom": False},
        "+Momentum":  {"sign": False, "mom": True},
        "+Both":      {"sign": True,  "mom": True},
    }
    F_TARGETS = [0.3, 0.5, 1.0]

    rows = []
    for mn in MODELS:
        mfn, jfn = SYNTHETIC_REGISTRY[mn]
        theta_true = TRUE_THETA[mn]
        xlo, xhi = X_RANGES[mn]
        for ft in F_TARGETS:
            for cname, cdict in CONFIGS.items():
                errs = []
                for trial in range(n_trials):
                    rng = np.random.default_rng(trial + 100)
                    x = np.linspace(xlo, xhi, 120)
                    y_clean = mfn(x, theta_true)
                    y = y_clean + rng.normal(0, 0.03, 120)
                    n_out = int(0.30 * 120)
                    idx = rng.choice(120, n_out, replace=False)
                    y[idx] += rng.normal(0, 1.0, n_out)

                    theta0 = default_init(mn)
                    cfg = SolverConfig(mode="dual", f_target=ft, max_iter=150,
                                       use_sign_detection=cdict["sign"],
                                       use_momentum=cdict["mom"],
                                       momentum_beta=0.1)
                    try:
                        res = solve_lf(x, y, theta0, mfn, jfn, cfg)
                        pe = np.linalg.norm(res["theta"] - theta_true)
                    except Exception:
                        pe = np.nan
                    errs.append(pe)
                rows.append({"Model": mn, "f_target": ft, "Config": cname,
                             "Mean_PE": np.nanmean(errs), "errors": errs})

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, mn in enumerate(MODELS):
        ax = axes[idx // 2, idx % 2]
        sub = [r for r in rows if r["Model"] == mn]
        positions = []
        data = []
        labels = []
        pos = 0
        for ft in F_TARGETS:
            for cname in CONFIGS:
                r = [s for s in sub if s["f_target"] == ft and s["Config"] == cname][0]
                data.append([e for e in r["errors"] if not np.isnan(e)])
                positions.append(pos)
                labels.append(f"{cname}\n$f$={ft}")
                pos += 1
            pos += 0.5  # gap between f groups

        bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True)
        config_colors = {"Baseline": "#2196F3", "+SignDet": "#4CAF50",
                         "+Momentum": "#FF9800", "+Both": "#F44336"}
        for i, patch in enumerate(bp["boxes"]):
            cname = list(CONFIGS.keys())[i % len(CONFIGS)]
            patch.set_facecolor(config_colors[cname])
            patch.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=6, rotation=45)
        ax.set_ylabel("Parameter Error")
        ax.set_yscale("log")
        ax.set_title(mn)

    # Legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.7)
                       for c in config_colors.values()]
    axes[0, 0].legend(legend_elements, list(CONFIGS.keys()), fontsize=7)

    plt.suptitle("Ablation: Sign Detection and Momentum (20 trials, 30% outliers)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{OUT}/fig4_ablation.{ext}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved fig4_ablation.{ext}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 5: Feature pruning
# ═══════════════════════════════════════════════════════════════════

def fig5(args):
    ext = get_ext(args)
    print("  Generating Figure 5: Feature pruning...")

    rng = np.random.default_rng(0)
    n, p, p_rel = 200, 15, 5
    theta_true = np.zeros(p)
    theta_true[:p_rel] = rng.standard_normal(p_rel) * 2
    X = rng.standard_normal((n, p))
    X[:, p_rel:p_rel+3] += 0.5 * X[:, :3]  # correlate features 5-7 with 0-2
    y_clean = X @ theta_true
    y = y_clean + rng.normal(0, 0.1, n)
    n_out = int(0.20 * n)
    idx_out = rng.choice(n, n_out, replace=False)
    y[idx_out] += rng.normal(0, 5.0, n_out)

    theta0 = np.zeros(p)
    res = lf_feature_pruning(X, y, theta0, f_target=0.5, tol=0.05,
                              min_features=1, verbose=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) Residual vs features
    ax = axes[0]
    n_features = [e["n_features"] for e in res["error_history"]]
    resid_norms = [e["resid_norm"] for e in res["error_history"]]
    ax.plot(n_features, resid_norms, "o-", color="#2196F3", lw=2)
    ax.axvline(p_rel, color="red", ls="--", alpha=0.7, label=f"True support ({p_rel})")
    ax.set_xlabel("Number of features")
    ax.set_ylabel("Residual norm")
    ax.set_title("(a) Residual vs retained features")
    ax.legend(fontsize=8)
    ax.invert_xaxis()

    # (b) Initial importance scores
    ax = axes[1]
    if res["importance_history"]:
        imp0 = res["importance_history"][0]["importance"]
        feats = sorted(imp0.keys())
        vals = [imp0[f] for f in feats]
        colors_bar = ["#4CAF50" if f < p_rel else "#F44336" for f in feats]
        ax.bar(range(len(feats)), vals, color=colors_bar, alpha=0.8)
        ax.set_xlabel("Feature index")
        ax.set_ylabel("SVD importance")
        ax.set_title("(b) Initial importance scores")
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="#4CAF50", alpha=0.8),
                           plt.Rectangle((0, 0), 1, 1, fc="#F44336", alpha=0.8)],
                  labels=["Relevant", "Irrelevant"], fontsize=7)

    # (c) Elimination order
    ax = axes[2]
    elim = res["eliminated_features"]
    if elim:
        colors_elim = ["#4CAF50" if f >= p_rel else "#F44336" for f in elim]
        ax.barh(range(len(elim)), [1] * len(elim), color=colors_elim, alpha=0.8)
        ax.set_yticks(range(len(elim)))
        ax.set_yticklabels([f"Feature {f}" for f in elim], fontsize=7)
    ax.set_xlabel("Elimination order (first removed at bottom)")
    ax.set_title("(c) Elimination path")

    plt.suptitle("SVD-Based Feature Pruning (15 features, 5 relevant, 20% outliers)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{OUT}/fig5_feature_pruning.{ext}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved fig5_feature_pruning.{ext}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 6: 2D Objective landscape
# ═══════════════════════════════════════════════════════════════════

def fig6(args):
    ext = get_ext(args)
    print("  Generating Figure 6: Objective landscape...")

    rng = np.random.default_rng(42)
    theta_true = TRUE_THETA["exp_decay"]
    x = np.linspace(0.0, 5.0, 120)
    y_clean = model_exp_decay(x, theta_true)
    y = y_clean + rng.normal(0, 0.03, 120)
    n_out = int(0.25 * 120)
    idx = rng.choice(120, n_out, replace=False)
    y[idx] += rng.normal(0, 1.0, n_out)

    # Handle negative parameters correctly
    t0_lo, t0_hi = sorted([theta_true[0] * 0.3, theta_true[0] * 1.7])
    t1_lo, t1_hi = sorted([theta_true[1] * 0.3, theta_true[1] * 1.7])
    t0_range = np.linspace(t0_lo, t0_hi, 80)
    t1_range = np.linspace(t1_lo, t1_hi, 80)
    T0, T1 = np.meshgrid(t0_range, t1_range)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax_idx, (f_val, eps_val) in enumerate([(2.0, 0.0), (1.0, 0.01), (0.5, 1e-6)]):
        Z = np.zeros_like(T0)
        for i in range(T0.shape[0]):
            for j in range(T0.shape[1]):
                # Vary first two params, fix remaining at true values
                theta_ij = theta_true.copy()
                theta_ij[0] = T0[i, j]
                theta_ij[1] = T1[i, j]
                d = residuals(x, y, theta_ij, model_exp_decay)
                Z[i, j] = lf_objective(d, f_val, max(eps_val, 1e-10))

        ax = axes[ax_idx]
        Z_log = np.log10(np.maximum(Z, 1e-15))
        cs = ax.contourf(T0, T1, Z_log, levels=30, cmap="viridis")
        ax.plot(theta_true[0], theta_true[1], "r*", ms=12, zorder=5)
        ax.set_xlabel("$\\theta_0$")
        ax.set_ylabel("$\\theta_1$")
        ax.set_title(f"$f={f_val}$, $\\varepsilon={eps_val}$")
        plt.colorbar(cs, ax=ax, label="$\\log_{10}\\mathcal{L}_f$")

    plt.suptitle("$L_f$ Objective Landscape (exp_decay, 25% outliers)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{OUT}/fig6_landscape.{ext}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved fig6_landscape.{ext}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 7: Hessian condition number vs f
# ═══════════════════════════════════════════════════════════════════

def fig7(args):
    ext = get_ext(args)
    print("  Generating Figure 7: Hessian conditioning...")

    MODELS = {
        "exp_decay":     (model_exp_decay, jac_exp_decay),
        "logistic":      (model_logistic, jac_logistic),
        "biexponential": (model_biexp, jac_biexp),
        "gaussian_rbf":  (model_gaussian_rbf, jac_gaussian_rbf),
    }
    X_RANGES = {
        "exp_decay": (0.0, 5.0), "logistic": (0.0, 5.0),
        "biexponential": (0.0, 5.0), "gaussian_rbf": (-3.0, 3.0),
    }

    f_values = np.linspace(0.2, 2.0, 30)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for idx, (mn, (mfn, jfn)) in enumerate(MODELS.items()):
        ax = axes[idx // 2, idx % 2]
        rng = np.random.default_rng(42)
        theta_true = TRUE_THETA[mn]
        xlo, xhi = X_RANGES[mn]
        x = np.linspace(xlo, xhi, 120)
        y_clean = mfn(x, theta_true)
        y = y_clean + rng.normal(0, 0.03, 120)
        n_out = int(0.25 * 120)
        out_idx = rng.choice(120, n_out, replace=False)
        y[out_idx] += rng.normal(0, 1.0, n_out)

        conds = []
        pd_flags = []
        for f_val in f_values:
            d = residuals(x, y, theta_true, mfn)
            w = irls_weights(d, f_val, 1e-6)
            J = jfn(x, theta_true)
            WJ = w[:, None] * J
            H = WJ.T @ WJ
            eigvals = np.linalg.eigvalsh(H)
            conds.append(max(eigvals) / max(min(eigvals), 1e-30))
            pd_flags.append(min(eigvals) > 0)

        for i, (f_v, c_v, pd) in enumerate(zip(f_values, conds, pd_flags)):
            color = "#4CAF50" if pd else "#F44336"
            ax.scatter(f_v, c_v, c=color, s=20, zorder=3)

        ax.axvline(1.0, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("Exponent $f$")
        ax.set_ylabel("Condition number")
        ax.set_yscale("log")
        ax.set_title(mn)

    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#4CAF50',
                              markersize=8, label='Positive definite'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336',
                              markersize=8, label='Not positive definite')]
    axes[0, 0].legend(handles=legend_elements, fontsize=7)

    plt.suptitle("Gauss-Newton Hessian Condition Number vs $f$ (25% outliers)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{OUT}/fig7_hessian_condition.{ext}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved fig7_hessian_condition.{ext}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 8: Leverage-point comparison
# ═══════════════════════════════════════════════════════════════════

def fig8(args):
    ext = get_ext(args)
    print("  Generating Figure 8: Leverage-point comparison...")

    csv_path = f"{OUT}/benchmark_leverage.csv"
    if not os.path.exists(csv_path):
        print(f"    ERROR: {csv_path} not found. Run: python run_benchmark.py --all27 --trials 20")
        return

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Leverage points from benchmark
    ax = axes[0]
    methods_lev = ["Lf_dual", "OLS", "Huber", "Cauchy", "Barron"]
    for m in methods_lev:
        sub = df[df["Method"] == m]
        if len(sub) == 0:
            continue
        ax.plot(sub["Leverage_Ratio"] * 100, sub["Mean_PE"], "o-",
                label=m, color=COLORS.get(m, "#666"), lw=2)
        if "Std_PE" in sub.columns:
            ax.fill_between(sub["Leverage_Ratio"] * 100,
                            sub["Mean_PE"] - sub["Std_PE"],
                            sub["Mean_PE"] + sub["Std_PE"],
                            alpha=0.12, color=COLORS.get(m, "#666"))
    ax.set_xlabel("Leverage-point contamination (%)")
    ax.set_ylabel("Mean Parameter Error")
    ax.set_title("(a) Leverage-point outliers (extreme in X-space)")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    # (b) Response-only outliers
    ax = axes[1]
    N, P = 200, 10
    n_trials = 20
    resp_rows = []
    for out_ratio in [0.0, 0.05, 0.10, 0.15, 0.20]:
        for trial in range(n_trials):
            rng = np.random.default_rng(trial + 3000)
            theta_true = rng.standard_normal(P)
            X = rng.standard_normal((N, P))
            y_clean = X @ theta_true
            y = y_clean + rng.normal(0, 0.1, N)
            if out_ratio > 0:
                n_out = int(round(out_ratio * N))
                oi = rng.choice(N, n_out, replace=False)
                y[oi] += rng.normal(0, 5.0, n_out)

            theta0 = np.zeros(P)
            # Lf_dual
            try:
                cfg = SolverConfig(mode="dual", f_target=0.5, max_iter=200,
                                   use_sign_detection=True, adaptive_f=True)
                res = solve_lf_adaptive(X, y, theta0.copy(), model_linear_nd,
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

            for method, pe in [("Lf_dual", pe_lf), ("OLS", pe_ols)]:
                resp_rows.append({"Outlier_Ratio": out_ratio, "Method": method,
                                  "Param_Error": pe})

    df_resp = pd.DataFrame(resp_rows)
    resp_agg = df_resp.groupby(["Outlier_Ratio", "Method"], as_index=False).agg(
        Mean_PE=("Param_Error", "mean"), Std_PE=("Param_Error", "std"))

    for m in ["Lf_dual", "OLS"]:
        sub = resp_agg[resp_agg["Method"] == m]
        ax.plot(sub["Outlier_Ratio"] * 100, sub["Mean_PE"], "o-",
                label=m, color=COLORS.get(m, "#666"), lw=2)
    ax.set_xlabel("Response-only contamination (%)")
    ax.set_ylabel("Mean Parameter Error")
    ax.set_title("(b) Response-only outliers (additive in Y)")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    plt.suptitle(f"Leverage vs Response Outliers ($n$={N}, $p$={P})", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{OUT}/fig8_leverage.{ext}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved fig8_leverage.{ext}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

FIGURE_MAP = {
    1: ("Outlier robustness (from CSV)", fig1),
    2: ("Convergence curves (solver run)", fig2),
    3: ("IRLS weight evolution (solver run)", fig3),
    4: ("Ablation box plots (solver run)", fig4),
    5: ("Feature pruning (solver run)", fig5),
    6: ("Objective landscape (solver run)", fig6),
    7: ("Hessian conditioning (solver run)", fig7),
    8: ("Leverage comparison (from CSV + solver run)", fig8),
}


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--only", nargs="+", type=int, default=None,
                        help="Generate only these figure numbers (e.g., --only 1 8)")
    parser.add_argument("--png", action="store_true",
                        help="Save as PNG instead of PDF")
    args = parser.parse_args()

    figs_to_gen = args.only if args.only else sorted(FIGURE_MAP.keys())
    ext = "PNG" if args.png else "PDF"

    print(f"Generating {len(figs_to_gen)} figure(s) as {ext}...\n")
    for fig_num in figs_to_gen:
        if fig_num not in FIGURE_MAP:
            print(f"  Unknown figure number: {fig_num}")
            continue
        desc, func = FIGURE_MAP[fig_num]
        print(f"  [Fig {fig_num}] {desc}")
        try:
            func(args)
        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. Output in {OUT}/")


if __name__ == "__main__":
    main()
