#!/usr/bin/env python3
"""
Comprehensive Benchmark: 20-trial aggregated results
=====================================================
Runs all 8 methods × 6 NIST datasets × 4 contamination levels × 20 trials.
Produces statistically robust tables with mean ± std, win counts, and
catastrophic failure rates.

Usage:
    python run_benchmark.py              # full 20-trial run
    python run_benchmark.py --trials 50  # custom trial count
    python run_benchmark.py --quick      # 5-trial quick check

Output:
    experiment_results/benchmark_full.csv          — raw per-trial results
    experiment_results/benchmark_summary.csv       — aggregated mean/std/median
    experiment_results/benchmark_wins.csv          — win counts per method
    experiment_results/benchmark_catastrophic.csv  — catastrophic failure rates
    experiment_results/benchmark_leverage.csv      — leverage-point results
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
    generate_nd_leverage,
    estimate_initial_scale, solve_barron_annealing,
)
from run_experiments import (
    NIST_DATASETS, _run_lf, _run_scipy, _run_irls, _run_barron,
    get_nist_datasets,
)

OUT = "experiment_results"
os.makedirs(OUT, exist_ok=True)


def print_header(title):
    w = 70
    print("\n" + "=" * w)
    print(f"  {title}")
    print("=" * w)


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 1: NIST Contaminated — Full 8-method comparison
# ═══════════════════════════════════════════════════════════════════

def run_nist_benchmark(n_trials=20, datasets=None):
    """Run all methods on all NIST datasets at all contamination levels."""
    if datasets is None:
        datasets = NIST_DATASETS
    n_ds = len(datasets)
    print_header(f"NIST BENCHMARK ({n_ds} datasets × {n_trials} trials per scenario)")

    # Main comparison: Lf_dual vs 7 external baselines
    METHODS = {
        "Lf_dual":  lambda x, y, t0, m, j: _run_lf(x, y, t0, m, j, "dual"),
        "OLS":      lambda x, y, t0, m, j: _run_scipy(x, y, t0, m, j, "linear"),
        "Huber":    lambda x, y, t0, m, j: _run_scipy(x, y, t0, m, j, "huber"),
        "Cauchy":   lambda x, y, t0, m, j: _run_scipy(x, y, t0, m, j, "cauchy"),
        "GM":       lambda x, y, t0, m, j: _run_irls(x, y, t0, m, j, "geman_mcclure"),
        "Welsch":   lambda x, y, t0, m, j: _run_irls(x, y, t0, m, j, "welsch"),
        "Tukey":    lambda x, y, t0, m, j: _run_irls(x, y, t0, m, j, "tukey"),
        "Barron":   lambda x, y, t0, m, j: _run_barron(x, y, t0, m, j, alpha_target=0.0),
    }
    # Ablation variant: Lf without SVD rank truncation
    ABLATION_METHODS = {
        "Lf_full":  lambda x, y, t0, m, j: _run_lf(x, y, t0, m, j, "full"),
    }

    OUTLIER_FRACS = [0.0, 0.10, 0.20, 0.30]

    rows = []
    total_combos = len(datasets) * len(OUTLIER_FRACS)
    done_combos = 0
    all_methods = {**METHODS, **ABLATION_METHODS}

    for ds_name, ds in datasets.items():
        x, y_clean = ds["x"], ds["y"]
        theta0 = ds["theta_start"]
        theta_cert = ds["theta_cert"]
        norm_cert = max(np.linalg.norm(theta_cert), 1e-30)
        y_range = y_clean.max() - y_clean.min()

        for frac in OUTLIER_FRACS:
            for trial in range(n_trials):
                rng = np.random.default_rng(trial + 2000)
                y = y_clean.copy()
                if frac > 0:
                    n_out = max(1, int(round(frac * len(y))))
                    idx = rng.choice(len(y), n_out, replace=False)
                    y[idx] += rng.normal(0, y_range * 0.5, n_out)

                for mname, runner in all_methods.items():
                    try:
                        t_start = time.perf_counter()
                        theta_hat, _ = runner(x, y, theta0.copy(),
                                              ds["model"], ds["jac"])
                        runtime_ms = (time.perf_counter() - t_start) * 1000
                        rel_err = np.linalg.norm(theta_hat - theta_cert) / norm_cert
                    except Exception:
                        rel_err = np.nan
                        runtime_ms = np.nan
                    rows.append({
                        "Dataset": ds_name,
                        "Difficulty": ds["difficulty"],
                        "Method": mname,
                        "Is_Ablation": mname in ABLATION_METHODS,
                        "Outlier_Frac": frac,
                        "Trial": trial,
                        "Rel_Error": rel_err,
                        "Runtime_ms": runtime_ms,
                    })

            done_combos += 1
            sys.stdout.write(f"\r  {ds_name:10s} {frac:3.0%} done  "
                             f"[{done_combos}/{total_combos} scenarios, "
                             f"{n_trials} trials × {len(all_methods)} methods each]")
            sys.stdout.flush()

    print()
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/benchmark_full.csv", index=False)
    print(f"  Raw results: {len(df)} rows saved to benchmark_full.csv")
    return df


def analyze_results(df):
    """Produce summary tables, win counts, catastrophic failure analysis,
    and ablation comparison (Lf_dual vs Lf_full)."""

    # Split main methods from ablation
    df_main = df[~df.get("Is_Ablation", False)].copy()
    df_ablation = df[df.get("Is_Ablation", False)].copy()
    has_ablation = len(df_ablation) > 0

    # ── 1. Summary statistics (all methods) ──
    summary = df.groupby(["Dataset", "Outlier_Frac", "Method"], as_index=False).agg(
        Mean_RE=("Rel_Error", "mean"),
        Std_RE=("Rel_Error", "std"),
        Median_RE=("Rel_Error", "median"),
        Min_RE=("Rel_Error", "min"),
        Max_RE=("Rel_Error", "max"),
        Mean_RT=("Runtime_ms", "mean"),
    )
    summary.to_csv(f"{OUT}/benchmark_summary.csv", index=False)

    # ── 2. Main comparison (excluding ablation) ──
    summary_main = summary[~summary["Method"].isin(["Lf_full"])]
    contam_main = summary_main[summary_main["Outlier_Frac"] > 0]

    print_header("MAIN COMPARISON: Lf_dual vs 6 external baselines")
    pivot_mean = contam_main.pivot_table(
        index=["Dataset", "Outlier_Frac"],
        columns="Method", values="Mean_RE"
    )
    print("\n  Mean Relative Error (lower is better):")
    print("  " + "-" * 90)
    print(pivot_mean.to_string(float_format="%.3e"))

    # ── 3. Win counts (main methods only) ──
    print_header("WIN COUNTS (main methods only)")
    main_methods = sorted(df_main["Method"].unique())
    wins = {m: 0 for m in main_methods}
    within_2x = {m: 0 for m in main_methods}
    n_scenarios = 0

    for (ds, frac), grp in contam_main.groupby(["Dataset", "Outlier_Frac"]):
        n_scenarios += 1
        best_method = grp.loc[grp["Mean_RE"].idxmin(), "Method"]
        best_val = grp["Mean_RE"].min()
        wins[best_method] += 1

        for _, row in grp.iterrows():
            if row["Mean_RE"] <= 2.0 * best_val:
                within_2x[row["Method"]] += 1

    print(f"\n  Total contaminated scenarios: {n_scenarios}")
    print(f"\n  {'Method':12s} {'Strict Wins':>12s} {'Within 2x':>10s}")
    print("  " + "-" * 38)
    for m in sorted(wins, key=wins.get, reverse=True):
        print(f"  {m:12s} {wins[m]:>8d}/{n_scenarios:<3d}  "
              f"{within_2x[m]:>6d}/{n_scenarios}")

    win_df = pd.DataFrame([
        {"Method": m, "Strict_Wins": wins[m], "Within_2x": within_2x[m],
         "Total_Scenarios": n_scenarios}
        for m in main_methods
    ])
    win_df.to_csv(f"{OUT}/benchmark_wins.csv", index=False)

    # ── 4. Catastrophic failure analysis (main methods only) ──
    print_header("CATASTROPHIC FAILURE ANALYSIS (main methods)")
    cat_rows = []
    for m in main_methods:
        m_contam = df_main[(df_main["Method"] == m) & (df_main["Outlier_Frac"] > 0)]
        n_total = len(m_contam)
        n_fail = int((m_contam["Rel_Error"] > 1.0).sum())
        n_severe = int((m_contam["Rel_Error"] > 10.0).sum())
        cat_rows.append({
            "Method": m,
            "Total_Trials": n_total,
            "Failures_RE>1": n_fail,
            "Failure_Rate_%": 100.0 * n_fail / max(n_total, 1),
            "Severe_RE>10": n_severe,
            "Severe_Rate_%": 100.0 * n_severe / max(n_total, 1),
        })
        if n_fail > 0 or m == "Lf_dual":
            print(f"  {m:12s}: {n_fail:4d}/{n_total} failures ({100*n_fail/max(n_total,1):.1f}%), "
                  f"{n_severe} severe (RE>10)")

    cat_df = pd.DataFrame(cat_rows)
    cat_df.to_csv(f"{OUT}/benchmark_catastrophic.csv", index=False)

    # ── 5. Overall method ranking (main methods only) ──
    print_header("OVERALL METHOD RANKING (main methods)")
    overall = contam_main.groupby("Method", as_index=False).agg(
        Grand_Mean_RE=("Mean_RE", "mean"),
        Median_of_Means=("Mean_RE", "median"),
        Worst_Case=("Mean_RE", "max"),
        Mean_RT=("Mean_RT", "mean"),
    ).sort_values("Grand_Mean_RE")
    print()
    print(overall.to_string(index=False, float_format="%.4f"))

    # ── 6. Ablation: Lf_dual vs Lf_full ──
    if has_ablation:
        print_header("ABLATION: Effect of SVD rank truncation (Lf_dual vs Lf_full)")
        contam_all = summary[summary["Outlier_Frac"] > 0]
        ablation_methods = ["Lf_dual", "Lf_full"]
        abl_data = contam_all[contam_all["Method"].isin(ablation_methods)]

        abl_overall = abl_data.groupby("Method", as_index=False).agg(
            Grand_Mean_RE=("Mean_RE", "mean"),
            Median_of_Means=("Mean_RE", "median"),
            Worst_Case=("Mean_RE", "max"),
            Mean_RT=("Mean_RT", "mean"),
        ).sort_values("Grand_Mean_RE")
        print("\n  Overall comparison:")
        print(abl_overall.to_string(index=False, float_format="%.4f"))

        # Catastrophic failures for ablation
        for m in ablation_methods:
            m_data = df[(df["Method"] == m) & (df["Outlier_Frac"] > 0)]
            n_total = len(m_data)
            n_fail = int((m_data["Rel_Error"] > 1.0).sum())
            n_severe = int((m_data["Rel_Error"] > 10.0).sum())
            print(f"\n  {m:12s}: {n_fail}/{n_total} catastrophic ({100*n_fail/max(n_total,1):.1f}%), "
                  f"{n_severe} severe (RE>10)")

        # Per-scenario comparison
        print("\n  Per-scenario mean RE (Lf_dual vs Lf_full):")
        print("  " + "-" * 55)
        abl_pivot = abl_data.pivot_table(
            index=["Dataset", "Outlier_Frac"],
            columns="Method", values="Mean_RE"
        )
        if "Lf_dual" in abl_pivot.columns and "Lf_full" in abl_pivot.columns:
            abl_pivot["Ratio_full/dual"] = abl_pivot["Lf_full"] / abl_pivot["Lf_dual"]
            print(abl_pivot.to_string(float_format="%.3e"))

        abl_data.to_csv(f"{OUT}/benchmark_ablation_rank.csv", index=False)

    return summary, win_df, cat_df


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 2: Leverage Points — 20-trial
# ═══════════════════════════════════════════════════════════════════

def run_leverage_benchmark(n_trials=20):
    """Run leverage-point experiment with proper trial count."""
    print_header(f"LEVERAGE-POINT BENCHMARK ({n_trials} trials)")

    N_SAMPLES = 200
    N_FEATURES = 10
    LEV_RATIOS = [0.0, 0.05, 0.10, 0.15, 0.20]

    rows = []
    for lev_ratio in LEV_RATIOS:
        for trial in range(n_trials):
            rng = np.random.default_rng(trial + 5000)
            theta_true = rng.standard_normal(N_FEATURES)
            X = rng.standard_normal((N_SAMPLES, N_FEATURES))
            y_clean = X @ theta_true
            y = y_clean + rng.normal(0, 0.1, N_SAMPLES)

            if lev_ratio > 0:
                n_lev = int(round(lev_ratio * N_SAMPLES))
                lev_idx = rng.choice(N_SAMPLES, n_lev, replace=False)
                for i in lev_idx:
                    X[i] = rng.normal(0, 8.0, N_FEATURES)
                    y[i] = X[i] @ theta_true + rng.normal(0, 16.0)

            theta0 = np.zeros(N_FEATURES)

            # Lf_dual (adaptive)
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
                theta_ols = theta0.copy()
                pe_ols = np.inf

            # Estimate MAD scale for baselines
            s_mad = estimate_initial_scale(X, y, theta0.copy(),
                                            model_linear_nd, jac_linear_nd)

            # PROTOCOL: scale-dependent baselines (Huber, Cauchy, Barron)
            # are initialized at the OLS solution -- standard practice for
            # redescending M-estimators and favorable to the baselines. From
            # the distant zero start, a redescending loss with a correctly
            # tight scale treats every observation as an outlier and stalls.
            # Huber (MAD-adaptive)
            try:
                from scipy.optimize import least_squares
                res_h = least_squares(
                    fun=lambda th: model_linear_nd(X, th) - y,
                    x0=theta_ols.copy(), jac=lambda th: jac_linear_nd(X, th),
                    method="trf", loss="huber", f_scale=s_mad, max_nfev=5000)
                pe_huber = float(np.linalg.norm(res_h.x - theta_true))
            except Exception:
                pe_huber = np.inf

            # Cauchy (MAD-adaptive)
            try:
                res_c = least_squares(
                    fun=lambda th: model_linear_nd(X, th) - y,
                    x0=theta_ols.copy(), jac=lambda th: jac_linear_nd(X, th),
                    method="trf", loss="cauchy", f_scale=s_mad, max_nfev=5000)
                pe_cauchy = float(np.linalg.norm(res_c.x - theta_true))
            except Exception:
                pe_cauchy = np.inf

            # Barron (MAD-adaptive, α → 0)
            try:
                res_b = solve_barron_annealing(
                    X, y, theta_ols.copy(), model_linear_nd, jac_linear_nd,
                    c=s_mad, alpha_target=0.0, max_iter=200)
                pe_barron = float(np.linalg.norm(res_b["theta"] - theta_true))
            except Exception:
                pe_barron = np.inf

            for method, pe in [("Lf_dual", pe_lf), ("OLS", pe_ols),
                                ("Huber", pe_huber), ("Cauchy", pe_cauchy),
                                ("Barron", pe_barron)]:
                rows.append({
                    "Leverage_Ratio": lev_ratio,
                    "Trial": trial,
                    "Method": method,
                    "Param_Error": pe,
                })

        sys.stdout.write(f"\r  leverage_ratio={lev_ratio:.0%} done")
        sys.stdout.flush()

    print()
    df_lev = pd.DataFrame(rows)

    lev_summary = df_lev.groupby(["Leverage_Ratio", "Method"], as_index=False).agg(
        Mean_PE=("Param_Error", "mean"),
        Std_PE=("Param_Error", "std"),
        Median_PE=("Param_Error", "median"),
    )
    lev_summary.to_csv(f"{OUT}/benchmark_leverage.csv", index=False)

    print("\n\n  Mean Parameter Error (Leverage Points):")
    print("  " + "-" * 60)
    pivot = lev_summary.pivot_table(
        index="Leverage_Ratio", columns="Method", values="Mean_PE")
    print(pivot.to_string(float_format="%.4f"))

    # Std
    print("\n  Std Parameter Error:")
    pivot_std = lev_summary.pivot_table(
        index="Leverage_Ratio", columns="Method", values="Std_PE")
    print(pivot_std.to_string(float_format="%.4f"))

    return df_lev


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Benchmark")
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of trials per scenario (default: 20)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (5 trials)")
    parser.add_argument("--all27", action="store_true",
                        help="Use all 27 NIST StRD datasets (requires nist_all_data.py)")
    args = parser.parse_args()

    n_trials = 5 if args.quick else args.trials

    # Select datasets
    datasets = get_nist_datasets(use_all_27=args.all27)
    print(f"  Using {len(datasets)} NIST datasets")

    t0 = time.time()

    # Run NIST benchmark
    df_nist = run_nist_benchmark(n_trials=n_trials, datasets=datasets)
    summary, wins, catastrophic = analyze_results(df_nist)

    # Run leverage benchmark
    df_lev = run_leverage_benchmark(n_trials=n_trials)

    elapsed = time.time() - t0

    # Final summary
    print_header("BENCHMARK COMPLETE")
    print(f"\n  NIST datasets: {len(datasets)}")
    print(f"  Trials per scenario: {n_trials}")
    print(f"  Total runtime: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"\n  Output files:")
    for f in sorted(os.listdir(OUT)):
        if f.startswith("benchmark"):
            fpath = os.path.join(OUT, f)
            size = os.path.getsize(fpath)
            print(f"    {f:45s}  {size:>8,} bytes")


if __name__ == "__main__":
    main()
