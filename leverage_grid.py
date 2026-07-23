#!/usr/bin/env python3
"""
leverage_grid.py — expanded leverage-point experiment.

Grid: n in {100, 200, 400} x p in {5, 10, 20} x leverage magnitude
m in {4, 8, 12} (predictor SDs), at 10% and 20% leverage contamination,
20 seeds, five methods (Lf_dual, OLS, Huber, Cauchy, Barron).

Protocol generalizes run_leverage_benchmark: predictors N(0,1); clean
response noise SD 0.1; leverage rows get X[i] ~ N(0, m) and
y[i] = X[i]@theta_true + N(0, 2m); seed default_rng(trial + 5000);
metric = ||theta_hat - theta_true||_2 (absolute PE). Lf_dual and OLS start
from theta0 = 0; the scale-dependent baselines (Huber, Cauchy, Barron)
receive the two-phase scale and are initialized at the OLS solution
(standard practice for redescending M-estimators, favorable to the
baselines). The (n=200, p=10, m=8) cell coincides with the single-setting
experiment of run_benchmark.py and serves as the reproduction anchor.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lf_norm import (SolverConfig, solve_lf_adaptive, solve_barron_annealing,
                     estimate_initial_scale, model_linear_nd, jac_linear_nd)

OUT_CSV = "experiment_results/leverage_grid.csv"
os.makedirs("experiment_results", exist_ok=True)

NS = [100, 200, 400]
PS = [5, 10, 20]
MAGS = [4.0, 8.0, 12.0]
RATIOS = [0.10, 0.20]
N_TRIALS = 20


def one_trial(n, p, mag, ratio, trial):
    rng = np.random.default_rng(trial + 5000)
    theta_true = rng.standard_normal(p)
    X = rng.standard_normal((n, p))
    y = X @ theta_true + rng.normal(0, 0.1, n)
    n_lev = int(round(ratio * n))
    lev_idx = rng.choice(n, n_lev, replace=False)
    for i in lev_idx:
        X[i] = rng.normal(0, mag, p)
        y[i] = X[i] @ theta_true + rng.normal(0, 2.0 * mag)
    th0 = np.zeros(p)
    # PROTOCOL: scale-dependent baselines (Huber, Cauchy, Barron) are
    # initialized at the OLS solution (standard practice for redescending
    # M-estimators; favorable to the baselines). Lf_dual and OLS use theta0=0.
    out = {}
    try:
        cfg = SolverConfig(mode="dual", f_target=0.5, max_iter=200,
                           use_sign_detection=True, adaptive_f=True)
        r = solve_lf_adaptive(X, y, th0.copy(), model_linear_nd,
                              jac_linear_nd, cfg)
        out["Lf_dual"] = float(np.linalg.norm(r["theta"] - theta_true))
    except Exception:
        out["Lf_dual"] = np.inf
    try:
        t_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
        out["OLS"] = float(np.linalg.norm(t_ols - theta_true))
    except Exception:
        t_ols = th0.copy()
        out["OLS"] = np.inf
    try:
        s = estimate_initial_scale(X, y, th0.copy(), model_linear_nd,
                                   jac_linear_nd)
    except Exception:
        s = 1.0
    for loss in ("huber", "cauchy"):
        try:
            r = least_squares(fun=lambda th: model_linear_nd(X, th) - y,
                              x0=t_ols.copy(),
                              jac=lambda th: jac_linear_nd(X, th),
                              method="trf", loss=loss, f_scale=s,
                              max_nfev=5000)
            out[loss.capitalize()] = float(np.linalg.norm(r.x - theta_true))
        except Exception:
            out[loss.capitalize()] = np.inf
    try:
        r = solve_barron_annealing(X, y, t_ols.copy(), model_linear_nd,
                                   jac_linear_nd, c=s, alpha_target=0.0,
                                   max_iter=200)
        out["Barron"] = float(np.linalg.norm(r["theta"] - theta_true))
    except Exception:
        out["Barron"] = np.inf
    return out


def main():
    done = set()
    if os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV)
        done = set(zip(prev.n, prev.p, prev.Magnitude))
        print(f"resuming; {len(done)} cells done", flush=True)
    first = not os.path.exists(OUT_CSV)
    for n in NS:
        for p in PS:
            for mag in MAGS:
                if (n, p, mag) in done:
                    continue
                t0 = time.time()
                rows = []
                for ratio in RATIOS:
                    for trial in range(N_TRIALS):
                        pe = one_trial(n, p, mag, ratio, trial)
                        for m, v in pe.items():
                            rows.append(dict(n=n, p=p, Magnitude=mag,
                                             Leverage_Ratio=ratio,
                                             Trial=trial, Method=m,
                                             Param_Error=v))
                pd.DataFrame(rows).to_csv(OUT_CSV, mode="a", index=False,
                                          header=first)
                first = False
                print(f"n={n} p={p} m={mag:g} in {time.time()-t0:5.1f}s",
                      flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
