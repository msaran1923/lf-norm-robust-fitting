#!/usr/bin/env python3
"""
scale_sensitivity.py — Baseline scale-parameter sensitivity analysis.

For every contaminated NIST scenario (27 datasets x {10,20,30}% x 20 seeds),
the two-phase MAD scale s is estimated once, then each scale-dependent
baseline (Huber, Cauchy, GM, Welsch, Tukey, Barron) is re-run with:
    s_eff in {0.5s, 1.0s, 2.0s, 4.0s}  and  s_oracle
where s_oracle = std of the certified-fit residuals on the clean
(non-contaminated) observations — information unavailable in practice.

OLS is excluded (no scale parameter). The 1.0x setting reproduces the
main-benchmark pipeline (verified bit-exact for Huber/Cauchy/GM/Welsch/Tukey
under numpy==1.26.4 / scipy==1.12.0; Barron matches to ~4e-5 relative).

Checkpointing: results are appended per dataset to OUT_CSV; completed
datasets are skipped on restart.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_experiments import get_nist_datasets
from lf_norm import (residuals, solve_method, SolverConfig,
                     estimate_initial_scale, solve_barron_annealing)

OUT_CSV = "experiment_results/scale_sensitivity.csv"
os.makedirs("experiment_results", exist_ok=True)

FRACS = [0.10, 0.20, 0.30]
N_TRIALS = 20
MULTS = [0.5, 1.0, 2.0, 4.0]
BASELINES = ["Huber", "Cauchy", "GM", "Welsch", "Tukey", "Barron"]


def run_baseline(name, x, y, theta0, model, jac, s_eff):
    if name in ("Huber", "Cauchy"):
        res = least_squares(fun=lambda th: model(x, th) - y, x0=theta0,
                            jac=lambda th: jac(x, th), method="trf",
                            loss=name.lower(), f_scale=s_eff, max_nfev=5000)
        return res.x
    if name in ("GM", "Welsch", "Tukey"):
        key = {"GM": "geman_mcclure", "Welsch": "welsch", "Tukey": "tukey"}[name]
        r = solve_method(key, x, y, theta0, model, jac, SolverConfig(),
                         noise_std=s_eff)
        return r["theta"]
    if name == "Barron":
        r = solve_barron_annealing(x, y, theta0, model, jac, c=s_eff,
                                   alpha_target=0.0, alpha0=2.0,
                                   rho_alpha=0.92, beta=5.0, mu0=1e-3,
                                   max_iter=200, max_inner=15)
        return r["theta"]
    raise ValueError(name)


def main():
    DS = get_nist_datasets(use_all_27=True)
    done = set()
    if os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV)
        done = set(zip(prev["Dataset"], prev["Outlier_Frac"]))
        print(f"resuming; {len(done)} (dataset,frac) chunks already done",
              flush=True)
    first_write = not os.path.exists(OUT_CSV)

    for di, (ds_name, ds) in enumerate(DS.items()):
        t_ds = time.time()
        x, yc = ds["x"], ds["y"]
        th0, thc = ds["theta_start"], ds["theta_cert"]
        model, jac = ds["model"], ds["jac"]
        norm_cert = max(np.linalg.norm(thc), 1e-30)
        y_range = yc.max() - yc.min()
        for frac in FRACS:
            if (ds_name, frac) in done:
                continue
            t_fr = time.time()
            rows = []
            for trial in range(N_TRIALS):
                rng = np.random.default_rng(trial + 2000)
                y = yc.copy()
                n_out = max(1, int(round(frac * len(y))))
                idx = rng.choice(len(y), n_out, replace=False)
                y[idx] += rng.normal(0, y_range * 0.5, n_out)
                clean = np.ones(len(y), bool); clean[idx] = False

                try:
                    s2p = estimate_initial_scale(x, y, th0.copy(), model, jac)
                except Exception:
                    s2p = np.nan
                d_cert = model(x, thc) - y
                s_orc = max(float(np.std(d_cert[clean])), 1e-12)

                settings = [(f"{m:g}x", m * s2p) for m in MULTS] \
                    + [("oracle", s_orc)]
                for meth in BASELINES:
                    for label, s_eff in settings:
                        try:
                            if not np.isfinite(s_eff):
                                raise ValueError("bad scale")
                            th = run_baseline(meth, x, y, th0.copy(),
                                              model, jac, s_eff)
                            re = np.linalg.norm(th - thc) / norm_cert
                        except Exception:
                            re = np.nan
                        rows.append(dict(Dataset=ds_name, Outlier_Frac=frac,
                                         Trial=trial, Method=meth,
                                         Setting=label, Scale_Used=s_eff,
                                         Rel_Error=re))
            pd.DataFrame(rows).to_csv(OUT_CSV, mode="a", index=False,
                                      header=first_write)
            first_write = False
            print(f"[{di+1:2d}/27] {ds_name:10s} frac={frac:.0%} done in "
                  f"{time.time()-t_fr:6.1f}s", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
