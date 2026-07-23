#!/usr/bin/env python3
"""
component_ablation.py — incremental C1-C7 component ablation over the full contaminated NIST benchmark (27 x {10,20,30}% x 20 seeds) plus
the clean suite (deterministic, one solve per dataset).

C1  Lf only: solved directly at f_target (f0=0.5, eps0=eps_min; no annealing),
    full rank, no polish, no adaptive-f
C2  C1 + warm-start annealing (f0=2, eps0=1)
C3  C2 + energy-only SVD truncation
C4  C2 + dual-criterion SVD truncation (core)
C5  C4 + dual-strategy polishing
C6  C4 + residual-adaptive f-target selection
C7  C4 - sign detection
(The full pipeline Lf_dual = C4 + polish + adaptive-f + sign detection is
the reference row, taken from benchmark_full.csv.)
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiments import get_nist_datasets
from lf_norm import SolverConfig, solve_lf, solve_lf_adaptive

OUT_CSV = "experiment_results/component_ablation.csv"
os.makedirs("experiment_results", exist_ok=True)
FRACS = [0.0, 0.10, 0.20, 0.30]
N_TRIALS = 20


def cfg_for(name):
    c = SolverConfig(mode="full", f_target=0.5, max_iter=200, eps0=1.0,
                     mu0=1e-4, use_sign_detection=True, use_momentum=False,
                     polish=False, adaptive_f=False)
    if name == "C1":
        c.f0 = 0.5; c.eps0 = 1e-8
    elif name == "C2":
        pass
    elif name == "C3":
        c.mode = "energy"
    elif name == "C4":
        c.mode = "dual"
    elif name == "C5":
        c.mode = "dual"; c.polish = True
    elif name == "C6":
        c.mode = "dual"; c.adaptive_f = True
    elif name == "C7":
        c.mode = "dual"; c.use_sign_detection = False
    else:
        raise ValueError(name)
    return c


CONFIGS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]


def main():
    DS = get_nist_datasets(use_all_27=True)
    done = set()
    if os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV)
        done = set(zip(prev["Dataset"], prev["Outlier_Frac"]))
        print(f"resuming; {len(done)} chunks done", flush=True)
    first = not os.path.exists(OUT_CSV)

    for di, (ds_name, ds) in enumerate(DS.items()):
        x, yc = ds["x"], ds["y"]
        th0, thc = ds["theta_start"], ds["theta_cert"]
        nrm = max(np.linalg.norm(thc), 1e-30)
        y_range = yc.max() - yc.min()
        for frac in FRACS:
            if (ds_name, frac) in done:
                continue
            t0 = time.time()
            rows = []
            trials = 1 if frac == 0 else N_TRIALS
            for trial in range(trials):
                rng = np.random.default_rng(trial + 2000)
                y = yc.copy()
                if frac > 0:
                    n_out = max(1, int(round(frac * len(y))))
                    idx = rng.choice(len(y), n_out, replace=False)
                    y[idx] += rng.normal(0, y_range * 0.5, n_out)
                for cname in CONFIGS:
                    try:
                        c = cfg_for(cname)
                        solver = (solve_lf_adaptive if c.adaptive_f
                                  else solve_lf)
                        res = solver(x, y, th0.copy(), ds["model"],
                                     ds["jac"], c)
                        re = np.linalg.norm(res["theta"] - thc) / nrm
                    except Exception:
                        re = np.nan
                    rows.append(dict(Dataset=ds_name, Outlier_Frac=frac,
                                     Trial=trial, Config=cname,
                                     Rel_Error=re))
            pd.DataFrame(rows).to_csv(OUT_CSV, mode="a", index=False,
                                      header=first)
            first = False
            print(f"[{di+1:2d}/27] {ds_name:10s} frac={frac:.0%} "
                  f"in {time.time()-t0:5.1f}s", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
