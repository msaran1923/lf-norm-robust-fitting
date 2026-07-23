#!/usr/bin/env python3
"""
hyperparam_sensitivity.py — one-at-a-time sensitivity for f_target, tau,
delta. Representative NIST subset (all three difficulty levels),
{0, 10, 30}% contamination, 20 seeds (clean run once: deterministic).

Isolation protocol: the core main-benchmark configuration (mode=dual, mu0=1e-4,
max_iter=200, sign detection on, momentum off, polish on) with the
residual-adaptive f-target selection DISABLED, since the adaptive wrapper
would override f_target with min(f_candidates). A 'published' reference config (the full pipeline: adaptive-f enabled,
defaults) is included for context.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiments import get_nist_datasets
from lf_norm import SolverConfig, solve_lf, solve_lf_adaptive

OUT_CSV = "experiment_results/hyperparam_sensitivity.csv"
os.makedirs("experiment_results", exist_ok=True)

DATASETS = ["Misra1a", "Chwirut2", "Lanczos3", "MGH17", "Hahn1",
            "Kirby2", "Nelson", "Thurber"]
FRACS = [0.0, 0.10, 0.30]
N_TRIALS = 20

def base_cfg(**kw):
    c = SolverConfig(mode="dual", f_target=0.5, tau=0.95, delta=0.05,
                     max_iter=200, eps0=1.0, mu0=1e-4,
                     use_sign_detection=True, use_momentum=False,
                     polish=True, adaptive_f=False)
    for k, v in kw.items():
        setattr(c, k, v)
    return c

CONFIGS = {
    "default":  base_cfg(),
    "f=0.3":    base_cfg(f_target=0.3),
    "f=0.8":    base_cfg(f_target=0.8),
    "f=1.0":    base_cfg(f_target=1.0),
    "tau=0.90": base_cfg(tau=0.90),
    "tau=0.99": base_cfg(tau=0.99),
    "delta=0.01": base_cfg(delta=0.01),
    "delta=0.10": base_cfg(delta=0.10),
    "published":  base_cfg(adaptive_f=True),   # adaptive-f + defaults
}


def main():
    DS = get_nist_datasets(use_all_27=True)
    done = set()
    if os.path.exists(OUT_CSV):
        done = set(pd.read_csv(OUT_CSV)["Dataset"].unique())
        print(f"resuming; done: {sorted(done)}", flush=True)
    first = not os.path.exists(OUT_CSV)

    for ds_name in DATASETS:
        if ds_name in done:
            continue
        t0 = time.time()
        ds = DS[ds_name]
        x, yc = ds["x"], ds["y"]
        th0, thc = ds["theta_start"], ds["theta_cert"]
        nrm = max(np.linalg.norm(thc), 1e-30)
        y_range = yc.max() - yc.min()
        rows = []
        for frac in FRACS:
            trials = 1 if frac == 0 else N_TRIALS
            for trial in range(trials):
                rng = np.random.default_rng(trial + 2000)
                y = yc.copy()
                if frac > 0:
                    n_out = max(1, int(round(frac * len(y))))
                    idx = rng.choice(len(y), n_out, replace=False)
                    y[idx] += rng.normal(0, y_range * 0.5, n_out)
                for cname, cfg in CONFIGS.items():
                    try:
                        c = base_cfg()  # fresh copy each solve
                        for k in ("f_target", "tau", "delta", "adaptive_f"):
                            setattr(c, k, getattr(cfg, k))
                        solver = solve_lf_adaptive if c.adaptive_f else solve_lf
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
        print(f"{ds_name:9s} done in {time.time()-t0:6.1f}s", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
