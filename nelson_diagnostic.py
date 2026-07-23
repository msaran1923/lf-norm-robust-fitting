#!/usr/bin/env python3
"""
nelson_diagnostic.py — traces the Nelson failure chain.

Reports, for the Nelson dataset (log y = b1 - b2*x1*exp(-b3*x2)):
  1. Jacobian conditioning at the standard NIST Start-2 values and at the
     certified solution, plus the collinearity |cos angle| between the b2
     and b3 columns.
  2. The OLS fit from the standard start: parameter-space RE, terminal b2,
     and residual SD versus the certified-fit residual SD (showing the
     drifted solution is nearly residual-equivalent).
  3. The two-phase scale trace (phase-1 OLS MAD, phase-2 Huber-refit MAD,
     final min) versus the certified-fit residual scale, on clean data and
     on a 10%-contaminated trial — showing the scale estimate stays within
     ~25% of the certified residual scale despite the OLS divergence.

Conclusion encoded here and in the paper (Sec. 4.2, "Anatomy of the Nelson
failure"): the Nelson catastrophic failures are parameter-IDENTIFIABILITY
failures (drift along a near-flat Jacobian direction toward
residual-equivalent solutions), not scale-estimation failures. The
scale-sensitivity experiment corroborates this: every baseline fails on
Nelson at every scale setting, including the oracle.
"""
import sys, warnings
import numpy as np
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from lf_norm import mad_scale, residuals
from run_experiments import get_nist_datasets


def main():
    ds = get_nist_datasets(use_all_27=True)["Nelson"]
    x, y = ds["x"], ds["y"]
    th0, thc = ds["theta_start"], ds["theta_cert"]
    model, jac = ds["model"], ds["jac"]
    print(f"n = {len(y)} | start (NIST Start-2) = {th0} | certified = {thc}")

    for name, th in [("Start-2", th0), ("certified", thc)]:
        J = jac(x, th)
        s = np.linalg.svd(J, compute_uv=False)
        c23 = abs(np.dot(J[:, 1], J[:, 2])
                  / (np.linalg.norm(J[:, 1]) * np.linalg.norm(J[:, 2])))
        print(f"J at {name:9s}: cond = {s[0]/s[-1]:.3e}  "
              f"|cos angle(b2,b3 cols)| = {c23:.6f}")

    def scale_trace(yy, label, clean_mask=None):
        r1 = least_squares(fun=lambda th: residuals(x, yy, th, model),
                           x0=th0.copy(), jac=lambda th: jac(x, th),
                           method="trf", loss="linear", max_nfev=2000)
        s1 = mad_scale(residuals(x, yy, r1.x, model))
        r2 = least_squares(fun=lambda th: residuals(x, yy, th, model),
                           x0=r1.x.copy(), jac=lambda th: jac(x, th),
                           method="trf", loss="huber", f_scale=s1,
                           max_nfev=2000)
        s2 = mad_scale(residuals(x, yy, r2.x, model))
        d_cert = residuals(x, yy, thc, model)
        ref = np.std(d_cert if clean_mask is None else d_cert[clean_mask])
        nrm = np.linalg.norm(thc)
        print(f"\n[{label}]")
        print(f"  phase-1 OLS:   RE = {np.linalg.norm(r1.x-thc)/nrm:.4g}  "
              f"b2 = {r1.x[1]:.4g}  resid SD = "
              f"{np.std(residuals(x, yy, r1.x, model)):.4g}  MAD s1 = {s1:.4g}")
        print(f"  phase-2 Huber: RE = {np.linalg.norm(r2.x-thc)/nrm:.4g}  "
              f"MAD s2 = {s2:.4g}")
        print(f"  final scale = {min(s1, s2):.4g}  vs certified-fit residual "
              f"SD = {ref:.4g}  (ratio {min(s1, s2)/ref:.3g})")

    scale_trace(y, "clean data")

    rng = np.random.default_rng(2000)  # trial 0 of the benchmark
    yc = y.copy()
    n_out = max(1, int(round(0.1 * len(y))))
    idx = rng.choice(len(y), n_out, replace=False)
    yc[idx] += rng.normal(0, (y.max() - y.min()) * 0.5, n_out)
    mask = np.ones(len(y), bool); mask[idx] = False
    scale_trace(yc, "10% contaminated (trial 0)", clean_mask=mask)


if __name__ == "__main__":
    main()
