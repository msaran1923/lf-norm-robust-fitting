"""
Lf-Norm Robust Nonlinear Regression — Implementation
=============================================================
Implements:
  - SVD-based solver with dual-rank selection, LM inner loop,
    warm-start scheduling, feature importance, backward elimination
  - Sign detection for indefinite Hessian (f<1), momentum
    acceleration, 27 NIST StRD benchmark functions with
    analytical Jacobians
"""

import math
import time
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import median_abs_deviation, kurtosis as _scipy_kurtosis


# ═══════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SyntheticConfig:
    model_name: str = "exp_decay"
    n_samples: int = 120
    x_min: float = 0.0
    x_max: float = 5.0
    noise_std: float = 0.03
    outlier_ratio: float = 0.20
    outlier_std: float = 1.0
    outlier_dist: str = "gaussian"
    seed: int = 0


@dataclass
class SolverConfig:
    """
    Solver hyperparameters.
    eps0=1.0: starting with large epsilon heavily smooths the Lf objective
    so early iterations behave like OLS. As eps decays toward 0, the true
    Lf behavior emerges, but by then we are near the solution basin.
    """
    mode: str = "dual"                # "full", "energy", or "dual"
    f0: float = 2.0                   # initial exponent (convex start)
    f_target: float = 0.5             # target exponent (robust)
    rho_f: float = 0.92               # multiplicative decay for f
    eps0: float = 1.0                 # initial smoothing
    eps_min: float = 1e-8             # smoothing floor
    rho_eps: float = 0.70             # multiplicative decay for epsilon
    tau: float = 0.95                 # SVD energy threshold
    delta: float = 0.05               # gradient capture threshold
    beta: float = 5.0                 # LM damping factor
    mu0: float = 1e-3                 # initial LM damping
    mu_min: float = 1e-8              # minimum LM damping
    max_iter: int = 120               # iteration budget
    max_inner: int = 15               # max inner LM iterations per outer step
    grad_tol: float = 1e-8            # gradient norm convergence
    step_tol: float = 1e-10           # step norm convergence
    obj_tol: float = 1e-12            # relative objective convergence
    use_sign_detection: bool = True   # enable dual-direction for f<1
    use_momentum: bool = False        # enable momentum acceleration
    momentum_beta: float = 0.1        # momentum coefficient
    capture_spectrum: bool = False    # record singular value history
    adaptive_damping: bool = True     # gain-ratio adaptive damping (Nielsen)
    rho_good: float = 0.75            # gain ratio above this → decrease damping aggressively
    rho_accept: float = 0.0           # gain ratio above this → accept step
    polish: bool = True               # L2 polishing phase after robust convergence
    polish_iter: int = 10             # max polishing iterations (GN with f=2, eps=0)
    adaptive_f: bool = True           # residual-adaptive f-target selection
    f_candidates: tuple = (0.5, 0.8, 1.0)  # candidate f_targets for adaptive selection
    use_rsvd: bool = False            # experimental: randomized SVD for p >> 200
    rsvd_threshold: int = 200         # p threshold (benefits require rapid spectral decay)
    rsvd_oversampling: int = 10       # oversampling parameter for randomized SVD


# ═══════════════════════════════════════════════════════════════════
# 2. CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def residuals(x, y, theta, model):
    return model(x, theta) - y


def lf_objective(d, f, eps):
    d2 = d * d
    return (1.0 / f) * np.sum((d2 + eps) ** (0.5 * f))


def irls_weights(d, f, eps):
    d2 = d * d
    return (d2 + eps) ** (0.25 * f - 0.5)


def surrogate_objective(d_new, w_frozen):
    """Q(theta; theta_k) = 0.5 * sum(w_k^2 * d_new^2)"""
    return 0.5 * np.sum(w_frozen**2 * d_new**2)


def rank_from_energy(svals, tau):
    energy = np.cumsum(svals**2)
    total = energy[-1] if energy[-1] > 0 else 1.0
    r = int(np.searchsorted(energy / total, tau) + 1)
    return min(max(r, 1), svals.size)


def gradient_capture_ratio(g, V_r):
    gn = np.linalg.norm(g)
    if gn == 0:
        return 0.0
    proj = V_r @ (V_r.T @ g)
    return np.linalg.norm(g - proj) / gn


def randomized_svd(A, target_rank, n_oversampling=10, n_power_iter=2):
    """
    Randomized SVD via Halko, Martinsson, Tropp (2011), Algorithm 4.4.

    Computes an approximate rank-k SVD: A ≈ U @ diag(S) @ Vt
    in O(n·p·k) instead of O(n·p²) for the full thin SVD.

    Parameters
    ----------
    A : (n, p) matrix
    target_rank : number of singular components to compute
    n_oversampling : additional columns for accuracy (default 10)
    n_power_iter : power iterations for spectral decay (default 2)

    Returns
    -------
    U : (n, k) left singular vectors
    svals : (k,) singular values (descending)
    Vt : (k, p) right singular vectors (transposed)
    """
    n, p = A.shape
    k = min(target_rank + n_oversampling, n, p)

    # If k >= p, fall back to exact SVD (no savings)
    if k >= p:
        return np.linalg.svd(A, full_matrices=False)

    rng = np.random.default_rng()
    Omega = rng.standard_normal((p, k))

    # Power iteration for better approximation of slowly decaying spectra
    Y = A @ Omega
    for _ in range(n_power_iter):
        Q, _ = np.linalg.qr(Y, mode='reduced')
        Z = A.T @ Q
        Q_z, _ = np.linalg.qr(Z, mode='reduced')
        Y = A @ Q_z

    Q, _ = np.linalg.qr(Y, mode='reduced')

    # Project A onto the range of Q and compute small SVD
    B = Q.T @ A                          # (k, p) — small matrix
    U_b, svals, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_b                          # (n, k)

    return U, svals, Vt


def compute_svd(WJ, cfg):
    """Dispatch to exact or randomized SVD based on config and problem size."""
    n, p = WJ.shape
    if cfg.use_rsvd and p > cfg.rsvd_threshold:
        # Target rank: capture enough of the spectrum for rank selection.
        # Use min(n, p) capped at a practical maximum to ensure speed gain.
        target = min(p - 1, max(int(0.8 * p), 20))
        U, svals, Vt = randomized_svd(WJ, target_rank=target,
                                       n_oversampling=cfg.rsvd_oversampling,
                                       n_power_iter=2)
        # Correct the total energy estimate using the Frobenius norm,
        # which equals sum(all singular values²). This is O(np) and exact.
        frob_sq = np.sum(WJ * WJ)
        captured_sq = np.sum(svals**2)
        if captured_sq < frob_sq * 0.999:
            # Append a synthetic singular value representing uncaptured energy.
            # This ensures rank_from_energy sees the correct total.
            residual_energy = max(frob_sq - captured_sq, 0.0)
            n_missing = p - len(svals)
            if n_missing > 0:
                # Distribute residual uniformly across missing components
                avg_missing = np.sqrt(residual_energy / n_missing)
                svals = np.append(svals, np.full(n_missing, avg_missing))
                # Extend V with zero columns (these won't be used in step)
                Vt = np.vstack([Vt, np.zeros((n_missing, p))])
                U = np.hstack([U, np.zeros((n, n_missing))])
    else:
        U, svals, Vt = np.linalg.svd(WJ, full_matrices=False)
    return U, svals, Vt


# ═══════════════════════════════════════════════════════════════════
# 3. SOLVER (with sign detection + momentum)
# ═══════════════════════════════════════════════════════════════════

def solve_lf(x, y, theta0, model, jacobian, cfg: SolverConfig):
    """
    Adaptive Lf-norm solver with:
      - SVD-based low-rank step computation
      - Levenberg-Marquardt inner loop with rank expansion
      - Convex-to-robust warm-start scheduling
      - Dual-direction sign detection for f < 1
      - Exponential momentum on the search direction
    """
    theta = theta0.astype(float).copy()
    f_k, eps_k = cfg.f0, cfg.eps0
    mu_k = max(cfg.mu0, cfg.mu_min)

    hist = {
        "obj": [], "f": [], "eps": [], "rank": [], "mu": [],
        "grad_norm": [], "step_norm": [], "weights_std": [],
        "spectrum": [], "accepted": [],
        "rho": [], "delta_lf": [], "delta_Q": [],
        "sign_flips": 0,       # count of times negative direction was chosen
        "momentum_applied": 0, # count of momentum-augmented steps
        "rejections": 0, "accepts": 0,
    }

    # Momentum buffer
    h_prev = None

    for k in range(cfg.max_iter):
        d = residuals(x, y, theta, model)
        obj = lf_objective(d, f_k, eps_k)
        w = irls_weights(d, f_k, eps_k)
        Wd = w * d
        J = jacobian(x, theta)
        WJ = w[:, None] * J
        g = WJ.T @ Wd
        grad_norm = np.linalg.norm(g)

        hist["obj"].append(obj)
        hist["f"].append(f_k)
        hist["eps"].append(eps_k)
        hist["mu"].append(mu_k)
        hist["grad_norm"].append(grad_norm)
        hist["weights_std"].append(float(np.std(w)))

        if grad_norm < cfg.grad_tol:
            hist["rank"].append(WJ.shape[1])
            hist["step_norm"].append(0.0)
            hist["accepted"].append(True)
            break

        # SVD of weighted Jacobian (exact or randomized)
        U, svals, Vt = compute_svd(WJ, cfg)
        V = Vt.T
        q = svals.size

        if cfg.capture_spectrum:
            hist["spectrum"].append({
                "iter": k, "svals": svals.copy(),
                "f": f_k, "eps": eps_k
            })

        # Rank selection
        if cfg.mode == "full":
            r = q
        else:
            r = rank_from_energy(svals, cfg.tau)
            if cfg.mode == "dual":
                while r < q:
                    if gradient_capture_ratio(g, V[:, :r]) <= cfg.delta:
                        break
                    r += 1

        # Surrogate value at current point
        Q_before = surrogate_objective(d, w)

        # Inner LM loop with bounded iteration count
        accepted = False
        local_r, local_mu = r, mu_k
        inner_iter = 0

        while not accepted and inner_iter < cfg.max_inner:
            inner_iter += 1
            V_r = V[:, :local_r]
            s_r = svals[:local_r]

            # Raw SVD step (with numerical floor on singular values)
            s_r_safe = np.maximum(s_r, 1e-15)
            h_raw = -V_r @ ((1.0 / (s_r_safe**2 + local_mu)) * (V_r.T @ g))

            # Apply momentum
            if cfg.use_momentum and h_prev is not None:
                h_step = (1.0 - cfg.momentum_beta) * h_raw + cfg.momentum_beta * h_prev
                hist["momentum_applied"] += 1
            else:
                h_step = h_raw

            sn = np.linalg.norm(h_step)
            if sn < cfg.step_tol:
                accepted = True
                hist["rank"].append(local_r)
                hist["step_norm"].append(sn)
                hist["accepted"].append(True)
                hist["rho"].append(1.0)
                hist["delta_lf"].append(0.0)
                hist["delta_Q"].append(0.0)
                break

            # --- Evaluate candidate step(s) ---
            if cfg.use_sign_detection and f_k < 1.0:
                # Sign detection: test both directions when f < 1
                theta_pos = theta + h_step
                theta_neg = theta - h_step

                d_pos = residuals(x, y, theta_pos, model)
                d_neg = residuals(x, y, theta_neg, model)

                obj_pos = lf_objective(d_pos, f_k, eps_k)
                obj_neg = lf_objective(d_neg, f_k, eps_k)

                if obj_pos <= obj_neg and obj_pos < obj:
                    theta_trial, d_trial, obj_trial = theta_pos, d_pos, obj_pos
                elif obj_neg < obj_pos and obj_neg < obj:
                    theta_trial, d_trial, obj_trial = theta_neg, d_neg, obj_neg
                    hist["sign_flips"] += 1
                else:
                    theta_trial, d_trial, obj_trial = None, None, None
            else:
                # Standard single-direction (f >= 1 or sign detection off)
                theta_trial = theta + h_step
                d_trial = residuals(x, y, theta_trial, model)
                obj_trial = lf_objective(d_trial, f_k, eps_k)
                if obj_trial >= obj:
                    theta_trial, d_trial, obj_trial = None, None, None

            # --- Accept or reject ---
            if theta_trial is not None:
                Q_after = surrogate_objective(d_trial, w)
                delta_lf = obj - obj_trial
                delta_Q = Q_before - Q_after
                rho = delta_lf / delta_Q if delta_Q > 1e-15 else 1.0

                theta = theta_trial
                accepted = True
                hist["accepts"] += 1
                hist["rank"].append(local_r)
                hist["step_norm"].append(sn)
                hist["accepted"].append(True)
                hist["rho"].append(float(rho))
                hist["delta_lf"].append(float(delta_lf))
                hist["delta_Q"].append(float(delta_Q))
                h_prev = h_step  # update momentum buffer

                # Gain-ratio adaptive damping (Nielsen's strategy)
                if cfg.adaptive_damping:
                    if rho > cfg.rho_good:
                        # Very good step: decrease damping aggressively
                        mu_k = max(cfg.mu_min, local_mu * max(1.0/3.0, 1.0 - (2.0*rho - 1.0)**3))
                    else:
                        # Acceptable step: standard decrease
                        mu_k = max(cfg.mu_min, local_mu / cfg.beta)
                else:
                    mu_k = max(cfg.mu_min, local_mu / cfg.beta)
            else:
                # Reject: increase damping and optionally rank
                hist["rejections"] += 1
                hist["accepted"].append(False)
                local_mu *= cfg.beta
                if local_r < q:
                    local_r += 1
                if local_mu > 1e12:
                    hist["rank"].append(local_r)
                    hist["step_norm"].append(sn)
                    return {
                        "theta": theta, "history": hist,
                        "success": False, "msg": "damping overflow"
                    }

        # If inner loop exhausted without acceptance, record stall
        if not accepted:
            hist["rank"].append(local_r)
            hist["step_norm"].append(0.0)
            hist["accepted"].append(False)
            hist["rho"].append(0.0)
            hist["delta_lf"].append(0.0)
            hist["delta_Q"].append(0.0)

        # Convergence checks
        if hist["step_norm"] and hist["step_norm"][-1] < cfg.step_tol * 10:
            break
        if len(hist["obj"]) >= 3 and accepted:
            recent = hist["obj"][-3:]
            if abs(recent[-1] - recent[-2]) < cfg.obj_tol * max(1.0, abs(recent[-1])):
                break

        # Anneal f and epsilon only on accepted steps
        if accepted:
            f_k = max(cfg.f_target, cfg.rho_f * f_k)
            eps_k = max(cfg.eps_min, cfg.rho_eps * eps_k)

    # ── Dual-strategy polishing phase ──
    # After robust convergence, try two polishing strategies and keep the better:
    #   (A) Full OLS polish: standard GN (f=2) — best on clean data
    #   (B) Weighted-OLS polish: GN with frozen robust weights as soft mask
    #       — best on contaminated data (outliers stay masked)
    # Each candidate is independently safety-checked, then the one with
    # lower OLS residual (better fit) is selected, provided Lf is acceptable.
    if cfg.polish and cfg.f_target < 2.0:
        theta_robust = theta.copy()
        d_robust = residuals(x, y, theta_robust, model)
        obj_lf_start = lf_objective(d_robust, f_k, eps_k)

        def _run_polish(theta_in, weight_vec=None):
            """Run GN polish, optionally with weight mask. Returns (theta, obj_lf)."""
            th = theta_in.copy()
            mu_p = cfg.mu0
            for p_k in range(cfg.polish_iter):
                d_p = residuals(x, y, th, model)
                J_p = jacobian(x, th)
                if weight_vec is not None:
                    WJ = weight_vec[:, None] * J_p
                    Wd = weight_vec * d_p
                else:
                    WJ = J_p
                    Wd = d_p
                g_p = WJ.T @ Wd
                if np.linalg.norm(g_p) < cfg.grad_tol:
                    break
                H_p = WJ.T @ WJ + mu_p * np.eye(WJ.shape[1])
                h_p = -np.linalg.solve(H_p, g_p)
                theta_trial = th + h_p
                d_trial = residuals(x, y, theta_trial, model)
                if weight_vec is not None:
                    fit_before = 0.5 * np.sum((weight_vec * d_p)**2)
                    fit_after = 0.5 * np.sum((weight_vec * d_trial)**2)
                else:
                    fit_before = 0.5 * np.sum(d_p**2)
                    fit_after = 0.5 * np.sum(d_trial**2)
                obj_lf_trial = lf_objective(d_trial, f_k, eps_k)
                lf_budget = obj_lf_start * 1.001 if weight_vec is None else obj_lf_start * 1.01
                if fit_after < fit_before and obj_lf_trial <= lf_budget:
                    th = theta_trial
                    mu_p = max(cfg.mu_min, mu_p / cfg.beta)
                else:
                    mu_p *= cfg.beta
                    if mu_p > 1e10:
                        break
            obj_lf_end = lf_objective(residuals(x, y, th, model), f_k, eps_k)
            drift = np.linalg.norm(th - theta_robust) / max(np.linalg.norm(theta_robust), 1e-15)
            drift_limit = 0.1 if weight_vec is None else 0.5
            if obj_lf_end > obj_lf_start or drift > drift_limit:
                return theta_robust, obj_lf_start
            return th, obj_lf_end

        # Candidate A: full OLS polish
        theta_ols, obj_ols = _run_polish(theta_robust, weight_vec=None)
        # Candidate B: weighted-OLS polish using robust weight mask
        w_mask = irls_weights(d_robust, f_k, eps_k)
        w_max = np.max(w_mask)
        if w_max > 0:
            w_mask = w_mask / w_max
        w_mask[w_mask < 0.01] = 0.0
        theta_wols, obj_wols = _run_polish(theta_robust, weight_vec=w_mask)

        # Pick the candidate with lower unweighted OLS residual
        ols_resid_a = 0.5 * np.sum(residuals(x, y, theta_ols, model)**2)
        ols_resid_b = 0.5 * np.sum(residuals(x, y, theta_wols, model)**2)
        if ols_resid_a <= ols_resid_b and obj_ols <= obj_lf_start * 1.001:
            theta = theta_ols
        elif obj_wols <= obj_lf_start * 1.01:
            theta = theta_wols
        # else: theta remains theta_robust (both rejected)

    return {"theta": theta, "history": hist, "success": True, "msg": "ok"}


def solve_lf_adaptive(x, y, theta0, model, jacobian, cfg: SolverConfig):
    """
    Residual-adaptive f-target selection wrapper.

    Strategy:
    1. Run the robust solver at f_min (most aggressive candidate) — safety baseline.
    2. Diagnose residuals: if outlier fraction < 5% and kurtosis < 4,
       data appears clean enough to try milder f values.
    3. For each milder f candidate (warm-started from the robust solution),
       accept only if ALL of:
       (a) Lf objective at f_min doesn't increase (within 1%)
       (b) OLS objective is strictly lower (better fit)
       (c) Median absolute residual doesn't increase > 10%
    4. Return the accepted candidate with lowest OLS, or the robust baseline.

    This ensures the solver never sacrifices robustness for fit quality:
    on contaminated data, diagnostics block the milder candidates; on clean
    data, milder f yields better accuracy (e.g., MGH17 RE improves ~5×).
    """
    if not cfg.adaptive_f or cfg.f_target >= max(cfg.f_candidates):
        return solve_lf(x, y, theta0, model, jacobian, cfg)

    from dataclasses import asdict

    f_min = min(cfg.f_candidates)
    # Ensure f_target matches the most aggressive candidate
    cfg_robust = SolverConfig(**{k: v for k, v in asdict(cfg).items() if k != 'f_candidates'})
    cfg_robust.f_target = f_min
    cfg_robust.adaptive_f = False  # prevent recursion

    res_robust = solve_lf(x, y, theta0.copy(), model, jacobian, cfg_robust)
    theta_robust = res_robust["theta"]
    d_robust = residuals(x, y, theta_robust, model)
    ols_robust = 0.5 * np.sum(d_robust**2)
    lf_robust = lf_objective(d_robust, f_min, 1e-8)
    median_robust = np.median(np.abs(d_robust))

    best_theta = theta_robust
    best_ols = ols_robust
    best_f = f_min
    best_res = res_robust

    # Residual diagnostics
    mad = median_abs_deviation(d_robust, scale='normal')
    if mad < 1e-15:
        mad = np.std(d_robust) + 1e-15
    standardized = np.abs(d_robust) / mad
    outlier_frac = float(np.mean(standardized > 3.0))
    kurt_val = float(_scipy_kurtosis(d_robust, fisher=True))

    # Conservative gate: only try milder if residuals look clean
    if outlier_frac < 0.05 and kurt_val < 4.0:
        for f_cand in sorted(cfg.f_candidates):
            if f_cand <= f_min:
                continue
            cfg_mild = SolverConfig(**{k: v for k, v in asdict(cfg).items() if k != 'f_candidates'})
            cfg_mild.f_target = f_cand
            cfg_mild.adaptive_f = False

            # Warm-start from robust solution
            res_mild = solve_lf(x, y, theta_robust.copy(), model, jacobian, cfg_mild)
            theta_mild = res_mild["theta"]
            d_mild = residuals(x, y, theta_mild, model)
            ols_mild = 0.5 * np.sum(d_mild**2)
            lf_mild = lf_objective(d_mild, f_min, 1e-8)
            median_mild = np.median(np.abs(d_mild))

            if (lf_mild <= lf_robust * 1.01 and
                ols_mild < best_ols and
                median_mild <= median_robust * 1.10):
                best_theta = theta_mild
                best_ols = ols_mild
                best_f = f_cand
                best_res = res_mild

    # Attach diagnostic info
    best_res["theta"] = best_theta
    best_res["f_selected"] = best_f
    best_res["outlier_frac"] = outlier_frac
    best_res["kurtosis"] = kurt_val
    return best_res


# ═══════════════════════════════════════════════════════════════════
# 4. SYNTHETIC 1D MODELS
# ═══════════════════════════════════════════════════════════════════

def model_exp_decay(x, theta):
    a, b, c = theta
    return a * np.exp(b * x) + c

def jac_exp_decay(x, theta):
    a, b, _ = theta
    e = np.exp(b * x)
    J = np.empty((x.size, 3))
    J[:, 0] = e;  J[:, 1] = a * x * e;  J[:, 2] = 1.0
    return J

def model_logistic(x, theta):
    L, k, x0, c = theta
    return L / (1.0 + np.exp(-k * (x - x0))) + c

def jac_logistic(x, theta):
    L, k, x0, _ = theta
    z = np.exp(-k * (x - x0));  d = 1.0 + z;  s = z / d**2
    J = np.empty((x.size, 4))
    J[:, 0] = 1/d;  J[:, 1] = L*(x-x0)*s;  J[:, 2] = -L*k*s;  J[:, 3] = 1.0
    return J

def model_biexp(x, theta):
    a1, b1, a2, b2, c = theta
    return a1*np.exp(b1*x) + a2*np.exp(b2*x) + c

def jac_biexp(x, theta):
    a1, b1, a2, b2, _ = theta
    e1, e2 = np.exp(b1*x), np.exp(b2*x)
    J = np.empty((x.size, 5))
    J[:,0]=e1; J[:,1]=a1*x*e1; J[:,2]=e2; J[:,3]=a2*x*e2; J[:,4]=1.0
    return J

def model_gaussian_rbf(x, theta):
    a, mu, sigma, c = theta
    return a * np.exp(-0.5 * ((x - mu) / sigma)**2) + c

def jac_gaussian_rbf(x, theta):
    a, mu, sigma, _ = theta
    z = (x - mu) / sigma
    g = np.exp(-0.5 * z**2)
    J = np.empty((x.size, 4))
    J[:, 0] = g
    J[:, 1] = a * g * z / sigma
    J[:, 2] = a * g * z**2 / sigma
    J[:, 3] = 1.0
    return J

def model_linear_nd(X, theta):
    return X @ theta

def jac_linear_nd(X, theta):
    return X

SYNTHETIC_REGISTRY = {
    "exp_decay":     (model_exp_decay,     jac_exp_decay),
    "logistic":      (model_logistic,      jac_logistic),
    "biexponential": (model_biexp,         jac_biexp),
    "gaussian_rbf":  (model_gaussian_rbf,  jac_gaussian_rbf),
}

TRUE_THETA = {
    "exp_decay":     np.array([2.5, -0.9, 0.4]),
    "logistic":      np.array([3.0, 2.0, 2.5, 0.2]),
    "biexponential": np.array([2.0, -1.2, 1.1, -0.25, 0.15]),
    "gaussian_rbf":  np.array([3.0, 2.5, 0.8, 0.5]),
}

INIT_OFFSETS = {
    3: np.array([0.5, 0.3, -0.2]),
    4: np.array([0.5, 0.3, -0.3, 0.2]),
    5: np.array([0.6, 0.3, -0.4, 0.1, -0.1]),
}

def default_init(model_name):
    t = TRUE_THETA[model_name]
    return t + INIT_OFFSETS[t.size]


# ═══════════════════════════════════════════════════════════════════
# 5. NIST StRD BENCHMARK FUNCTIONS (27 functions)
# ═══════════════════════════════════════════════════════════════════

# --- 1. Misra1a ---
def misra1a(x, b):
    return b[0] * (1 - np.exp(-b[1] * x))

def misra1a_jacobian(x, b):
    J = np.zeros((x.shape[0], b.shape[0]))
    J[:, 0] = (1 - np.exp(-b[1] * x)).flatten()
    J[:, 1] = (b[0] * x * np.exp(-b[1] * x)).flatten()
    return J

# --- 2. Bennett5 ---
def bennett5(x, b):
    return b[0] * ((b[1] + x) ** (-1.0 / b[2]))

def bennett5_jacobian(x, b):
    J = np.zeros((x.shape[0], b.shape[0]))
    J[:, 0] = ((b[1] + x) ** (-1.0 / b[2])).flatten()
    J[:, 1] = (-b[0] / b[2] * (b[1] + x) ** (-1.0 / b[2] - 1)).flatten()
    J[:, 2] = (b[0] * ((b[1] + x) ** (-1.0 / b[2])) * np.log(b[1] + x) / (b[2] ** 2)).flatten()
    return J

# --- 3. Lanczos1 ---
def lanczos1(x, b):
    return b[0] * np.exp(-b[1] * x) + b[2] * np.exp(-b[3] * x) + b[4] * np.exp(-b[5] * x)

def lanczos1_jacobian(x, b):
    J = np.zeros((x.shape[0], b.shape[0]))
    J[:, 0] = np.exp(-b[1] * x).flatten()
    J[:, 1] = (-b[0] * x * np.exp(-b[1] * x)).flatten()
    J[:, 2] = np.exp(-b[3] * x).flatten()
    J[:, 3] = (-b[2] * x * np.exp(-b[3] * x)).flatten()
    J[:, 4] = np.exp(-b[5] * x).flatten()
    J[:, 5] = (-b[4] * x * np.exp(-b[5] * x)).flatten()
    return J

# --- 4. Kirby2 ---
def kirby2(x, b):
    num = b[0] + b[1] * x + b[2] * x**2
    den = 1 + b[3] * x + b[4] * x**2
    return num / den

def kirby2_jacobian(x, b):
    J = np.zeros((x.shape[0], b.shape[0]))
    num = b[0] + b[1] * x + b[2] * x**2
    den = 1 + b[3] * x + b[4] * x**2
    J[:, 0] = (1 / den).flatten()
    J[:, 1] = (x / den).flatten()
    J[:, 2] = (x**2 / den).flatten()
    J[:, 3] = (-x * num / den**2).flatten()
    J[:, 4] = (-x**2 * num / den**2).flatten()
    return J

# --- 5. Chwirut2 ---
def chwirut2(x, b):
    return np.exp(-b[0] * x) / (b[1] + b[2] * x)

def chwirut2_jacobian(x, b):
    J = np.zeros((x.shape[0], b.shape[0]))
    exp_term = np.exp(-b[0] * x)
    J[:, 0] = (-x * exp_term / (b[1] + b[2] * x)).flatten()
    J[:, 1] = (-exp_term / ((b[1] + b[2] * x)**2)).flatten()
    J[:, 2] = (-exp_term * x / ((b[1] + b[2] * x)**2)).flatten()
    return J

# --- 6. Thurber ---
def thurber(x, b):
    return (b[0] + b[1]*x + b[2]*x**2 + b[3]*x**3) / (1 + b[4]*x + b[5]*x**2 + b[6]*x**3)

def thurber_jacobian(x, b):
    J = np.zeros((x.shape[0], b.shape[0]))
    num = b[0] + b[1]*x + b[2]*x**2 + b[3]*x**3
    den = 1 + b[4]*x + b[5]*x**2 + b[6]*x**3
    J[:, 0] = (1 / den).flatten()
    J[:, 1] = (x / den).flatten()
    J[:, 2] = (x**2 / den).flatten()
    J[:, 3] = (x**3 / den).flatten()
    J[:, 4] = (-num * x / den**2).flatten()
    J[:, 5] = (-num * x**2 / den**2).flatten()
    J[:, 6] = (-num * x**3 / den**2).flatten()
    return J

# --- 7. Roszman1 ---
def roszman1(x, b):
    return b[0] - b[1] * x - (1 / np.pi) * np.arctan(b[2] / (x - b[3]))

def roszman1_jacobian(x, b):
    A = b[2] / (x - b[3])
    J = np.zeros((x.shape[0], b.shape[0]))
    J[:, 0] = np.ones_like(x).flatten()
    J[:, 1] = (-x).flatten()
    J[:, 2] = ((-1 / np.pi) * (1 / (1 + A**2)) * (1 / (x - b[3]))).flatten()
    J[:, 3] = ((1 / np.pi) * (1 / (1 + A**2)) * b[2] / ((x - b[3])**2)).flatten()
    return J

# --- 8. Eckerle4 ---
def eckerle4(x, b):
    return (b[0] / b[1]) * np.exp(-((x - b[2])**2) / (2 * b[1]**2))

def eckerle4_jacobian(x, b):
    J = np.zeros((x.shape[0], b.shape[0]))
    exp_term = np.exp(-((x - b[2])**2) / (2 * b[1]**2))
    J[:, 0] = (exp_term / b[1]).flatten()
    J[:, 1] = ((b[0] / b[1]) * exp_term * (-1 / b[1] + (x - b[2])**2 / b[1]**3)).flatten()
    J[:, 2] = ((b[0] / b[1]) * (x - b[2]) / b[1]**2 * exp_term).flatten()
    return J

# --- 9. Rat43 ---
def rat43(x, b):
    return b[0] / ((1 + np.exp(b[1] - b[2] * x)) ** (1 / b[3]))

def rat43_jacobian(x, b):
    J = np.zeros((x.shape[0], b.shape[0]))
    exp_term = np.exp(b[1] - b[2] * x)
    J[:, 0] = ((1 + exp_term) ** (-1.0 / b[3])).flatten()
    J[:, 1] = (-b[0] * (1 / b[3]) * exp_term * (1 + exp_term) ** (-1.0/b[3] - 1)).flatten()
    J[:, 2] = (b[0] / b[3] * x * exp_term * (1 + exp_term) ** (-1.0/b[3] - 1)).flatten()
    J[:, 3] = (b[0] / b[3]**2 * np.log(1 + exp_term) * (1 + exp_term) ** (-1.0/b[3])).flatten()
    return J

# --- 10. MGH09 ---
def mgh09(x, b):
    return (b[0] * (x**2 + x * b[1])) / (x**2 + x * b[2] + b[3])

def mgh09_jacobian(x, b):
    J = np.zeros((x.shape[0], b.shape[0]))
    num = b[0] * (x**2 + x * b[1])
    den = x**2 + x * b[2] + b[3]
    J[:, 0] = ((x**2 + x * b[1]) / den).flatten()
    J[:, 1] = (b[0] * x / den).flatten()
    J[:, 2] = (-x * num / den**2).flatten()
    J[:, 3] = (-num / den**2).flatten()
    return J

# --- 11. ENSO ---
def enso(x, b):
    return (b[0] + b[1] * np.cos(2*np.pi*x / 12)
            + b[2] * np.sin(2*np.pi*x / 12)
            + b[4] * np.cos(2*np.pi*x / b[3])
            + b[5] * np.sin(2*np.pi*x / b[3])
            + b[7] * np.cos(2*np.pi*x / b[6])
            + b[8] * np.sin(2*np.pi*x / b[6]))

def enso_jacobian(x, b):
    J = np.zeros((x.shape[0], b.shape[0]))
    A = 2 * np.pi * x / 12
    B = 2 * np.pi * x / b[3]
    C = 2 * np.pi * x / b[6]
    J[:, 0] = np.ones_like(x).flatten()
    J[:, 1] = np.cos(A).flatten()
    J[:, 2] = np.sin(A).flatten()
    J[:, 3] = ((2*np.pi*x) / b[3]**2 * (b[4]*np.sin(B) - b[5]*np.cos(B))).flatten()
    J[:, 4] = np.cos(B).flatten()
    J[:, 5] = np.sin(B).flatten()
    J[:, 6] = ((2*np.pi*x) / b[6]**2 * (b[7]*np.sin(C) - b[8]*np.cos(C))).flatten()
    J[:, 7] = np.cos(C).flatten()
    J[:, 8] = np.sin(C).flatten()
    return J

# --- 12. Chwirut1 ---
def chwirut1(x, b):
    return np.exp(-b[0] * x) / (b[1] + b[2] * x)

def chwirut1_jacobian(x, b):
    J = np.zeros((x.shape[0], b.shape[0]))
    u = np.exp(-b[0] * x)
    v = b[1] + b[2] * x
    J[:, 0] = (-x * u / v).flatten()
    J[:, 1] = (-u / v**2).flatten()         
    J[:, 2] = (-x * u / v**2).flatten()
    return J

# --- 13. Lanczos3 ---
def lanczos3(x, b):
    return b[0]*np.exp(-b[1]*x) + b[2]*np.exp(-b[3]*x) + b[4]*np.exp(-b[5]*x)

def lanczos3_jacobian(x, b):
    x_flat = x.flatten()
    E1 = np.exp(-b[1] * x_flat)
    E2 = np.exp(-b[3] * x_flat)
    E3 = np.exp(-b[5] * x_flat)
    J = np.zeros((len(x_flat), len(b)))
    J[:, 0] = E1
    J[:, 1] = (-b[0] * x_flat * E1)
    J[:, 2] = E2
    J[:, 3] = (-b[2] * x_flat * E2)
    J[:, 4] = E3
    J[:, 5] = (-b[4] * x_flat * E3)
    return J

# --- 14. Gauss1 ---
def gauss1(x, b):
    return (b[0] * np.exp(-b[1] * x)
            + b[2] * np.exp(-((x - b[3])**2) / b[4]**2)
            + b[5] * np.exp(-((x - b[6])**2) / b[7]**2))

def gauss1_jacobian(x, b):
    return _gauss_common_jacobian(x, b)

# --- 15. Gauss2 ---
def gauss2(x, b):
    return (b[0] * np.exp(-b[1] * x)
            + b[2] * np.exp(-((x - b[3])**2) / b[4]**2)
            + b[5] * np.exp(-((x - b[6])**2) / b[7]**2))

def gauss2_jacobian(x, b):
    return _gauss_common_jacobian(x, b)

# --- 16. DanWood ---
def danwood(x, b):
    return b[0] * (x ** b[1])

def danwood_jacobian(x, b):
    J = np.zeros((len(x), len(b)))
    J[:, 0] = (x ** b[1]).flatten()
    J[:, 1] = (b[0] * x ** b[1] * np.log(x)).flatten()
    return J

# --- 17. Misra1b ---
def misra1b(x, b):
    return b[0] * (1 - 1 / (1 + b[1] * x / 2)**2)

def misra1b_jacobian(x, b):
    denom = 1 + b[1] * x / 2
    J = np.zeros((len(x), len(b)))
    J[:, 0] = (1 - 1 / denom**2).flatten()
    J[:, 1] = (b[0] * x * denom**(-3)).flatten()
    return J

# --- 18. Hahn1 --- 
def hahn1(x, b):
    num = b[0] + b[1]*x + b[2]*x**2 + b[3]*x**3
    den = 1 + b[4]*x + b[5]*x**2 + b[6]*x**3
    return num / den

def hahn1_jacobian(x, b):
    num = b[0] + b[1]*x + b[2]*x**2 + b[3]*x**3
    den = 1 + b[4]*x + b[5]*x**2 + b[6]*x**3
    J = np.zeros((len(x), len(b)))
    J[:, 0] = (1 / den).flatten()
    J[:, 1] = (x / den).flatten()
    J[:, 2] = (x**2 / den).flatten()
    J[:, 3] = (x**3 / den).flatten()
    J[:, 4] = (-num * x / den**2).flatten()       
    J[:, 5] = (-num * x**2 / den**2).flatten()     
    J[:, 6] = (-num * x**3 / den**2).flatten()     
    return J

# --- 19. Nelson ---
def nelson(x, b):
    return b[0] - b[1] * x * np.exp(-b[2] * x)

def nelson_jacobian(x, b):
    exp_term = np.exp(-b[2] * x).flatten()
    J = np.zeros((len(x), len(b)))
    J[:, 0] = np.ones_like(x).flatten()
    J[:, 1] = (-x * exp_term).flatten()
    J[:, 2] = (b[1] * x**2 * exp_term).flatten()
    return J

# --- 20. MGH17 ---
def mgh17(x, b):
    return b[0] + b[1] * np.exp(-b[3] * x) + b[2] * np.exp(-b[4] * x)

def mgh17_jacobian(x, b):
    E1 = np.exp(-b[3] * x).flatten()
    E2 = np.exp(-b[4] * x).flatten()
    x_flat = x.flatten()
    J = np.zeros((len(x_flat), len(b)))
    J[:, 0] = 1.0
    J[:, 1] = E1
    J[:, 2] = E2
    J[:, 3] = (-b[1] * x_flat * E1)
    J[:, 4] = (-b[2] * x_flat * E2)
    return J

# --- 21. Lanczos2 ---
def lanczos2(x, b):
    return b[0]*np.exp(-b[1]*x) + b[2]*np.exp(-b[3]*x) + b[4]*np.exp(-b[5]*x)

def lanczos2_jacobian(x, b):
    return lanczos3_jacobian(x, b)

# --- 22. Misra1c ---
def misra1c(x, b):
    return b[0] * (1 - 1 / np.sqrt(1 + 2 * b[1] * x))

def misra1c_jacobian(x, b):
    denom = np.sqrt(1 + 2 * b[1] * x)
    J = np.zeros((len(x), len(b)))
    J[:, 0] = (1 - 1 / denom).flatten()
    J[:, 1] = (b[0] * x / denom**3).flatten()
    return J

# --- 23. Misra1d ---
def misra1d(x, b):
    return (b[0] * b[1] * x) / (1 + b[1] * x)

def misra1d_jacobian(x, b):
    den = 1 + b[1] * x
    J = np.zeros((len(x), len(b)))
    J[:, 0] = (b[1] * x / den).flatten()
    J[:, 1] = (b[0] * x / den - b[0] * b[1] * x**2 / den**2).flatten()
    return J

# --- 24. Gauss3 ---
def gauss3(x, b):
    return (b[0] * np.exp(-b[1] * x)
            + b[2] * np.exp(-((x - b[3])**2) / b[4]**2)
            + b[5] * np.exp(-((x - b[6])**2) / b[7]**2))

def _gauss_common_jacobian(x, b):
    """Shared Jacobian for Gauss1, Gauss2, Gauss3 (same functional form)."""
    E = np.exp(-b[1] * x)
    G1 = np.exp(-((x - b[3])**2) / b[4]**2)
    G2 = np.exp(-((x - b[6])**2) / b[7]**2)
    J = np.zeros((len(x), len(b)))
    J[:, 0] = E.flatten()
    J[:, 1] = (-b[0] * x * E).flatten()
    J[:, 2] = G1.flatten()
    J[:, 3] = (b[2] * 2 * (x - b[3]) / b[4]**2 * G1).flatten()
    J[:, 4] = (b[2] * 2 * (x - b[3])**2 / b[4]**3 * G1).flatten()
    J[:, 5] = G2.flatten()
    J[:, 6] = (b[5] * 2 * (x - b[6]) / b[7]**2 * G2).flatten()
    J[:, 7] = (b[5] * 2 * (x - b[6])**2 / b[7]**3 * G2).flatten()
    return J

def gauss3_jacobian(x, b):
    return _gauss_common_jacobian(x, b)

# --- 25. BoxBOD ---
def boxbod(x, b):
    return b[0] * (1 - np.exp(-b[1] * x))

def boxbod_jacobian(x, b):
    E = np.exp(-b[1] * x)
    J = np.zeros((len(x), len(b)))
    J[:, 0] = (1 - E).flatten()
    J[:, 1] = (b[0] * x * E).flatten()
    return J

# --- 26. Rat42 ---
def rat42(x, b):
    return b[0] / (1 + np.exp(b[1] - b[2] * x))

def rat42_jacobian(x, b):
    exp_term = np.exp(b[1] - b[2] * x)
    den = (1 + exp_term)**2
    J = np.zeros((len(x), len(b)))
    J[:, 0] = (1 / (1 + exp_term)).flatten()
    J[:, 1] = (-b[0] * exp_term / den).flatten()
    J[:, 2] = (b[0] * x * exp_term / den).flatten()
    return J

# --- 27. MGH10 ---
def mgh10(x, b):
    return b[0] * np.exp(b[1] / (x + b[2]))

def mgh10_jacobian(x, b):
    den = (x + b[2]).flatten()
    exp_term = np.exp(b[1] / den)
    J = np.zeros((len(x), len(b)))
    J[:, 0] = exp_term
    J[:, 1] = (b[0] / den * exp_term)
    J[:, 2] = (-b[0] * b[1] / den**2 * exp_term)
    return J


# ═══════════════════════════════════════════════════════════════════
# 6. NIST REGISTRY
# ═══════════════════════════════════════════════════════════════════

NIST_REGISTRY = {
    "misra1a":   {"fn": misra1a,   "jac": misra1a_jacobian,   "p": 2, "difficulty": "Lower"},
    "chwirut2":  {"fn": chwirut2,  "jac": chwirut2_jacobian,  "p": 3, "difficulty": "Lower"},
    "chwirut1":  {"fn": chwirut1,  "jac": chwirut1_jacobian,  "p": 3, "difficulty": "Lower"},
    "lanczos3":  {"fn": lanczos3,  "jac": lanczos3_jacobian,  "p": 6, "difficulty": "Lower"},
    "gauss1":    {"fn": gauss1,    "jac": gauss1_jacobian,    "p": 8, "difficulty": "Lower"},
    "gauss2":    {"fn": gauss2,    "jac": gauss2_jacobian,    "p": 8, "difficulty": "Lower"},
    "danwood":   {"fn": danwood,   "jac": danwood_jacobian,   "p": 2, "difficulty": "Lower"},
    "misra1b":   {"fn": misra1b,   "jac": misra1b_jacobian,   "p": 2, "difficulty": "Lower"},

    "kirby2":    {"fn": kirby2,    "jac": kirby2_jacobian,    "p": 5, "difficulty": "Average"},
    "hahn1":     {"fn": hahn1,     "jac": hahn1_jacobian,     "p": 7, "difficulty": "Average"},
    "nelson":    {"fn": nelson,    "jac": nelson_jacobian,    "p": 3, "difficulty": "Average"},
    "mgh17":     {"fn": mgh17,     "jac": mgh17_jacobian,     "p": 5, "difficulty": "Average"},
    "lanczos1":  {"fn": lanczos1,  "jac": lanczos1_jacobian,  "p": 6, "difficulty": "Average"},
    "lanczos2":  {"fn": lanczos2,  "jac": lanczos2_jacobian,  "p": 6, "difficulty": "Average"},
    "gauss3":    {"fn": gauss3,    "jac": gauss3_jacobian,    "p": 8, "difficulty": "Average"},
    "misra1c":   {"fn": misra1c,   "jac": misra1c_jacobian,   "p": 2, "difficulty": "Average"},
    "misra1d":   {"fn": misra1d,   "jac": misra1d_jacobian,   "p": 2, "difficulty": "Average"},
    "roszman1":  {"fn": roszman1,  "jac": roszman1_jacobian,  "p": 4, "difficulty": "Average"},
    "enso":      {"fn": enso,      "jac": enso_jacobian,      "p": 9, "difficulty": "Average"},

    "thurber":   {"fn": thurber,   "jac": thurber_jacobian,   "p": 7, "difficulty": "Higher"},
    "boxbod":    {"fn": boxbod,    "jac": boxbod_jacobian,    "p": 2, "difficulty": "Higher"},
    "rat42":     {"fn": rat42,     "jac": rat42_jacobian,     "p": 3, "difficulty": "Higher"},
    "mgh09":     {"fn": mgh09,     "jac": mgh09_jacobian,     "p": 4, "difficulty": "Higher"},
    "eckerle4":  {"fn": eckerle4,  "jac": eckerle4_jacobian,  "p": 3, "difficulty": "Higher"},
    "rat43":     {"fn": rat43,     "jac": rat43_jacobian,     "p": 4, "difficulty": "Higher"},
    "bennett5":  {"fn": bennett5,  "jac": bennett5_jacobian,  "p": 3, "difficulty": "Higher"},
    "mgh10":     {"fn": mgh10,     "jac": mgh10_jacobian,     "p": 3, "difficulty": "Higher"},
}


# ═══════════════════════════════════════════════════════════════════
# 7. M-ESTIMATOR BASELINES
# ═══════════════════════════════════════════════════════════════════

def solve_scipy_baseline(x, y, theta0, model, jacobian,
                         loss="linear", f_scale=1.0, max_nfev=2000):
    t0 = time.perf_counter()
    res = least_squares(
        fun=lambda th: residuals(x, y, th, model),
        x0=theta0,
        jac=lambda th: jacobian(x, th),
        method="trf", loss=loss, f_scale=f_scale, max_nfev=max_nfev)
    rt = time.perf_counter() - t0
    d = residuals(x, y, res.x, model)
    return {"theta": res.x, "success": bool(res.success),
            "runtime_s": rt, "n_iters": int(getattr(res, 'nfev', 0)),
            "rejections": 0, "accepts": int(getattr(res, 'nfev', 0)),
            "avg_rank": np.nan,
            "final_obj": float(0.5 * np.sum(d**2))}


def _irls_generic(x, y, theta0, model, jacobian,
                   weight_fn, obj_fn, c=1.0, beta=5.0,
                   mu0=1e-3, max_iter=120):
    theta = theta0.astype(float).copy()
    mu = mu0; rej = 0; acc = 0
    t0 = time.perf_counter()
    for k in range(max_iter):
        d = residuals(x, y, theta, model)
        w = weight_fn(d, c)
        J = jacobian(x, theta)
        WJ = w[:, None] * J
        g = WJ.T @ (w * d)
        if np.linalg.norm(g) < 1e-8:
            break
        H = WJ.T @ WJ + mu * np.eye(WJ.shape[1])
        step = -np.linalg.solve(H, g)
        if np.linalg.norm(step) < 1e-10:
            break
        obj = obj_fn(d, c)
        t_trial = theta + step
        if obj_fn(residuals(x, y, t_trial, model), c) < obj:
            theta = t_trial; acc += 1; mu = max(1e-8, mu / beta)
        else:
            rej += 1; mu *= beta
            if mu > 1e12:
                break
    rt = time.perf_counter() - t0
    d_fin = residuals(x, y, theta, model)
    return {"theta": theta, "success": True, "runtime_s": rt,
            "n_iters": k + 1, "rejections": rej, "accepts": acc,
            "avg_rank": np.nan, "final_obj": float(obj_fn(d_fin, c))}


# Geman-McClure
def _gm_w(d, c):
    d2 = d*d; return (c*c) / (d2 + c*c)**2

def _gm_obj(d, c):
    d2 = d*d; return 0.5 * np.sum(d2 / (d2 + c*c))

# Welsch
def _welsch_w(d, c):
    return np.exp(-d**2 / (2*c*c))

def _welsch_obj(d, c):
    return np.sum(1.0 - np.exp(-d**2 / (2*c*c)))

# Tukey bisquare
def _tukey_w(d, c):
    mask = np.abs(d) <= c
    w = np.zeros_like(d)
    w[mask] = (1.0 - (d[mask]/c)**2)**2
    return w

def _tukey_obj(d, c):
    mask = np.abs(d) <= c
    r = d / c
    obj = np.full_like(d, c*c/6.0)
    obj[mask] = (c*c/6.0) * (1.0 - (1.0 - r[mask]**2)**3)
    return np.sum(obj)


# ── Barron's General Adaptive Robust Loss (Barron, CVPR 2019) ──
# ρ(r, α, c) = (|α-2|/α) · [((r/c)²/|α-2| + 1)^(α/2) - 1]
# Continuously interpolates: α=2 → L2, α=1 → pseudo-Huber/Charbonnier,
# α=0 → Cauchy/Lorentzian, α=-2 → Geman-McClure, α→-∞ → Welsch.

def _barron_obj(d, alpha, c):
    """Barron's general robust loss objective. Handles special cases."""
    z2 = (d / c) ** 2
    abs_am2 = abs(alpha - 2.0)
    if abs(alpha) < 1e-8:
        # α → 0: Cauchy/Lorentzian
        return np.sum(0.5 * np.log(z2 + 1.0))
    elif abs_am2 < 1e-8:
        # α → 2: L2
        return np.sum(0.5 * z2)
    else:
        return np.sum((abs_am2 / alpha) * ((z2 / abs_am2 + 1.0) ** (alpha / 2.0) - 1.0))


def _barron_w(d, alpha, c):
    """IRLS weights for Barron's loss: w = ψ(r)/r = (1/c²)·(z²/|α-2|+1)^(α/2-1).
    Returns w such that ∂ρ/∂r = w · r (for IRLS formulation)."""
    z2 = (d / c) ** 2
    abs_am2 = abs(alpha - 2.0)
    if abs(alpha - 2.0) < 1e-8:
        return np.ones_like(d) / (c * c)
    elif abs(alpha) < 1e-8:
        # α → 0: w = 1/(c²·(z² + 1))
        return 1.0 / (c * c * (z2 + 1.0))
    else:
        return (1.0 / (c * c)) * (z2 / abs_am2 + 1.0) ** (alpha / 2.0 - 1.0)


def solve_barron_annealing(x, y, theta0, model, jacobian, c=1.0,
                            alpha_target=0.0, alpha0=2.0,
                            rho_alpha=0.92, beta=5.0, mu0=1e-3,
                            max_iter=120, max_inner=15):
    """
    Barron adaptive robust loss solver with α-annealing from convex (α=2)
    to robust (α=alpha_target), mirroring the Lf solver's continuation strategy.

    Parameters
    ----------
    x, y : data
    theta0 : initial parameter guess
    model, jacobian : model function and Jacobian
    c : scale parameter (should be set from MAD estimate)
    alpha_target : target shape parameter (0 → Cauchy, -2 → Geman-McClure, etc.)
    alpha0 : starting shape (2.0 = L2)
    rho_alpha : multiplicative annealing factor per accepted step
    beta : LM damping increase factor on rejection
    mu0 : initial LM damping
    max_iter : outer iteration budget
    max_inner : inner LM iterations per outer step

    Returns
    -------
    dict with 'theta', 'success', 'runtime_s', 'n_iters', 'history'
    """
    theta = theta0.astype(float).copy()
    alpha_k = alpha0
    mu_k = max(mu0, 1e-8)
    t0 = time.perf_counter()

    hist = {"obj": [], "alpha": [], "mu": [],
            "rejections": 0, "accepts": 0}

    for k in range(max_iter):
        d = residuals(x, y, theta, model)
        obj = _barron_obj(d, alpha_k, c)
        w = _barron_w(d, alpha_k, c)

        # Ensure weights are positive and finite
        w = np.clip(w, 1e-15, 1e10)
        w_sqrt = np.sqrt(w)

        J = jacobian(x, theta)
        WJ = w_sqrt[:, None] * J
        g = J.T @ (w * d)   # exact gradient: sum_i w_i d_i J_i
        grad_norm = np.linalg.norm(g)

        hist["obj"].append(obj)
        hist["alpha"].append(alpha_k)
        hist["mu"].append(mu_k)

        if grad_norm < 1e-8:
            break

        # SVD of weighted Jacobian for stable step computation
        U, svals, Vt = np.linalg.svd(WJ, full_matrices=False)
        V = Vt.T

        accepted = False
        for inner in range(max_inner):
            # Damped SVD step
            S_inv = svals / (svals**2 + mu_k)
            step = -V @ (S_inv * (U.T @ (w_sqrt * d)))

            step_norm = np.linalg.norm(step)
            if step_norm < 1e-10:
                break

            theta_trial = theta + step
            obj_trial = _barron_obj(residuals(x, y, theta_trial, model),
                                     alpha_k, c)

            if obj_trial < obj:
                # Gain ratio for adaptive damping
                pred_reduction = -g @ step - 0.5 * step @ (WJ.T @ WJ + mu_k * np.eye(len(step))) @ step
                actual_reduction = obj - obj_trial
                if pred_reduction > 0:
                    rho = actual_reduction / pred_reduction
                else:
                    rho = 1.0

                theta = theta_trial
                hist["accepts"] += 1
                accepted = True

                # Nielsen-style adaptive damping
                if rho > 0.75:
                    mu_k = mu_k * max(1.0/3.0, 1.0 - (2.0*rho - 1.0)**3)
                else:
                    mu_k = mu_k / beta
                mu_k = max(mu_k, 1e-8)

                break
            else:
                hist["rejections"] += 1
                mu_k *= beta
                if mu_k > 1e12:
                    break

        # Anneal alpha on accepted steps only
        if accepted:
            if alpha_target < alpha_k:
                alpha_k = max(alpha_target, rho_alpha * alpha_k +
                              (1.0 - rho_alpha) * alpha_target)
            # Convergence check
            if k > 5 and len(hist["obj"]) >= 3:
                recent = hist["obj"][-3:]
                if abs(recent[-1] - recent[-2]) < 1e-12 * (abs(recent[-2]) + 1e-15):
                    if abs(alpha_k - alpha_target) < 1e-6:
                        break

    # Polish: run a few iterations at target alpha
    for kp in range(10):
        d = residuals(x, y, theta, model)
        obj = _barron_obj(d, alpha_target, c)
        w = _barron_w(d, alpha_target, c)
        w = np.clip(w, 1e-15, 1e10)
        w_sqrt = np.sqrt(w)
        J = jacobian(x, theta)
        WJ = w_sqrt[:, None] * J
        g = J.T @ (w * d)
        if np.linalg.norm(g) < 1e-8:
            break
        U, svals, Vt = np.linalg.svd(WJ, full_matrices=False)
        V = Vt.T
        S_inv = svals / (svals**2 + mu_k)
        step = -V @ (S_inv * (U.T @ (w_sqrt * d)))
        theta_trial = theta + step
        if _barron_obj(residuals(x, y, theta_trial, model), alpha_target, c) < obj:
            theta = theta_trial

    rt = time.perf_counter() - t0
    d_fin = residuals(x, y, theta, model)
    return {
        "theta": theta, "success": True, "runtime_s": rt,
        "n_iters": k + 1,
        "rejections": hist["rejections"], "accepts": hist["accepts"],
        "avg_rank": np.nan,
        "final_obj": float(_barron_obj(d_fin, alpha_target, c)),
        "history": hist,
    }


# ── MAD-adaptive scale estimation ──

def mad_scale(residuals_vec):
    """Median absolute deviation scale estimate (robust)."""
    return 1.4826 * np.median(np.abs(residuals_vec - np.median(residuals_vec)))


def estimate_initial_scale(x, y, theta0, model, jacobian):
    """
    Two-phase robust scale estimation for baseline calibration.

    Phase 1: OLS fit → MAD of residuals (may be inflated if OLS is
             distorted by outliers).
    Phase 2: Huber fit with Phase-1 scale → MAD of Huber residuals
             (more robust to contamination).

    Returns the minimum of both MAD estimates, which is the more
    conservative (tighter) scale — appropriate for robust baselines
    that use scale to separate inliers from outliers.

    Returns
    -------
    float : MAD-based scale estimate (with floor to avoid degeneracy)
    """
    floor = 1e-6 * (np.max(np.abs(y)) + 1e-15)

    # Phase 1: OLS
    theta_ols = theta0.copy()
    try:
        res_ols = least_squares(
            fun=lambda th: residuals(x, y, th, model),
            x0=theta0.copy(),
            jac=lambda th: jacobian(x, th),
            method="trf", loss="linear", max_nfev=2000)
        theta_ols = res_ols.x
        d_ols = residuals(x, y, theta_ols, model)
        s_ols = mad_scale(d_ols)
    except Exception:
        d_ols = residuals(x, y, theta0, model)
        s_ols = mad_scale(d_ols)

    s_ols = max(s_ols, floor)

    # Phase 2: Huber refit with Phase-1 scale → refined MAD
    try:
        res_hub = least_squares(
            fun=lambda th: residuals(x, y, th, model),
            x0=theta_ols.copy(),
            jac=lambda th: jacobian(x, th),
            method="trf", loss="huber", f_scale=s_ols, max_nfev=2000)
        d_hub = residuals(x, y, res_hub.x, model)
        s_hub = mad_scale(d_hub)
        s_hub = max(s_hub, floor)
    except Exception:
        s_hub = s_ols

    # Use the tighter (more conservative) scale estimate
    return min(s_ols, s_hub)


def solve_method(method, x, y, theta0, model, jacobian,
                 solver_cfg: SolverConfig, noise_std=0.03):
    t0 = time.perf_counter()
    if method in {"full", "energy", "dual"}:
        cfg = SolverConfig(**asdict(solver_cfg))
        cfg.mode = method
        if cfg.adaptive_f:
            res = solve_lf_adaptive(x, y, theta0, model, jacobian, cfg)
        else:
            res = solve_lf(x, y, theta0, model, jacobian, cfg)
        h = res["history"]
        return {
            "theta": res["theta"], "success": res["success"],
            "runtime_s": time.perf_counter() - t0,
            "n_iters": len(h["obj"]),
            "rejections": h["rejections"], "accepts": h["accepts"],
            "sign_flips": h.get("sign_flips", 0),
            "momentum_applied": h.get("momentum_applied", 0),
            "avg_rank": float(np.mean(h["rank"])) if h["rank"] else np.nan,
            "final_obj": h["obj"][-1] if h["obj"] else np.nan,
            "history": h,
        }
    s = max(2.5 * noise_std, 1e-3)
    c_rob = max(3 * noise_std, 0.1)
    if method == "ols":
        r = solve_scipy_baseline(x, y, theta0, model, jacobian, "linear", s)
    elif method == "huber":
        r = solve_scipy_baseline(x, y, theta0, model, jacobian, "huber", s)
    elif method == "cauchy":
        r = solve_scipy_baseline(x, y, theta0, model, jacobian, "cauchy", s)
    elif method == "geman_mcclure":
        r = _irls_generic(x, y, theta0, model, jacobian, _gm_w, _gm_obj, c=c_rob)
    elif method == "welsch":
        r = _irls_generic(x, y, theta0, model, jacobian, _welsch_w, _welsch_obj, c=c_rob)
    elif method == "tukey":
        r = _irls_generic(x, y, theta0, model, jacobian, _tukey_w, _tukey_obj,
                          c=4.685 * max(noise_std, 0.01))
    elif method == "barron":
        r = solve_barron_annealing(x, y, theta0, model, jacobian,
                                    c=c_rob, alpha_target=0.0)
    else:
        raise ValueError(f"Unknown method: {method}")
    r["runtime_s"] = time.perf_counter() - t0
    return r

ALL_METHODS = ["full", "energy", "dual", "ols", "huber", "cauchy",
               "geman_mcclure", "welsch", "tukey", "barron"]


# ═══════════════════════════════════════════════════════════════════
# 8. SVD-BASED FEATURE IMPORTANCE & BACKWARD ELIMINATION
# ═══════════════════════════════════════════════════════════════════

def feature_importance_svd(WJ):
    """Compute SVD-based importance for each parameter column."""
    U, svals, Vt = np.linalg.svd(WJ, full_matrices=False)
    V = Vt.T
    importance = (svals**2)[None, :] * (V**2)
    importance_per_param = importance.sum(axis=1)
    total = importance_per_param.sum()
    if total > 0:
        importance_per_param /= total
    return {"importance": importance_per_param, "svals": svals, "V": V}


def lf_feature_pruning(X, y, theta0, f_target=0.5, tol=0.05,
                        min_features=1, max_rounds=None, verbose=True):
    """
    Backward elimination using SVD importance from the Lf-norm fit.

    Parameters
    ----------
    X : (n, p) data matrix
    y : (n,) response
    theta0 : (p,) initial guess
    f_target : target f for Lf solver
    tol : max allowed relative increase in residual norm
    min_features : stop when this many features remain
    max_rounds : max elimination rounds
    verbose : print progress

    Returns
    -------
    dict with selected_features, importance_history, error_history,
         theta_final, elimination_order
    """
    n, p = X.shape
    active = list(range(p))
    elimination_order = []
    importance_history = []
    error_history = []
    if max_rounds is None:
        max_rounds = p - min_features

    cfg = SolverConfig(mode="dual", f_target=f_target, max_iter=150)
    res_full = solve_lf(X, y, theta0, model_linear_nd, jac_linear_nd, cfg)
    d_full = residuals(X, y, res_full["theta"], model_linear_nd)
    base_obj = lf_objective(d_full, f_target, 1e-6)
    base_resid = np.linalg.norm(d_full)
    error_history.append({"n_features": p, "resid_norm": base_resid,
                          "lf_obj": base_obj,
                          "eliminated": None, "importance": None})

    if verbose:
        print(f"Full model ({p} features): Lf_obj = {base_obj:.6f}, ||resid|| = {base_resid:.6f}")

    for round_i in range(max_rounds):
        if len(active) <= min_features:
            break

        X_a = X[:, active]
        theta0_a = np.zeros(len(active))
        res = solve_lf(X_a, y, theta0_a, model_linear_nd, jac_linear_nd, cfg)

        d = residuals(X_a, y, res["theta"], model_linear_nd)
        w = irls_weights(d, f_target, 1e-6)
        WJ = w[:, None] * jac_linear_nd(X_a, res["theta"])
        fi = feature_importance_svd(WJ)
        imp = fi["importance"]

        importance_history.append({
            "round": round_i,
            "active_features": list(active),
            "importance": {active[j]: float(imp[j]) for j in range(len(active))},
        })

        # Find least important
        j_min = int(np.argmin(imp))
        feat_to_remove = active[j_min]
        imp_val = float(imp[j_min])

        # Trial fit without that feature
        trial_active = [f for f in active if f != feat_to_remove]
        X_trial = X[:, trial_active]
        theta0_trial = np.zeros(len(trial_active))
        res_trial = solve_lf(X_trial, y, theta0_trial, model_linear_nd, jac_linear_nd, cfg)
        d_trial = residuals(X_trial, y, res_trial["theta"], model_linear_nd)
        trial_obj = lf_objective(d_trial, f_target, 1e-6)
        trial_resid = np.linalg.norm(d_trial)

        # Tolerance check on robust Lf objective, not L2 norm
        rel_increase = (trial_obj - base_obj) / max(abs(base_obj), 1e-15)

        if rel_increase <= tol:
            active = trial_active
            elimination_order.append(feat_to_remove)
            base_obj = trial_obj
            base_resid = trial_resid
            error_history.append({
                "n_features": len(active), "resid_norm": trial_resid,
                "lf_obj": trial_obj,
                "eliminated": feat_to_remove, "importance": imp_val,
            })
            if verbose:
                print(f"  Round {round_i+1}: removed feature {feat_to_remove} "
                      f"(imp={imp_val:.4f}), Lf_obj={trial_obj:.6f} "
                      f"(+{rel_increase*100:.1f}%)")
        else:
            if verbose:
                print(f"  Round {round_i+1}: stopping, removing feature "
                      f"{feat_to_remove} would increase residual by {rel_increase*100:.1f}%")
            break

    return {
        "selected_features": active,
        "eliminated_features": elimination_order,
        "importance_history": importance_history,
        "error_history": error_history,
        "n_selected": len(active),
        "n_original": p,
    }


def stability_selection_svd(X, y, f_target=0.5, n_bootstrap=50,
                            subsample_ratio=0.7, seed=0):
    """
    Stability selection wrapper for SVD-based feature importance.
    Returns selection frequencies for each feature.
    """
    rng = np.random.default_rng(seed)
    n, p = X.shape
    n_sub = int(subsample_ratio * n)
    selection_count = np.zeros(p)
    cfg = SolverConfig(mode="dual", f_target=f_target, max_iter=100)

    for b in range(n_bootstrap):
        idx = rng.choice(n, n_sub, replace=False)
        X_b, y_b = X[idx], y[idx]
        theta0_b = np.zeros(p)
        res = solve_lf(X_b, y_b, theta0_b, model_linear_nd, jac_linear_nd, cfg)
        d = residuals(X_b, y_b, res["theta"], model_linear_nd)
        w = irls_weights(d, f_target, 1e-6)
        WJ = w[:, None] * jac_linear_nd(X_b, res["theta"])
        fi = feature_importance_svd(WJ)
        imp = fi["importance"]
        median_imp = np.median(imp)
        selection_count[imp > median_imp] += 1

    frequencies = selection_count / n_bootstrap
    return {
        "frequencies": frequencies,
        "selected": np.where(frequencies > 0.6)[0].tolist(),
        "n_bootstrap": n_bootstrap,
    }


# ═══════════════════════════════════════════════════════════════════
# 9. HESSIAN ANALYSIS UTILITIES
# ═══════════════════════════════════════════════════════════════════

def analyze_hessian(x, y, theta, model, jacobian, f_vals, eps=1e-4):
    """Analyze Hessian structure (density, PD, condition) across f values."""
    rows = []
    d = residuals(x, y, theta, model)
    J = jacobian(x, theta)
    for f in f_vals:
        w = irls_weights(d, f, eps)
        WJ = w[:, None] * J if w.ndim == 1 else (w.reshape(-1, 1) * J)
        H = WJ.T @ WJ
        eigvals = np.linalg.eigvalsh(H)
        is_pd = bool(np.all(eigvals > 0))
        cond = float(eigvals[-1] / max(eigvals[0], 1e-15))
        nnz = np.count_nonzero(np.abs(H) > 1e-12)
        density = nnz / H.size
        rows.append({
            "f": f, "is_pd": is_pd, "condition": cond,
            "density": density, "min_eigval": float(eigvals[0]),
            "max_eigval": float(eigvals[-1]),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# 10. DATA GENERATION
# ═══════════════════════════════════════════════════════════════════

def generate_1d_data(model_name, cfg: SyntheticConfig):
    rng = np.random.default_rng(cfg.seed)
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.n_samples)
    model_fn, _ = SYNTHETIC_REGISTRY[model_name]
    theta_true = TRUE_THETA[model_name]
    y_clean = model_fn(x, theta_true)
    y = y_clean + rng.normal(0, cfg.noise_std, cfg.n_samples)
    n_out = int(cfg.outlier_ratio * cfg.n_samples)
    outlier_idx = rng.choice(cfg.n_samples, n_out, replace=False)
    if cfg.outlier_dist == "gaussian":
        y[outlier_idx] += rng.normal(0, cfg.outlier_std, n_out)
    elif cfg.outlier_dist == "uniform":
        y[outlier_idx] += rng.uniform(-cfg.outlier_std, cfg.outlier_std, n_out)
    return x, y, y_clean, outlier_idx


def generate_nd_data(n_samples=200, n_features=10, noise_std=0.1,
                     outlier_ratio=0.20, outlier_std=5.0, seed=0,
                     correlation="none"):
    """Generate nD linear data with optional correlation and outliers."""
    rng = np.random.default_rng(seed)
    theta_true = rng.standard_normal(n_features)
    X = rng.standard_normal((n_samples, n_features))
    if correlation == "dense":
        # Add correlation between features
        for j in range(1, n_features):
            X[:, j] += 0.5 * X[:, 0] + 0.3 * X[:, max(0, j-1)]
    y_clean = X @ theta_true
    y = y_clean + rng.normal(0, noise_std, n_samples)
    n_out = int(outlier_ratio * n_samples)
    idx = rng.choice(n_samples, n_out, replace=False)
    y[idx] += rng.normal(0, outlier_std, n_out)
    return X, y, y_clean, theta_true, idx


def generate_nd_leverage(n_samples=200, n_features=10, noise_std=0.1,
                         leverage_ratio=0.15, leverage_magnitude=8.0, seed=0):
    """Generate data with leverage-point outliers (extreme in X-space)."""
    rng = np.random.default_rng(seed)
    theta_true = rng.standard_normal(n_features)
    X = rng.standard_normal((n_samples, n_features))
    y_clean = X @ theta_true
    y = y_clean + rng.normal(0, noise_std, n_samples)
    n_lev = int(round(leverage_ratio * n_samples))
    lev_idx = rng.choice(n_samples, n_lev, replace=False)
    for i in lev_idx:
        X[i] = rng.normal(0, leverage_magnitude, n_features)
        y[i] = X[i] @ theta_true + rng.normal(0, leverage_magnitude * 2)
    return X, y, y_clean, theta_true, lev_idx


# ═══════════════════════════════════════════════════════════════════
# 11. ABLATION & BENCHMARKING INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════

def run_ablation(model_names, methods, n_trials=30,
                 noise_std=0.03, outlier_ratio=0.25, outlier_std=1.5,
                 outlier_dist="gaussian", solver_cfg=None):
    """Run all methods × all models × n_trials."""
    if solver_cfg is None:
        solver_cfg = SolverConfig()
    rows = []
    for mn in model_names:
        model_fn, jac_fn = SYNTHETIC_REGISTRY[mn]
        theta_true = TRUE_THETA[mn]
        theta0 = default_init(mn)
        for trial in range(n_trials):
            cfg_d = SyntheticConfig(
                model_name=mn, seed=trial,
                noise_std=noise_std, outlier_ratio=outlier_ratio,
                outlier_std=outlier_std, outlier_dist=outlier_dist)
            x, y, y_clean, _ = generate_1d_data(mn, cfg_d)
            for m in methods:
                res = solve_method(m, x, y, theta0, model_fn, jac_fn,
                                  solver_cfg, noise_std)
                pe = float(np.linalg.norm(res["theta"] - theta_true))
                rows.append({
                    "model": mn, "method": m, "trial": trial,
                    "param_error": pe,
                    "final_obj": res.get("final_obj", np.nan),
                    "runtime_s": res["runtime_s"],
                    "rejections": res.get("rejections", 0),
                    "accepts": res.get("accepts", 0),
                    "sign_flips": res.get("sign_flips", 0),
                    "avg_rank": res.get("avg_rank", np.nan),
                })
    return pd.DataFrame(rows)


def aggregate(df, by):
    return df.groupby(by, as_index=False).agg(
        mean_pe=("param_error", "mean"), std_pe=("param_error", "std"),
        mean_obj=("final_obj", "mean"), mean_rt=("runtime_s", "mean"),
        mean_rej=("rejections", "mean"), mean_acc=("accepts", "mean"),
        mean_sign_flips=("sign_flips", "mean"),
        mean_rank=("avg_rank", "mean"),
    )


# ═══════════════════════════════════════════════════════════════════
# 12. QUICK SELF-TEST
# ═══════════════════════════════════════════════════════════════════

def self_test():
    """Quick sanity checks."""
    print("=" * 60)
    print("SELF-TEST: Lf-Norm Merged Solver")
    print("=" * 60)

    # 1. f=2, eps=0 → OLS equivalence
    rng = np.random.default_rng(42)
    d_test = rng.standard_normal(50)
    ols_obj = 0.5 * np.sum(d_test**2)
    lf2_obj = lf_objective(d_test, f=2.0, eps=0.0)
    assert np.isclose(ols_obj, lf2_obj), "f=2 OLS equivalence FAILED"
    print(f"  [PASS] f=2 → OLS: {ols_obj:.8f} == {lf2_obj:.8f}")

    # 2. f=1, eps→0 → L1 approximation
    lf1_obj = lf_objective(d_test, f=1.0, eps=1e-12)
    l1_obj = np.sum(np.abs(d_test))
    assert np.isclose(lf1_obj, l1_obj, rtol=1e-4), "f=1 L1 approx FAILED"
    print(f"  [PASS] f=1 → L1:  {lf1_obj:.8f} ≈ {l1_obj:.8f}")

    # 3. Solver runs without errors (exp_decay with outliers)
    cfg_d = SyntheticConfig(seed=0, outlier_ratio=0.25, outlier_std=1.5)
    x, y, _, _ = generate_1d_data("exp_decay", cfg_d)
    theta0 = default_init("exp_decay")
    theta_true = TRUE_THETA["exp_decay"]

    # Without new features
    cfg_base = SolverConfig(use_sign_detection=False, use_momentum=False)
    res_base = solve_lf(x, y, theta0, model_exp_decay, jac_exp_decay, cfg_base)
    pe_base = np.linalg.norm(res_base["theta"] - theta_true)
    print(f"  [PASS] Solver (base):     PE = {pe_base:.6f}")

    # With sign detection + momentum
    cfg_full = SolverConfig(use_sign_detection=True, use_momentum=True)
    res_full = solve_lf(x, y, theta0, model_exp_decay, jac_exp_decay, cfg_full)
    pe_full = np.linalg.norm(res_full["theta"] - theta_true)
    sf = res_full["history"]["sign_flips"]
    mm = res_full["history"]["momentum_applied"]
    print(f"  [PASS] Solver (full):     PE = {pe_full:.6f}, "
          f"sign_flips={sf}, momentum_steps={mm}")

    # 4. NIST function count
    print(f"  [INFO] NIST functions registered: {len(NIST_REGISTRY)}")
    for diff in ["Lower", "Average", "Higher"]:
        cnt = sum(1 for v in NIST_REGISTRY.values() if v["difficulty"] == diff)
        print(f"         {diff}: {cnt}")

    # 5. Jacobian check on a NIST function
    x_test = np.array([1.0, 2.0, 3.0])
    b_test = np.array([0.16, 0.005])
    J_analytical = misra1a_jacobian(x_test, b_test)
    J_numerical = np.zeros_like(J_analytical)
    eps_fd = 1e-7
    for j in range(len(b_test)):
        b_plus = b_test.copy(); b_plus[j] += eps_fd
        b_minus = b_test.copy(); b_minus[j] -= eps_fd
        J_numerical[:, j] = (misra1a(x_test, b_plus) - misra1a(x_test, b_minus)) / (2 * eps_fd)
    jac_err = np.max(np.abs(J_analytical - J_numerical))
    assert jac_err < 1e-5, f"Misra1a Jacobian check FAILED (err={jac_err})"
    print(f"  [PASS] Misra1a Jacobian:  max_err = {jac_err:.2e}")

    # 6. Chwirut1 Jacobian (was buggy)
    x_test2 = np.array([0.5, 1.0, 2.0])
    b_test2 = np.array([0.1, 1.0, 0.5])
    J_a = chwirut1_jacobian(x_test2, b_test2)
    J_n = np.zeros_like(J_a)
    for j in range(len(b_test2)):
        bp = b_test2.copy(); bp[j] += eps_fd
        bm = b_test2.copy(); bm[j] -= eps_fd
        J_n[:, j] = (chwirut1(x_test2, bp) - chwirut1(x_test2, bm)) / (2 * eps_fd)
    jac_err2 = np.max(np.abs(J_a - J_n))
    assert jac_err2 < 1e-5, f"Chwirut1 Jacobian check FAILED (err={jac_err2})"
    print(f"  [PASS] Chwirut1 Jacobian: max_err = {jac_err2:.2e} (bug was fixed)")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    self_test()
