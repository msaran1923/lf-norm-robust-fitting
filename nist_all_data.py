#!/usr/bin/env python3
"""
All 27 NIST StRD Nonlinear Regression Datasets
================================================
Self-contained module that downloads data from NIST on first use,
caches locally, and provides model functions with analytical Jacobians.

Usage:
    from nist_all_data import NIST_ALL_DATASETS
    # NIST_ALL_DATASETS is a dict keyed by dataset name, each entry has:
    #   x, y, model, jac, theta_cert, theta_start, p, difficulty

Datasets (27 total):
  Lower (8):   Misra1a, Chwirut2, Chwirut1, Lanczos3, Gauss1, Gauss2, DanWood, Misra1b
  Average (11): Kirby2, Hahn1, Nelson, MGH17, Lanczos1, Lanczos2, Gauss3,
                Misra1c, Misra1d, Roszman1, ENSO
  Higher (8):  MGH09, Thurber, BoxBOD, Rat42, MGH10, Eckerle4, Rat43, Bennett5
"""

import json
import os
import re
import urllib.request
import numpy as np

_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nist_cache.json")

# ═══════════════════════════════════════════════════════════════════
# 1. MODEL FUNCTIONS AND JACOBIANS
# ═══════════════════════════════════════════════════════════════════

# --- Misra1a, BoxBOD: y = b1*(1 - exp(-b2*x)) ---
def _m_exp_saturation(x, t):
    return t[0] * (1.0 - np.exp(-t[1] * x))
def _j_exp_saturation(x, t):
    e = np.exp(-t[1] * x)
    J = np.empty((x.size, 2)); J[:,0] = 1.0 - e; J[:,1] = t[0]*x*e
    return J

# --- Misra1b: y = b1*(1 - (1+b2*x/2)^(-2)) ---
def _m_misra1b(x, t):
    return t[0] * (1.0 - (1.0 + t[1]*x/2.0)**(-2))
def _j_misra1b(x, t):
    u = 1.0 + t[1]*x/2.0
    J = np.empty((x.size, 2)); J[:,0] = 1.0 - u**(-2); J[:,1] = t[0]*x*u**(-3)
    return J

# --- Misra1c: y = b1*(1 - (1+2*b2*x)^(-1/2)) ---
def _m_misra1c(x, t):
    return t[0] * (1.0 - (1.0 + 2.0*t[1]*x)**(-0.5))
def _j_misra1c(x, t):
    u = 1.0 + 2.0*t[1]*x
    J = np.empty((x.size, 2)); J[:,0] = 1.0 - u**(-0.5); J[:,1] = t[0]*x*u**(-1.5)
    return J

# --- Misra1d: y = b1*b2*x/(1+b2*x) ---
def _m_misra1d(x, t):
    return t[0]*t[1]*x / (1.0 + t[1]*x)
def _j_misra1d(x, t):
    u = 1.0 + t[1]*x
    J = np.empty((x.size, 2)); J[:,0] = t[1]*x/u; J[:,1] = t[0]*x/u**2
    return J

# --- Chwirut1, Chwirut2: y = exp(-b1*x)/(b2+b3*x) ---
def _m_chwirut(x, t):
    return np.exp(-t[0]*x) / (t[1] + t[2]*x)
def _j_chwirut(x, t):
    e = np.exp(-t[0]*x); v = t[1]+t[2]*x
    J = np.empty((x.size, 3)); J[:,0]=-x*e/v; J[:,1]=-e/v**2; J[:,2]=-x*e/v**2
    return J

# --- DanWood: y = b1*x^b2 ---
def _m_danwood(x, t):
    return t[0] * x**t[1]
def _j_danwood(x, t):
    J = np.empty((x.size, 2))
    J[:,0] = x**t[1]; J[:,1] = t[0]*x**t[1]*np.log(np.maximum(x, 1e-30))
    return J

# --- Lanczos1/2/3: y = b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x) ---
def _m_lanczos(x, t):
    return t[0]*np.exp(-t[1]*x) + t[2]*np.exp(-t[3]*x) + t[4]*np.exp(-t[5]*x)
def _j_lanczos(x, t):
    e1=np.exp(-t[1]*x); e2=np.exp(-t[3]*x); e3=np.exp(-t[5]*x)
    J = np.empty((x.size, 6))
    J[:,0]=e1; J[:,1]=-t[0]*x*e1; J[:,2]=e2; J[:,3]=-t[2]*x*e2
    J[:,4]=e3; J[:,5]=-t[4]*x*e3
    return J

# --- Gauss1/2/3: y = b1*exp(-b2*x) + b3*exp(-((x-b4)/b5)^2) + b6*exp(-((x-b7)/b8)^2) ---
def _m_gauss(x, t):
    z1 = (x-t[3])/t[4]; z2 = (x-t[6])/t[7]
    return t[0]*np.exp(-t[1]*x) + t[2]*np.exp(-z1**2) + t[5]*np.exp(-z2**2)
def _j_gauss(x, t):
    e0 = np.exp(-t[1]*x)
    z1 = (x-t[3])/t[4]; e1 = np.exp(-z1**2)
    z2 = (x-t[6])/t[7]; e2 = np.exp(-z2**2)
    J = np.empty((x.size, 8))
    J[:,0]=e0; J[:,1]=-t[0]*x*e0
    J[:,2]=e1; J[:,3]=t[2]*2*z1/t[4]*e1; J[:,4]=t[2]*2*z1**2/t[4]*e1
    J[:,5]=e2; J[:,6]=t[5]*2*z2/t[7]*e2; J[:,7]=t[5]*2*z2**2/t[7]*e2
    return J

# --- MGH09: y = b1*(x^2+b2*x)/(x^2+b3*x+b4) ---
def _m_mgh09(x, t):
    return t[0]*(x**2+t[1]*x)/(x**2+t[2]*x+t[3])
def _j_mgh09(x, t):
    x2=x**2; num=x2+t[1]*x; den=x2+t[2]*x+t[3]
    J = np.empty((x.size, 4))
    J[:,0]=num/den; J[:,1]=t[0]*x/den; J[:,2]=-t[0]*num*x/den**2; J[:,3]=-t[0]*num/den**2
    return J

# --- MGH10: y = b1*exp(b2/(x+b3)) ---
def _m_mgh10(x, t):
    return t[0]*np.exp(t[1]/(x+t[2]))
def _j_mgh10(x, t):
    u=x+t[2]; e=np.exp(t[1]/u)
    J = np.empty((x.size, 3)); J[:,0]=e; J[:,1]=t[0]*e/u; J[:,2]=-t[0]*t[1]*e/u**2
    return J

# --- MGH17: y = b1 + b2*exp(-x*b4) + b3*exp(-x*b5) ---
def _m_mgh17(x, t):
    return t[0]+t[1]*np.exp(-x*t[3])+t[2]*np.exp(-x*t[4])
def _j_mgh17(x, t):
    e1=np.exp(-x*t[3]); e2=np.exp(-x*t[4])
    J = np.empty((x.size, 5))
    J[:,0]=1.0; J[:,1]=e1; J[:,2]=e2; J[:,3]=-t[1]*x*e1; J[:,4]=-t[2]*x*e2
    return J

# --- Eckerle4: y = (b1/b2)*exp(-0.5*((x-b3)/b2)^2) ---
def _m_eckerle4(x, t):
    z=(x-t[2])/t[1]; return (t[0]/t[1])*np.exp(-0.5*z**2)
def _j_eckerle4(x, t):
    z=(x-t[2])/t[1]; g=np.exp(-0.5*z**2); v=(t[0]/t[1])*g
    J = np.empty((x.size, 3))
    J[:,0]=g/t[1]; J[:,1]=v*(z**2-1.0)/t[1]; J[:,2]=v*z/t[1]
    return J

# --- Rat42: y = b1/(1+exp(b2-b3*x)) ---
def _m_rat42(x, t):
    return t[0]/(1.0+np.exp(t[1]-t[2]*x))
def _j_rat42(x, t):
    e=np.exp(t[1]-t[2]*x); d=(1.0+e)**2
    J = np.empty((x.size, 3)); J[:,0]=1.0/(1.0+e); J[:,1]=-t[0]*e/d; J[:,2]=t[0]*x*e/d
    return J

# --- Rat43: y = b1/((1+exp(b2-b3*x))^(1/b4)) ---
def _m_rat43(x, t):
    e=np.exp(t[1]-t[2]*x); return t[0]/(1.0+e)**(1.0/t[3])
def _j_rat43(x, t):
    e=np.exp(t[1]-t[2]*x); u=1.0+e; ib4=1.0/t[3]; f=t[0]*u**(-ib4)
    J = np.empty((x.size, 4))
    J[:,0]=u**(-ib4); J[:,1]=-f*ib4*e/u; J[:,2]=f*ib4*x*e/u
    J[:,3]=f*np.log(np.maximum(u, 1e-30))/t[3]**2
    return J

# --- Bennett5: y = b1*(b2+x)^(-1/b3) ---
def _m_bennett5(x, t):
    return t[0]*(t[1]+x)**(-1.0/t[2])
def _j_bennett5(x, t):
    u=t[1]+x; ib3=1.0/t[2]; f=u**(-ib3)
    J = np.empty((x.size, 3))
    J[:,0]=f; J[:,1]=-t[0]*ib3*u**(-ib3-1.0)
    J[:,2]=t[0]*f*np.log(np.maximum(u, 1e-30))/t[2]**2
    return J

# --- Kirby2: y = (b1+b2*x+b3*x^2)/(1+b4*x+b5*x^2) ---
def _m_kirby2(x, t):
    x2=x**2; return (t[0]+t[1]*x+t[2]*x2)/(1.0+t[3]*x+t[4]*x2)
def _j_kirby2(x, t):
    x2=x**2; den=1.0+t[3]*x+t[4]*x2; num=t[0]+t[1]*x+t[2]*x2
    J = np.empty((x.size, 5))
    J[:,0]=1.0/den; J[:,1]=x/den; J[:,2]=x2/den
    J[:,3]=-num*x/den**2; J[:,4]=-num*x2/den**2
    return J

# --- Hahn1, Thurber: y = (b1+b2*x+b3*x^2+b4*x^3)/(1+b5*x+b6*x^2+b7*x^3) ---
def _m_rational7(x, t):
    x2=x**2; x3=x**3
    return (t[0]+t[1]*x+t[2]*x2+t[3]*x3)/(1.0+t[4]*x+t[5]*x2+t[6]*x3)
def _j_rational7(x, t):
    x2=x**2; x3=x**3
    num=t[0]+t[1]*x+t[2]*x2+t[3]*x3; den=1.0+t[4]*x+t[5]*x2+t[6]*x3
    J = np.empty((x.size, 7))
    J[:,0]=1.0/den; J[:,1]=x/den; J[:,2]=x2/den; J[:,3]=x3/den
    J[:,4]=-num*x/den**2; J[:,5]=-num*x2/den**2; J[:,6]=-num*x3/den**2
    return J

# --- Nelson: log(y) = b1 - b2*x1*exp(-b3*x2) ---
#     Nelson has 2 predictors packed as (n,2) array; y is log-transformed
def _m_nelson(x, t):
    return t[0] - t[1]*x[:,0]*np.exp(-t[2]*x[:,1])
def _j_nelson(x, t):
    e=np.exp(-t[2]*x[:,1])
    J = np.empty((x.shape[0], 3))
    J[:,0]=1.0; J[:,1]=-x[:,0]*e; J[:,2]=t[1]*x[:,0]*x[:,1]*e
    return J

# --- Roszman1: y = b1 - b2*x - arctan(b3/(x-b4))/pi ---
def _m_roszman1(x, t):
    return t[0] - t[1]*x - np.arctan(t[2]/(x-t[3]))/np.pi
def _j_roszman1(x, t):
    u=x-t[3]; v=t[2]/u; d=1.0+v**2
    J = np.empty((x.size, 4))
    J[:,0]=1.0; J[:,1]=-x; J[:,2]=-1.0/(np.pi*u*d); J[:,3]=-t[2]/(np.pi*u**2*d)
    return J

# --- ENSO: y = b1 + b2*cos(2π x/12) + b3*sin(2π x/12)
#              + b5*cos(2π x/b4) + b6*sin(2π x/b4)
#              + b8*cos(2π x/b7) + b9*sin(2π x/b7) ---
def _m_enso(x, t):
    w1=2*np.pi*x/12.0; w2=2*np.pi*x/t[3]; w3=2*np.pi*x/t[6]
    return (t[0]+t[1]*np.cos(w1)+t[2]*np.sin(w1)
            +t[4]*np.cos(w2)+t[5]*np.sin(w2)+t[7]*np.cos(w3)+t[8]*np.sin(w3))
def _j_enso(x, t):
    w1=2*np.pi*x/12.0; w2=2*np.pi*x/t[3]; w3=2*np.pi*x/t[6]
    J = np.empty((x.size, 9))
    J[:,0]=1.0; J[:,1]=np.cos(w1); J[:,2]=np.sin(w1)
    dw2=2*np.pi*x/t[3]**2
    J[:,3]=t[4]*np.sin(w2)*dw2 - t[5]*np.cos(w2)*dw2
    J[:,4]=np.cos(w2); J[:,5]=np.sin(w2)
    dw3=2*np.pi*x/t[6]**2
    J[:,6]=t[7]*np.sin(w3)*dw3 - t[8]*np.cos(w3)*dw3
    J[:,7]=np.cos(w3); J[:,8]=np.sin(w3)
    return J


# Model registry: name → (model_fn, jac_fn)
_MODEL_REGISTRY = {
    "Misra1a":  (_m_exp_saturation, _j_exp_saturation),
    "Misra1b":  (_m_misra1b, _j_misra1b),
    "Misra1c":  (_m_misra1c, _j_misra1c),
    "Misra1d":  (_m_misra1d, _j_misra1d),
    "Chwirut1": (_m_chwirut, _j_chwirut),
    "Chwirut2": (_m_chwirut, _j_chwirut),
    "DanWood":  (_m_danwood, _j_danwood),
    "BoxBOD":   (_m_exp_saturation, _j_exp_saturation),
    "Lanczos1": (_m_lanczos, _j_lanczos),
    "Lanczos2": (_m_lanczos, _j_lanczos),
    "Lanczos3": (_m_lanczos, _j_lanczos),
    "Gauss1":   (_m_gauss, _j_gauss),
    "Gauss2":   (_m_gauss, _j_gauss),
    "Gauss3":   (_m_gauss, _j_gauss),
    "MGH09":    (_m_mgh09, _j_mgh09),
    "MGH10":    (_m_mgh10, _j_mgh10),
    "MGH17":    (_m_mgh17, _j_mgh17),
    "Eckerle4": (_m_eckerle4, _j_eckerle4),
    "Rat42":    (_m_rat42, _j_rat42),
    "Rat43":    (_m_rat43, _j_rat43),
    "Bennett5": (_m_bennett5, _j_bennett5),
    "Kirby2":   (_m_kirby2, _j_kirby2),
    "Hahn1":    (_m_rational7, _j_rational7),
    "Thurber":  (_m_rational7, _j_rational7),
    "Nelson":   (_m_nelson, _j_nelson),
    "Roszman1": (_m_roszman1, _j_roszman1),
    "ENSO":     (_m_enso, _j_enso),
}

# ═══════════════════════════════════════════════════════════════════
# 2. DATASET METADATA
# ═══════════════════════════════════════════════════════════════════

# (name, dat_filename, difficulty, n_params, n_obs, n_predictors)
_DATASET_INFO = [
    ("Misra1a",  "Misra1a.dat",  "Lower",   2,  14, 1),
    ("Chwirut2", "Chwirut2.dat", "Lower",   3,  54, 1),
    ("Chwirut1", "Chwirut1.dat", "Lower",   3, 214, 1),
    ("Lanczos3", "Lanczos3.dat", "Lower",   6,  24, 1),
    ("Gauss1",   "Gauss1.dat",   "Lower",   8, 250, 1),
    ("Gauss2",   "Gauss2.dat",   "Lower",   8, 250, 1),
    ("DanWood",  "DanWood.dat",  "Lower",   2,   6, 1),
    ("Misra1b",  "Misra1b.dat",  "Lower",   2,  14, 1),
    ("Kirby2",   "Kirby2.dat",   "Average",  5, 151, 1),
    ("Hahn1",    "Hahn1.dat",    "Average",  7, 236, 1),
    ("Nelson",   "Nelson.dat",   "Average",  3, 128, 2),
    ("MGH17",    "MGH17.dat",    "Average",  5,  33, 1),
    ("Lanczos1", "Lanczos1.dat", "Average",  6,  24, 1),
    ("Lanczos2", "Lanczos2.dat", "Average",  6,  24, 1),
    ("Gauss3",   "Gauss3.dat",   "Average",  8, 250, 1),
    ("Misra1c",  "Misra1c.dat",  "Average",  2,  14, 1),
    ("Misra1d",  "Misra1d.dat",  "Average",  2,  14, 1),
    ("Roszman1", "Roszman1.dat", "Average",  4,  25, 1),
    ("ENSO",     "ENSO.dat",     "Average",  9, 168, 1),
    ("MGH09",    "MGH09.dat",    "Higher",   4,  11, 1),
    ("Thurber",  "Thurber.dat",  "Higher",   7,  37, 1),
    ("BoxBOD",   "BoxBOD.dat",   "Higher",   2,   6, 1),
    ("Rat42",    "Rat42.dat",    "Higher",   3,   9, 1),
    ("MGH10",    "MGH10.dat",    "Higher",   3,  16, 1),
    ("Eckerle4", "Eckerle4.dat", "Higher",   3,  35, 1),
    ("Rat43",    "Rat43.dat",    "Higher",   4,  15, 1),
    ("Bennett5", "Bennett5.dat", "Higher",   3, 154, 1),
]


# ═══════════════════════════════════════════════════════════════════
# 3. DOWNLOAD AND PARSE
# ═══════════════════════════════════════════════════════════════════

_BASE_URL = "https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA"


def _download_dat(filename):
    """Download a single .dat file from NIST StRD."""
    url = f"{_BASE_URL}/{filename}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _parse_dat(text, n_params, n_predictors=1):
    """Parse a NIST StRD .dat file to extract data and certified values.

    Returns dict with x (list), y (list), certified (list), start2 (list).
    """
    lines = text.strip().split("\n")

    # --- Extract certified params and starting values ---
    # Look for the parameter table: lines with "Start 1" and "Start 2" header
    param_lines = []
    in_section = False
    for line in lines[:60]:
        s = line.strip()
        if "Start 1" in s and "Start 2" in s:
            in_section = True
            continue
        if in_section:
            nums = re.findall(r'[-+]?\d+\.?\d*(?:[eE][+-]?\d+)?', s)
            if len(nums) >= 3:
                param_lines.append(nums)
            elif len(nums) < 2 and s:
                in_section = False

    # Each parameter line in NIST .dat files has format:
    #   bN =  Start1_value  Start2_value  Certified_value  StdDev_value
    # The regex also captures N from "bN", giving 5 numbers per line.
    # Use negative indices to reliably extract from the END:
    #   pl[-4] = Start 1 (far),  pl[-3] = Start 2 (close),
    #   pl[-2] = Certified,      pl[-1] = StdDev
    certified = [float(pl[-2]) for pl in param_lines[:n_params]] if param_lines else []
    start2 = [float(pl[-3]) for pl in param_lines[:n_params]] if param_lines else []

    # --- Extract data (starts at line 61, i.e. index 60) ---
    x_data, y_data = [], []
    for dline in lines[60:]:
        dline = dline.strip()
        if not dline:
            continue
        parts = dline.split()
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue

        if n_predictors == 1 and len(vals) >= 2:
            y_data.append(vals[0])
            x_data.append(vals[1])
        elif n_predictors == 2 and len(vals) >= 3:
            y_data.append(vals[0])
            x_data.append([vals[1], vals[2]])
        elif n_predictors > 2 and len(vals) >= n_predictors + 1:
            y_data.append(vals[0])
            x_data.append(vals[1:n_predictors+1])

    return {
        "x": x_data, "y": y_data,
        "certified": certified, "start2": start2,
    }


def download_all(force=False):
    """Download all 27 datasets, parse, and cache to JSON file.

    Returns dict of parsed data keyed by dataset name.
    """
    if os.path.exists(_CACHE_FILE) and not force:
        with open(_CACHE_FILE, "r") as f:
            cached = json.load(f)
        if len(cached) == 27:
            return cached

    print("Downloading NIST StRD nonlinear regression datasets...")
    all_data = {}
    for name, filename, difficulty, n_params, n_obs, n_pred in _DATASET_INFO:
        try:
            text = _download_dat(filename)
            parsed = _parse_dat(text, n_params, n_pred)
            parsed["difficulty"] = difficulty
            parsed["n_params"] = n_params
            parsed["n_predictors"] = n_pred
            all_data[name] = parsed
            print(f"  {name:12s}: {len(parsed['y']):4d} obs, "
                  f"{len(parsed['certified'])} params — OK")
        except Exception as e:
            print(f"  {name:12s}: FAILED — {e}")

    # Cache
    with open(_CACHE_FILE, "w") as f:
        json.dump(all_data, f)
    print(f"Cached {len(all_data)} datasets to {_CACHE_FILE}")
    return all_data


# ═══════════════════════════════════════════════════════════════════
# 4. BUILD NIST_ALL_DATASETS DICT
# ═══════════════════════════════════════════════════════════════════

def load_datasets(exclude_nelson=False):
    """Load all 27 NIST datasets into the standard dict format.

    Parameters
    ----------
    exclude_nelson : bool
        If True, exclude Nelson which has 2 predictors (x is a 2D array).
        Default False — Nelson works correctly with all solvers.

    Returns
    -------
    dict : keyed by dataset name, each entry has:
        x : np.ndarray — predictor(s)
        y : np.ndarray — response
        model : callable(x, theta) → predictions
        jac : callable(x, theta) → Jacobian
        theta_cert : np.ndarray — certified parameter values
        theta_start : np.ndarray — starting values ("close" start)
        p : int — number of parameters
        difficulty : str — "Lower", "Average", or "Higher"
    """
    raw = download_all()

    datasets = {}
    for name, data in raw.items():
        if exclude_nelson and name == "Nelson":
            continue

        model_fn, jac_fn = _MODEL_REGISTRY[name]

        x = np.array(data["x"])
        y = np.array(data["y"])
        cert = np.array(data["certified"])
        start = np.array(data["start2"]) if data["start2"] else cert.copy()

        datasets[name] = {
            "x": x, "y": y,
            "model": model_fn, "jac": jac_fn,
            "theta_cert": cert,
            "theta_start": start,
            "p": data["n_params"],
            "difficulty": data["difficulty"],
        }

    return datasets


# Lazy-loaded global — populated on first access
_LOADED = {}

def _ensure_loaded():
    global _LOADED
    if not _LOADED:
        _LOADED.update(load_datasets(exclude_nelson=False))

class _DatasetProxy(dict):
    """Dict that auto-downloads on first access."""
    def __getitem__(self, key):
        _ensure_loaded()
        return _LOADED[key]
    def __contains__(self, key):
        _ensure_loaded()
        return key in _LOADED
    def __iter__(self):
        _ensure_loaded()
        return iter(_LOADED)
    def __len__(self):
        _ensure_loaded()
        return len(_LOADED)
    def items(self):
        _ensure_loaded()
        return _LOADED.items()
    def keys(self):
        _ensure_loaded()
        return _LOADED.keys()
    def values(self):
        _ensure_loaded()
        return _LOADED.values()
    def get(self, key, default=None):
        _ensure_loaded()
        return _LOADED.get(key, default)

NIST_ALL_DATASETS = _DatasetProxy()


# ═══════════════════════════════════════════════════════════════════
# 5. SELF-TEST
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    datasets = load_datasets(exclude_nelson=False)
    print(f"\nLoaded {len(datasets)} datasets.\n")

    # Verify models compute without error and Jacobians have correct shape
    n_ok, n_fail = 0, 0
    for name, ds in sorted(datasets.items()):
        x, y = ds["x"], ds["y"]
        theta = ds["theta_cert"]
        try:
            yhat = ds["model"](x, theta)
            J = ds["jac"](x, theta)
            resid = yhat - y
            sse = np.sum(resid**2)
            n = len(y) if isinstance(y, np.ndarray) else y.shape[0]
            assert yhat.shape == (n,), f"yhat shape {yhat.shape} != ({n},)"
            assert J.shape == (n, ds["p"]), f"J shape {J.shape} != ({n}, {ds['p']})"
            print(f"  {name:12s}: {ds['difficulty']:8s}  n={n:4d}  p={ds['p']}  "
                  f"SSE={sse:.6e}  ✓")
            n_ok += 1
        except Exception as e:
            print(f"  {name:12s}: FAILED — {e}")
            n_fail += 1

    print(f"\n{n_ok} OK, {n_fail} failed.")
