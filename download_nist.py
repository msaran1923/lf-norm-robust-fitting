#!/usr/bin/env python3
"""
Download and parse all 27 NIST StRD Nonlinear Regression datasets.
Run once to generate nist_all_data.py containing hardcoded data arrays.

Usage:
    python download_nist.py

Output:
    nist_all_data.py  — Python module with all dataset arrays
"""

import urllib.request
import re
import os
import sys
import json
import textwrap

BASE_URL = "https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA"

# All 27 datasets: (name, filename, difficulty, n_params)
DATASETS = [
    # Lower difficulty
    ("Misra1a",  "Misra1a.dat",  "Lower",  2),
    ("Chwirut2", "Chwirut2.dat", "Lower",  3),
    ("Chwirut1", "Chwirut1.dat", "Lower",  3),
    ("Lanczos3", "Lanczos3.dat", "Lower",  6),
    ("Gauss1",   "Gauss1.dat",   "Lower",  8),
    ("Gauss2",   "Gauss2.dat",   "Lower",  8),
    ("DanWood",  "DanWood.dat",  "Lower",  2),
    ("Misra1b",  "Misra1b.dat",  "Lower",  2),
    # Average difficulty
    ("Kirby2",   "Kirby2.dat",   "Average", 5),
    ("Hahn1",    "Hahn1.dat",    "Average", 7),
    ("Nelson",   "Nelson.dat",   "Average", 3),
    ("MGH17",    "MGH17.dat",    "Average", 5),
    ("Lanczos1", "Lanczos1.dat", "Average", 6),
    ("Lanczos2", "Lanczos2.dat", "Average", 6),
    ("Gauss3",   "Gauss3.dat",   "Average", 8),
    ("Misra1c",  "Misra1c.dat",  "Average", 2),
    ("Misra1d",  "Misra1d.dat",  "Average", 2),
    ("Roszman1", "Roszman1.dat", "Average", 4),
    ("ENSO",     "ENSO.dat",     "Average", 9),
    # Higher difficulty
    ("MGH09",    "MGH09.dat",    "Higher",  4),
    ("Thurber",  "Thurber.dat",  "Higher",  7),
    ("BoxBOD",   "BoxBOD.dat",   "Higher",  2),
    ("Rat42",    "Rat42.dat",    "Higher",  3),
    ("MGH10",    "MGH10.dat",    "Higher",  3),
    ("Eckerle4", "Eckerle4.dat", "Higher",  3),
    ("Rat43",    "Rat43.dat",    "Higher",  4),
    ("Bennett5", "Bennett5.dat", "Higher",  3),
]


def download_dat(filename):
    """Download a .dat file from NIST StRD."""
    url = f"{BASE_URL}/{filename}"
    print(f"  Downloading {url} ...", end=" ")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        print("OK")
        return text
    except Exception as e:
        print(f"FAILED: {e}")
        return None


def parse_dat(text, n_params):
    """Parse a NIST StRD .dat file.

    Returns dict with:
        x, y: data arrays (lists of floats)
        certified_params: certified parameter values
        start1: "far" starting values (Start 1)
        start2: "close" starting values (Start 2)
    """
    lines = text.strip().split("\n")

    # Find certified values: look for "Certified Values" section
    # Format: b1 = value  value
    # or: B = value  value  value
    certified = []
    start1 = []
    start2 = []

    # Parse certified values and starting values from header (lines 1-60)
    for line in lines[:60]:
        line = line.strip()
        # Match certified parameter values: "b1 = 2.3894212918E+02  2.706..."
        # Some files have "B =" and some have "b1 ="
        m = re.match(r'^\s*[bB]\d?\s*=\s*([-+]?\d+\.\d+[eE][+-]?\d+)', line)
        if m:
            certified.append(float(m.group(1)))

        # Match starting values: "Start 1  Start 2  Parameter"
        # Lines with two numbers before the certified value
        # Format varies: some have "500.  250.  b1"
        m2 = re.match(
            r'^\s*([-+]?\d+\.?\d*[eE]?[+-]?\d*)\s+'
            r'([-+]?\d+\.?\d*[eE]?[+-]?\d*)\s+'
            r'([-+]?\d+\.?\d*[eE]?[+-]?\d*)\s*$',
            line
        )
        if m2 and len(certified) == 0:
            # This is in the starting values section (before certified values)
            pass

    # More robust parsing: scan for the parameter table
    # Look for lines matching the pattern of start1, start2, certified
    param_lines = []
    in_param_section = False
    for i, line in enumerate(lines[:60]):
        stripped = line.strip()
        if "Start 1" in stripped and "Start 2" in stripped:
            in_param_section = True
            continue
        if in_param_section:
            # Try to extract numeric values
            nums = re.findall(r'[-+]?\d+\.?\d*(?:[eE][+-]?\d+)?', stripped)
            if len(nums) >= 3:
                param_lines.append(nums)
            elif stripped == "" or "Residual" in stripped:
                in_param_section = False

    if param_lines:
        start1 = [float(pl[-4]) for pl in param_lines[:n_params]]
        start2 = [float(pl[-3]) for pl in param_lines[:n_params]]
        certified = [float(pl[-2]) for pl in param_lines[:n_params]]

    # Parse data: starts at line 61 (0-indexed: 60)
    data_lines = lines[60:]
    x_data = []
    y_data = []
    for dline in data_lines:
        dline = dline.strip()
        if not dline:
            continue
        parts = dline.split()
        if len(parts) >= 2:
            try:
                vals = [float(p) for p in parts]
                y_data.append(vals[0])  # y is typically first column
                x_data.append(vals[1] if len(vals) == 2 else vals[1:])
            except ValueError:
                continue

    return {
        "x": x_data,
        "y": y_data,
        "certified": certified,
        "start1": start1,
        "start2": start2,
    }


def format_array(arr, name, indent=4):
    """Format a list as a numpy array string."""
    prefix = " " * indent
    if not arr:
        return f"{prefix}{name} = np.array([])"

    # Check if elements are lists (multi-column x)
    if isinstance(arr[0], list):
        lines = [f"{prefix}{name} = np.array(["]
        for row in arr:
            lines.append(f"{prefix}    {row},")
        lines.append(f"{prefix}])")
        return "\n".join(lines)

    # Single array
    per_line = 6
    s = f"{prefix}{name} = np.array([\n"
    for i in range(0, len(arr), per_line):
        chunk = arr[i:i+per_line]
        s += prefix + "    " + ", ".join(f"{v}" for v in chunk) + ",\n"
    s += f"{prefix}])"
    return s


def generate_python(all_data):
    """Generate nist_all_data.py with all datasets."""

    code = '''#!/usr/bin/env python3
"""
All 27 NIST StRD Nonlinear Regression Datasets
================================================
Auto-generated by download_nist.py from https://www.itl.nist.gov/div898/strd/nls/

Contains: data arrays, certified parameters, starting values,
          model functions, and analytical Jacobians for all 27 datasets.

Datasets by difficulty:
  Lower (8):   Misra1a, Chwirut2, Chwirut1, Lanczos3, Gauss1, Gauss2, DanWood, Misra1b
  Average (11): Kirby2, Hahn1, Nelson, MGH17, Lanczos1, Lanczos2, Gauss3,
                Misra1c, Misra1d, Roszman1, ENSO
  Higher (8):  MGH09, Thurber, BoxBOD, Rat42, MGH10, Eckerle4, Rat43, Bennett5
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# MODEL FUNCTIONS AND JACOBIANS
# ═══════════════════════════════════════════════════════════════════

# --- Misra1a: y = b1*(1 - exp(-b2*x)) ---
def m_misra1a(x, t):
    return t[0] * (1.0 - np.exp(-t[1] * x))

def j_misra1a(x, t):
    e = np.exp(-t[1] * x)
    J = np.empty((x.size, 2))
    J[:, 0] = 1.0 - e
    J[:, 1] = t[0] * x * e
    return J


# --- Misra1b: y = b1*(1 - (1+b2*x/2)^(-2)) ---
def m_misra1b(x, t):
    return t[0] * (1.0 - (1.0 + t[1] * x / 2.0) ** (-2))

def j_misra1b(x, t):
    u = 1.0 + t[1] * x / 2.0
    J = np.empty((x.size, 2))
    J[:, 0] = 1.0 - u ** (-2)
    J[:, 1] = t[0] * x * u ** (-3)
    return J


# --- Misra1c: y = b1*(1 - (1+2*b2*x)^(-1/2)) ---
def m_misra1c(x, t):
    return t[0] * (1.0 - (1.0 + 2.0 * t[1] * x) ** (-0.5))

def j_misra1c(x, t):
    u = 1.0 + 2.0 * t[1] * x
    J = np.empty((x.size, 2))
    J[:, 0] = 1.0 - u ** (-0.5)
    J[:, 1] = t[0] * x * u ** (-1.5)
    return J


# --- Misra1d: y = b1*b2*x*(1+b2*x)^(-1) ---
def m_misra1d(x, t):
    return t[0] * t[1] * x / (1.0 + t[1] * x)

def j_misra1d(x, t):
    u = 1.0 + t[1] * x
    J = np.empty((x.size, 2))
    J[:, 0] = t[1] * x / u
    J[:, 1] = t[0] * x / (u ** 2)
    return J


# --- Chwirut1, Chwirut2: y = exp(-b1*x)/(b2+b3*x) ---
def m_chwirut(x, t):
    return np.exp(-t[0] * x) / (t[1] + t[2] * x)

def j_chwirut(x, t):
    e = np.exp(-t[0] * x)
    v = t[1] + t[2] * x
    J = np.empty((x.size, 3))
    J[:, 0] = -x * e / v
    J[:, 1] = -e / v ** 2
    J[:, 2] = -x * e / v ** 2
    return J


# --- DanWood: y = b1*x^b2 ---
def m_danwood(x, t):
    return t[0] * x ** t[1]

def j_danwood(x, t):
    J = np.empty((x.size, 2))
    J[:, 0] = x ** t[1]
    J[:, 1] = t[0] * x ** t[1] * np.log(np.maximum(x, 1e-30))
    return J


# --- BoxBOD: y = b1*(1 - exp(-b2*x))  [same functional form as Misra1a] ---
def m_boxbod(x, t):
    return t[0] * (1.0 - np.exp(-t[1] * x))

def j_boxbod(x, t):
    e = np.exp(-t[1] * x)
    J = np.empty((x.size, 2))
    J[:, 0] = 1.0 - e
    J[:, 1] = t[0] * x * e
    return J


# --- Lanczos1, Lanczos2, Lanczos3: y = b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x) ---
def m_lanczos(x, t):
    return t[0]*np.exp(-t[1]*x) + t[2]*np.exp(-t[3]*x) + t[4]*np.exp(-t[5]*x)

def j_lanczos(x, t):
    e1 = np.exp(-t[1]*x); e2 = np.exp(-t[3]*x); e3 = np.exp(-t[5]*x)
    J = np.empty((x.size, 6))
    J[:, 0] = e1
    J[:, 1] = -t[0]*x*e1
    J[:, 2] = e2
    J[:, 3] = -t[2]*x*e2
    J[:, 4] = e3
    J[:, 5] = -t[4]*x*e3
    return J


# --- Gauss1, Gauss2, Gauss3: y = b1*exp(-b2*x) + b3*exp(-((x-b4)/b5)^2) + b6*exp(-((x-b7)/b8)^2) ---
def m_gauss(x, t):
    z1 = (x - t[3]) / t[4]
    z2 = (x - t[6]) / t[7]
    return t[0]*np.exp(-t[1]*x) + t[2]*np.exp(-z1**2) + t[5]*np.exp(-z2**2)

def j_gauss(x, t):
    e0 = np.exp(-t[1]*x)
    z1 = (x - t[3]) / t[4]; e1 = np.exp(-z1**2)
    z2 = (x - t[6]) / t[7]; e2 = np.exp(-z2**2)
    J = np.empty((x.size, 8))
    J[:, 0] = e0
    J[:, 1] = -t[0]*x*e0
    J[:, 2] = e1
    J[:, 3] = t[2] * 2*z1/t[4] * e1
    J[:, 4] = t[2] * 2*z1**2/t[4] * e1
    J[:, 5] = e2
    J[:, 6] = t[5] * 2*z2/t[7] * e2
    J[:, 7] = t[5] * 2*z2**2/t[7] * e2
    return J


# --- MGH09: y = b1*(x^2+b2*x)/(x^2+b3*x+b4) ---
def m_mgh09(x, t):
    num = t[0] * (x**2 + t[1]*x)
    den = x**2 + t[2]*x + t[3]
    return num / den

def j_mgh09(x, t):
    x2 = x**2
    num = x2 + t[1]*x
    den = x2 + t[2]*x + t[3]
    J = np.empty((x.size, 4))
    J[:, 0] = num / den
    J[:, 1] = t[0] * x / den
    J[:, 2] = -t[0] * num * x / den**2
    J[:, 3] = -t[0] * num / den**2
    return J


# --- MGH10: y = b1*exp(b2/(x+b3)) ---
def m_mgh10(x, t):
    return t[0] * np.exp(t[1] / (x + t[2]))

def j_mgh10(x, t):
    u = x + t[2]
    e = np.exp(t[1] / u)
    J = np.empty((x.size, 3))
    J[:, 0] = e
    J[:, 1] = t[0] * e / u
    J[:, 2] = -t[0] * t[1] * e / u**2
    return J


# --- MGH17: y = b1 + b2*exp(-x*b4) + b3*exp(-x*b5) ---
def m_mgh17(x, t):
    return t[0] + t[1]*np.exp(-x*t[3]) + t[2]*np.exp(-x*t[4])

def j_mgh17(x, t):
    e1 = np.exp(-x*t[3]); e2 = np.exp(-x*t[4])
    J = np.empty((x.size, 5))
    J[:, 0] = 1.0
    J[:, 1] = e1
    J[:, 2] = e2
    J[:, 3] = -t[1]*x*e1
    J[:, 4] = -t[2]*x*e2
    return J


# --- Eckerle4: y = (b1/b2)*exp(-0.5*((x-b3)/b2)^2) ---
def m_eckerle4(x, t):
    z = (x - t[2]) / t[1]
    return (t[0] / t[1]) * np.exp(-0.5 * z**2)

def j_eckerle4(x, t):
    z = (x - t[2]) / t[1]
    g = np.exp(-0.5 * z**2)
    v = (t[0] / t[1]) * g
    J = np.empty((x.size, 3))
    J[:, 0] = g / t[1]
    J[:, 1] = v * (z**2 - 1.0) / t[1]
    J[:, 2] = v * z / t[1]
    return J


# --- Rat42: y = b1/(1+exp(b2-b3*x)) ---
def m_rat42(x, t):
    return t[0] / (1.0 + np.exp(t[1] - t[2]*x))

def j_rat42(x, t):
    e = np.exp(t[1] - t[2]*x)
    d = (1.0 + e)**2
    J = np.empty((x.size, 3))
    J[:, 0] = 1.0 / (1.0 + e)
    J[:, 1] = -t[0] * e / d
    J[:, 2] = t[0] * x * e / d
    return J


# --- Rat43: y = b1/((1+exp(b2-b3*x))^(1/b4)) ---
def m_rat43(x, t):
    e = np.exp(t[1] - t[2]*x)
    return t[0] / (1.0 + e) ** (1.0/t[3])

def j_rat43(x, t):
    e = np.exp(t[1] - t[2]*x)
    u = 1.0 + e
    inv_b4 = 1.0/t[3]
    f = t[0] * u**(-inv_b4)
    J = np.empty((x.size, 4))
    J[:, 0] = u**(-inv_b4)
    J[:, 1] = -f * inv_b4 * e / u
    J[:, 2] = f * inv_b4 * x * e / u
    J[:, 3] = f * np.log(u) / t[3]**2
    return J


# --- Bennett5: y = b1*(b2+x)^(-1/b3) ---
def m_bennett5(x, t):
    return t[0] * (t[1] + x) ** (-1.0/t[2])

def j_bennett5(x, t):
    u = t[1] + x
    inv_b3 = 1.0/t[2]
    f = u ** (-inv_b3)
    J = np.empty((x.size, 3))
    J[:, 0] = f
    J[:, 1] = -t[0] * inv_b3 * u ** (-inv_b3 - 1.0)
    J[:, 2] = t[0] * f * np.log(np.maximum(u, 1e-30)) / t[2]**2
    return J


# --- Kirby2: y = (b1+b2*x+b3*x^2)/(1+b4*x+b5*x^2) ---
def m_kirby2(x, t):
    x2 = x**2
    return (t[0] + t[1]*x + t[2]*x2) / (1.0 + t[3]*x + t[4]*x2)

def j_kirby2(x, t):
    x2 = x**2
    den = 1.0 + t[3]*x + t[4]*x2
    num = t[0] + t[1]*x + t[2]*x2
    J = np.empty((x.size, 5))
    J[:, 0] = 1.0 / den
    J[:, 1] = x / den
    J[:, 2] = x2 / den
    J[:, 3] = -num * x / den**2
    J[:, 4] = -num * x2 / den**2
    return J


# --- Hahn1: y = (b1+b2*x+b3*x^2+b4*x^3)/(1+b5*x+b6*x^2+b7*x^3) ---
def m_hahn1(x, t):
    x2 = x**2; x3 = x**3
    return (t[0]+t[1]*x+t[2]*x2+t[3]*x3) / (1.0+t[4]*x+t[5]*x2+t[6]*x3)

def j_hahn1(x, t):
    x2 = x**2; x3 = x**3
    num = t[0]+t[1]*x+t[2]*x2+t[3]*x3
    den = 1.0+t[4]*x+t[5]*x2+t[6]*x3
    J = np.empty((x.size, 7))
    J[:, 0] = 1.0/den; J[:, 1] = x/den; J[:, 2] = x2/den; J[:, 3] = x3/den
    J[:, 4] = -num*x/den**2; J[:, 5] = -num*x2/den**2; J[:, 6] = -num*x3/den**2
    return J


# --- Thurber: y = (b1+b2*x+b3*x^2+b4*x^3)/(1+b5*x+b6*x^2+b7*x^3) ---
#     Same model as Hahn1
m_thurber = m_hahn1
j_thurber = j_hahn1


# --- Nelson: log(y) = b1 - b2*x1*exp(-b3*x2) ---
#     Nelson has 2 predictors. x is (n,2) array: x[:,0]=x1 (log of amplitude), x[:,1]=x2 (time)
#     y is already log-transformed in the NIST file
def m_nelson(x, t):
    # x is (n,2): col0=x1, col1=x2
    return t[0] - t[1] * x[:, 0] * np.exp(-t[2] * x[:, 1])

def j_nelson(x, t):
    e = np.exp(-t[2] * x[:, 1])
    J = np.empty((x.shape[0], 3))
    J[:, 0] = 1.0
    J[:, 1] = -x[:, 0] * e
    J[:, 2] = t[1] * x[:, 0] * x[:, 1] * e
    return J


# --- Roszman1: y = b1 - b2*x - arctan(b3/(x-b4))/pi ---
def m_roszman1(x, t):
    return t[0] - t[1]*x - np.arctan(t[2]/(x - t[3])) / np.pi

def j_roszman1(x, t):
    u = x - t[3]
    v = t[2] / u
    d = 1.0 + v**2
    J = np.empty((x.size, 4))
    J[:, 0] = 1.0
    J[:, 1] = -x
    J[:, 2] = -1.0 / (np.pi * u * d)
    J[:, 3] = -t[2] / (np.pi * u**2 * d)
    return J


# --- ENSO: y = b1 + b2*cos(2*pi*x/12) + b3*sin(2*pi*x/12)
#              + b5*cos(2*pi*x/b4) + b6*sin(2*pi*x/b4)
#              + b8*cos(2*pi*x/b7) + b9*sin(2*pi*x/b7)  ---
def m_enso(x, t):
    w1 = 2*np.pi*x/12.0
    w2 = 2*np.pi*x/t[3]
    w3 = 2*np.pi*x/t[6]
    return (t[0] + t[1]*np.cos(w1) + t[2]*np.sin(w1)
            + t[4]*np.cos(w2) + t[5]*np.sin(w2)
            + t[7]*np.cos(w3) + t[8]*np.sin(w3))

def j_enso(x, t):
    w1 = 2*np.pi*x/12.0
    w2 = 2*np.pi*x/t[3]
    w3 = 2*np.pi*x/t[6]
    J = np.empty((x.size, 9))
    J[:, 0] = 1.0
    J[:, 1] = np.cos(w1)
    J[:, 2] = np.sin(w1)
    J[:, 3] = t[4]*np.sin(w2)*2*np.pi*x/t[3]**2 - t[5]*np.cos(w2)*2*np.pi*x/t[3]**2
    J[:, 4] = np.cos(w2)
    J[:, 5] = np.sin(w2)
    J[:, 6] = t[7]*np.sin(w3)*2*np.pi*x/t[6]**2 - t[8]*np.cos(w3)*2*np.pi*x/t[6]**2
    J[:, 7] = np.cos(w3)
    J[:, 8] = np.sin(w3)
    return J


# ═══════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "Misra1a":  (m_misra1a, j_misra1a),
    "Misra1b":  (m_misra1b, j_misra1b),
    "Misra1c":  (m_misra1c, j_misra1c),
    "Misra1d":  (m_misra1d, j_misra1d),
    "Chwirut1": (m_chwirut, j_chwirut),
    "Chwirut2": (m_chwirut, j_chwirut),
    "DanWood":  (m_danwood, j_danwood),
    "BoxBOD":   (m_boxbod, j_boxbod),
    "Lanczos1": (m_lanczos, j_lanczos),
    "Lanczos2": (m_lanczos, j_lanczos),
    "Lanczos3": (m_lanczos, j_lanczos),
    "Gauss1":   (m_gauss, j_gauss),
    "Gauss2":   (m_gauss, j_gauss),
    "Gauss3":   (m_gauss, j_gauss),
    "MGH09":    (m_mgh09, j_mgh09),
    "MGH10":    (m_mgh10, j_mgh10),
    "MGH17":    (m_mgh17, j_mgh17),
    "Eckerle4": (m_eckerle4, j_eckerle4),
    "Rat42":    (m_rat42, j_rat42),
    "Rat43":    (m_rat43, j_rat43),
    "Bennett5": (m_bennett5, j_bennett5),
    "Kirby2":   (m_kirby2, j_kirby2),
    "Hahn1":    (m_hahn1, j_hahn1),
    "Thurber":  (m_thurber, j_thurber),
    "Nelson":   (m_nelson, j_nelson),
    "Roszman1": (m_roszman1, j_roszman1),
    "ENSO":     (m_enso, j_enso),
}
'''
    return code


def main():
    print("=" * 60)
    print("  NIST StRD Nonlinear Regression — Dataset Downloader")
    print("=" * 60)

    all_data = {}
    for name, filename, difficulty, n_params in DATASETS:
        text = download_dat(filename)
        if text is None:
            print(f"  WARNING: Could not download {name}. Skipping.")
            continue

        parsed = parse_dat(text, n_params)
        parsed["difficulty"] = difficulty
        parsed["n_params"] = n_params
        all_data[name] = parsed

        n_obs = len(parsed["y"])
        n_cert = len(parsed["certified"])
        print(f"    → {name}: {n_obs} obs, {n_cert} certified params, "
              f"difficulty={difficulty}")

    print(f"\nParsed {len(all_data)}/{len(DATASETS)} datasets.")

    # Save raw parsed data as JSON for inspection
    json_path = "nist_parsed_data.json"
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved parsed data to {json_path}")

    # Generate the Python module
    code = generate_python(all_data)

    # Add data arrays
    code += "\n\n# ═══════════════════════════════════════════════════════════════════\n"
    code += "# DATASET DEFINITIONS (auto-generated from NIST .dat files)\n"
    code += "# ═══════════════════════════════════════════════════════════════════\n\n"
    code += "NIST_ALL_DATASETS = {\n"

    for name, filename, difficulty, n_params in DATASETS:
        if name not in all_data:
            continue
        d = all_data[name]
        model_fn, jac_fn = f"m_{name.lower()}", f"j_{name.lower()}"

        # Map to the actual function names in our registry
        registry_key = name

        code += f'    "{name}": {{\n'

        # x data
        x = d["x"]
        if isinstance(x[0], list):
            # Multi-column x (Nelson)
            code += f'        "x": np.array({json.dumps(x)}),\n'
        else:
            code += f'        "x": np.array({json.dumps(x)}),\n'

        # y data
        code += f'        "y": np.array({json.dumps(d["y"])}),\n'

        # Model and Jacobian
        mfn, jfn = MODEL_REGISTRY_NAMES[name]
        code += f'        "model": {mfn}, "jac": {jfn},\n'

        # Certified values
        code += f'        "theta_cert": np.array({json.dumps(d["certified"])}),\n'

        # Starting values (use "close" start = start2 for better convergence)
        if d["start2"]:
            code += f'        "theta_start": np.array({json.dumps(d["start2"])}),\n'
        elif d["start1"]:
            code += f'        "theta_start": np.array({json.dumps(d["start1"])}),\n'
        else:
            code += f'        "theta_start": np.array({json.dumps(d["certified"])}),\n'

        code += f'        "p": {n_params}, "difficulty": "{difficulty}",\n'
        code += f'    }},\n'

    code += "}\n"

    out_path = "nist_all_data.py"
    with open(out_path, "w") as f:
        f.write(code)
    print(f"Generated {out_path}")
    print("Done!")


# Map dataset names to function names in the generated module
MODEL_REGISTRY_NAMES = {
    "Misra1a":  ("m_misra1a", "j_misra1a"),
    "Misra1b":  ("m_misra1b", "j_misra1b"),
    "Misra1c":  ("m_misra1c", "j_misra1c"),
    "Misra1d":  ("m_misra1d", "j_misra1d"),
    "Chwirut1": ("m_chwirut", "j_chwirut"),
    "Chwirut2": ("m_chwirut", "j_chwirut"),
    "DanWood":  ("m_danwood", "j_danwood"),
    "BoxBOD":   ("m_boxbod", "j_boxbod"),
    "Lanczos1": ("m_lanczos", "j_lanczos"),
    "Lanczos2": ("m_lanczos", "j_lanczos"),
    "Lanczos3": ("m_lanczos", "j_lanczos"),
    "Gauss1":   ("m_gauss", "j_gauss"),
    "Gauss2":   ("m_gauss", "j_gauss"),
    "Gauss3":   ("m_gauss", "j_gauss"),
    "MGH09":    ("m_mgh09", "j_mgh09"),
    "MGH10":    ("m_mgh10", "j_mgh10"),
    "MGH17":    ("m_mgh17", "j_mgh17"),
    "Eckerle4": ("m_eckerle4", "j_eckerle4"),
    "Rat42":    ("m_rat42", "j_rat42"),
    "Rat43":    ("m_rat43", "j_rat43"),
    "Bennett5": ("m_bennett5", "j_bennett5"),
    "Kirby2":   ("m_kirby2", "j_kirby2"),
    "Hahn1":    ("m_hahn1", "j_hahn1"),
    "Thurber":  ("m_hahn1", "j_hahn1"),  # same model type
    "Nelson":   ("m_nelson", "j_nelson"),
    "Roszman1": ("m_roszman1", "j_roszman1"),
    "ENSO":     ("m_enso", "j_enso"),
}


if __name__ == "__main__":
    main()
