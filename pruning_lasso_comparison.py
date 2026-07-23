#!/usr/bin/env python3
"""
pruning_lasso_comparison.py — compares the SVD feature screener against
LassoCV and ElasticNetCV on the identical Experiment-3 synthetic design.
Linear design: n=200, p=15, 5 relevant
(coefficients U(2,6)), mild correlation, 20% outliers (SD 10),
seed default_rng(trial + 100), 10 trials.
Support for Lasso/EN = non-zero coefficients (5-fold CV regularization).
Output: experiment_results/pruning_lasso_comparison.csv
"""
import warnings, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LassoCV, ElasticNetCV
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
from lf_norm import lf_feature_pruning

P, REL, N, RATIO, STD = 15, 5, 200, 0.20, 10.0
rows = []
for trial in range(10):
    rng = np.random.default_rng(trial + 100)
    tt = np.zeros(P); tt[:REL] = rng.uniform(2.0, 6.0, REL)
    X = rng.standard_normal((N, P)); X[:, REL:REL+3] += 0.3 * X[:, :3]
    y = X @ tt + rng.normal(0, 0.5, N)
    oi = rng.choice(N, int(RATIO * N), replace=False)
    y[oi] += rng.normal(0, STD, len(oi))
    true_rel = set(range(REL))

    def score(sel, name):
        tp, fp, fn = len(sel & true_rel), len(sel - true_rel), len(true_rel - sel)
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        rows.append(dict(Trial=trial, Method=name, Selected=len(sel),
                         Precision=p, Recall=r, F1=2*p*r/max(p+r, 1e-10)))

    th0 = np.zeros(P)
    score(set(lf_feature_pruning(X, y, th0, f_target=0.5, tol=0.20,
              min_features=1, verbose=False)["selected_features"]),
          "Lf screener (f=0.5)")
    score(set(lf_feature_pruning(X, y, th0, f_target=2.0, tol=0.20,
              min_features=1, verbose=False)["selected_features"]),
          "OLS screener (f=2)")
    las = LassoCV(cv=5, random_state=0, max_iter=50000).fit(X, y)
    score(set(np.flatnonzero(np.abs(las.coef_) > 1e-8)), "LassoCV")
    en = ElasticNetCV(l1_ratio=[0.5, 0.9], cv=5, random_state=0,
                      max_iter=50000).fit(X, y)
    score(set(np.flatnonzero(np.abs(en.coef_) > 1e-8)), "ElasticNetCV")

d = pd.DataFrame(rows)
d.to_csv("experiment_results/pruning_lasso_comparison.csv", index=False)
print(d.groupby("Method")[["Selected", "Precision", "Recall", "F1"]]
      .mean().round(3).to_string())
