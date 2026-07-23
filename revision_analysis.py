#!/usr/bin/env python3
"""
revision_analysis.py — statistical analyses for the benchmark results.

Computes, from per-trial results:
  1. Aggregate summary per method: mean, median, geometric mean, IQR of RE,
     catastrophic (%RE>1) and severe (%RE>10) failure rates, median runtime.
     -> LaTeX rows for the aggregate summary table in the paper.
  2. 95% bootstrap confidence intervals for the per-method mean RE.
  3. Paired two-sided Wilcoxon signed-rank tests: Lf_dual vs each baseline,
     paired by (dataset, contamination, seed), with Holm-Bonferroni correction
     and rank-biserial effect sizes.
  4. Per-dataset catastrophic-failure counts (for Appendix A).

INPUT: per_trial_results.csv with columns:
    dataset, contamination, seed, method, re[, runtime_ms]

INPUT: the released per-trial file benchmark_full.csv works directly
(columns Dataset, Difficulty, Method, Is_Ablation, Outlier_Frac, Trial,
Rel_Error, Runtime_ms). Rows with Is_Ablation == True (Lf_full) are excluded
from the summary and tests; clean-data rows (Outlier_Frac == 0) likewise.

Usage:  python revision_analysis.py benchmark_full.csv

Usage:  python revision_analysis.py results/per_trial_results.csv
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

PROPOSED = "Lf_dual"
N_BOOT = 10_000
RNG = np.random.default_rng(0)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, g in df.groupby("method"):
        re = g["re"].to_numpy(float)
        n_inf = np.isinf(re).sum() + np.isnan(re).sum()
        # error statistics on finite values; rates over the full trial count
        fin = re[np.isfinite(re)]
        q25, q75 = np.percentile(fin, [25, 75])
        # geometric mean: floor tiny/zero REs to avoid log(0); report floor used
        gm = float(np.exp(np.mean(np.log(np.maximum(fin, 1e-15)))))
        boot = RNG.choice(fin, size=(N_BOOT, fin.size), replace=True).mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append(dict(
            method=m, n=re.size, n_crash=int(n_inf),
            mean=fin.mean(), mean_ci_lo=lo, mean_ci_hi=hi,
            median=np.median(fin), geomean=gm, iqr=q75 - q25, max=fin.max(),
            cat_pct=100 * (fin > 1.0).sum() / re.size,
            sev_pct=100 * (fin > 10.0).sum() / re.size,
            runtime_ms=(g["runtime_ms"].median()
                        if "runtime_ms" in g else np.nan),
        ))
    return (pd.DataFrame(rows)
            .sort_values("cat_pct")
            .reset_index(drop=True))


def wilcoxon_vs_proposed(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "contamination", "seed"]
    wide = df.pivot_table(index=keys, columns="method", values="re")
    out = []
    baselines = [c for c in wide.columns if c != PROPOSED]
    for b in baselines:
        pair = wide[[PROPOSED, b]].dropna()
        x, y = pair[PROPOSED].to_numpy(), pair[b].to_numpy()
        # crashed solves (inf) -> rank-preserving sentinel above all finite REs
        fin_max = np.nanmax(np.concatenate([x[np.isfinite(x)],
                                            y[np.isfinite(y)], [1.0]]))
        big = 10.0 * fin_max
        x = np.where(np.isfinite(x), x, big)
        y = np.where(np.isfinite(y), y, big)
        d = x - y
        nz = d != 0
        res = stats.wilcoxon(x[nz], y[nz], alternative="two-sided",
                             zero_method="wilcox", method="approx")
        # rank-biserial effect size r = 1 - 2*W_min / (n(n+1)/2); sign from medians
        n = nz.sum()
        total = n * (n + 1) / 2
        r_rb = 1 - 2 * res.statistic / total
        sign = np.sign(np.median(y[nz]) - np.median(x[nz]))  # + => proposed better
        out.append(dict(baseline=b, n_pairs=int(n), W=float(res.statistic),
                        p_raw=float(res.pvalue),
                        effect_rank_biserial=float(sign * abs(r_rb)),
                        median_diff=float(np.median(d))))
    res_df = pd.DataFrame(out).sort_values("p_raw").reset_index(drop=True)
    # Holm-Bonferroni correction
    m = len(res_df)
    holm = np.minimum.accumulate(
        (res_df["p_raw"] * (m - np.arange(m)))[::-1])[::-1]  # enforce monotone
    res_df["p_holm"] = np.minimum(1.0, np.maximum.accumulate(
        res_df["p_raw"] * (m - np.arange(m))))
    return res_df.drop(columns=[]) if holm is not None else res_df


def failure_counts(df: pd.DataFrame) -> pd.DataFrame:
    f = (df.assign(fail=df["re"] > 1.0)
           .groupby(["dataset", "method"])["fail"].sum()
           .unstack("method").astype(int))
    return f


def latex_summary_rows(s: pd.DataFrame) -> str:
    def fmt(v, nd=3):
        return "--" if pd.isna(v) else f"{v:,.{nd}f}"
    lines = []
    for _, r in s.iterrows():
        name = r"\texttt{Lf\_dual}" if r.method == PROPOSED else r.method
        lines.append(
            f"{name} & {fmt(r['mean'])} & {fmt(r['median'])} & {fmt(r['iqr'])}"
            f" & {r.cat_pct:.1f} & {r.sev_pct:.1f} & {fmt(r.runtime_ms, 0)} \\\\")
    return "\n".join(lines)


COLMAP = {"Dataset": "dataset", "Outlier_Frac": "contamination",
          "Trial": "seed", "Method": "method", "Rel_Error": "re",
          "Runtime_ms": "runtime_ms"}

# Policy for solver crashes (NaN RE): "fail" counts them as catastrophic
# (RE := inf), "drop" excludes them. Use the SAME policy the paper used for
# its 4.1% / 11-37.5% failure rates, and state it in Section 4.1.
NAN_POLICY = "drop"  # matches the published failure rates (crashes excluded)


def main(path: str) -> None:
    df = pd.read_csv(path).rename(columns=COLMAP)
    if "Is_Ablation" in df.columns:
        n_ab = int(df["Is_Ablation"].sum())
        if n_ab:
            print(f"note: excluding {n_ab} ablation-variant rows (Lf_full)")
            df = df[~df["Is_Ablation"]].copy()
    need = {"dataset", "contamination", "seed", "method", "re"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"missing columns: {missing}")

    # contaminated trials only (the paper's 1,620-trial aggregate excludes 0%)
    n0 = (df["contamination"] == 0).sum()
    if n0:
        print(f"note: excluding {n0} clean-data rows (contamination == 0)")
        df = df[df["contamination"] > 0].copy()

    n_nan = df["re"].isna().sum()
    if n_nan:
        if NAN_POLICY == "fail":
            print(f"note: {n_nan} NaN REs (solver crashes) counted as "
                  f"catastrophic failures (RE := inf)")
            df.loc[df["re"].isna(), "re"] = np.inf
        else:
            print(f"note: {n_nan} crashed solves (NaN RE) excluded from error "
                  f"statistics but kept in the trial count (published "
                  f"convention; favors the affected baselines)")

    s = summarize(df)
    print("\n=== Aggregate summary per method ===")
    print(s.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print("\n=== LaTeX rows for tab:summary ===")
    print(latex_summary_rows(s))

    w = wilcoxon_vs_proposed(df)
    print("\n=== Paired Wilcoxon signed-rank: Lf_dual vs baselines "
          "(Holm-corrected) ===")
    print(w.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    fc = failure_counts(df)
    fc.to_csv("per_dataset_failure_counts.csv")
    print("\nPer-dataset catastrophic-failure counts -> "
          "per_dataset_failure_counts.csv")
    s.to_csv("aggregate_summary.csv", index=False)
    w.to_csv("wilcoxon_tests.csv", index=False)
    print("Saved: aggregate_summary.csv, wilcoxon_tests.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "per_trial_results.csv")
