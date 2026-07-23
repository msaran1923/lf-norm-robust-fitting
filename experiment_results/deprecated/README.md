# Deprecated result files
 
## benchmark_leverage_v1_prerelease.csv
Leverage-point results as reported in the originally submitted manuscript.
The Huber, Cauchy, and Barron values in this file were produced by a
pre-release revision of the scale-estimation code and are NOT reproducible
from the released implementation (the Lf_dual and OLS values reproduce
bit-exactly; neither uses the scale pipeline). Superseded by
experiment_results/benchmark_leverage.csv, which is produced by the released
run_benchmark.py under the protocol stated in the paper (two-phase scale;
scale-dependent baselines initialized at the OLS solution). Kept here for
provenance only; do not use for analysis or figures.
