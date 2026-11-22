"""Summary of repeated nested CV: 3 random partitions x 5 folds = 15 outer folds."""
import os, sys, json, itertools
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS, CKPT

RUNS = {42: CKPT, 7: CKPT + "_seed7", 2026: CKPT + "_seed2026"}
pm, cons, sigs, sels = [], [], [], {}
for seed, d in RUNS.items():
    for f in range(5):
        c = json.load(open(f"{d}/fold{f}.json"))
        for r in c["per_method"]:
            pm.append({**r, "seed": seed})
        for r in c["models"]:
            if r["model"] == "KNN" and r["features"] == "consensus-15":
                cons.append({**r, "seed": seed})
        sigs.append(c["signature"])
        for m, g in c["selected"].items():
            sels.setdefault(m, []).append(set(g))
pm, cons = pd.DataFrame(pm), pd.DataFrame(cons)
K, R = 5, 3
corr = 1 / (K * R) + 0.25


def ci(v):
    v = np.asarray(v); m, s = v.mean(), v.std(ddof=1); h = stats.t.ppf(0.975, len(v) - 1) * s / np.sqrt(len(v))
    return m, s, m - h, m + h


rows = []
for m, d in pm.groupby("method"):
    d = d.sort_values(["seed", "fold"])
    mb, sb, lo, hi = ci(d.balanced_accuracy)
    jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(sels[m], 2)]
    rows.append({"method": m, "bal_mean": mb, "bal_sd": sb, "ci_low": lo, "ci_high": hi, "jaccard": np.mean(jac)})
cons = cons.sort_values(["seed", "fold"])
mb, sb, lo, hi = ci(cons.balanced_accuracy)
jac = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in itertools.combinations(sigs, 2)]
rows.append({"method": "CONSENSUS", "bal_mean": mb, "bal_sd": sb, "ci_low": lo, "ci_high": hi, "jaccard": np.mean(jac)})
t = pd.DataFrame(rows).sort_values("bal_mean", ascending=False).reset_index(drop=True)
t.to_csv(f"{RESULTS}/revision/repeated_cv_per_method.csv", index=False)
tests = {}
for m in ["Kruskal-Wallis", "LASSO (L1)", "Stability Selection", "ACO", "MIFS", "ANOVA", "Variance Threshold"]:
    a = pm[pm.method == m].sort_values(["seed", "fold"]).balanced_accuracy.values
    d = a - cons.balanced_accuracy.values
    tt = d.mean() / np.sqrt(corr * d.var(ddof=1)); p = 2 * stats.t.sf(abs(tt), K * R - 1)
    tests[m] = {"mean_difference": float(d.mean()), "t": float(tt), "p": float(p)}
ps = np.array([v["p"] for v in tests.values()]); order = np.argsort(ps); run = 0; adj = np.empty(len(ps))
for rank, i in enumerate(order):
    run = max(run, min(1.0, ps[i] * (len(ps) - rank))); adj[i] = run
for (k, v), a in zip(tests.items(), adj):
    v["p_holm"] = float(a)
from collections import Counter
cnt = Counter(g for s in sigs for g in s)
out = {"n_outer_folds": len(sigs),
       "consensus": t[t.method == "CONSENSUS"].round(4).to_dict("records")[0],
       "consensus_rank_of_33_kNN": int(t.index[t.method == "CONSENSUS"][0]) + 1,
       "median_selector_bal": float(t[t.method != "CONSENSUS"].bal_mean.median()),
       "median_selector_jaccard": float(t[t.method != "CONSENSUS"].jaccard.median()),
       "top_method": t.iloc[0].method, "top_method_bal": float(t.iloc[0].bal_mean),
       "plpp r4_top_in_folds": None,
       "genes_in_all_15": sorted([g for g, n in cnt.items() if n == 15]),
       "genes_in_at_least_12": sorted([g for g, n in cnt.items() if n >= 12]),
       "plppr4_rank1_in_folds": int(sum(s[0] == "PLPPR4" for s in sigs)),
       "corrected_tests": tests,
       "key_methods": t[t.method.isin(["Kruskal-Wallis", "LASSO (L1)", "Stability Selection", "ACO", "MIFS"])].round(4).to_dict("records")}
out.pop("plpp r4_top_in_folds")
json.dump(out, open(f"{RESULTS}/revision/repeated_cv.json", "w"), indent=2)
print(json.dumps(out, indent=1))
print(t.head(10).round(3).to_string())
