"""
Reconstruct how Data.csv was built from the GEO series matrices.

Quantile normalisation preserves the order of genes within each sample, so for
every analysed sample the Spearman correlation between Data.csv and a candidate
gene-level matrix built from the GEO series values measures how well that
candidate reproduces the original preprocessing. Tested on 200 random genes
(150 with several probes, which discriminate between collapsing rules).
"""
import json, io, sys, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS, DATA
OUT = f"{RESULTS}/preprocessing"
TR = "/root/.claude/projects/-home-claude/c7b3b980-392c-58e4-b56b-c608df3699e5/tool-results/"
FILES = ["mcp-remote-devices-Claude_Browser__javascript_tool-1795358880493-89anz8.txt",
         "mcp-remote-devices-Claude_Browser__javascript_tool-1795358917693-dp3lbw.txt",
         "mcp-remote-devices-Claude_Browser__javascript_tool-1795358925541-z5e1yf.txt"]


def load_series():
    ser = {}
    for f in FILES:
        p = TR + f if os.path.exists(TR + f) else f"{OUT}/{f}"
        s = open(p).read(); v, _ = json.JSONDecoder().raw_decode(s); t = v[0]["text"]
        if t.lstrip().startswith('"'): t, _ = json.JSONDecoder().raw_decode(t.lstrip())
        for block in t.split("##")[1:]:
            name, body = block.split("\n", 1)
            ser[name] = pd.read_csv(io.StringIO(body), sep="\t", index_col=0)
    return ser


def collapse(M, probes, rule, log):
    V = np.log2(M.clip(lower=1)) if log else M
    out = {}
    for g, ps in probes.groupby("symbol").probe:
        ps = [p for p in ps if p in V.index]
        if not ps: continue
        sub = V.loc[ps]
        if rule == "max_mean": out[g] = sub.loc[sub.mean(1).idxmax()]
        elif rule == "max_iqr": out[g] = sub.loc[(sub.quantile(.75, 1) - sub.quantile(.25, 1)).idxmax()]
        elif rule == "mean": out[g] = sub.mean()
        elif rule == "median": out[g] = sub.median()
        elif rule == "max_per_sample": out[g] = sub.max()
        elif rule == "first": out[g] = sub.loc[sorted(ps)[0]]
    return pd.DataFrame(out)  # samples x genes


def main():
    probes = pd.read_csv(f"{OUT}/test_probes.tsv", sep="\t")
    D = pd.read_csv(DATA, index_col=0).drop(index="Class").astype(float)
    ser = load_series()
    rows = []
    for name, M in ser.items():
        samples = [c for c in M.columns if c in D.columns]
        M = M[samples].astype(float)
        is_log = M.max().max() < 30
        for rule in ["max_mean", "max_iqr", "mean", "median", "max_per_sample", "first"]:
            for log in ([False] if is_log else [True, False]):
                C = collapse(M, probes, rule, log)
                g = [x for x in C.columns if x in D.index]
                rho = [spearmanr(C.loc[s, g], D.loc[g, s]).correlation for s in samples]
                rows.append({"series": name, "n_samples": len(samples), "n_genes": len(g), "rule": rule,
                             "averaged_on": "log2" if (log or is_log) else "linear",
                             "median_rho": float(np.median(rho)), "min_rho": float(np.min(rho))})
    r = pd.DataFrame(rows).sort_values(["series", "median_rho"], ascending=[True, False])
    r.to_csv(f"{OUT}/collapse_rule_test.csv", index=False)
    print(r.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
