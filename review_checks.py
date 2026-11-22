"""Checks run during the final manuscript review (all nested where it matters)."""
import os, sys, json, warnings, itertools
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, RESULTS, CKPT
from core import load_data, evaluate_predictions, RANDOM_STATE
from step import knn, inner_accuracy, redundancy_weights, consensus, PRIOR_MARKERS
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
REV = os.path.join(RESULTS, "review"); os.makedirs(REV, exist_ok=True)
X, y, b, _ = load_data(DATA, verbose=False)
Xv, yv = X.values, y.values
genes = np.array(X.columns); gi = {g: i for i, g in enumerate(genes)}
folds = list(StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE).split(Xv, yv))
CK = [json.load(open(f"{CKPT}/fold{f}.json")) for f in range(5)]

def fold_sel(f):
    return {m: np.array([gi[g] for g in gs]) for m, gs in CK[f]["selected"].items()}

def part1():
    out = {}
    # 1. overlap between methods' full-data selections
    S = pd.read_csv(f"{RESULTS}/selected_features_per_method.csv")
    sets = {m: set(S[m].dropna()) for m in S.columns}
    ov = {(a, c): len(sets[a] & sets[c]) for a, c in itertools.combinations(sets, 2)}
    out["pairwise_overlap_min"] = min(ov.values()); out["pairwise_overlap_median"] = float(np.median(list(ov.values())))
    out["pairs_with_zero_overlap"] = sum(v == 0 for v in ov.values()); out["n_pairs"] = len(ov)
    t1 = pd.read_csv(f"{RESULTS}/table1_method_benchmark.csv").set_index("method")
    top5 = list(t1.sort_values("balanced_accuracy", ascending=False).index[:5])
    out["top5_methods"] = top5
    out["top5_pairwise_overlap"] = {f"{a}|{c}": len(sets[a] & sets[c]) for a, c in itertools.combinations(top5, 2)}
    # 2. leaky (full-data selection, then CV) vs nested accuracy, all methods
    rows = []
    for m in sets:
        cols = [gi[g] for g in sets[m]]
        acc = np.mean([(knn().fit(Xv[tr][:, cols], yv[tr]).predict(Xv[te][:, cols]) == yv[te]).mean() for tr, te in folds])
        rows.append({"method": m, "leaky_acc": acc, "nested_acc": t1.loc[m, "accuracy"]})
    d = pd.DataFrame(rows); d["inflation"] = d.leaky_acc - d.nested_acc
    d.to_csv(f"{REV}/selection_bias_per_method.csv", index=False)
    out["inflation_median"] = float(d.inflation.median()); out["inflation_mean"] = float(d.inflation.mean())
    out["inflation_min"] = float(d.inflation.min()); out["inflation_max"] = float(d.inflation.max())
    out["n_methods_inflated"] = int((d.inflation > 0).sum())
    an = d.set_index("method").loc["ANOVA"]; out["anova_leaky_vs_nested"] = [float(an.leaky_acc), float(an.nested_acc)]
    # 3. nested null model: each fold's own top-5 on its own test set
    def f1_fold(cols_per_fold):
        return float(np.mean([evaluate_predictions(yv[te], knn().fit(Xv[tr][:, c], yv[tr]).predict(Xv[te][:, c]))["f1_macro"]
                              for (tr, te), c in zip(folds, cols_per_fold)]))
    obs = f1_fold([[gi[g] for g in CK[f]["signature"][:5]] for f in range(5)])
    rng = np.random.default_rng(RANDOM_STATE)
    null = np.array([f1_fold([rng.choice(len(genes), 5, replace=False)] * 5) for _ in range(500)])
    np.save(f"{REV}/null_nested.npy", null)
    out["null_nested"] = {"observed": obs, "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
                          "null_max": float(null.max()), "empirical_p": float(((null >= obs).sum() + 1) / (len(null) + 1))}
    json.dump(out, open(f"{REV}/part1.json", "w"), indent=2)
    print(json.dumps(out, indent=1))

def part2():
    out = {}
    # 4. nested single-gene comparison: fold-specific consensus top-30 vs prior markers
    rows = []
    for f, (tr, te) in enumerate(folds):
        sel = fold_sel(f)
        perf = {m: inner_accuracy(Xv[tr], yv[tr], c, RANDOM_STATE + 1000 * (f + 1)) for m, c in sel.items()}
        w, _ = redundancy_weights(sel)
        top30 = list(consensus(sel, w, perf, genes).index[:30])
        for grp, gl in [("proposed", top30), ("prior", [g for g in PRIOR_MARKERS if g in gi])]:
            for g in gl:
                j = gi[g]
                m = RandomForestClassifier(200, random_state=RANDOM_STATE, n_jobs=-1).fit(Xv[tr][:, [j]], yv[tr])
                rows.append({"fold": f, "group": grp, "gene": g,
                             "f1": evaluate_predictions(yv[te], m.predict(Xv[te][:, [j]]))["f1_macro"]})
    d = pd.DataFrame(rows); d.to_csv(f"{REV}/single_gene_nested.csv", index=False)
    p, q = d[d.group == "proposed"].f1, d[d.group == "prior"].f1
    out["single_gene_nested"] = {"proposed_mean": float(p.mean()), "prior_mean": float(q.mean()),
                                 "n_proposed_gene_folds": int(len(p)), "n_prior_gene_folds": int(len(q)),
                                 "mannwhitney_p": float(stats.mannwhitneyu(p, q, alternative="greater").pvalue),
                                 "per_fold_means": d.groupby(["fold", "group"]).f1.mean().unstack().round(3).to_dict()}
    json.dump(out, open(f"{REV}/part2.json", "w"), indent=2)
    print(json.dumps(out, indent=1))

def part3():
    out = {}
    from extra_analyses import cell_scores, residualise_in_fold
    top60_all = set(np.load(f"{CKPT}/top_genes.npy", allow_pickle=True))
    excl = set().union(top60_all, *[set(c["signature"]) for c in CK])
    S, used = cell_scores(X, exclude=excl)
    C_all, C_ker = S.values, S[["keratinocyte"]].values
    def run(task_mask, labels):
        res = {k: [] for k in ["signature", "adj_keratinocyte", "adj_all", "keratinocyte_only", "all_scores_only"]}
        for f, (tr, te) in enumerate(folds):
            tr = tr[task_mask[tr]]; te = te[task_mask[te]]
            idx = [gi[g] for g in CK[f]["signature"]]
            G = Xv[:, idx]
            def ev(Ftr, Fte):
                m = knn().fit(Ftr, yv[tr]); return evaluate_predictions(yv[te], m.predict(Fte), labels=labels)["balanced_accuracy"]
            res["signature"].append(ev(G[tr], G[te]))
            for key, C in [("adj_keratinocyte", C_ker), ("adj_all", C_all)]:
                full = np.r_[tr, te]; Gtr, Gte = residualise_in_fold(G, C, tr, te); res[key].append(ev(Gtr, Gte))
            res["keratinocyte_only"].append(ev(C_ker[tr], C_ker[te]))
            res["all_scores_only"].append(ev(C_all[tr], C_all[te]))
        return {k: round(float(np.mean(v)), 3) for k, v in res.items()}
    out["three_class"] = run(np.ones(len(yv), bool), (1, 2, 3))
    out["primary_vs_metastatic"] = run(np.isin(yv, [2, 3]), (2, 3))
    out["markers_used"] = used
    # 6. ablation of the consensus design
    variants = {"weighted+redundancy": (True, True), "weighted only": (True, False),
                "redundancy only": (False, True), "plain vote count": (False, False)}
    abl = {}
    for name, (wt, red) in variants.items():
        sigs, accs, bals = [], [], []
        for f, (tr, te) in enumerate(folds):
            sel = fold_sel(f)
            perf = ({m: inner_accuracy(Xv[tr], yv[tr], c, RANDOM_STATE + 1000 * (f + 1)) for m, c in sel.items()}
                    if wt else {m: 1.0 for m in sel})
            w = redundancy_weights(sel)[0] if red else {m: 1.0 for m in sel}
            sc = consensus(sel, w, perf, genes)
            # deterministic tie-break by ANOVA F on training data for vote counts
            top = list(sc.index[:15]); sigs.append(top)
            idx = [gi[g] for g in top]
            r = evaluate_predictions(yv[te], knn().fit(Xv[tr][:, idx], yv[tr]).predict(Xv[te][:, idx]))
            accs.append(r["accuracy"]); bals.append(r["balanced_accuracy"])
        core = set(sigs[0]).intersection(*map(set, sigs[1:]))
        mean_j = np.mean([len(set(a) & set(c)) / len(set(a) | set(c)) for a, c in itertools.combinations(sigs, 2)])
        abl[name] = {"accuracy": round(float(np.mean(accs)), 3), "balanced_accuracy": round(float(np.mean(bals)), 3),
                     "genes_in_all_5_folds": len(core), "mean_pairwise_jaccard": round(float(mean_j), 3)}
    out["ablation"] = abl
    json.dump(out, open(f"{REV}/part3.json", "w"), indent=2, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "markers_used"}, indent=1))

if __name__ == "__main__":
    globals()[sys.argv[1]]()
