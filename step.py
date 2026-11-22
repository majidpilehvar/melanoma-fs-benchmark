"""
Resumable driver. Each invocation does one unit of work and checkpoints it,
so a long study can be run in short slices:

    python3 step.py fold 0 ... fold 4     leak-free outer CV, one fold each
    python3 step.py discovery             consensus ranking on the full data
    python3 step.py loso 0 ... loso 3     leave-one-study-out
    python3 step.py de                    differential expression + FDR
    python3 step.py singlegene            proposed vs prior markers
    python3 step.py null                  random-signature null model
    python3 step.py collect               assemble the final tables
"""
import os, sys, json, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from core import load_data, evaluate_predictions, RANDOM_STATE
from fs_methods import METHODS, FAMILY
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import stats

from config import DATA, RESULTS as OUT, CKPT
K, N_FOLDS, JACCARD_REDUNDANT = 15, 5, 0.80

PRIOR_MARKERS = ["PRAME", "PMEL", "TYRP1", "TYR", "MLANA", "MITF", "SOX10",
                 "S100B", "DCT", "MAGEA1", "MAGEA3", "CDKN2A", "BRAF", "NRAS"]


def knn():
    return KNeighborsClassifier(n_neighbors=5)


def mlp():
    """Keras network when TensorFlow is installed, else the sklearn equivalent."""
    try:
        from fs_methods_reference import KerasMLP
        if KerasMLP is not None:
            return KerasMLP(random_state=RANDOM_STATE)
    except ImportError:
        pass
    return Pipeline([("sc", StandardScaler()),
                     ("nn", MLPClassifier(hidden_layer_sizes=(150, 50, 20),
                                          max_iter=800, random_state=RANDOM_STATE,
                                          early_stopping=True))])


def inner_accuracy(Xtr, ytr, cols, seed):
    """Subset accuracy estimated inside the training fold only, so consensus
    weights never see held-out data."""
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    return float(np.mean([
        (knn().fit(Xtr[a][:, cols], ytr[a]).predict(Xtr[b][:, cols]) == ytr[b]).mean()
        for a, b in cv.split(Xtr[:, cols], ytr)]))


def redundancy_weights(sel_sets, thr=JACCARD_REDUNDANT):
    """Methods whose outputs overlap above `thr` share a single vote."""
    names = list(sel_sets)
    parent = {n: n for n in names}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A, B = set(sel_sets[a]), set(sel_sets[b])
            if len(A & B) / max(len(A | B), 1) >= thr:
                parent[find(a)] = find(b)
    clusters = {}
    for n in names:
        clusters.setdefault(find(n), []).append(n)
    w = {n: 1.0 / len(m) for m in clusters.values() for n in m}
    return w, clusters


def consensus(sel_sets, weights, perf, genes):
    score = pd.Series(0.0, index=genes)
    for m, cols in sel_sets.items():
        score.iloc[list(cols)] += weights.get(m, 1.0) * perf[m]
    return score.sort_values(ascending=False)


def select_all(Xtr, ytr, seed_offset):
    """Run every method on this training set. Returns sets + inner weights."""
    sel, perf, times = {}, {}, {}
    for name, fn in METHODS.items():
        t = time.time()
        rng = np.random.default_rng(RANDOM_STATE + seed_offset)
        try:
            cols = np.unique(np.asarray(fn(Xtr, ytr, K, rng), dtype=int))[:K]
        except Exception as e:
            print(f"  {name}: FAILED {e}", flush=True)
            continue
        sel[name] = cols
        perf[name] = inner_accuracy(Xtr, ytr, cols, RANDOM_STATE + seed_offset)
        times[name] = time.time() - t
    return sel, perf, times


def load():
    X, y, batch, report = load_data(DATA, verbose=False)
    return X, y, batch, report, np.array(X.columns)


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        prev = min(prev, p[i] * n / (n - rank))
        q[i] = prev
    return q


# ------------------------------------------------------------------ steps
SPLIT_SEED = int(os.environ.get("MELANOMA_CV_SEED", RANDOM_STATE))
GROUPED = os.environ.get("MELANOMA_GROUPED") == "1"  # patient-grouped folds from results/revision/patient_groups.csv
CKPT_RUN = CKPT if SPLIT_SEED == RANDOM_STATE else CKPT + f"_seed{SPLIT_SEED}"
if GROUPED:
    CKPT_RUN = CKPT + "_grouped"
os.makedirs(CKPT_RUN, exist_ok=True)


def step_fold(fold):
    X, y, batch, report, genes = load()
    json.dump(report, open(f"{OUT}/cleaning_report.json", "w"),
              indent=2, default=str)
    Xv, yv = X.values, y.values
    if GROUPED:
        from sklearn.model_selection import StratifiedGroupKFold
        groups = pd.read_csv(f"{OUT}/revision/patient_groups.csv", index_col=0).loc[X.index, "patient_group"].values
        tr, te = list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SPLIT_SEED).split(Xv, yv, groups))[fold]
        assert not set(groups[tr]) & set(groups[te]), "patient in both train and test"
    else:
        cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SPLIT_SEED)
        tr, te = list(cv.split(Xv, yv))[fold]
    Xtr, Xte, ytr, yte = Xv[tr], Xv[te], yv[tr], yv[te]

    sel, perf, times = select_all(Xtr, ytr, 1000 * (fold + 1))
    rows = []
    for name, cols in sel.items():
        m = knn().fit(Xtr[:, cols], ytr)
        r = {"method": name, "family": FAMILY.get(name, ""), "fold": fold,
             "runtime_s": times[name]}
        r.update(evaluate_predictions(yte, m.predict(Xte[:, cols]),
                                      m.predict_proba(Xte[:, cols])))
        rows.append(r)

    w, _ = redundancy_weights(sel)
    sc = consensus(sel, w, perf, genes)
    top = list(sc.index[:K])
    cidx = [list(genes).index(g) for g in top]
    mrows = []
    for label, model in [("KNN", knn()), ("MLP", mlp())]:
        for feats, cols in [("consensus-15", cidx), ("all-genes", slice(None))]:
            model.fit(Xtr[:, cols], ytr)
            pred = model.predict(Xte[:, cols])
            proba = model.predict_proba(Xte[:, cols])
            r = {"model": label, "features": feats, "fold": fold}
            r.update(evaluate_predictions(yte, pred, proba))
            mrows.append(r)

    json.dump({"per_method": rows, "models": mrows, "signature": top,
               "selected": {k: genes[v].tolist() for k, v in sel.items()}},
              open(f"{CKPT_RUN}/fold{fold}.json", "w"), indent=2, default=str)
    print(f"seed {SPLIT_SEED} fold {fold} done | consensus top5 {top[:5]}", flush=True)


def step_discovery():
    X, y, batch, report, genes = load()
    sel, perf, times = select_all(X.values, y.values, 0)
    w, clusters = redundancy_weights(sel)
    sc = consensus(sel, w, perf, genes)
    sc.head(300).rename("consensus_score").to_csv(f"{OUT}/table2_gene_ranking.csv")
    pd.DataFrame({m: genes[c] for m, c in sel.items()}).to_csv(
        f"{OUT}/selected_features_per_method.csv", index=False)
    json.dump({"redundant_clusters": {k: v for k, v in clusters.items() if len(v) > 1},
               "weights": w, "inner_accuracy": perf},
              open(f"{OUT}/method_redundancy.json", "w"), indent=2, default=str)
    np.save(f"{CKPT}/top_genes.npy", np.array(sc.index[:60]))
    print("discovery done | top20:", list(sc.index[:20]), flush=True)


def step_loso(i):
    X, y, batch, report, genes = load()
    Xv, yv, bv = X.values, y.values, batch.values
    held = sorted(np.unique(bv))[i]
    tr = np.where(bv != held)[0]
    te = np.where(bv == held)[0]
    out = {"held_out_study": int(held), "n_test": int(len(te)),
           "classes_in_test": sorted(map(int, np.unique(yv[te]))),
           "classes_in_train": sorted(map(int, np.unique(yv[tr])))}
    if len(np.unique(yv[tr])) < 3:
        out["note"] = "training set lacks a class; comparison not meaningful"
        json.dump(out, open(f"{CKPT}/loso{i}.json", "w"), indent=2)
        print(out, flush=True)
        return
    sel, perf, _ = select_all(Xv[tr], yv[tr], 500 + i)
    w, _ = redundancy_weights(sel)
    sc = consensus(sel, w, perf, genes)
    top = list(sc.index[:K])
    cidx = [list(genes).index(g) for g in top]
    out["signature"] = top
    out["results"] = []
    for label, model in [("KNN", knn()), ("MLP", mlp())]:
        model.fit(Xv[tr][:, cidx], yv[tr])
        r = {"model": label}
        r.update(evaluate_predictions(yv[te], model.predict(Xv[te][:, cidx]),
                                      labels=sorted(np.unique(yv))))
        out["results"].append(r)
    json.dump(out, open(f"{CKPT}/loso{i}.json", "w"), indent=2, default=str)
    print(f"loso {held}: " + " | ".join(
        f"{r['model']} bal_acc={r['balanced_accuracy']:.3f} f1={r['f1_macro']:.3f}"
        for r in out["results"]), flush=True)


def step_de():
    X, y, _, _, genes = load()
    Xv, yv = X.values, y.values
    h, a, m = Xv[yv == 1], Xv[yv == 2], Xv[yv == 3]
    _, p1 = stats.ttest_ind(a, h, axis=0, equal_var=False)
    _, p2 = stats.ttest_ind(m, h, axis=0, equal_var=False)
    _, p3 = stats.ttest_ind(m, a, axis=0, equal_var=False)
    de = pd.DataFrame({
        "gene": genes,
        "log2FC_atrisk_vs_healthy": a.mean(0) - h.mean(0),
        "p_atrisk_vs_healthy": p1, "fdr_atrisk_vs_healthy": bh_fdr(p1),
        "log2FC_meta_vs_healthy": m.mean(0) - h.mean(0),
        "p_meta_vs_healthy": p2, "fdr_meta_vs_healthy": bh_fdr(p2),
        "log2FC_meta_vs_atrisk": m.mean(0) - a.mean(0),
        "p_meta_vs_atrisk": p3, "fdr_meta_vs_atrisk": bh_fdr(p3)})
    de.to_csv(f"{OUT}/differential_expression_all_genes.csv", index=False)
    print(f"DE done | FDR<0.05 meta-vs-healthy: "
          f"{int((de.fdr_meta_vs_healthy < .05).sum())}/{len(de)}", flush=True)


def step_singlegene():
    X, y, _, _, genes = load()
    Xv, yv = X.values, y.values
    de = pd.read_csv(f"{OUT}/differential_expression_all_genes.csv").set_index("gene")
    top = list(np.load(f"{CKPT}/top_genes.npy", allow_pickle=True))[:30]
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for group, gl in [("proposed", top), ("prior_literature", PRIOR_MARKERS)]:
        for g in gl:
            if g not in de.index:
                continue
            j = list(genes).index(g)
            accs, f1s = [], []
            for tr, te in cv.split(Xv, yv):
                mdl = RandomForestClassifier(200, random_state=RANDOM_STATE,
                                             n_jobs=-1).fit(Xv[tr][:, [j]], yv[tr])
                r = evaluate_predictions(yv[te], mdl.predict(Xv[te][:, [j]]))
                accs.append(r["accuracy"]); f1s.append(r["f1_macro"])
            rows.append({"gene": g, "group": group,
                         "rf_accuracy": np.mean(accs), "rf_f1": np.mean(f1s),
                         "log2FC_meta_vs_healthy": de.loc[g, "log2FC_meta_vs_healthy"],
                         "fdr_meta_vs_healthy": de.loc[g, "fdr_meta_vs_healthy"]})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/table5_single_gene.csv", index=False)
    pa, pb = df[df.group == "proposed"].rf_f1, df[df.group == "prior_literature"].rf_f1
    u = stats.mannwhitneyu(pa, pb, alternative="greater")
    json.dump({"proposed_mean_f1": float(pa.mean()), "prior_mean_f1": float(pb.mean()),
               "n_proposed": int(len(pa)), "n_prior": int(len(pb)),
               "mannwhitney_p": float(u.pvalue)},
              open(f"{OUT}/single_gene_test.json", "w"), indent=2)
    print(f"single-gene | proposed {pa.mean():.3f} vs prior {pb.mean():.3f}, "
          f"p={u.pvalue:.4g}", flush=True)


def step_null(n_perm=500):
    X, y, _, _, genes = load()
    Xv, yv = X.values, y.values
    folds = [json.load(open(f"{CKPT}/fold{f}.json"))["signature"]
             for f in range(N_FOLDS) if os.path.exists(f"{CKPT}/fold{f}.json")]
    core = [g for g in folds[0] if all(g in s for s in folds)]
    sig = core[:5] if len(core) >= 5 else folds[0][:5]
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    def f1_of(cols):
        return float(np.mean([
            evaluate_predictions(
                yv[te], knn().fit(Xv[tr][:, cols], yv[tr]).predict(Xv[te][:, cols])
            )["f1_macro"] for tr, te in cv.split(Xv, yv)]))

    idx = [list(genes).index(g) for g in sig]
    obs = f1_of(idx)
    rng = np.random.default_rng(RANDOM_STATE)
    null = np.array([f1_of(rng.choice(len(genes), 5, replace=False))
                     for _ in range(n_perm)])
    res = {"signature": sig, "observed_macro_f1": obs,
           "null_mean": float(null.mean()),
           "null_p95": float(np.percentile(null, 95)),
           "null_max": float(null.max()),
           "empirical_p": float((null >= obs).mean()), "n_permutations": n_perm,
           "stable_core_size": len(core), "stable_core": core}
    json.dump(res, open(f"{OUT}/table6_null_model.json", "w"), indent=2)
    print(json.dumps(res, indent=2), flush=True)


def step_collect():
    pm, mods, sigs = [], [], []
    for f in range(N_FOLDS):
        p = f"{CKPT}/fold{f}.json"
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        pm += d["per_method"]; mods += d["models"]; sigs.append(d["signature"])
    pm = pd.DataFrame(pm)
    t1 = (pm.groupby(["method", "family"], as_index=False)
            .mean(numeric_only=True).drop(columns=["fold"])
            .sort_values("balanced_accuracy", ascending=False))
    t1.to_csv(f"{OUT}/table1_method_benchmark.csv", index=False)
    t3 = (pd.DataFrame(mods).groupby(["model", "features"], as_index=False)
            .mean(numeric_only=True).drop(columns=["fold"]))
    t3.to_csv(f"{OUT}/table3_model_comparison.csv", index=False)

    losos = []
    for i in range(4):
        p = f"{CKPT}/loso{i}.json"
        if os.path.exists(p):
            d = json.load(open(p))
            for r in d.get("results", [{}]):
                losos.append({**{k: v for k, v in d.items()
                                 if k not in ("results", "signature")}, **r})
    if losos:
        pd.DataFrame(losos).to_csv(f"{OUT}/table4_leave_one_study_out.csv", index=False)

    core = [g for g in sigs[0] if all(g in s for s in sigs)] if sigs else []
    json.dump({"per_fold": sigs, "in_all_folds": core,
               "n_in_all_folds": len(core)},
              open(f"{OUT}/signature_stability.json", "w"), indent=2)
    print(f"collected | {len(t1)} methods, signature core {core}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    arg = int(sys.argv[2]) if len(sys.argv) > 2 else None
    {"fold": lambda: step_fold(arg), "discovery": step_discovery,
     "loso": lambda: step_loso(arg), "de": step_de,
     "singlegene": step_singlegene, "null": step_null,
     "collect": step_collect}[cmd]()
