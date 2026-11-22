"""
Robustness of the main results to the three implementation substitutions.

Reference configuration = the 31 unchanged selectors (their per-fold
selections are reused from results/ckpt, so they are identical) plus
  * Boruta  -> full Boruta algorithm (fs_methods_faithful.boruta_full)
  * MIC     -> exact MIC for a categorical target, added as method 33
  * MLP     -> Keras-equivalent dropout network (DropoutMLP)

    python3 sensitivity.py fold 0..4     consensus + kNN + MLP on consensus genes
    python3 sensitivity.py mlpall 0..4   dropout network on all genes
    python3 sensitivity.py discovery     full-data ranking, 33 methods
    python3 sensitivity.py collect       comparison tables
"""
import os, sys, json, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, RESULTS, CKPT
from core import load_data, evaluate_predictions, RANDOM_STATE
from step import knn, inner_accuracy, redundancy_weights, consensus, K, N_FOLDS
from fs_methods_faithful import boruta_full, mic_exact, DropoutMLP
from sklearn.model_selection import StratifiedKFold

SENS = os.path.join(RESULTS, "sensitivity")
os.makedirs(SENS, exist_ok=True)
NEW = {"Boruta": boruta_full, "MIC": mic_exact}


def _load():
    X, y, b, _ = load_data(DATA, verbose=False)
    return X, y, b, np.array(X.columns)


def _split(Xv, yv, fold):
    return list(StratifiedKFold(N_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE).split(Xv, yv))[fold]


def step_fold(fold):
    X, y, b, genes = _load()
    Xv, yv = X.values, y.values
    gi = {g: i for i, g in enumerate(genes)}
    tr, te = _split(Xv, yv, fold)
    Xtr, Xte, ytr, yte = Xv[tr], Xv[te], yv[tr], yv[te]
    old = json.load(open(f"{CKPT}/fold{fold}.json"))
    sel = {m: np.array([gi[g] for g in gs]) for m, gs in old["selected"].items()
           if m != "Boruta"}
    rows = []
    for name, fn in NEW.items():
        t = time.time()
        cols = np.asarray(fn(Xtr, ytr, K, np.random.default_rng(RANDOM_STATE + 1000 * (fold + 1))))
        sel[name] = cols
        m = knn().fit(Xtr[:, cols], ytr)
        r = {"method": name + " (reference)", "fold": fold, "runtime_s": time.time() - t}
        r.update(evaluate_predictions(yte, m.predict(Xte[:, cols]), m.predict_proba(Xte[:, cols])))
        rows.append(r)
    perf = {m: inner_accuracy(Xtr, ytr, c, RANDOM_STATE + 1000 * (fold + 1)) for m, c in sel.items()}
    w, clusters = redundancy_weights(sel)
    top = list(consensus(sel, w, perf, genes).index[:K])
    idx = [gi[g] for g in top]
    models = []
    for label, model in [("KNN", knn()), ("DropoutMLP", DropoutMLP(random_state=RANDOM_STATE))]:
        model.fit(Xtr[:, idx], ytr)
        r = {"model": label, "features": "consensus-15", "fold": fold}
        r.update(evaluate_predictions(yte, model.predict(Xte[:, idx]), model.predict_proba(Xte[:, idx])))
        models.append(r)
    json.dump({"per_method": rows, "models": models, "signature": top,
               "original_signature": old["signature"],
               "selected_new": {k: genes[v].tolist() for k, v in sel.items() if k in NEW},
               "redundant_clusters": {k: v for k, v in clusters.items() if len(v) > 1}},
              open(f"{SENS}/fold{fold}.json", "w"), indent=2, default=str)
    ov = len(set(top) & set(old["signature"]))
    print(f"fold {fold}: overlap with original signature {ov}/15 | "
          + " | ".join(f"{r['method']} acc={r['accuracy']:.3f}" for r in rows)
          + " | " + " | ".join(f"{r['model']} acc={r['accuracy']:.3f}" for r in models), flush=True)


def step_mlpall(fold):
    X, y, b, genes = _load()
    Xv, yv = X.values, y.values
    tr, te = _split(Xv, yv, fold)
    m = DropoutMLP(random_state=RANDOM_STATE).fit(Xv[tr], yv[tr])
    r = {"model": "DropoutMLP", "features": "all-genes", "fold": fold, "epochs": m.n_epochs_}
    r.update(evaluate_predictions(yv[te], m.predict(Xv[te]), m.predict_proba(Xv[te])))
    json.dump(r, open(f"{SENS}/mlpall{fold}.json", "w"), indent=2, default=str)
    print(f"mlpall fold {fold}: acc={r['accuracy']:.3f} f1={r['f1_macro']:.3f} epochs={m.n_epochs_}", flush=True)


def step_discovery():
    X, y, b, genes = _load()
    Xv, yv = X.values, y.values
    gi = {g: i for i, g in enumerate(genes)}
    old = pd.read_csv(f"{RESULTS}/selected_features_per_method.csv")
    sel = {m: np.array([gi[g] for g in old[m].dropna()]) for m in old.columns if m != "Boruta"}
    for name, fn in NEW.items():
        sel[name] = np.asarray(fn(Xv, yv, K, np.random.default_rng(RANDOM_STATE)))
    perf = {m: inner_accuracy(Xv, yv, c, RANDOM_STATE) for m, c in sel.items()}
    w, _ = redundancy_weights(sel)
    sc = consensus(sel, w, perf, genes)
    sc.head(300).rename("consensus_score").to_csv(f"{SENS}/gene_ranking_reference.csv")
    orig = pd.read_csv(f"{RESULTS}/table2_gene_ranking.csv", index_col=0).index[:60]
    new60 = list(sc.index[:60])
    out = {"top15_reference": new60[:15], "top15_original": list(orig[:15]),
           "overlap_top15": len(set(new60[:15]) & set(orig[:15])),
           "overlap_top60": len(set(new60) & set(orig)),
           "mic_selected": genes[sel["MIC"]].tolist(),
           "boruta_full_selected": genes[sel["Boruta"]].tolist()}
    json.dump(out, open(f"{SENS}/discovery_comparison.json", "w"), indent=2)
    print(json.dumps(out, indent=1), flush=True)


def step_collect():
    pm, md, sig_new, sig_old = [], [], [], []
    for f in range(N_FOLDS):
        d = json.load(open(f"{SENS}/fold{f}.json"))
        pm += d["per_method"]; md += d["models"]
        sig_new.append(d["signature"]); sig_old.append(d["original_signature"])
        p = f"{SENS}/mlpall{f}.json"
        if os.path.exists(p):
            md.append(json.load(open(p)))
    pm = pd.DataFrame(pm).groupby("method").mean(numeric_only=True).drop(columns="fold")
    md = pd.DataFrame(md).groupby(["model", "features"]).mean(numeric_only=True).drop(columns="fold")
    orig_pm = pd.read_csv(f"{RESULTS}/table1_method_benchmark.csv").set_index("method")
    orig_md = pd.read_csv(f"{RESULTS}/table3_model_comparison.csv").set_index(["model", "features"])
    core_new = sorted(set(sig_new[0]).intersection(*map(set, sig_new[1:])))
    core_old = sorted(set(sig_old[0]).intersection(*map(set, sig_old[1:])))
    summary = {
        "per_fold_overlap_with_original": [len(set(a) & set(b)) for a, b in zip(sig_new, sig_old)],
        "core_in_all_folds_reference": core_new, "core_in_all_folds_original": core_old,
        "boruta_lite_bal_acc": float(orig_pm.loc["Boruta", "balanced_accuracy"]),
        "boruta_full_bal_acc": float(pm.loc["Boruta (reference)", "balanced_accuracy"]),
        "mic_exact_bal_acc": float(pm.loc["MIC (reference)", "balanced_accuracy"]),
        "knn_consensus_original": orig_md.loc[("KNN", "consensus-15")][["accuracy", "balanced_accuracy", "f1_macro"]].round(3).to_dict(),
        "knn_consensus_reference": md.loc[("KNN", "consensus-15")][["accuracy", "balanced_accuracy", "f1_macro"]].round(3).to_dict(),
        "mlp_consensus_sklearn": orig_md.loc[("MLP", "consensus-15")][["accuracy", "balanced_accuracy", "f1_macro"]].round(3).to_dict(),
        "mlp_consensus_dropout": md.loc[("DropoutMLP", "consensus-15")][["accuracy", "balanced_accuracy", "f1_macro"]].round(3).to_dict(),
        "mlp_allgenes_sklearn": orig_md.loc[("MLP", "all-genes")][["accuracy", "balanced_accuracy", "f1_macro"]].round(3).to_dict(),
    }
    if ("DropoutMLP", "all-genes") in md.index:
        summary["mlp_allgenes_dropout"] = md.loc[("DropoutMLP", "all-genes")][["accuracy", "balanced_accuracy", "f1_macro"]].round(3).to_dict()
    json.dump(summary, open(f"{RESULTS}/table12_sensitivity.json", "w"), indent=2)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    cmd, arg = sys.argv[1], (int(sys.argv[2]) if len(sys.argv) > 2 else None)
    {"fold": lambda: step_fold(arg), "mlpall": lambda: step_mlpall(arg),
     "discovery": step_discovery, "collect": step_collect}[cmd]()
