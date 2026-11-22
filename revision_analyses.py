"""
Analyses added in response to the pre-submission review.

    python3 revision_analyses.py stats        uncertainty, corrected paired tests, 5,000-draw null
    python3 revision_analyses.py classifiers  conclusions under logistic regression and random forest
    python3 revision_analyses.py batch        in-fold ComBat-style correction, LOSO with study centring
    python3 revision_analyses.py dups         near-duplicate samples (proxy for repeated patients)
Outputs: results/revision/*.json|csv
"""
import os, sys, json, itertools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, RESULTS, CKPT
from core import load_data, evaluate_predictions, RANDOM_STATE
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

REV = os.path.join(RESULTS, "revision"); os.makedirs(REV, exist_ok=True)
X, y, b, _ = load_data(DATA, verbose=False)
Xv, yv, bv = X.values, y.values, b.values
genes = np.array(X.columns); gi = {g: i for i, g in enumerate(genes)}
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE).split(Xv, yv))
CK = [json.load(open(f"{CKPT}/fold{f}.json")) for f in range(5)]
N_TEST_OVER_TRAIN = 0.25  # 5-fold CV


def ci95(v):
    v = np.asarray(v, float); m = v.mean(); s = v.std(ddof=1)
    h = stats.t.ppf(0.975, len(v) - 1) * s / np.sqrt(len(v))
    return m, s, m - h, m + h


def corrected_ttest(a, c):
    """Nadeau & Bengio (2003) corrected resampled t-test for paired CV scores."""
    d = np.asarray(a) - np.asarray(c); k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / k + N_TEST_OVER_TRAIN) * var)
    return float(t), float(2 * stats.t.sf(abs(t), k - 1))


def stats_part():
    out = {}
    pm = pd.DataFrame([r for c in CK for r in c["per_method"]])
    cons = pd.DataFrame([r for c in CK for r in c["models"] if r["model"] == "KNN" and r["features"] == "consensus-15"])
    rows = []
    for m, d in pm.groupby("method"):
        d = d.sort_values("fold")
        mb, sb, lo, hi = ci95(d.balanced_accuracy)
        sets = [set(c["selected"][m]) for c in CK]
        jac = [len(p & q) / len(p | q) for p, q in itertools.combinations(sets, 2)]
        rows.append({"method": m, "bal_acc_mean": mb, "bal_acc_sd": sb, "bal_acc_ci_low": lo, "bal_acc_ci_high": hi,
                     "jaccard_mean": np.mean(jac), "jaccard_sd": np.std(jac, ddof=1)})
    cons = cons.sort_values("fold")
    mb, sb, lo, hi = ci95(cons.balanced_accuracy)
    sigs = [set(c["signature"]) for c in CK]
    jac = [len(p & q) / len(p | q) for p, q in itertools.combinations(sigs, 2)]
    rows.append({"method": "CONSENSUS", "bal_acc_mean": mb, "bal_acc_sd": sb, "bal_acc_ci_low": lo, "bal_acc_ci_high": hi,
                 "jaccard_mean": np.mean(jac), "jaccard_sd": np.std(jac, ddof=1)})
    t = pd.DataFrame(rows).sort_values("bal_acc_mean", ascending=False)
    t.to_csv(f"{REV}/uncertainty_per_method.csv", index=False)
    out["consensus"] = t[t.method == "CONSENSUS"].round(4).to_dict("records")[0]
    # paired corrected tests: consensus vs key selectors
    tests = {}
    for m in ["Kruskal-Wallis", "LASSO (L1)", "Stability Selection", "ACO", "MIFS", "ANOVA", "Variance Threshold"]:
        a = pm[pm.method == m].sort_values("fold").balanced_accuracy.values
        tt, p = corrected_ttest(a, cons.balanced_accuracy.values)
        tests[m] = {"mean_difference": float(np.mean(a - cons.balanced_accuracy.values)), "t": tt, "p": p}
    # Holm correction across the comparisons
    ps = np.array([v["p"] for v in tests.values()]); order = np.argsort(ps); m_ = len(ps); adj = np.empty(m_)
    running = 0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, ps[i] * (m_ - rank))); adj[i] = running
    for (k, v), a_ in zip(tests.items(), adj):
        v["p_holm"] = float(a_)
    out["corrected_tests_vs_consensus"] = tests
    # 5,000-draw null for the nested top-5 signature
    def f1_fold(cols_per_fold):
        return float(np.mean([evaluate_predictions(yv[te], KNeighborsClassifier(5).fit(Xv[tr][:, c], yv[tr]).predict(Xv[te][:, c]))["f1_macro"]
                              for (tr, te), c in zip(FOLDS, cols_per_fold)]))
    obs = f1_fold([[gi[g] for g in CK[f]["signature"][:5]] for f in range(5)])
    rng = np.random.default_rng(RANDOM_STATE + 7)
    null = np.array([f1_fold([rng.choice(len(genes), 5, replace=False)] * 5) for _ in range(5000)])
    np.save(f"{REV}/null_5000.npy", null)
    out["null_5000"] = {"observed": obs, "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
                        "null_max": float(null.max()), "n_ge_observed": int((null >= obs).sum()),
                        "empirical_p": float(((null >= obs).sum() + 1) / (len(null) + 1))}
    json.dump(out, open(f"{REV}/stats.json", "w"), indent=2)
    print(json.dumps(out, indent=1))


def classifiers_part():
    models = {"kNN": lambda: KNeighborsClassifier(5),
              "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)),
              "random_forest": lambda: RandomForestClassifier(300, random_state=RANDOM_STATE, n_jobs=-1)}
    rows = []
    sets = {m: [[gi[g] for g in c["selected"][m]] for c in CK] for m in CK[0]["selected"]}
    sets["CONSENSUS"] = [[gi[g] for g in c["signature"]] for c in CK]
    for name, mk in models.items():
        for m, per_fold in sets.items():
            ba = [evaluate_predictions(yv[te], mk().fit(Xv[tr][:, c], yv[tr]).predict(Xv[te][:, c]))["balanced_accuracy"]
                  for (tr, te), c in zip(FOLDS, per_fold)]
            rows.append({"classifier": name, "method": m, "bal_acc": float(np.mean(ba))})
    d = pd.DataFrame(rows); d.to_csv(f"{REV}/classifier_robustness.csv", index=False)
    piv = d.pivot(index="method", columns="classifier", values="bal_acc")
    out = {}
    for c in piv.columns:
        r = piv[c].rank(ascending=False)
        out[c] = {"consensus_bal_acc": float(piv.loc["CONSENSUS", c]), "consensus_rank_of_33": int(r["CONSENSUS"]),
                  "median_selector": float(piv[c].drop("CONSENSUS").median()),
                  "kruskal": float(piv.loc["Kruskal-Wallis", c]), "lasso": float(piv.loc["LASSO (L1)", c]),
                  "stability_selection": float(piv.loc["Stability Selection", c]),
                  "best_method": piv[c].drop("CONSENSUS").idxmax(), "best_value": float(piv[c].drop("CONSENSUS").max())}
    out["rank_correlation_kNN_vs_logistic"] = float(stats.spearmanr(piv["kNN"], piv["logistic"]).correlation)
    out["rank_correlation_kNN_vs_rf"] = float(stats.spearmanr(piv["kNN"], piv["random_forest"]).correlation)
    json.dump(out, open(f"{REV}/classifiers.json", "w"), indent=2)
    print(json.dumps(out, indent=1))


def _eb_location(Ztr, btr, ytr, Zte=None, bte=None):
    """ComBat-style batch adjustment, location only, with class as a protected
    covariate. Parameters are estimated on training samples; test samples are
    shifted by their batch's estimated effect, which needs no test labels."""
    classes = np.unique(ytr)
    D = np.column_stack([np.ones(len(ytr))] + [(ytr == c).astype(float) for c in classes[1:]])
    B = np.column_stack([(btr == k).astype(float) for k in np.unique(btr)])
    Bc = B - B.mean(0)
    full = np.column_stack([D, Bc])
    coef, *_ = np.linalg.lstsq(full, Ztr, rcond=None)
    resid = Ztr - full @ coef
    sigma = np.sqrt((resid ** 2).sum(0) / max(len(ytr) - full.shape[1], 1)) + 1e-8
    gamma_hat = coef[D.shape[1]:] / sigma                       # batch x genes, standardised
    n_b = B.sum(0)
    gbar = gamma_hat.mean(1, keepdims=True); tau2 = gamma_hat.var(1, ddof=1, keepdims=True) + 1e-8
    gamma_star = (n_b[:, None] * tau2 * gamma_hat + gbar) / (n_b[:, None] * tau2 + 1)   # EB shrinkage
    batches = list(np.unique(btr))
    shift_tr = (Bc @ (gamma_star * sigma))
    Atr = Ztr - shift_tr
    if Zte is None:
        return Atr
    Bte = np.column_stack([(bte == k).astype(float) for k in batches]) - B.mean(0)
    Ate = Zte - Bte @ (gamma_star * sigma)
    return Atr, Ate


def batch_part():
    out = {}
    sig = [[gi[g] for g in c["signature"]] for c in CK]
    raw, cor = [], []
    rf_raw, rf_cor = [], []
    for f, (tr, te) in enumerate(FOLDS):
        Atr, Ate = _eb_location(Xv[tr], bv[tr], yv[tr], Xv[te], bv[te])
        c = sig[f]
        raw.append(evaluate_predictions(yv[te], KNeighborsClassifier(5).fit(Xv[tr][:, c], yv[tr]).predict(Xv[te][:, c]))["balanced_accuracy"])
        cor.append(evaluate_predictions(yv[te], KNeighborsClassifier(5).fit(Atr[:, c], yv[tr]).predict(Ate[:, c]))["balanced_accuracy"])
        # residual study information after correction (study prediction, in-fold)
        rf_raw.append((RandomForestClassifier(200, random_state=0, n_jobs=-1).fit(Xv[tr], bv[tr]).predict(Xv[te]) == bv[te]).mean())
        rf_cor.append((RandomForestClassifier(200, random_state=0, n_jobs=-1).fit(Atr, bv[tr]).predict(Ate) == bv[te]).mean())
    out["pooled_cv_consensus_bal_acc"] = {"uncorrected": float(np.mean(raw)), "in_fold_corrected": float(np.mean(cor))}
    out["study_prediction_accuracy"] = {"uncorrected": float(np.mean(rf_raw)), "in_fold_corrected": float(np.mean(rf_cor)),
                                        "majority_baseline": float(pd.Series(bv).value_counts(normalize=True).max())}
    # LOSO: the held-out study has no training samples, so its batch effect cannot be
    # estimated from labelled data; the only label-free option is centring each study
    loso = {}
    for held in np.unique(bv):
        tr, te = np.where(bv != held)[0], np.where(bv == held)[0]
        c = [gi[g] for g in json.load(open(f"{CKPT}/loso{int(held)}.json")).get("signature", CK[0]["signature"])]
        Xc = Xv.copy()
        for k in np.unique(bv):
            Xc[bv == k] -= Xc[bv == k].mean(0)
        r0 = evaluate_predictions(yv[te], KNeighborsClassifier(5).fit(Xv[tr][:, c], yv[tr]).predict(Xv[te][:, c]), labels=(1, 2, 3))["balanced_accuracy"]
        r1 = evaluate_predictions(yv[te], KNeighborsClassifier(5).fit(Xc[tr][:, c], yv[tr]).predict(Xc[te][:, c]), labels=(1, 2, 3))["balanced_accuracy"]
        loso[f"study_{int(held)+1}"] = {"uncorrected": float(r0), "study_centred": float(r1),
                                        "classes_present": sorted(map(int, np.unique(yv[te])))}
    out["loso_study_centring"] = loso
    # full-data correction for the PCA figure only
    A = _eb_location(Xv, bv, yv)
    np.save(f"{REV}/combat_full_for_pca.npy", A.astype(np.float32))
    json.dump(out, open(f"{REV}/batch.json", "w"), indent=2)
    print(json.dumps(out, indent=1))


def dups_part():
    Z = (Xv - Xv.mean(1, keepdims=True)) / Xv.std(1, keepdims=True)
    R = (Z @ Z.T) / Xv.shape[1]
    iu = np.triu_indices(len(R), 1)
    r = R[iu]
    thr = 0.98
    pairs = [(int(i), int(j), float(R[i, j])) for i, j in zip(*iu) if R[i, j] >= thr]
    fold_of = np.empty(len(yv), int)
    for f, (_, te) in enumerate(FOLDS):
        fold_of[te] = f
    rows = [{"sample_a": X.index[i], "sample_b": X.index[j], "r": round(v, 4), "class_a": int(yv[i]), "class_b": int(yv[j]),
             "study_a": int(bv[i]) + 1, "study_b": int(bv[j]) + 1, "split_across_folds": bool(fold_of[i] != fold_of[j])}
            for i, j, v in pairs]
    pd.DataFrame(rows).to_csv(f"{REV}/near_duplicates.csv", index=False)
    out = {"threshold_r": thr, "n_pairs_above": len(rows), "max_r": float(r.max()),
           "pct99_r": float(np.percentile(r, 99)), "median_r": float(np.median(r)),
           "within_study_median_r": {int(k) + 1: float(np.median(R[np.ix_(bv == k, bv == k)][np.triu_indices((bv == k).sum(), 1)])) for k in np.unique(bv)},
           "pairs_split_across_folds": int(sum(x["split_across_folds"] for x in rows))}
    json.dump(out, open(f"{REV}/duplicates.json", "w"), indent=2)
    print(json.dumps(out, indent=1))
    if rows:
        print(pd.DataFrame(rows).head(20).to_string(index=False))


if __name__ == "__main__":
    {"stats": stats_part, "classifiers": classifiers_part, "batch": batch_part, "dups": dups_part}[sys.argv[1]]()
