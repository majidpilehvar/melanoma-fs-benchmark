"""
Additional analyses requested for the manuscript.

  composition  marker-based cell-type scores and a composition-adjusted
               re-evaluation of the signature (proxy for CIBERSORTx/xCell)
  network      co-expression network of the top-60 consensus genes with
               betweenness centrality (offline complement to STRING)
  roc          out-of-fold one-vs-rest ROC curves for the nested consensus

Marker genes that appear in the top-60 consensus list are excluded from the
marker sets, otherwise adjusting the signature for "keratinocyte content"
would partly adjust it for itself.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core import load_data, evaluate_predictions, RANDOM_STATE
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve, auc

from config import DATA, RESULTS as OUT, CKPT

MARKERS = {
    "keratinocyte": ["KRT5", "KRT14", "KRT1", "KRT10", "KRT15", "DSG1", "DSC3",
                     "LOR", "FLG", "IVL", "SFN", "TP63", "PERP", "KRT2", "DSP",
                     "SPRR1A", "SPRR1B"],
    "melanocyte": ["PMEL", "MLANA", "TYR", "DCT", "MITF", "SOX10", "TYRP1", "S100B"],
    "fibroblast": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRB", "FBLN1"],
    "endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "EMCN"],
    "immune": ["PTPRC", "CD3E", "CD2", "MS4A1", "CD14", "CD68", "LYZ", "CD53"],
}


def knn():
    return KNeighborsClassifier(5)


def cell_scores(X, exclude):
    """Mean z-score of marker genes per sample; signature genes excluded."""
    Z = (X - X.mean()) / X.std(ddof=0)
    out, used = {}, {}
    for ct, genes in MARKERS.items():
        g = [x for x in genes if x in X.columns and x not in exclude]
        used[ct] = g
        out[ct] = Z[g].mean(axis=1)
    return pd.DataFrame(out), used


def cv_eval(F, y, cv):
    rows = []
    for tr, te in cv.split(F, y):
        m = knn().fit(F[tr], y[tr])
        rows.append(evaluate_predictions(y[te], m.predict(F[te]), m.predict_proba(F[te])))
    return pd.DataFrame(rows).mean().to_dict()


def residualise_in_fold(G, C, tr, te):
    """Regress each gene on composition scores using TRAINING samples only,
    then remove that fitted component from train and test alike."""
    Gtr, Gte = G[tr].copy(), G[te].copy()
    for j in range(G.shape[1]):
        lr = LinearRegression().fit(C[tr], G[tr, j])
        Gtr[:, j] = G[tr, j] - lr.predict(C[tr])
        Gte[:, j] = G[te, j] - lr.predict(C[te])
    return Gtr, Gte


def composition():
    X, y, batch, _ = load_data(DATA, verbose=False)
    yv = y.values
    top60 = list(np.load(f"{CKPT}/top_genes.npy", allow_pickle=True))
    sig = json.load(open(f"{OUT}/signature_stability.json"))
    core3 = sig["in_all_folds"]
    sig15 = top60[:15]

    S, used = cell_scores(X, exclude=set(top60))
    S.to_csv(f"{OUT}/cell_type_scores.csv")
    by_class = S.groupby(y.values).mean().round(3)
    by_class.index = ["normal skin", "primary melanoma", "metastatic melanoma"]

    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    G = X[sig15].values
    C_ker = S[["keratinocyte"]].values
    C_all = S.values

    res = {}
    res["signature_15"] = cv_eval(G, yv, cv)
    res["keratinocyte_score_only"] = cv_eval(C_ker, yv, cv)
    res["all_5_cell_scores_only"] = cv_eval(C_all, yv, cv)

    for label, C in [("signature_15_adj_keratinocyte", C_ker),
                     ("signature_15_adj_all_cell_types", C_all)]:
        rows = []
        for tr, te in cv.split(G, yv):
            Gtr, Gte = residualise_in_fold(G, C, tr, te)
            m = knn().fit(Gtr, yv[tr])
            rows.append(evaluate_predictions(yv[te], m.predict(Gte), m.predict_proba(Gte)))
        res[label] = pd.DataFrame(rows).mean().to_dict()

    # tumour-only task: primary vs metastatic melanoma
    les = np.isin(yv, [2, 3])
    yl = yv[les]
    Gl, Ckl, Cal = G[les], C_ker[les], C_all[les]
    cvl = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    def cv_bin(F):
        rows = []
        for tr, te in cvl.split(F, yl):
            m = knn().fit(F[tr], yl[tr])
            rows.append(evaluate_predictions(yl[te], m.predict(F[te]), labels=(2, 3)))
        return pd.DataFrame(rows).mean().to_dict()

    lesion = {"signature_15": cv_bin(Gl),
              "keratinocyte_score_only": cv_bin(Ckl),
              "all_5_cell_scores_only": cv_bin(Cal)}
    rows = []
    for tr, te in cvl.split(Gl, yl):
        Gtr, Gte = residualise_in_fold(Gl, Ckl, tr, te)
        m = knn().fit(Gtr, yl[tr])
        rows.append(evaluate_predictions(yl[te], m.predict(Gte), labels=(2, 3)))
    lesion["signature_15_adj_keratinocyte"] = pd.DataFrame(rows).mean().to_dict()
    rows = []
    for tr, te in cvl.split(Gl, yl):
        Gtr, Gte = residualise_in_fold(Gl, Cal, tr, te)
        m = knn().fit(Gtr, yl[tr])
        rows.append(evaluate_predictions(yl[te], m.predict(Gte), labels=(2, 3)))
    lesion["signature_15_adj_all_cell_types"] = pd.DataFrame(rows).mean().to_dict()

    # per-gene correlation with composition
    corr = pd.DataFrame({ct: [np.corrcoef(X[g], S[ct])[0, 1] for g in sig15]
                         for ct in S.columns}, index=sig15).round(3)

    out = {"marker_genes_used": used,
           "cell_scores_by_class": by_class.to_dict(),
           "three_class": {k: {m: round(v, 3) for m, v in d.items()} for k, d in res.items()},
           "primary_vs_metastatic": {k: {m: round(v, 3) for m, v in d.items()} for k, d in lesion.items()},
           "gene_composition_correlation": corr.to_dict()}
    json.dump(out, open(f"{OUT}/table8_composition.json", "w"), indent=2)
    corr.to_csv(f"{OUT}/table8b_gene_composition_correlation.csv")
    print("cell scores by class:\n", by_class.to_string())
    print("\nthree-class (bal_acc / f1):")
    for k, d in res.items():
        print(f"  {k:34s} {d['balanced_accuracy']:.3f} / {d['f1_macro']:.3f}")
    print("\nprimary vs metastatic (bal_acc / f1):")
    for k, d in lesion.items():
        print(f"  {k:34s} {d['balanced_accuracy']:.3f} / {d['f1_macro']:.3f}")
    print("\ncorrelation with keratinocyte score:\n", corr["keratinocyte"].to_string())


def network(thr=0.7):
    import networkx as nx
    X, y, _, _ = load_data(DATA, verbose=False)
    top60 = [g for g in np.load(f"{CKPT}/top_genes.npy", allow_pickle=True)
             if g in X.columns]
    R = X[top60].corr().values
    Gx = nx.Graph()
    Gx.add_nodes_from(top60)
    for i in range(len(top60)):
        for j in range(i + 1, len(top60)):
            if abs(R[i, j]) >= thr:
                Gx.add_edge(top60[i], top60[j], weight=float(abs(R[i, j])),
                            sign=int(np.sign(R[i, j])))
    iso = [n for n in Gx.nodes if Gx.degree(n) == 0]
    H = Gx.copy()
    H.remove_nodes_from(iso)
    bc = nx.betweenness_centrality(H)
    deg = dict(H.degree())
    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    tab = pd.DataFrame({"degree": deg, "betweenness": bc}).sort_values(
        "betweenness", ascending=False)
    tab["component"] = [next(i for i, c in enumerate(comps) if n in c) for n in tab.index]
    tab.to_csv(f"{OUT}/table9_coexpression_network.csv")
    nx.write_edgelist(H, f"{OUT}/coexpression_edges.tsv", delimiter="\t",
                      data=["weight", "sign"])
    summary = {"threshold_abs_r": thr, "nodes": H.number_of_nodes(),
               "edges": H.number_of_edges(), "isolated_removed": iso,
               "components": [sorted(c) for c in comps],
               "top_betweenness": tab.head(10).round(4).reset_index()
                                     .rename(columns={"index": "gene"}).to_dict("records")}
    json.dump(summary, open(f"{OUT}/table9_network_summary.json", "w"), indent=2)
    print(f"network |r|>={thr}: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges, "
          f"{len(comps)} components; isolated: {iso}")
    print(tab.head(12).round(4).to_string())


def roc():
    X, y, _, _ = load_data(DATA, verbose=False)
    Xv, yv = X.values, y.values
    genes = list(X.columns)
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    proba = np.zeros((len(yv), 3))
    for f, (tr, te) in enumerate(cv.split(Xv, yv)):
        sig = json.load(open(f"{CKPT}/fold{f}.json"))["signature"]
        idx = [genes.index(g) for g in sig]
        m = knn().fit(Xv[tr][:, idx], yv[tr])
        proba[te] = m.predict_proba(Xv[te][:, idx])
    curves = {}
    for i, c in enumerate([1, 2, 3]):
        fpr, tpr, _ = roc_curve((yv == c).astype(int), proba[:, i])
        curves[c] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))}
    json.dump(curves, open(f"{OUT}/roc_out_of_fold.json", "w"))
    print({c: round(v["auc"], 3) for c, v in curves.items()})


if __name__ == "__main__":
    {"composition": composition, "network": network, "roc": roc}[sys.argv[1]]()
