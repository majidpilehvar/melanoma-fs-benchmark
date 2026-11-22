"""
External validation on TCGA-SKCM RNA-seq (RSEM, from the cBioPortal API, study
skcm_tcga), primary vs metastatic only (TCGA has no normal skin).

The 15 genes are FIXED (full-data consensus from the microarray cohort); the
classifier is trained on the microarray primary and metastatic samples only and
applied unchanged to TCGA. Each platform is z-scored within itself, a label-free
step that removes platform scale but assumes broadly similar class mixes.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, RESULTS
from core import load_data, evaluate_predictions
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import stats

T = pd.read_csv(f"{RESULTS}/revision/tcga_skcm_signature_rnaseq.tsv", sep="\t", index_col=0)
T = T[T.type.isin(["Primary", "Metastasis"])].dropna()
lab = np.where(T.type == "Primary", 2, 3)
sig = list(pd.read_csv(f"{RESULTS}/table2_gene_ranking.csv", index_col=0).index[:15])
B = np.log2(T[sig].astype(float) + 1)
X, y, _, _ = load_data(DATA, verbose=False)
keep = y.isin([2, 3]).values
A, ya = X.loc[keep, sig], y[keep].values
z = lambda M: (M - M.mean()) / M.std()
Az, Bz = z(A), z(B)
res = {"n_tcga_primary": int((lab == 2).sum()), "n_tcga_metastatic": int((lab == 3).sum()), "genes": sig}
for name, mdl in [("kNN", KNeighborsClassifier(5)), ("logistic", LogisticRegression(max_iter=3000, C=1.0))]:
    mdl.fit(Az.values, ya)
    pred = mdl.predict(Bz.values); p = mdl.predict_proba(Bz.values)[:, 1]
    r = evaluate_predictions(lab, pred, labels=(2, 3)); r["auc"] = roc_auc_score((lab == 3).astype(int), p)
    res[name] = {k: round(float(v), 3) for k, v in r.items()}
# direction of each gene, microarray vs TCGA (metastatic minus primary)
dirs = {}
for g in sig:
    da = A[g][ya == 3].mean() - A[g][ya == 2].mean()
    db = B[g][lab == 3].mean() - B[g][lab == 2].mean()
    pb = stats.mannwhitneyu(B[g][lab == 3], B[g][lab == 2]).pvalue
    dirs[g] = {"microarray_diff": round(float(da), 2), "tcga_diff": round(float(db), 2), "tcga_p": float(f"{pb:.2g}")}
res["direction_agreement"] = int(sum(np.sign(v["microarray_diff"]) == np.sign(v["tcga_diff"]) for v in dirs.values()))
res["per_gene"] = dirs
json.dump(res, open(f"{RESULTS}/revision/tcga_validation.json", "w"), indent=2)
print(json.dumps({k: v for k, v in res.items() if k != "per_gene"}, indent=1))
print(pd.DataFrame(dirs).T.to_string())
