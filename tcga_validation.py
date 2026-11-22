"""
External validation on TCGA-SKCM RNA-seq (primary vs metastatic only; TCGA has
almost no normal skin, so the three-class model cannot be validated there).

The signature is FIXED: the 15 consensus genes from the full microarray data.
Nothing is selected or tuned on TCGA. Models are trained on the microarray
primary (class 2) and metastatic (class 3) samples and applied to TCGA.

    pip install requests pandas scikit-learn
    python3 tcga_validation.py
Needs internet. Not run in the environment that produced the manuscript.
"""
import io, os, sys, gzip, json
import numpy as np, pandas as pd, requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, RESULTS
from core import load_data, evaluate_predictions
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

URLS = ["https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.SKCM.sampleMap%2FHiSeqV2.gz",
        "https://tcga.xenahubs.net/download/TCGA.SKCM.sampleMap/HiSeqV2.gz"]
ALIAS = {"LPPR4": "PLPPR4", "NOV": "CCN3", "IGJ": "JCHAIN", "FAIM3": "FCMR"}


def harmonise(sym):
    s = ALIAS.get(sym, sym)
    if s.startswith("MARCH") and s[5:].isdigit():
        return "MARCHF" + s[5:]
    if s.startswith("SEPT") and s[4:].isdigit():
        return "SEPTIN" + s[4:]
    return s


def load_tcga():
    for u in URLS:
        try:
            r = requests.get(u, timeout=300); r.raise_for_status()
            df = pd.read_csv(io.BytesIO(gzip.decompress(r.content)), sep="\t", index_col=0)
            df.index = [harmonise(g) for g in df.index]
            return df.groupby(level=0).mean()
        except Exception as e:
            print("download failed:", u, e)
    raise SystemExit("Could not download TCGA-SKCM expression from UCSC Xena.")


def within_sample_rank(M):
    """Percentile rank of each gene within each sample: platform- and
    composition-independent scale for cross-platform transfer."""
    return M.rank(axis=1, pct=True)


def main():
    X, y, _, _ = load_data(DATA, verbose=False)
    sig = list(pd.read_csv(f"{RESULTS}/table2_gene_ranking.csv", index_col=0).index[:15])
    T = load_tcga().T                                  # samples x genes
    code = T.index.str[13:15]
    lab = pd.Series(np.where(code == "01", 2, np.where(code == "06", 3, 0)), index=T.index)
    T, lab = T[lab > 0], lab[lab > 0]
    common = sorted(set(X.columns) & set(T.columns))
    missing = [g for g in sig if g not in common]
    sig = [g for g in sig if g in common]
    keep = y.isin([2, 3])
    Xm, ym = X.loc[keep, common], y[keep].values
    res = {"n_tcga_primary": int((lab == 2).sum()), "n_tcga_metastatic": int((lab == 3).sum()),
           "signature_genes_used": sig, "signature_genes_missing_in_tcga": missing}
    for scale in ["within_platform_z", "within_sample_rank"]:
        if scale == "within_platform_z":
            A = (Xm[sig] - Xm[sig].mean()) / Xm[sig].std()
            B = (T[sig] - T[sig].mean()) / T[sig].std()
        else:
            A = within_sample_rank(Xm)[sig]; B = within_sample_rank(T[common])[sig]
        for name, mdl in [("kNN", KNeighborsClassifier(5)),
                          ("logistic", LogisticRegression(max_iter=3000))]:
            mdl.fit(A.values, ym)
            pred = mdl.predict(B.values); proba = mdl.predict_proba(B.values)[:, 1]
            r = evaluate_predictions(lab.values, pred, labels=(2, 3))
            r["auc"] = roc_auc_score((lab.values == 3).astype(int), proba)
            res[f"{scale}/{name}"] = {k: round(float(v), 3) for k, v in r.items()}
    os.makedirs(f"{RESULTS}/revision", exist_ok=True)
    json.dump(res, open(f"{RESULTS}/revision/tcga_validation.json", "w"), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
