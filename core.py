"""
Core utilities: data loading/cleaning, batch assignment, leak-free evaluation.

Melanoma biomarker discovery - corrected pipeline.
"""
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score)

RANDOM_STATE = 42

# Excel silently converts these gene symbols to dates. Map them back.
# MARCH* -> MARCHF* (membrane associated ring-CH-type finger)
# SEPT*  -> SEPTIN*
_MONTH = {"Mar": "MARCHF", "Sep": "SEPTIN"}


def fix_excel_gene_symbols(index):
    """Repair gene symbols corrupted into dates by Excel (Mar-01 -> MARCHF1)."""
    out, fixed = [], {}
    for g in index:
        m = re.match(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-(\d+)$", str(g))
        if m and m.group(1) in _MONTH:
            new = f"{_MONTH[m.group(1)]}{int(m.group(2))}"
            fixed[g] = new
            out.append(new)
        else:
            out.append(str(g))
    return out, fixed


# Obsolete symbols -> current HGNC symbols (checked for the genes this study reports)
SYMBOL_UPDATES = {
    "LPPR4": "PLPPR4",
    "NOV": "CCN3",
    "IGJ": "JCHAIN",
    "FAIM3": "FCMR",
    "KRT18P38": "KRT18P38",  # pseudogene, unchanged
}


def load_data(path, fix_symbols=True, collapse_duplicates=True, verbose=True):
    """Load the gene-expression matrix.

    Returns
    -------
    X : DataFrame  (samples x genes)
    y : Series     class label per sample (1 healthy, 2 at-risk, 3 metastatic)
    batch : Series study/batch label per sample, inferred from GSM accession blocks
    report : dict  what the cleaning step changed
    """
    df = pd.read_csv(path, index_col=0)
    y = df.loc["Class"].astype(int)
    X = df.drop(index="Class").astype(float)

    report = {"n_genes_raw": X.shape[0], "n_samples": X.shape[1]}

    if fix_symbols:
        new_index, fixed = fix_excel_gene_symbols(X.index)
        X.index = new_index
        X = X.rename(index=SYMBOL_UPDATES)
        report["excel_date_symbols_fixed"] = fixed

    if collapse_duplicates:
        dup = X.index[X.index.duplicated()].unique().tolist()
        if dup:
            X = X.groupby(level=0).mean()
        report["duplicate_genes_collapsed"] = dup

    report["n_genes_clean"] = X.shape[0]

    batch = infer_batches(X.columns)
    X = X.T  # samples x genes

    if verbose:
        print(f"[data] {X.shape[0]} samples x {X.shape[1]} genes")
        print(f"[data] classes: {dict(y.value_counts().sort_index())}")
        print(f"[data] batches: {dict(batch.value_counts().sort_index())}")
        if report.get("excel_date_symbols_fixed"):
            print(f"[data] repaired {len(report['excel_date_symbols_fixed'])} "
                  f"Excel-corrupted symbols")
        if report.get("duplicate_genes_collapsed"):
            print(f"[data] collapsed duplicates: {report['duplicate_genes_collapsed']}")
    return X, y, batch, report


def infer_batches(sample_ids, gap=10000):
    """Group GSM accessions into source studies by numeric gaps."""
    num = pd.Series({s: int(re.sub(r"\D", "", s)) for s in sample_ids})
    order = num.sort_values()
    lab, gid, prev = {}, 0, None
    for s, v in order.items():
        if prev is not None and v - prev > gap:
            gid += 1
        lab[s] = gid
        prev = v
    return pd.Series(lab)[list(sample_ids)]


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def macro_specificity(y_true, y_pred, labels):
    """One-vs-rest specificity averaged over classes.

    The original code read cm[0,0] and cm[0,1] only, which gives a single
    class' value on a 3x3 matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    specs = []
    for i in range(len(labels)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        specs.append(tn / (tn + fp) if (tn + fp) else np.nan)
    return float(np.nanmean(specs))


def evaluate_predictions(y_true, y_pred, y_proba=None, labels=(1, 2, 3)):
    labels = list(labels)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro",
                                           zero_division=0, labels=labels),
        # macro recall == macro sensitivity; reported once, not twice
        "sensitivity_macro": recall_score(y_true, y_pred, average="macro",
                                          zero_division=0, labels=labels),
        "f1_macro": f1_score(y_true, y_pred, average="macro",
                             zero_division=0, labels=labels),
        "specificity_macro": macro_specificity(y_true, y_pred, labels),
    }
    if y_proba is not None:
        try:
            out["auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr",
                                           average="macro", labels=labels)
        except Exception:
            out["auc_ovr"] = np.nan
    return out


# --------------------------------------------------------------------------
# selector wrapper so every method plugs into a Pipeline identically
# --------------------------------------------------------------------------
class FeatureSelector(BaseEstimator, TransformerMixin):
    """Wraps a `fn(X, y, k, rng) -> index array` as a sklearn transformer.

    Fitting happens on training folds only, which is the whole point.
    """

    def __init__(self, fn=None, k=15, random_state=RANDOM_STATE, name=""):
        self.fn = fn
        self.k = k
        self.random_state = random_state
        self.name = name

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.random_state)
        idx = np.asarray(self.fn(np.asarray(X), np.asarray(y), self.k, rng),
                         dtype=int)
        self.support_ = np.unique(idx)[: self.k]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]

    def get_support(self, indices=True):
        return self.support_
