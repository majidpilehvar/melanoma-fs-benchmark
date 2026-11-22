"""Repeat the nested composition adjustment (Table 5) with xCell or MCP-counter
scores produced by deconvolution.R, instead of the marker averages."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, RESULTS, CKPT
from core import load_data, evaluate_predictions, RANDOM_STATE
from extra_analyses import residualise_in_fold
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

X, y, _, _ = load_data(DATA, verbose=False)
Xv, yv = X.values, y.values; genes = list(X.columns)
folds = list(StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE).split(Xv, yv))
out = {}
for name in ["xcell", "mcpcounter"]:
    p = f"{RESULTS}/revision/{name}_scores.csv"
    if not os.path.exists(p):
        continue
    S = pd.read_csv(p, index_col=0).loc[X.index]
    S = S.loc[:, S.std() > 0]
    C = ((S - S.mean()) / S.std()).values
    sig_bal, adj_bal, only_bal = [], [], []
    for f, (tr, te) in enumerate(folds):
        idx = [genes.index(g) for g in json.load(open(f"{CKPT}/fold{f}.json"))["signature"]]
        G = Xv[:, idx]
        ev = lambda A, B: evaluate_predictions(yv[te], KNeighborsClassifier(5).fit(A, yv[tr]).predict(B))["balanced_accuracy"]
        sig_bal.append(ev(G[tr], G[te]))
        Gtr, Gte = residualise_in_fold(G, C, tr, te); adj_bal.append(ev(Gtr, Gte))
        only_bal.append(ev(C[tr], C[te]))
    out[name] = {"signature": float(np.mean(sig_bal)), "signature_adjusted": float(np.mean(adj_bal)),
                 "scores_alone": float(np.mean(only_bal)), "n_cell_types": int(C.shape[1])}
json.dump(out, open(f"{RESULTS}/revision/deconvolution_adjustment.json", "w"), indent=2)
print(json.dumps(out, indent=1))
