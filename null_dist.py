import os, sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_data, evaluate_predictions, RANDOM_STATE
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from config import DATA, RESULTS
X, y, _, _ = load_data(DATA, verbose=False)
Xv, yv = X.values, y.values
cv = list(StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE).split(Xv, yv))
def f1_of(cols):
    return float(np.mean([evaluate_predictions(yv[te], KNeighborsClassifier(5).fit(Xv[tr][:, cols], yv[tr]).predict(Xv[te][:, cols]))["f1_macro"] for tr, te in cv]))
rng = np.random.default_rng(RANDOM_STATE)
null = [f1_of(rng.choice(Xv.shape[1], 5, replace=False)) for _ in range(500)]
np.save(os.path.join(RESULTS, "null_distribution.npy"), np.array(null))
print("saved", np.mean(null))
