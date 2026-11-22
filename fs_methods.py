"""
Feature-selection methods, all with the same signature:

    fn(X, y, k, rng) -> array of selected column indices

X and y are ALWAYS the training part of a fold. Nothing here may touch
test data; that is what the original implementation got wrong.

Wrapper / metaheuristic methods are expensive on 12k features, so they
pre-filter to `PREFILTER` candidates by ANOVA F *within the same training
fold*. That keeps the procedure leak-free while remaining tractable.
"""
import numpy as np
from scipy import stats
from sklearn.feature_selection import (f_classif, chi2, mutual_info_classif)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.inspection import permutation_importance

PREFILTER = 300
INNER_CV = 3


def _topk(scores, k):
    s = np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf)
    return np.argsort(-s)[:k]


def _prefilter(X, y, n=PREFILTER):
    F, _ = f_classif(X, y)
    return _topk(F, min(n, X.shape[1]))


def _fitness(X, y, cols, rng_seed=0):
    """Internal CV accuracy of a candidate subset (training data only)."""
    cols = np.asarray(cols, dtype=int)
    if cols.size == 0:
        return 0.0
    cv = StratifiedKFold(INNER_CV, shuffle=True, random_state=rng_seed)
    return float(cross_val_score(KNeighborsClassifier(5), X[:, cols], y,
                                 cv=cv, scoring="accuracy").mean())


# ---------------------------------------------------------------- filters
def anova(X, y, k, rng):
    F, _ = f_classif(X, y)
    return _topk(F, k)


def chi_square(X, y, k, rng):
    Xp = X - X.min(axis=0)          # chi2 needs non-negative input
    s, _ = chi2(Xp, y)
    return _topk(s, k)


def mutual_info(X, y, k, rng):
    # kNN-based MI scales linearly in n_features and dominates runtime; use a
    # generous ANOVA shortlist computed on the same training fold.
    pre = _prefilter(X, y, 3000)
    s = mutual_info_classif(X[:, pre], y, random_state=int(rng.integers(1 << 31)))
    return pre[_topk(s, k)]


def information_gain(X, y, k, rng):
    """MI on equal-frequency discretised expression -> classic info gain."""
    pre = _prefilter(X, y, 3000)
    Xd = np.apply_along_axis(
        lambda c: np.digitize(c, np.quantile(c, [.2, .4, .6, .8])), 0, X[:, pre])
    s = mutual_info_classif(Xd, y, discrete_features=True,
                            random_state=int(rng.integers(1 << 31)))
    return pre[_topk(s, k)]


def fisher_score(X, y, k, rng):
    classes = np.unique(y)
    mu, var = X.mean(0), X.var(0) + 1e-12
    num = np.zeros(X.shape[1])
    den = np.zeros(X.shape[1])
    for c in classes:
        Xc = X[y == c]
        num += len(Xc) * (Xc.mean(0) - mu) ** 2
        den += len(Xc) * Xc.var(0)
    return _topk(num / (den + 1e-12), k)


def pearson(X, y, k, rng):
    yc = y - y.mean()
    Xc = X - X.mean(0)
    r = (Xc * yc[:, None]).sum(0) / (
        np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum()) + 1e-12)
    return _topk(np.abs(r), k)


def spearman(X, y, k, rng):
    Xr = np.apply_along_axis(stats.rankdata, 0, X)
    return pearson(Xr, stats.rankdata(y).astype(float), k, rng)


def variance_threshold(X, y, k, rng):
    return _topk(X.var(0), k)


def kruskal(X, y, k, rng):
    groups = [X[y == c] for c in np.unique(y)]
    H = np.array([stats.kruskal(*[g[:, j] for g in groups]).statistic
                  for j in range(X.shape[1])])
    return _topk(H, k)


# -------------------------------------------------------------- embedded
def lasso(X, y, k, rng):
    """Multinomial L1 via one-vs-rest; liblinear has no native multiclass
    support in current scikit-learn."""
    pre = _prefilter(X, y, 1000)
    base = LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                              max_iter=5000,
                              random_state=int(rng.integers(1 << 31)))
    m = OneVsRestClassifier(base).fit(X[:, pre], y)
    coef = np.vstack([e.coef_ for e in m.estimators_])
    return pre[_topk(np.abs(coef).max(0), k)]


def elastic_net(X, y, k, rng):
    pre = _prefilter(X, y, 1000)
    Xs = StandardScaler().fit_transform(X[:, pre])
    m = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                           C=0.1, max_iter=300, tol=1e-3,
                           random_state=int(rng.integers(1 << 31))).fit(Xs, y)
    return pre[_topk(np.abs(m.coef_).max(0), k)]


def ridge_sel(X, y, k, rng):
    m = Ridge(alpha=1.0, random_state=int(rng.integers(1 << 31))).fit(X, y)
    return _topk(np.abs(m.coef_), k)


def rf_importance(X, y, k, rng):
    m = RandomForestClassifier(300, n_jobs=-1,
                               random_state=int(rng.integers(1 << 31))).fit(X, y)
    return _topk(m.feature_importances_, k)


def extra_trees(X, y, k, rng):
    m = ExtraTreesClassifier(300, n_jobs=-1,
                             random_state=int(rng.integers(1 << 31))).fit(X, y)
    return _topk(m.feature_importances_, k)


def permutation_imp(X, y, k, rng):
    pre = _prefilter(X, y)
    m = RandomForestClassifier(200, n_jobs=-1,
                               random_state=int(rng.integers(1 << 31)))
    m.fit(X[:, pre], y)
    r = permutation_importance(m, X[:, pre], y, n_repeats=5,
                               random_state=int(rng.integers(1 << 31)), n_jobs=-1)
    return pre[_topk(r.importances_mean, k)]


def stability_selection(X, y, k, rng):
    """Bootstrapped L1 selection frequency."""
    pre = _prefilter(X, y, 1000)
    Xp = X[:, pre]
    counts = np.zeros(Xp.shape[1])
    n = Xp.shape[0]
    base = LogisticRegression(penalty="l1", solver="liblinear", C=0.05,
                              max_iter=2000)
    for _ in range(20):
        idx = rng.choice(n, int(0.75 * n), replace=False)
        if len(np.unique(y[idx])) < 2:
            continue
        m = OneVsRestClassifier(base).fit(Xp[idx], y[idx])
        coef = np.vstack([e.coef_ for e in m.estimators_])
        counts += (np.abs(coef).max(0) > 0).astype(float)
    return pre[_topk(counts, k)]


def boruta_lite(X, y, k, rng):
    """Shadow-feature importance test (Boruta's core idea, fixed iterations)."""
    pre = _prefilter(X, y, 500)
    Xp = X[:, pre]
    hits = np.zeros(Xp.shape[1])
    for _ in range(8):
        shadow = np.apply_along_axis(rng.permutation, 0, Xp)
        Z = np.hstack([Xp, shadow])
        m = RandomForestClassifier(200, n_jobs=-1,
                                   random_state=int(rng.integers(1 << 31))).fit(Z, y)
        imp = m.feature_importances_
        hits += (imp[:Xp.shape[1]] > imp[Xp.shape[1]:].max()).astype(float)
    return pre[_topk(hits, k)]


# ------------------------------------------------------------ projection
def pca_loading(X, y, k, rng):
    p = PCA(n_components=min(10, X.shape[0] - 1),
            random_state=int(rng.integers(1 << 31))).fit(X)
    load = (np.abs(p.components_).T * p.explained_variance_ratio_).sum(1)
    return _topk(load, k)


def lda_coef(X, y, k, rng):
    pre = _prefilter(X, y)
    m = LinearDiscriminantAnalysis().fit(X[:, pre], y)
    return pre[_topk(np.abs(m.coef_).max(0), k)]


def cca_loading(X, y, k, rng):
    pre = _prefilter(X, y)
    Y = np.zeros((len(y), len(np.unique(y))))
    for i, c in enumerate(np.unique(y)):
        Y[:, i] = (y == c).astype(float)
    m = CCA(n_components=2, max_iter=500).fit(X[:, pre], Y)
    return pre[_topk(np.abs(m.x_weights_).max(1), k)]


# --------------------------------------------- information-theoretic greedy
def mrmr(X, y, k, rng):
    pre = _prefilter(X, y)
    Xp = X[:, pre]
    rel = mutual_info_classif(Xp, y, random_state=0)
    sel = [int(np.argmax(rel))]
    corr = np.abs(np.corrcoef(Xp, rowvar=False))
    np.fill_diagonal(corr, 0)
    while len(sel) < k:
        cand = [j for j in range(Xp.shape[1]) if j not in sel]
        score = [rel[j] - corr[j, sel].mean() for j in cand]
        sel.append(cand[int(np.argmax(score))])
    return pre[np.array(sel)]


def mifs(X, y, k, rng, beta=0.5):
    pre = _prefilter(X, y)
    Xp = X[:, pre]
    rel = mutual_info_classif(Xp, y, random_state=0)
    corr = np.abs(np.corrcoef(Xp, rowvar=False))
    np.fill_diagonal(corr, 0)
    sel = [int(np.argmax(rel))]
    while len(sel) < k:
        cand = [j for j in range(Xp.shape[1]) if j not in sel]
        score = [rel[j] - beta * corr[j, sel].sum() for j in cand]
        sel.append(cand[int(np.argmax(score))])
    return pre[np.array(sel)]


# ------------------------------------------------------- wrappers / search
def rfe(X, y, k, rng):
    pre = _prefilter(X, y)
    cols = list(pre)
    while len(cols) > k:
        m = LogisticRegression(max_iter=1000, random_state=0).fit(X[:, cols], y)
        imp = np.abs(m.coef_).max(0)
        drop = max(1, int(len(cols) * 0.2))
        keep = np.argsort(-imp)[: max(k, len(cols) - drop)]
        cols = [cols[i] for i in sorted(keep)]
    return np.array(cols)


def forward_selection(X, y, k, rng):
    pre = _prefilter(X, y, 150)
    sel, remaining = [], list(range(len(pre)))
    seed = int(rng.integers(1 << 31))
    while len(sel) < k:
        best, best_s = None, -1
        for j in remaining:
            s = _fitness(X[:, pre], y, sel + [j], seed)
            if s > best_s:
                best, best_s = j, s
        sel.append(best)
        remaining.remove(best)
    return pre[np.array(sel)]


def backward_selection(X, y, k, rng):
    pre = _prefilter(X, y, 60)
    cols = list(range(len(pre)))
    seed = int(rng.integers(1 << 31))
    while len(cols) > k:
        scores = [(_fitness(X[:, pre], y, [c for c in cols if c != j], seed), j)
                  for j in cols]
        cols.remove(max(scores)[1])
    return pre[np.array(cols)]


def beam_search(X, y, k, rng, width=3):
    pre = _prefilter(X, y, 80)
    seed = int(rng.integers(1 << 31))
    beam = [[]]
    for _ in range(k):
        cands = []
        for b in beam:
            for j in range(len(pre)):
                if j in b:
                    continue
                cands.append((_fitness(X[:, pre], y, b + [j], seed), b + [j]))
        cands.sort(key=lambda t: -t[0])
        beam = [c[1] for c in cands[:width]]
    return pre[np.array(beam[0])]


# ------------------------------------------------------------ metaheuristics
def _ga(X, y, k, rng, pop_size=20, gens=15):
    pre = _prefilter(X, y)
    n = len(pre)
    seed = int(rng.integers(1 << 31))
    pop = [rng.choice(n, k, replace=False) for _ in range(pop_size)]
    fit = [_fitness(X[:, pre], y, p, seed) for p in pop]
    for _ in range(gens):
        order = np.argsort(fit)[::-1]
        pop = [pop[i] for i in order]
        fit = [fit[i] for i in order]
        children = []
        for _ in range(pop_size // 2):
            a, b = pop[rng.integers(pop_size // 2)], pop[rng.integers(pop_size // 2)]
            union = np.unique(np.concatenate([a, b]))
            child = rng.choice(union, k, replace=False)
            if rng.random() < 0.3:
                child[rng.integers(k)] = rng.integers(n)
            children.append(np.unique(child))
        for c in children:
            if len(c) < k:
                extra = rng.choice([j for j in range(n) if j not in c],
                                   k - len(c), replace=False)
                c = np.concatenate([c, extra])
            s = _fitness(X[:, pre], y, c, seed)
            if s > fit[-1]:
                pop[-1], fit[-1] = c, s
                order = np.argsort(fit)[::-1]
                pop = [pop[i] for i in order]
                fit = [fit[i] for i in order]
    return pre[pop[int(np.argmax(fit))]]


def _swarm(X, y, k, rng, agents=20, iters=15, kind="pso"):
    """Shared skeleton for PSO / ABC / ACO / DE: population of subsets with
    different neighbourhood-update rules."""
    pre = _prefilter(X, y)
    n = len(pre)
    seed = int(rng.integers(1 << 31))
    pop = [rng.choice(n, k, replace=False) for _ in range(agents)]
    fit = np.array([_fitness(X[:, pre], y, p, seed) for p in pop])
    gbest = pop[int(np.argmax(fit))].copy()
    gfit = fit.max()
    pher = np.ones(n)
    for _ in range(iters):
        for i in range(agents):
            cur = pop[i].copy()
            if kind == "pso":
                take = rng.integers(1, k)
                cand = np.unique(np.concatenate(
                    [rng.choice(gbest, take, replace=False),
                     rng.choice(cur, k - take, replace=False)]))
            elif kind == "abc":
                cand = cur.copy()
                cand[rng.integers(k)] = rng.integers(n)
                cand = np.unique(cand)
            elif kind == "aco":
                p = pher / pher.sum()
                cand = rng.choice(n, k, replace=False, p=p)
            else:  # de
                a, b = pop[rng.integers(agents)], pop[rng.integers(agents)]
                pool = np.unique(np.concatenate([cur, a, b]))
                cand = rng.choice(pool, k, replace=False)
            if len(cand) < k:
                extra = rng.choice([j for j in range(n) if j not in cand],
                                   k - len(cand), replace=False)
                cand = np.concatenate([cand, extra])
            s = _fitness(X[:, pre], y, cand, seed)
            if s > fit[i]:
                pop[i], fit[i] = cand, s
                if s > gfit:
                    gbest, gfit = cand.copy(), s
            if kind == "aco":
                pher[cand] += s
        if kind == "aco":
            pher *= 0.9
    return pre[gbest]


def _sa(X, y, k, rng, iters=200):
    pre = _prefilter(X, y)
    n = len(pre)
    seed = int(rng.integers(1 << 31))
    cur = rng.choice(n, k, replace=False)
    cf = _fitness(X[:, pre], y, cur, seed)
    best, bf = cur.copy(), cf
    T = 0.1
    for t in range(iters):
        cand = cur.copy()
        cand[rng.integers(k)] = rng.integers(n)
        cand = np.unique(cand)
        if len(cand) < k:
            continue
        f = _fitness(X[:, pre], y, cand, seed)
        if f > cf or rng.random() < np.exp((f - cf) / max(T, 1e-6)):
            cur, cf = cand, f
            if f > bf:
                best, bf = cand.copy(), f
        T *= 0.98
    return pre[best]


def genetic_algorithm(X, y, k, rng):
    return _ga(X, y, k, rng)


def pso(X, y, k, rng):
    return _swarm(X, y, k, rng, kind="pso")


def abc(X, y, k, rng):
    return _swarm(X, y, k, rng, kind="abc")


def aco(X, y, k, rng):
    return _swarm(X, y, k, rng, kind="aco")


def differential_evolution(X, y, k, rng):
    return _swarm(X, y, k, rng, kind="de")


def simulated_annealing(X, y, k, rng):
    return _sa(X, y, k, rng)


# --------------------------------------------------------------- registry
METHODS = {
    # filter
    "ANOVA": anova,
    "Chi-square": chi_square,
    "Mutual Information": mutual_info,
    "Information Gain": information_gain,
    "Fisher Score": fisher_score,
    "Pearson Correlation": pearson,
    "Spearman Correlation": spearman,
    "Kruskal-Wallis": kruskal,
    "Variance Threshold": variance_threshold,
    # embedded
    "LASSO (L1)": lasso,
    "Elastic Net": elastic_net,
    "Ridge (L2)": ridge_sel,
    "Random Forest Importance": rf_importance,
    "Extra Trees Importance": extra_trees,
    "Permutation Importance": permutation_imp,
    "Stability Selection": stability_selection,
    "Boruta": boruta_lite,
    # projection
    "PCA Loading": pca_loading,
    "LDA Coefficient": lda_coef,
    "CCA": cca_loading,
    # information-theoretic greedy
    "mRMR": mrmr,
    "MIFS": mifs,
    # wrappers
    "RFE": rfe,
    "Forward Selection": forward_selection,
    "Backward Selection": backward_selection,
    "Beam Search": beam_search,
    # metaheuristics
    "Genetic Algorithm": genetic_algorithm,
    "PSO": pso,
    "ABC": abc,
    "ACO": aco,
    "Differential Evolution": differential_evolution,
    "Simulated Annealing": simulated_annealing,
}

FAMILY = {  # used only for reporting; redundancy is measured empirically
    "ANOVA": "filter", "Chi-square": "filter", "Mutual Information": "filter",
    "Information Gain": "filter", "Fisher Score": "filter",
    "Pearson Correlation": "filter", "Spearman Correlation": "filter",
    "Kruskal-Wallis": "filter", "Variance Threshold": "filter",
    "LASSO (L1)": "embedded", "Elastic Net": "embedded", "Ridge (L2)": "embedded",
    "Random Forest Importance": "embedded", "Extra Trees Importance": "embedded",
    "Permutation Importance": "embedded", "Stability Selection": "embedded",
    "Boruta": "embedded",
    "PCA Loading": "projection", "LDA Coefficient": "projection", "CCA": "projection",
    "mRMR": "information", "MIFS": "information",
    "RFE": "wrapper", "Forward Selection": "wrapper",
    "Backward Selection": "wrapper", "Beam Search": "wrapper",
    "Genetic Algorithm": "metaheuristic", "PSO": "metaheuristic",
    "ABC": "metaheuristic", "ACO": "metaheuristic",
    "Differential Evolution": "metaheuristic", "Simulated Annealing": "metaheuristic",
}


# ------------------------------------------------- reference implementations
# When minepy / boruta_py are installed, add MIC and replace the built-in
# Boruta with the reference package (see fs_methods_reference.py).
try:
    from fs_methods_reference import REFERENCE_METHODS
    for _name, _fn in REFERENCE_METHODS.items():
        METHODS[_name] = _fn
        FAMILY.setdefault(_name, "filter" if _name == "MIC" else "embedded")
except ImportError:
    pass
