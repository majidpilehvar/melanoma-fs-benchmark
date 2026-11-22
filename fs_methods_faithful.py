"""
Faithful re-implementations of the three components that the main analysis
replaced with simpler substitutes. They need only NumPy, SciPy and
scikit-learn, so they run anywhere and are used for the robustness analysis
(sensitivity.py).

boruta_full    The complete Boruta algorithm as in boruta_py (Kursa & Rudnicki
               2010): shadow features, per-iteration hits against the maximum
               shadow importance, two-step binomial testing (Benjamini-Hochberg
               then Bonferroni over iterations), removal of rejected features,
               and the 'auto' tree-count rule.

mic_exact      Maximal Information Coefficient (Reshef et al. 2011) with
               B(n) = n^0.6, computed EXACTLY for a categorical target. With
               three ordered classes the target axis admits only three
               contiguous partitions; for each, the optimal partition of the
               gene axis into at most B/ny bins is found by dynamic programming
               over class-boundary points, which is exact for entropy
               minimisation (Fayyad & Irani 1992). minepy's ApproxMaxMI is a
               heuristic for the general continuous case; here no heuristic is
               needed.

DropoutMLP     The thesis Keras network (150-50-20 ReLU, dropout 0.5, softmax,
               Adam lr 1e-3, batch 8, up to 500 epochs) in NumPy, with Glorot
               uniform initialisation as in Keras and early stopping on a
               stratified 15% split of the training fold.
"""
import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif


def _prefilter(X, y, n):
    F, _ = f_classif(X, y)
    return np.argsort(-np.nan_to_num(F, nan=-np.inf))[:min(n, X.shape[1])]


def _bh(p, alpha):
    """Benjamini-Hochberg rejection mask (as statsmodels fdrcorrection)."""
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return np.zeros(0, bool)
    order = np.argsort(p)
    thresh = alpha * np.arange(1, n + 1) / n
    below = p[order] <= thresh
    k = np.max(np.where(below)[0]) + 1 if below.any() else 0
    mask = np.zeros(n, bool)
    mask[order[:k]] = True
    return mask


# --------------------------------------------------------------- Boruta
def _tree_num(n_feat, depth=5, f_repr=100):
    multi = (n_feat * 2) / (np.sqrt(n_feat * 2) * depth)
    return int(multi * f_repr)


def boruta_full(X, y, k, rng, prefilter=300, max_iter=50, alpha=0.05, max_depth=5):
    pre = _prefilter(X, y, prefilter)
    Xp = X[:, pre]
    n_feat = Xp.shape[1]
    dec = np.zeros(n_feat, int)     # 0 tentative, 1 confirmed, -1 rejected
    hits = np.zeros(n_feat, int)
    it = 1
    while np.any(dec == 0) and it < max_iter:
        active = np.where(dec >= 0)[0]
        Xa = Xp[:, active]
        shadow = np.apply_along_axis(rng.permutation, 0, Xa)
        rf = RandomForestClassifier(n_estimators=max(_tree_num(len(active), max_depth), 50),
                                    max_depth=max_depth, class_weight="balanced",
                                    n_jobs=-1, random_state=int(rng.integers(1 << 31)))
        rf.fit(np.hstack([Xa, shadow]), y)
        imp = rf.feature_importances_
        real, sha = imp[:len(active)], imp[len(active):]
        hits[active[real > sha.max()]] += 1
        # two-step test exactly as boruta_py._do_tests
        h = hits[active]
        p_acc = stats.binom.sf(h - 1, it, 0.5)
        p_rej = stats.binom.cdf(h, it, 0.5)
        acc = _bh(p_acc, alpha) & (p_acc <= alpha / it)
        rej = _bh(p_rej, alpha) & (p_rej <= alpha / it)
        tent = dec[active] == 0
        dec[active[acc & tent]] = 1
        dec[active[rej & tent]] = -1
        it += 1
    # confirmed first, then tentative, then rejected; ties broken by hits
    score = hits + 10_000 * (dec == 1) + 5_000 * (dec == 0)
    return pre[np.argsort(-score, kind="stable")[:k]]


# ------------------------------------------------------------------ MIC
def _best_partition_mi(x, q, kmax):
    """Max I(P;Q) over contiguous partitions P of sorted x into <= kmax bins.
    Returns array best[k] for k = 1..kmax (natural-log mutual information)."""
    order = np.argsort(x, kind="stable")
    qs = q[order]
    labels = np.unique(qs)
    n = len(qs)
    # clumps: maximal runs of identical label in x order (boundary points)
    change = np.r_[True, qs[1:] != qs[:-1]]
    starts = np.where(change)[0]
    m = len(starts)
    counts = np.zeros((m, len(labels)))
    ends = np.r_[starts[1:], n]
    for c, lab in enumerate(labels):
        csum = np.r_[0, np.cumsum(qs == lab)]
        counts[:, c] = csum[ends] - csum[starts]
    cum = np.vstack([np.zeros(len(labels)), np.cumsum(counts, 0)])   # (m+1, L)
    # cost[i, j] = n_ij * H(Q | clumps i..j-1), for 0 <= i < j <= m
    seg = cum[None, :, :] - cum[:, None, :]                           # (m+1, m+1, L)
    tot = seg.sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tot[..., None] > 0, seg / tot[..., None], 0)
        ent = -(np.where(p > 0, p * np.log(p), 0)).sum(-1)
    cost = tot * ent
    cost[np.tril_indices(m + 1)] = np.inf
    pq = counts.sum(0) / n
    hq = -(pq[pq > 0] * np.log(pq[pq > 0])).sum()
    dp = cost[0].copy()                 # 1 bin ending at clump boundary j
    best = np.empty(kmax)
    best[0] = hq - dp[m] / n
    for kk in range(2, kmax + 1):
        dp = np.min(dp[:, None] + cost, axis=0)
        best[kk - 1] = hq - dp[m] / n
    return best


def mic_score(x, y, alpha=0.6):
    """Exact MIC between a continuous x and an ordinal categorical y."""
    n = len(x)
    B = n ** alpha
    classes = np.unique(y)
    L = len(classes)
    yi = np.searchsorted(classes, y)
    best = 0.0
    # contiguous partitions of the ordered classes into ny groups
    from itertools import combinations
    for ny in range(2, L + 1):
        kmax = int(np.floor(B / ny))
        if kmax < 2:
            continue
        for cuts in combinations(range(1, L), ny - 1):
            q = np.searchsorted(np.array(cuts), yi, side="right")
            mi = _best_partition_mi(x, q, kmax)
            for nx in range(2, kmax + 1):
                best = max(best, mi[nx - 1] / np.log(min(nx, ny)))
    return min(best, 1.0)


def mic_exact(X, y, k, rng, prefilter=2000):
    pre = _prefilter(X, y, prefilter)
    s = np.array([mic_score(X[:, j], y) for j in pre])
    return pre[np.argsort(-s, kind="stable")[:k]]


# ------------------------------------------------------------- network
class DropoutMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden=(150, 50, 20), dropout=0.5, lr=1e-3, batch_size=8,
                 epochs=500, patience=30, val_frac=0.15, random_state=42):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.val_frac = val_frac
        self.random_state = random_state

    def _forward(self, X, train, rng):
        acts, masks = [X], []
        h = X
        for i, (W, b) in enumerate(self.params_[:-1]):
            h = np.maximum(h @ W + b, 0)
            if train:
                m = ((rng.random(h.shape) >= self.dropout) / (1 - self.dropout)).astype(np.float32)
                h = h * m
                masks.append(m)
            acts.append(h)
        W, b = self.params_[-1]
        z = h @ W + b
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        return p, acts, masks

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        self.classes_ = np.unique(y)
        yi = np.searchsorted(self.classes_, y)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-9
        Xs = ((X - self.mu_) / self.sd_).astype(np.float32)
        # stratified validation split (Keras would take the last rows, which
        # in a class-sorted matrix is a single class)
        val = np.zeros(len(yi), bool)
        for c in np.unique(yi):
            idx = np.where(yi == c)[0]
            val[rng.choice(idx, max(1, int(round(self.val_frac * len(idx)))), replace=False)] = True
        Xt, yt, Xv, yv = Xs[~val], yi[~val], Xs[val], yi[val]
        sizes = [Xs.shape[1], *self.hidden, len(self.classes_)]
        self.params_ = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            lim = np.sqrt(6 / (a + b))                      # Glorot uniform
            self.params_.append([rng.uniform(-lim, lim, (a, b)).astype(np.float32),
                                 np.zeros(b, np.float32)])
        m1 = [[np.zeros_like(W), np.zeros_like(b)] for W, b in self.params_]
        m2 = [[np.zeros_like(W), np.zeros_like(b)] for W, b in self.params_]
        b1, b2, eps, step = 0.9, 0.999, 1e-7, 0
        Y = np.eye(len(self.classes_))
        best_loss, best_params, wait = np.inf, None, 0
        for ep in range(self.epochs):
            perm = rng.permutation(len(yt))
            for s in range(0, len(perm), self.batch_size):
                bi = perm[s:s + self.batch_size]
                p, acts, masks = self._forward(Xt[bi], True, rng)
                d = (p - Y[yt[bi]]) / len(bi)
                grads = []
                for li in range(len(self.params_) - 1, -1, -1):
                    W, _ = self.params_[li]
                    grads.append((acts[li].T @ d, d.sum(0)))
                    if li > 0:
                        d = (d @ W.T) * masks[li - 1] * (acts[li] > 0)
                grads = grads[::-1]
                step += 1
                for li, (gW, gb) in enumerate(grads):
                    for j, g in enumerate((gW, gb)):
                        m1[li][j] = b1 * m1[li][j] + (1 - b1) * g
                        m2[li][j] = b2 * m2[li][j] + (1 - b2) * g * g
                        mh = m1[li][j] / (1 - b1 ** step)
                        vh = m2[li][j] / (1 - b2 ** step)
                        self.params_[li][j] -= self.lr * mh / (np.sqrt(vh) + eps)
            pv, _, _ = self._forward(Xv, False, rng)
            loss = -np.mean(np.log(pv[np.arange(len(yv)), yv] + 1e-12))
            if loss < best_loss - 1e-6:
                best_loss, wait = loss, 0
                best_params = [[W.copy(), b.copy()] for W, b in self.params_]
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.params_ = best_params
        self.n_epochs_ = ep + 1
        return self

    def predict_proba(self, X):
        p, _, _ = self._forward(((X - self.mu_) / self.sd_).astype(np.float32), False, None)
        return p

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]
