"""
Reference implementations of the three methods that were substituted in the
offline run. They activate automatically when the libraries are installed:

    pip install minepy boruta tensorflow

If any import fails, the pipeline silently keeps the substitutes defined in
fs_methods.py and step.py, so results remain reproducible either way.

Notes
-----
* boruta_py 0.3 uses the removed alias `np.int`; with NumPy >= 1.24 install
  the patched release (`pip install boruta>=0.4`) or pin `numpy<1.24`.
* These functions follow the same contract as fs_methods.py:
  fn(X, y, k, rng) -> indices, fitted on the training fold only.
* Not executed in the environment that produced the manuscript tables; run
  `python3 step.py fold 0` once after installing and compare with
  results/ckpt/fold0.json before re-running everything.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import f_classif

REFERENCE_METHODS = {}


def _prefilter(X, y, n):
    F, _ = f_classif(X, y)
    return np.argsort(-np.nan_to_num(F, nan=-np.inf))[:min(n, X.shape[1])]


# ------------------------------------------------------------------ MIC
try:
    from minepy import MINE

    def mic(X, y, k, rng):
        """Maximal Information Coefficient (Reshef et al., 2011).
        MINE is O(n^1.2) per pair; pre-filter to 2,000 genes within the fold."""
        pre = _prefilter(X, y, 2000)
        m = MINE(alpha=0.6, c=15)
        yv = y.astype(float)
        s = np.empty(len(pre))
        for i, j in enumerate(pre):
            m.compute_score(X[:, j], yv)
            s[i] = m.mic()
        return pre[np.argsort(-s)[:k]]

    REFERENCE_METHODS["MIC"] = mic
except ImportError:
    pass


# --------------------------------------------------------------- Boruta
try:
    from boruta import BorutaPy
    from sklearn.ensemble import RandomForestClassifier

    def boruta_reference(X, y, k, rng):
        """boruta_py (Kursa & Rudnicki, 2010) on a 1,000-gene in-fold shortlist."""
        pre = _prefilter(X, y, 1000)
        rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5,
                                    random_state=int(rng.integers(1 << 31)))
        b = BorutaPy(rf, n_estimators="auto", max_iter=100,
                     random_state=int(rng.integers(1 << 31)), verbose=0)
        b.fit(X[:, pre], y)
        order = np.lexsort((-b.support_.astype(int), b.ranking_))
        return pre[order[:k]]

    REFERENCE_METHODS["Boruta"] = boruta_reference
except ImportError:
    pass


# ------------------------------------------------------------- Keras MLP
try:
    import tensorflow as tf
    from tensorflow import keras

    class KerasMLP(BaseEstimator, ClassifierMixin):
        """Keras network matching the thesis architecture (150-50-20, dropout
        0.5), with early stopping on an internal validation split so that the
        held-out fold is never used for stopping."""

        def __init__(self, epochs=500, batch_size=8, random_state=42):
            self.epochs = epochs
            self.batch_size = batch_size
            self.random_state = random_state

        def fit(self, X, y):
            tf.keras.utils.set_random_seed(self.random_state)
            self.classes_ = np.unique(y)
            yi = np.searchsorted(self.classes_, y)
            self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-9
            Xs = (X - self.mu_) / self.sd_
            m = keras.Sequential([
                keras.layers.Input(shape=(X.shape[1],)),
                keras.layers.Dense(150, activation="relu"), keras.layers.Dropout(0.5),
                keras.layers.Dense(50, activation="relu"), keras.layers.Dropout(0.5),
                keras.layers.Dense(20, activation="relu"), keras.layers.Dropout(0.5),
                keras.layers.Dense(len(self.classes_), activation="softmax")])
            m.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
            m.fit(Xs, yi, epochs=self.epochs, batch_size=self.batch_size,
                  validation_split=0.15, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(patience=30,
                                                           restore_best_weights=True)])
            self.model_ = m
            return self

        def predict_proba(self, X):
            return self.model_.predict((X - self.mu_) / self.sd_, verbose=0)

        def predict(self, X):
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

except ImportError:
    KerasMLP = None
