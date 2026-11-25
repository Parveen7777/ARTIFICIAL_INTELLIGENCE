"""
smartgrid.py  -  synthetic substation records + from-scratch Naive Bayes (maximum-likelihood counts, Laplace smoothing).

Dataset (documented assumptions)
    load_pct      substation load as % of rated capacity          U(20, 100)
    voltage_dev   voltage deviation from nominal, %                U(-8, +8)
    temp_c        transformer temperature, deg C                   30 + 0.35*load + N(0, 5)
    thd_pct       total harmonic distortion, %                     2 + 0.15*|voltage_dev| + N(0, 1)
    hazardous     1 if the operating point leaves the SAFE ELLIPSE ((load-60)/32)^2 + (voltage/6)^2 <= 1
                  (so hazardous and normal states are NOT linearly separable), then 3 % of labels flipped.
"""
import numpy as np

FEATURES = ["load_pct", "voltage_dev", "temp_c", "thd_pct"]


def make_smart_grid(n=1500, seed=0, noise=0.03):
    rng = np.random.RandomState(seed)
    load = rng.uniform(20, 100, n)
    volt = rng.uniform(-8, 8, n)
    temp = 30 + 0.35 * load + rng.normal(0, 5, n)
    thd = 2 + 0.15 * np.abs(volt) + rng.normal(0, 1, n)
    y = (((load - 60) / 32) ** 2 + (volt / 6) ** 2 > 1).astype(int)
    flip = rng.rand(n) < noise
    y = np.where(flip, 1 - y, y)
    return np.column_stack([load, volt, temp, thd]), y


class QuantileDiscretizer:
    """Equal-frequency binning; bin edges are learnt on TRAINING data only."""
    def __init__(self, n_bins=10):
        self.n_bins = n_bins

    def fit(self, X):
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges_ = [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]
        self.n_values_ = [len(e) + 1 for e in self.edges_]
        return self

    def transform(self, X):
        return np.column_stack([np.searchsorted(self.edges_[j], X[:, j], side="right") for j in range(X.shape[1])])


class CategoricalNaiveBayes:
    """
    Parameter estimation:   P(c) = N_c / N                      (maximum likelihood)
                            P(x_j = v | c) = (N_{j,v,c} + alpha) / (N_c + alpha * V_j)     (alpha = 0 -> pure ML)
    Classification:         argmax_c  log P(c) + SUM_j log P(x_j | c)
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y, n_values):
        self.classes_ = np.unique(y)
        self.n_values_ = list(n_values)
        n = len(y)
        self.class_count_ = np.array([(y == c).sum() for c in self.classes_], float)
        self.log_prior_ = np.log(self.class_count_ / n)
        self.count_ = []                                   # count_[j][c, v]
        for j, V in enumerate(self.n_values_):
            cnt = np.zeros((len(self.classes_), V))
            for ci, c in enumerate(self.classes_):
                cnt[ci] = np.bincount(X[y == c, j], minlength=V)
            self.count_.append(cnt)
        self.zero_counts_ = int(sum((c == 0).sum() for c in self.count_))
        return self

    def _log_likelihood_tables(self):
        tabs = []
        with np.errstate(divide="ignore"):
            for j, V in enumerate(self.n_values_):
                num = self.count_[j] + self.alpha
                den = self.class_count_[:, None] + self.alpha * V
                tabs.append(np.log(num / den))            # log(0) = -inf when alpha = 0 and the count is 0
        return tabs

    def predict_proba(self, X):
        """Posterior probabilities.  Returns (proba, n_undefined): rows where every class has zero likelihood
        (only possible with alpha = 0) are undefined; they are given the class prior as a documented fall-back."""
        tabs = self._log_likelihood_tables()
        scores = np.tile(self.log_prior_, (len(X), 1))
        for j in range(len(self.n_values_)):
            scores = scores + tabs[j][:, X[:, j]].T
        proba = np.zeros_like(scores)
        undefined = np.isneginf(scores).all(axis=1)
        for i in range(len(X)):
            if undefined[i]:
                proba[i] = np.exp(self.log_prior_)
            else:
                s = scores[i] - scores[i].max()
                e = np.exp(s)
                proba[i] = e / e.sum()
        return proba, int(undefined.sum())

    def predict(self, X):
        p, _ = self.predict_proba(X)
        return self.classes_[p.argmax(axis=1)]
