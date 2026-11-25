"""
em_gmm.py  -  EM for a mixture of K one-dimensional Gaussians.

Two versions of the E-step are provided:

  faulty    exactly the pseudocode given in the assignment:
                r[i][k] <- pi[k] * Normal(X[i] | mu[k], var[k])          (NOT normalised over k)
  repaired  the ONLY change is the extra normalisation line
                r[i][k] <- r[i][k] / SUM_j r[i][j]

Everything else (M-step formulas, initialisation, loop structure) is identical, so the
experiment isolates the effect of the single missing line.
"""
import numpy as np

VAR_FLOOR = 1e-6          # numerical guard, identical in both versions


def normal_pdf(x, mu, var):
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


def log_likelihood(X, pi, mu, var):
    """log p(X) = SUM_i log SUM_k pi_k N(x_i | mu_k, var_k).   Same function for both versions."""
    dens = np.array([pi[k] * normal_pdf(X, mu[k], var[k]) for k in range(len(pi))])   # K x n
    return float(np.sum(np.log(np.maximum(dens.sum(axis=0), 1e-300))))


def e_step(X, pi, mu, var, normalise):
    r = np.array([pi[k] * normal_pdf(X, mu[k], var[k]) for k in range(len(pi))]).T      # n x K
    if normalise:                                                                       # <-- THE REPAIR
        r = r / np.maximum(r.sum(axis=1, keepdims=True), 1e-300)
    return r


def m_step(X, r):
    n = len(X)
    Nk = r.sum(axis=0)
    mu = (r * X[:, None]).sum(axis=0) / Nk
    var = np.maximum((r * (X[:, None] - mu[None, :]) ** 2).sum(axis=0) / Nk, VAR_FLOOR)
    pi = Nk / n
    return pi, mu, var, Nk


def fit_em(X, K, T, init, repaired):
    """Run T EM iterations from init=(pi, mu, var). Returns final params and per-iteration history."""
    pi, mu, var = [np.array(a, dtype=float).copy() for a in init]
    hist = dict(ll=[log_likelihood(X, pi, mu, var)], pi_sum=[pi.sum()], mu=[mu.copy()], var=[var.copy()], pi=[pi.copy()])
    for _ in range(T):
        r = e_step(X, pi, mu, var, normalise=repaired)
        pi, mu, var, _ = m_step(X, r)
        hist["ll"].append(log_likelihood(X, pi, mu, var))
        hist["pi_sum"].append(float(pi.sum()))
        hist["mu"].append(mu.copy()); hist["var"].append(var.copy()); hist["pi"].append(pi.copy())
    return pi, mu, var, hist


def one_iteration_trace(X, init, repaired):
    """Return every intermediate quantity of ONE iteration (used for the manual trace of Task 3.2)."""
    pi, mu, var = [np.array(a, dtype=float) for a in init]
    r = e_step(X, pi, mu, var, normalise=repaired)
    pi2, mu2, var2, Nk = m_step(X, r)
    return dict(r=r, row_sums=r.sum(axis=1), Nk=Nk, mu=mu2, var=var2, pi=pi2, pi_sum=pi2.sum())


def random_init(X, K, rng):
    """Random initialisation: K distinct data points as means, global variance, equal weights."""
    mu = rng.choice(X, size=K, replace=False)
    var = np.full(K, X.var())
    pi = np.full(K, 1.0 / K)
    return pi, mu, var
