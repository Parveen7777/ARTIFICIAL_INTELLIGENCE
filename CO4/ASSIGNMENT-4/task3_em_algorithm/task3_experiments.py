"""
TASK 3.2 / 3.4 / 3.5  -  trace, faulty-vs-repaired experiment, regression check.

Run:  python task3_em_algorithm/task3_experiments.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from em_gmm import fit_em, one_iteration_trace, random_init, log_likelihood  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
np.set_printoptions(precision=4, suppress=True)


def show_trace(title, X, init, repaired):
    t = one_iteration_trace(X, init, repaired)
    print("\n--- %s ---" % title)
    print("responsibilities r[i][k] (columns = component 1, 2):")
    for i, (row, s) in enumerate(zip(t["r"], t["row_sums"]), 1):
        print("   x%d=%.1f  r = [%.6f, %.6f]   row-sum = %.6f %s" % (i, X[i - 1], row[0], row[1], s,
                                                                      "" if abs(s - 1) < 1e-9 else "  <-- should be 1"))
    print("   N_k = %s   (SUM of N_k should equal n = %d, here %.4f)" % (t["Nk"], len(X), t["Nk"].sum()))
    print("   mu_new = %s   var_new = %s" % (t["mu"], t["var"]))
    print("   pi_new = %s   SUM pi = %.6f %s" % (t["pi"], t["pi_sum"], "" if abs(t["pi_sum"] - 1) < 1e-9 else "  <-- should be 1"))


def main():
    # ------------------------------------------------------------------ Task 3.2  student-designed failing input
    X = np.array([1.0, 1.2, 4.8, 5.0])
    init = ([0.5, 0.5], [1.0, 5.0], [1.0, 1.0])
    print("=" * 78)
    print("TASK 3.2  FAILING INPUT  X = %s  K = 2   pi0 = (0.5, 0.5)  mu0 = (1, 5)  var0 = (1, 1)" % X)
    print("=" * 78)
    show_trace("FAULTY version, iteration 1", X, init, repaired=False)
    show_trace("REPAIRED version, iteration 1", X, init, repaired=True)

    # ------------------------------------------------------------------ Task 3.4  20 random initialisations
    rng_data = np.random.RandomState(7)
    data = np.concatenate([rng_data.normal(-2.0, 0.8, 150), rng_data.normal(3.0, 1.2, 100)])
    K, T = 2, 60
    print("\n" + "=" * 78)
    print("TASK 3.4  DATA: 250 points from 0.6*N(-2,0.8^2)+0.4*N(3,1.2^2)   K=%d  T=%d  20 random initialisations" % (K, T))
    print("=" * 78)
    res = {False: [], True: []}
    for seed in range(20):
        init = random_init(data, K, np.random.RandomState(seed))
        for rep in (False, True):
            res[rep].append(fit_em(data, K, T, init, repaired=rep))

    def summarize(rep):
        runs = res[rep]
        monotone = sum(bool((np.diff(h["ll"]) >= -1e-9).all()) for *_, h in runs)
        final_ll = np.array([h["ll"][-1] for *_, h in runs])
        final_sum = np.array([h["pi_sum"][-1] for *_, h in runs])
        worst_drop = min(np.diff(h["ll"]).min() for *_, h in runs)
        means_sorted = np.array([np.sort(m) for _, m, _, _ in runs])
        return monotone, final_ll, final_sum, worst_drop, means_sorted

    for rep in (False, True):
        mono, fl, fs, wd, ms = summarize(rep)
        print("\n%s version:" % ("REPAIRED" if rep else "FAULTY  "))
        print("  runs whose log-likelihood never decreased : %d / 20" % mono)
        print("  largest single-iteration drop in log-lik  : %.4f" % wd)
        print("  final log-likelihood  mean=%.2f  min=%.2f  max=%.2f" % (fl.mean(), fl.min(), fl.max()))
        print("  final SUM of mixing weights  mean=%.3e  min=%.3e  max=%.3e" % (fs.mean(), fs.min(), fs.max()))
        print("  recovered means (sorted) mean = %s   (true = [-2, 3])" % ms.mean(axis=0))
        print("  recovered means std over inits = %s" % ms.std(axis=0))
    pi_r, mu_r, var_r, _ = res[True][0]
    print("\nREPAIRED seed-0 final parameters: pi=%s mu=%s var=%s" % (pi_r, mu_r, var_r))
    pi_f, mu_f, var_f, _ = res[False][0]
    print("FAULTY   seed-0 final parameters: pi=%s mu=%s var=%s" % (pi_f, mu_f, var_f))

    # figure
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for rep, col in ((False, 0), (True, 1)):
        name = "Repaired" if rep else "Faulty"
        for *_, h in res[rep]:
            ax[0, col].plot(h["ll"], alpha=.55, lw=1)
            ax[1, col].plot(h["pi_sum"], alpha=.55, lw=1)
        ax[0, col].set_title("%s: log-likelihood per iteration (20 inits)" % name)
        ax[0, col].set_xlabel("iteration"); ax[0, col].set_ylabel("log-likelihood")
        ax[1, col].set_title("%s: sum of mixing weights" % name)
        ax[1, col].set_xlabel("iteration"); ax[1, col].set_ylabel("sum of pi"); ax[1, col].axhline(1, color="k", ls="--", lw=1)
        ax[1, col].set_yscale("log" if not rep else "linear")
        for a in ax[:, col]:
            a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "task3_loglik_and_weights.png"), dpi=160); plt.close(fig)

    # ------------------------------------------------------------------ Task 3.5  regression check
    print("\n" + "=" * 78)
    print("TASK 3.5  REGRESSION CHECK - an input on which the original already behaved correctly")
    print("=" * 78)
    Xr = np.concatenate([np.full(5, -3.0) + np.linspace(-.1, .1, 5), np.full(5, 3.0) + np.linspace(-.1, .1, 5)])
    initr = ([0.5, 0.5], [-3.0, 3.0], [0.1, 0.1])
    print("Well-separated, balanced, tight clusters near -3 and +3; initial means placed on the clusters.")
    f = fit_em(Xr, 2, 30, initr, repaired=False)
    g = fit_em(Xr, 2, 30, initr, repaired=True)
    rf = np.argmax(np.array([f[0][k] * np.exp(-0.5 * (Xr - f[1][k]) ** 2 / f[2][k]) / np.sqrt(f[2][k]) for k in range(2)]), axis=0)
    rg = np.argmax(np.array([g[0][k] * np.exp(-0.5 * (Xr - g[1][k]) ** 2 / g[2][k]) / np.sqrt(g[2][k]) for k in range(2)]), axis=0)
    print("  faulty   : mu=%s  hard assignments=%s" % (f[1], rf))
    print("  repaired : mu=%s  hard assignments=%s  pi=%s" % (g[1], rg, g[0]))
    print("  hard assignments identical      : %s" % bool((rf == rg).all()))
    print("  max |mu_faulty - mu_repaired|   : %.2e" % np.abs(f[1] - g[1]).max())
    print("  repaired log-likelihood non-decreasing: %s" % bool((np.diff(g[3]["ll"]) >= -1e-9).all()))
    print("  sum(pi): faulty=%.3e   repaired=%.6f" % (f[0].sum(), g[0].sum()))
    print("  -> The original recovers the right cluster means and labels here, so it 'looks fine'; but only the")
    print("     repaired version also returns weights that sum to 1 (0.5 / 0.5).")


if __name__ == "__main__":
    main()
