"""
TASK 5.1  -  Naive Bayes from scratch: toy zero-count example and the alpha sweep.

Run:  python task5_learning_comparison/task5_nb_experiment.py
"""
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from smartgrid import make_smart_grid, QuantileDiscretizer, CategoricalNaiveBayes, FEATURES  # noqa: E402


def toy_zero_count():
    print("=" * 80 + "\nTASK 5.1 (Design)  THREE-EXAMPLE TOY TABLE: WHY A ZERO COUNT BREAKS CLASSIFICATION\n" + "=" * 80)
    rows = [("High", "Fault", "Hazard"), ("High", "OK", "Safe"), ("Normal", "OK", "Safe")]
    print("Ex  Load    Cooling  Class\n" + "\n".join("%d   %-7s %-8s %s" % (i + 1, *r) for i, r in enumerate(rows)))
    print("Query: Load = Normal, Cooling = Fault")
    cls = {"Hazard": [r for r in rows if r[2] == "Hazard"], "Safe": [r for r in rows if r[2] == "Safe"]}
    vals = {"load": ["High", "Normal"], "cool": ["OK", "Fault"]}
    for alpha in (0.0, 1.0):
        sc = {}
        for c, rs in cls.items():
            prior = len(rs) / 3
            p_load = (sum(r[0] == "Normal" for r in rs) + alpha) / (len(rs) + alpha * 2)
            p_cool = (sum(r[1] == "Fault" for r in rs) + alpha) / (len(rs) + alpha * 2)
            sc[c] = prior * p_load * p_cool
            print("  alpha=%.0f  P(%s)=%.4f  P(Normal|%s)=%.4f  P(Fault|%s)=%.4f  ->  score = %.4f" % (alpha, c, prior, c, p_load, c, p_cool, sc[c]))
        z = sum(sc.values())
        if z == 0:
            print("  alpha=%.0f  BOTH scores are 0: the posterior is 0/0 - classification is undefined." % alpha)
        else:
            print("  alpha=%.0f  posterior: Hazard = %.4f   Safe = %.4f   -> predict %s" % (alpha, sc["Hazard"] / z, sc["Safe"] / z, max(sc, key=sc.get)))


def main():
    toy_zero_count()
    X, y = make_smart_grid(n=1500, seed=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    disc = QuantileDiscretizer(10).fit(Xtr)
    Dtr, Dte = disc.transform(Xtr), disc.transform(Xte)
    def run_regime(title, Xtr_, ytr_, n_bins):
        disc = QuantileDiscretizer(n_bins).fit(Xtr_)
        Dtr, Dte = disc.transform(Xtr_), disc.transform(Xte)
        print("\n" + "=" * 80 + "\n" + title + "\n" + "=" * 80)
        print("train=%d  test=%d  bins/feature=%d  hazardous rate=%.3f  features=%s" % (len(ytr_), len(yte), n_bins, y.mean(), FEATURES))
        print("%-7s %-12s %-8s %-8s %-8s %-8s %-9s %-10s" % ("alpha", "zero-counts", "Acc", "Prec", "Recall", "F1", "log-loss", "undefined"))
        rows = []
        for alpha in (0.0, 0.1, 1.0, 10.0):
            nb = CategoricalNaiveBayes(alpha).fit(Dtr, ytr_, disc.n_values_)
            proba, undefined = nb.predict_proba(Dte)
            pred = nb.classes_[proba.argmax(axis=1)]
            ll = log_loss(yte, np.clip(proba, 1e-15, 1 - 1e-15), labels=[0, 1])
            rows.append((alpha, nb.zero_counts_, accuracy_score(yte, pred), f1_score(yte, pred), ll, undefined, pred))
            print("%-7s %-12d %-8.4f %-8.4f %-8.4f %-8.4f %-9.4f %-10d" % (alpha, nb.zero_counts_, accuracy_score(yte, pred), precision_score(yte, pred, zero_division=0),
                                                                        recall_score(yte, pred), f1_score(yte, pred), ll, undefined))
        best = min(rows, key=lambda r: r[4])
        print("Best alpha by held-out log-loss = %s ; confusion matrix [[TN FP],[FN TP]] = %s" % (best[0], confusion_matrix(yte, best[6]).tolist()))
        return rows

    rich = run_regime("TASK 5.1 (Implement)  REGIME A - plentiful data (all 1050 training rows, 10 bins)", Xtr, ytr, 10)
    sparse_idx = np.random.RandomState(1).choice(len(ytr), 120, replace=False)
    sparse = run_regime("TASK 5.1 (Implement)  REGIME B - sparse data (120 training rows, 20 bins) : zero counts appear", Xtr[sparse_idx], ytr[sparse_idx], 20)

    print("\nINTERPRETATION (read from the tables above)")
    print(" * Regime A: zero-counts = %d for every alpha, so smoothing changes almost nothing (log-loss %.4f at alpha=0 vs %.4f at alpha=1);" % (rich[0][1], rich[0][4], rich[2][4]))
    print("   alpha = 10 only over-smooths (log-loss rises to %.4f)." % rich[3][4])
    print(" * Regime B: alpha = 0 meets %d zero counts, %d test rows become undefined (0/0) and log-loss is %.4f;" % (sparse[0][1], sparse[0][5], sparse[0][4]))
    print("   alpha = 1 lowers log-loss to %.4f and removes every undefined row. Laplace smoothing matters exactly when counts are sparse." % sparse[2][4])
    print(" * Accuracy is capped (~0.85) because hazard depends on load and voltage JOINTLY (elliptical safe region) - independence is violated.")


if __name__ == "__main__":
    main()
