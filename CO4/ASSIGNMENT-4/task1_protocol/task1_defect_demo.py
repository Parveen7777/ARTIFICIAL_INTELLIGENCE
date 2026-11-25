"""
TASK 1.3  -  Demonstrate the defect of the junior engineer's protocol.

The original pseudocode trains and tests on the SAME data.  This script
(1) reproduces that protocol, (2) shows how optimistic it is compared with
cross-validated and held-out estimates, (3) shows why accuracy alone misleads
on imbalanced data (majority-class baseline) and (4) implements the REPAIRED
protocol of Task 1.2 end to end.

Run:  python task1_protocol/task1_defect_demo.py
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

SEED = 42


def make_student_data(seed=SEED):
    """Synthetic 'student at-risk' data: 1000 students, 8 features, ~15 % at-risk, 8 % label noise."""
    X, y = make_classification(n_samples=1000, n_features=8, n_informative=4, n_redundant=2,
                               weights=[0.85, 0.15], flip_y=0.08, class_sep=0.8,
                               random_state=seed)
    return X, y


def metrics(y_true, y_pred):
    return dict(acc=accuracy_score(y_true, y_pred),
                prec=precision_score(y_true, y_pred, zero_division=0),
                rec=recall_score(y_true, y_pred, zero_division=0),
                f1=f1_score(y_true, y_pred, zero_division=0))


def main():
    X, y = make_student_data()
    print("=" * 74)
    print("DATASET: synthetic student at-risk data  n=%d  features=%d  at-risk rate=%.3f" % (len(y), X.shape[1], y.mean()))
    print("=" * 74)

    # ---------------- A. The DEFECTIVE protocol (train == test) ----------------
    print("\n[A] DEFECTIVE protocol  (Read Data -> Train on Data -> Test on Data -> PRINT accuracy)")
    models = {"Decision tree (unpruned)": lambda: DecisionTreeClassifier(random_state=SEED),
              "1-Nearest-Neighbour": lambda: KNeighborsClassifier(n_neighbors=1)}
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    print("%-26s %-12s %-12s %-12s %-10s" % ("Model", "Train-on-all", "5-fold CV", "Held-out", "Optimism"))
    for name, mk in models.items():
        m_all = mk().fit(X, y)
        acc_defect = accuracy_score(y, m_all.predict(X))            # what the junior engineer prints
        cv = cross_val_score(mk(), Xtr, ytr, cv=skf, scoring="accuracy")
        m_tr = mk().fit(Xtr, ytr)
        acc_held = accuracy_score(yte, m_tr.predict(Xte))
        print("%-26s %-12.4f %-12.4f %-12.4f %+.4f" % (name, acc_defect, cv.mean(), acc_held, acc_defect - acc_held))

    # ---------------- B. Accuracy alone on imbalanced data -------------------
    print("\n[B] Majority-class baseline and imbalance (held-out test set, n=%d)" % len(yte))
    base = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    tree = DecisionTreeClassifier(random_state=SEED).fit(Xtr, ytr)
    print("%-26s %-8s %-8s %-8s %-8s" % ("Model", "Acc", "Prec", "Recall", "F1"))
    for name, mdl in [("Majority baseline", base), ("Unpruned tree", tree)]:
        mt = metrics(yte, mdl.predict(Xte))
        print("%-26s %-8.4f %-8.4f %-8.4f %-8.4f" % (name, mt["acc"], mt["prec"], mt["rec"], mt["f1"]))
    print("  -> The baseline that flags NOBODY as at-risk already scores %.1f %% accuracy but recall = 0." %
          (100 * accuracy_score(yte, base.predict(Xte))))

    # ---------------- C. Optimism averaged over 10 random splits -------------
    print("\n[C] Optimism gap over 10 different random splits (unpruned tree)")
    gaps, trains, tests = [], [], []
    for s in range(10):
        a, b, c, d = train_test_split(X, y, test_size=0.25, stratify=y, random_state=s)
        m = DecisionTreeClassifier(random_state=s).fit(a, c)
        trains.append(accuracy_score(c, m.predict(a)))
        tests.append(accuracy_score(d, m.predict(b)))
        gaps.append(trains[-1] - tests[-1])
    print("  train acc  = %.4f +/- %.4f" % (np.mean(trains), np.std(trains)))
    print("  held-out   = %.4f +/- %.4f" % (np.mean(tests), np.std(tests)))
    print("  optimism   = %.4f +/- %.4f   (train minus held-out)" % (np.mean(gaps), np.std(gaps)))

    # ---------------- D. Uncontrolled randomness -----------------------------
    print("\n[D] Uncontrolled randomness: same code, no seed, 5 runs (held-out accuracy)")
    accs = []
    for _ in range(5):
        a, b, c, d = train_test_split(X, y, test_size=0.25)          # NO random_state on purpose
        accs.append(accuracy_score(d, DecisionTreeClassifier().fit(a, c).predict(b)))
    print("  accuracies:", np.round(accs, 4), " spread = %.4f" % (max(accs) - min(accs)))

    # ---------------- E. REPAIRED protocol (Task 1.2 executed) ---------------
    print("\n[E] REPAIRED protocol: split -> 5-fold CV model selection -> test set used ONCE")
    grid = {"max_depth": [1, 2, 3, 4, 5, 8, None], "min_samples_leaf": [1, 5, 10, 20]}
    gs = GridSearchCV(DecisionTreeClassifier(random_state=SEED), grid, cv=skf, scoring="f1")
    gs.fit(Xtr, ytr)
    print("  best hyper-parameters (CV F1 = %.4f): %s" % (gs.best_score_, gs.best_params_))
    final = gs.best_estimator_
    pred = final.predict(Xte)                                           # test set touched exactly once
    mt = metrics(yte, pred)
    print("  TEST  acc=%.4f prec=%.4f recall=%.4f F1=%.4f" % (mt["acc"], mt["prec"], mt["rec"], mt["f1"]))
    print("  confusion matrix [[TN FP],[FN TP]] =", confusion_matrix(yte, pred).tolist())
    mb = metrics(yte, base.predict(Xte))
    print("  baseline F1 = %.4f  ->  model beats baseline by %.4f F1" % (mb["f1"], mt["f1"] - mb["f1"]))
    print("  train acc of repaired tree = %.4f (now close to held-out %.4f)" % (accuracy_score(ytr, final.predict(Xtr)), mt["acc"]))


if __name__ == "__main__":
    main()
