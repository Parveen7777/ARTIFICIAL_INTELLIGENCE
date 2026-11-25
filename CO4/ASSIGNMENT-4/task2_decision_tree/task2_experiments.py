"""
TASK 2.5 / 2.6 / 2.7 (implement part)

A. Verify the from-scratch learner against scikit-learn on the SAME split (entropy and Gini).
B. Report accuracy, precision, recall, F1 and the confusion matrix (held-out test data only).
C. Overfitting and pruning: depth, leaves, train / validation / test accuracy.
D. Identifier-like attribute on a larger dataset: information gain vs gain ratio.

Dataset: Breast Cancer Wisconsin (Diagnostic), a public UCI dataset shipped with scikit-learn.
Run:  python task2_decision_tree/task2_experiments.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from decision_tree import DecisionTree  # noqa: E402

SEED = 42
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load():
    d = load_breast_cancer(as_frame=True)
    X = d.data.copy()
    y = d.target.map({0: "malignant", 1: "benign"})
    return X, y


def scores(y_true, y_pred, pos="malignant"):
    return dict(acc=accuracy_score(y_true, y_pred),
                prec=precision_score(y_true, y_pred, pos_label=pos),
                rec=recall_score(y_true, y_pred, pos_label=pos),
                f1=f1_score(y_true, y_pred, pos_label=pos))


def main():
    X, y = load()
    print("DATASET: Breast Cancer Wisconsin (Diagnostic)  n=%d  features=%d  classes=%s" % (len(y), X.shape[1], y.value_counts().to_dict()))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    print("Split: train=%d  test=%d  (stratified, seed=%d)" % (len(ytr), len(yte), SEED))

    # ---------------------------------------------------------------- A + B
    print("\n" + "=" * 78 + "\n[A/B] FROM-SCRATCH LEARNER vs scikit-learn  (full-depth trees, same split)\n" + "=" * 78)
    rows = []
    for crit in ["entropy", "gini"]:
        mine = DecisionTree(criterion=crit).fit(Xtr, ytr)
        sk = DecisionTreeClassifier(criterion=crit, random_state=SEED).fit(Xtr, ytr)
        pm, ps = mine.predict(Xte), sk.predict(Xte)
        agree = (pm == ps).mean()
        for name, pred in [("scratch-" + crit, pm), ("sklearn-" + crit, ps)]:
            s = scores(yte, pred)
            rows.append([name, s["acc"], s["prec"], s["rec"], s["f1"]])
        print("criterion=%-8s  prediction agreement with sklearn on test set = %.4f   (depth scratch=%d, sklearn=%d; leaves %d vs %d)" % (
            crit, agree, mine.depth(), sk.get_depth(), mine.n_leaves(), sk.get_n_leaves()))
        sk_root_feat = X.columns[sk.tree_.feature[0]]
        print("  ROOT split  scratch: %s <= %.5f   sklearn: %s <= %.5f   -> %s" % (
            mine.root.feature, mine.root.threshold, sk_root_feat, sk.tree_.threshold[0],
            "IDENTICAL" if (mine.root.feature == sk_root_feat and abs(mine.root.threshold - sk.tree_.threshold[0]) < 1e-6) else "different"))
        cm_m = confusion_matrix(yte, pm, labels=["malignant", "benign"])
        cm_s = confusion_matrix(yte, ps, labels=["malignant", "benign"])
        print("  confusion matrix  scratch  [[TP FN],[FP TN]] =", cm_m.tolist(), "   sklearn =", cm_s.tolist())
    print("(Residual disagreement comes from equally-good splits: sklearn breaks exact ties with a random feature order, the scratch tree uses first-attribute-wins.)")
    print("\n%-18s %-8s %-10s %-8s %-8s   (positive class = malignant)" % ("Model", "Acc", "Precision", "Recall", "F1"))
    for r in rows:
        print("%-18s %-8.4f %-10.4f %-8.4f %-8.4f" % tuple(r))

    # ---------------------------------------------------------------- C
    print("\n" + "=" * 78 + "\n[C] OVERFITTING AND PRUNING (60 / 20 / 20 train / validation / test)\n" + "=" * 78)
    Xa, Xte2, ya, yte2 = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    Xtr2, Xva, ytr2, yva = train_test_split(Xa, ya, test_size=0.25, stratify=ya, random_state=SEED)
    print("train=%d validation=%d test=%d" % (len(ytr2), len(yva), len(yte2)))
    table = []

    def record(name, tree):
        acc = lambda A, b: accuracy_score(b, tree.predict(A))
        table.append([name, tree.depth(), tree.n_leaves(), acc(Xtr2, ytr2), acc(Xva, yva), acc(Xte2, yte2)])

    full = DecisionTree(criterion="entropy").fit(Xtr2, ytr2)
    record("full tree (no limit)", full)
    for d in [1, 2, 3, 4, 5, 6, 8]:
        record("max_depth=%d" % d, DecisionTree(criterion="entropy", max_depth=d).fit(Xtr2, ytr2))
    for m in [5, 10, 20]:
        record("min_samples_leaf=%d" % m, DecisionTree(criterion="entropy", min_samples_leaf=m).fit(Xtr2, ytr2))
    pruned = DecisionTree(criterion="entropy").fit(Xtr2, ytr2).prune_reduced_error(Xva, yva)
    record("reduced-error pruning", pruned)
    print("%-24s %-6s %-7s %-8s %-8s %-8s %-8s" % ("Setting", "Depth", "Leaves", "Train", "Valid", "Test", "Gap(tr-te)"))
    for r in table:
        print("%-24s %-6d %-7d %-8.4f %-8.4f %-8.4f %+.4f" % (r[0], r[1], r[2], r[3], r[4], r[5], r[3] - r[5]))
    best_val = max(table, key=lambda r: (r[4], -r[2]))
    print("\nBest validation accuracy: '%s' (val %.4f) -> test %.4f   [test set looked at only for this final report]" % (best_val[0], best_val[4], best_val[5]))

    depth_rows = [r for r in table if r[0].startswith("max_depth")] + [table[0]]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xs = [r[1] for r in depth_rows]
    order = np.argsort(xs)
    for idx, lab, col in [(3, "Train", "#1f77b4"), (4, "Validation", "#ff7f0e"), (5, "Test", "#2ca02c")]:
        ax.plot(np.array(xs)[order], np.array([r[idx] for r in depth_rows])[order], "o-", label=lab, color=col)
    ax.set_xlabel("Tree depth"); ax.set_ylabel("Accuracy"); ax.set_title("Task 2.6  Overfitting: accuracy vs tree depth")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "task2_depth_vs_accuracy.png"), dpi=160); plt.close(fig)

    # ---------------------------------------------------------------- D
    print("\n" + "=" * 78 + "\n[D] IDENTIFIER-LIKE ATTRIBUTE ON REAL DATA (StudentID-style unique string per row)\n" + "=" * 78)
    Xid = X.copy()
    Xid.insert(0, "RecordID", ["R%04d" % i for i in range(len(Xid))])
    Xtr3, Xte3, ytr3, yte3 = train_test_split(Xid, y, test_size=0.30, stratify=y, random_state=SEED)
    print("%-34s %-14s %-8s %-8s %-8s" % ("Learner", "Root", "Depth", "Leaves", "Test acc"))
    configs = [("entropy (information gain)", dict(criterion="entropy")),
               ("gain ratio (C4.5 repair)", dict(criterion="gain_ratio")),
               ("entropy + min_samples_leaf=5", dict(criterion="entropy", min_samples_leaf=5))]
    for name, kw in configs:
        t = DecisionTree(**kw).fit(Xtr3, ytr3)
        print("%-34s %-14s %-8d %-8d %-8.4f" % (name, t.root.feature, t.depth(), t.n_leaves(), accuracy_score(yte3, t.predict(Xte3))))
    maj = ytr3.value_counts().idxmax()
    print("Majority-class baseline accuracy on the same test set = %.4f" % (yte3 == maj).mean())
    print("-> With plain information gain the unique ID wins the root (gain = H(Y)); the tree memorises the")
    print("   training set and, because test IDs are unseen, falls back to the root majority: no better than the baseline.")


if __name__ == "__main__":
    main()
