"""
TASK 5.2  -  Linear SVM vs RBF SVM (5-fold CV over C and gamma) vs Naive Bayes vs the Task-2 tree, identical split.

Features used for the comparison and for the 2-D decision-boundary plots: load_pct and voltage_dev.
Run:  python task5_learning_comparison/task5_kernel_experiment.py
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "task2_decision_tree"))
from smartgrid import make_smart_grid, QuantileDiscretizer, CategoricalNaiveBayes  # noqa: E402
from decision_tree import DecisionTree  # noqa: E402

FIG_DIR = os.path.join(HERE, "..", "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
SEED = 42


def row(name, ytrue, pred, train_s, extra=""):
    return [name, accuracy_score(ytrue, pred), precision_score(ytrue, pred), recall_score(ytrue, pred), f1_score(ytrue, pred), train_s, extra]


def main():
    X4, y = make_smart_grid(n=1500, seed=0)
    X = X4[:, :2]                                   # load_pct, voltage_dev
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    print("=" * 84)
    print("TASK 5.2  SMART-GRID, 2 FEATURES (load_pct, voltage_dev); train=%d test=%d (same split for all four models)" % (len(ytr), len(yte)))
    print("Hazardous rate = %.3f.  Safe region is an ellipse -> classes are NOT linearly separable." % y.mean())
    print("=" * 84)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    # ----------------------------------------------------------- linear SVM
    g_lin = GridSearchCV(SVC(kernel="linear"), {"C": [0.01, 0.1, 1, 10, 100]}, cv=cv, scoring="accuracy").fit(Ztr, ytr)
    t = time.perf_counter(); lin = SVC(kernel="linear", **g_lin.best_params_).fit(Ztr, ytr); t_lin = time.perf_counter() - t
    print("\nLinear SVM  5-fold CV grid over C: best = %s, CV accuracy = %.4f" % (g_lin.best_params_, g_lin.best_score_))
    # ----------------------------------------------------------- RBF SVM
    grid = {"C": [0.1, 1, 10, 100], "gamma": [0.01, 0.1, 1, 10]}
    g_rbf = GridSearchCV(SVC(kernel="rbf"), grid, cv=cv, scoring="accuracy").fit(Ztr, ytr)
    t = time.perf_counter(); rbf = SVC(kernel="rbf", **g_rbf.best_params_).fit(Ztr, ytr); t_rbf = time.perf_counter() - t
    print("RBF SVM     5-fold CV grid over C x gamma (16 settings): best = %s, CV accuracy = %.4f" % (g_rbf.best_params_, g_rbf.best_score_))
    res = pd.DataFrame(g_rbf.cv_results_)[["param_C", "param_gamma", "mean_test_score"]]
    print("\nRBF CV accuracy table (rows = C, columns = gamma):")
    print(res.pivot(index="param_C", columns="param_gamma", values="mean_test_score").round(4).to_string())

    # ----------------------------------------------------------- Naive Bayes (scratch) on the same two features
    disc = QuantileDiscretizer(10).fit(Xtr)
    t = time.perf_counter(); nb = CategoricalNaiveBayes(1.0).fit(disc.transform(Xtr), ytr, disc.n_values_); t_nb = time.perf_counter() - t
    # ----------------------------------------------------------- Task-2 tree (scratch) on the same two features
    dfr = pd.DataFrame(Xtr, columns=["load_pct", "voltage_dev"]); dfe = pd.DataFrame(Xte, columns=["load_pct", "voltage_dev"])
    t = time.perf_counter(); tree = DecisionTree(criterion="entropy", min_samples_leaf=5).fit(dfr, ytr); t_tree = time.perf_counter() - t

    p_lin, p_rbf = lin.predict(Zte), rbf.predict(Zte)
    p_nb = nb.predict(disc.transform(Xte))
    p_tree = tree.predict(dfe).astype(int)
    rows = [row("Linear SVM", yte, p_lin, t_lin, "SV=%d" % lin.n_support_.sum()),
            row("RBF SVM", yte, p_rbf, t_rbf, "SV=%d" % rbf.n_support_.sum()),
            row("Naive Bayes (scratch, a=1)", yte, p_nb, t_nb, "-"),
            row("Decision tree (Task 2)", yte, p_tree, t_tree, "leaves=%d" % tree.n_leaves())]
    print("\nHELD-OUT TEST RESULTS (identical split)")
    print("%-28s %-8s %-9s %-8s %-8s %-12s %-10s" % ("Model", "Acc", "Precision", "Recall", "F1", "Train time", "Size"))
    for r in rows:
        print("%-28s %-8.4f %-9.4f %-8.4f %-8.4f %-12s %-10s" % (r[0], r[1], r[2], r[3], r[4], "%.4f s" % r[5], r[6]))
    print("\nSupport vectors: linear = %d of %d training points (%.1f%%), RBF = %d (%.1f%%)" % (
        lin.n_support_.sum(), len(ytr), 100 * lin.n_support_.sum() / len(ytr), rbf.n_support_.sum(), 100 * rbf.n_support_.sum() / len(ytr)))
    print("Majority-class baseline accuracy = %.4f" % max(yte.mean(), 1 - yte.mean()))

    # ----------------------------------------------------------- decision-boundary plots
    xx, yy = np.meshgrid(np.linspace(18, 102, 220), np.linspace(-8.5, 8.5, 220))
    grid_pts = np.c_[xx.ravel(), yy.ravel()]
    gz = sc.transform(grid_pts)
    grid_df = pd.DataFrame(grid_pts, columns=["load_pct", "voltage_dev"])
    panels = [("Linear SVM", lin.predict(gz), rows[0][4]),
              ("RBF SVM", rbf.predict(gz), rows[1][4]),
              ("Naive Bayes (scratch)", nb.predict(disc.transform(grid_pts)), rows[2][4]),
              ("Decision tree (scratch, Task 2)", tree.predict(grid_df).astype(int), rows[3][4])]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, (name, pr, f1_) in zip(axes.ravel(), panels):
        ax.contourf(xx, yy, np.asarray(pr).reshape(xx.shape), alpha=.28, levels=[-.5, .5, 1.5], colors=["#2ca02c", "#d62728"])
        ax.scatter(Xte[yte == 0, 0], Xte[yte == 0, 1], s=9, c="#2ca02c", label="normal (test)")
        ax.scatter(Xte[yte == 1, 0], Xte[yte == 1, 1], s=9, c="#d62728", label="hazardous (test)")
        if name in ("Linear SVM", "RBF SVM"):
            m = lin if name == "Linear SVM" else rbf
            sv = sc.inverse_transform(m.support_vectors_)
            ax.scatter(sv[:, 0], sv[:, 1], s=34, facecolors="none", edgecolors="k", linewidths=.6, label="support vectors")
        ax.set_title("%s  (test F1 = %.3f)" % (name, f1_), fontsize=10)
        ax.set_xlabel("load_pct"); ax.set_ylabel("voltage_dev")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Task 5.2  Decision boundaries on the smart-grid test set (identical split)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "task5_decision_boundaries.png"), dpi=150); plt.close(fig)
    print("\nSaved figure: outputs/figures/task5_decision_boundaries.png")


if __name__ == "__main__":
    main()
