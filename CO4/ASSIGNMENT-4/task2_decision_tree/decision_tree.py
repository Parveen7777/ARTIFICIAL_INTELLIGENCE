"""
decision_tree.py  -  Decision-tree learner written from scratch (no scikit-learn inside).

Supports
  * categorical attributes  -> multi-way split, attribute removed from the child's candidate list (ID3)
  * numeric attributes      -> binary split  x <= t  at mid-points between distinct sorted values
  * criteria                -> "entropy" (information gain), "gini", "gain_ratio" (C4.5 repair)
  * stopping / pre-pruning  -> max_depth, min_samples_split, min_samples_leaf
  * post-pruning            -> reduced-error pruning on a validation set
  * optional step-by-step trace of every entropy / gain computation (used for the desk-check)

Algorithm logic is kept separate from any input/output: nothing here reads files or plots.
"""
from collections import Counter
import math
import numpy as np
import pandas as pd

EPS = 1e-12


# ----------------------------------------------------------------------------- impurity measures
def entropy_from_counts(counts):
    """H = - sum p log2 p   (0 log 0 := 0)."""
    n = float(sum(counts))
    if n == 0:
        return 0.0
    h = -sum((c / n) * math.log2(c / n) for c in counts if c > 0)
    return h if h > 0 else 0.0                      # avoid printing -0.0


def gini_from_counts(counts):
    """G = 1 - sum p^2."""
    n = float(sum(counts))
    if n == 0:
        return 0.0
    return 1.0 - sum((c / n) ** 2 for c in counts)


IMPURITY = {"entropy": entropy_from_counts, "gain_ratio": entropy_from_counts, "gini": gini_from_counts}


class Node:
    """A tree node.  Leaf if `children` is empty."""
    def __init__(self, label, counts, depth):
        self.label = label                # majority class at this node (used for leaf output and fall-back)
        self.counts = counts              # dict class -> count of training examples reaching the node
        self.depth = depth
        self.feature = None
        self.threshold = None             # None => categorical split
        self.children = {}                # categorical: value -> Node ; numeric: {"<=": Node, ">": Node}

    @property
    def is_leaf(self):
        return not self.children


class DecisionTree:
    def __init__(self, criterion="entropy", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_gain=0.0, trace=False):
        assert criterion in IMPURITY
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.trace = trace
        self.trace_lines = []
        self.root = None

    # ------------------------------------------------------------------ public API
    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        self.classes_ = sorted(y.unique().tolist(), key=str)
        self.features_ = list(X.columns)
        # any column that is not a real number (object / str / category / bool) is categorical
        self.is_cat_ = {c: (not pd.api.types.is_numeric_dtype(X[c])) or pd.api.types.is_bool_dtype(X[c])
                        for c in X.columns}
        majority = self._majority(y)
        self.root = self._build(X, y, list(X.columns), majority, depth=0, path="root")
        return self

    def predict(self, X):
        X = pd.DataFrame(X).reset_index(drop=True)
        return np.array([self._predict_row(self.root, X.iloc[i]) for i in range(len(X))], dtype=object)

    def depth(self):
        def d(n):
            return 0 if n.is_leaf else 1 + max(d(c) for c in n.children.values())
        return d(self.root)

    def n_leaves(self):
        def f(n):
            return 1 if n.is_leaf else sum(f(c) for c in n.children.values())
        return f(self.root)

    def prune_reduced_error(self, Xval, yval):
        """Bottom-up reduced-error pruning: replace a subtree by a leaf if validation accuracy does not drop."""
        Xval = pd.DataFrame(Xval).reset_index(drop=True)
        yval = pd.Series(yval).reset_index(drop=True)

        def acc_subtree(node, idx):
            if len(idx) == 0:
                return 0
            return sum(self._predict_row(node, Xval.iloc[i]) == yval.iloc[i] for i in idx)

        def route(node, idx):
            """Partition validation row indices among node's children."""
            parts = {k: [] for k in node.children}
            for i in idx:
                row = Xval.iloc[i]
                if node.threshold is None:
                    v = row[node.feature]
                    if v in node.children:
                        parts[v].append(i)
                else:
                    parts["<=" if row[node.feature] <= node.threshold else ">"].append(i)
            return parts

        def visit(node, idx):
            if node.is_leaf:
                return
            for k, child in node.children.items():
                visit(child, route(node, idx)[k])
            as_subtree = acc_subtree(node, idx)
            as_leaf = sum(node.label == yval.iloc[i] for i in idx)
            if as_leaf >= as_subtree:           # pruning never hurts validation accuracy
                node.children, node.feature, node.threshold = {}, None, None

        visit(self.root, list(range(len(Xval))))
        return self

    def to_text(self):
        lines = []

        def rec(n, indent):
            if n.is_leaf:
                lines.append("%s-> %s  %s" % ("  " * indent, n.label, dict(n.counts)))
                return
            for k, c in n.children.items():
                cond = ("%s = %s" % (n.feature, k)) if n.threshold is None else ("%s %s %.4g" % (n.feature, k, n.threshold))
                lines.append("%s[%s]" % ("  " * indent, cond))
                rec(c, indent + 1)
        rec(self.root, 0)
        return "\n".join(lines)

    # ------------------------------------------------------------------ construction (the pseudocode of Task 2.3)
    def _build(self, X, y, attrs, parent_majority, depth, path):
        # stopping condition 1: no examples -> parent's majority class
        if len(y) == 0:
            self._log(path, "no examples -> leaf = parent majority '%s'" % parent_majority)
            return Node(parent_majority, {}, depth)
        counts = Counter(y.tolist())
        majority = self._majority(y)
        node = Node(majority, dict(counts), depth)
        # stopping condition 2: all examples have the same class
        if len(counts) == 1:
            self._log(path, "pure node %s -> leaf '%s'" % (dict(counts), majority))
            return node
        # stopping condition 3: no attributes remaining (or user limits reached)
        if not attrs:
            self._log(path, "no attributes remaining -> leaf = majority '%s' %s" % (majority, dict(counts)))
            return node
        if (self.max_depth is not None and depth >= self.max_depth) or len(y) < self.min_samples_split:
            self._log(path, "depth/size limit reached -> leaf = majority '%s'" % majority)
            return node

        best = self._choose_attribute(X, y, attrs, path)
        if best is None or best["score"] <= self.min_gain + EPS:
            self._log(path, "no split with positive gain -> leaf = majority '%s'" % majority)
            return node

        node.feature, node.threshold = best["feature"], best["threshold"]
        self._log(path, "CHOOSE attribute '%s'%s (score %.4f)" % (
            best["feature"], "" if best["threshold"] is None else " <= %.4g" % best["threshold"], best["score"]))
        if best["threshold"] is None:                              # categorical, multi-way
            remaining = [a for a in attrs if a != best["feature"]]
            for v in sorted(X[best["feature"]].unique(), key=str):
                mask = X[best["feature"]] == v
                node.children[v] = self._build(X[mask].reset_index(drop=True), y[mask].reset_index(drop=True),
                                               remaining, majority, depth + 1, "%s/%s=%s" % (path, best["feature"], v))
        else:                                                      # numeric, binary (attribute stays available)
            m = X[best["feature"]] <= best["threshold"]
            node.children["<="] = self._build(X[m].reset_index(drop=True), y[m].reset_index(drop=True), attrs,
                                              majority, depth + 1, path + "/" + best["feature"] + "<=")
            node.children[">"] = self._build(X[~m].reset_index(drop=True), y[~m].reset_index(drop=True), attrs,
                                             majority, depth + 1, path + "/" + best["feature"] + ">")
        return node

    # ------------------------------------------------------------------ attribute selection
    def _choose_attribute(self, X, y, attrs, path):
        imp = IMPURITY[self.criterion]
        classes = self.classes_
        parent_counts = [int((y == c).sum()) for c in classes]
        parent_imp = imp(parent_counts)
        self._log(path, "parent counts %s  impurity(%s) = %.4f" % (dict(zip(classes, parent_counts)), self.criterion, parent_imp))
        best = None
        for a in attrs:                                            # earlier attribute wins ties (documented rule)
            cand = self._best_split_for_attribute(X[a], y, a, parent_imp, imp, classes)
            if cand is None:
                continue
            self._log(path, "   %-14s gain = %.4f%s" % (a, cand["gain"], "" if self.criterion != "gain_ratio" else
                                                         "  split-info = %.4f  gain-ratio = %.4f" % (cand["split_info"], cand["score"])),
                      detail=cand.get("detail"))
            if best is None or cand["score"] > best["score"] + EPS:
                best = cand
        return best

    def _best_split_for_attribute(self, col, y, name, parent_imp, imp, classes):
        n = len(y)
        if self.is_cat_[name]:
            groups = {}
            for v, yy in zip(col.tolist(), y.tolist()):
                groups.setdefault(v, []).append(yy)
            if len(groups) < 2:
                return None
            if any(len(g) < self.min_samples_leaf for g in groups.values()):
                return None
            remainder, split_info, detail = 0.0, 0.0, []
            for v in sorted(groups, key=str):
                g = groups[v]
                cnt = [g.count(c) for c in classes]
                w = len(g) / n
                h = imp(cnt)
                remainder += w * h
                split_info -= w * math.log2(w)
                detail.append("%s: %d/%d %s imp=%.4f" % (v, len(g), n, dict(zip(classes, cnt)), h))
            gain = parent_imp - remainder
            score = gain / split_info if self.criterion == "gain_ratio" and split_info > EPS else gain
            return dict(feature=name, threshold=None, gain=gain, score=score, split_info=split_info, detail=detail)
        # numeric: scan sorted thresholds with cumulative class counts
        vals = col.to_numpy(dtype=float)
        order = np.argsort(vals, kind="mergesort")
        sv = vals[order]
        yy = y.to_numpy()[order]
        onehot = np.array([[1 if yv == c else 0 for c in classes] for yv in yy])
        left_cum = np.cumsum(onehot, axis=0)
        total = left_cum[-1]
        best = None
        for i in range(n - 1):
            if sv[i] == sv[i + 1]:
                continue
            nl, nr = i + 1, n - i - 1
            if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                continue
            lc, rc = left_cum[i], total - left_cum[i]
            remainder = (nl / n) * imp(lc.tolist()) + (nr / n) * imp(rc.tolist())
            gain = parent_imp - remainder
            si = -(nl / n) * math.log2(nl / n) - (nr / n) * math.log2(nr / n)
            score = gain / si if self.criterion == "gain_ratio" and si > EPS else gain
            if best is None or score > best["score"] + EPS:
                best = dict(feature=name, threshold=(sv[i] + sv[i + 1]) / 2.0, gain=gain, score=score, split_info=si)
        return best

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _majority(y):
        c = Counter(y.tolist())
        top = max(c.values())
        return sorted([k for k, v in c.items() if v == top], key=str)[0]      # deterministic tie-break

    def _predict_row(self, node, row):
        while not node.is_leaf:
            if node.threshold is None:
                v = row[node.feature]
                if v not in node.children:           # unseen category -> majority of this node
                    return node.label
                node = node.children[v]
            else:
                node = node.children["<=" if float(row[node.feature]) <= node.threshold else ">"]
        return node.label

    def _log(self, path, msg, detail=None):
        if self.trace:
            self.trace_lines.append("[%s] %s" % (path, msg))
            if detail:
                for d in detail:
                    self.trace_lines.append("        " + d)
