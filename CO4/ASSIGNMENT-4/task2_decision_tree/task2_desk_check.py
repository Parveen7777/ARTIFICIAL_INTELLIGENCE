"""
TASK 2.4 / 2.7  -  Desk-check table, modified case and identifier-bias demonstration.

The manual trace in the report is verified here: every entropy and information-gain value
printed by the code must match the hand calculation.

Run:  python task2_decision_tree/task2_desk_check.py
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from decision_tree import DecisionTree, entropy_from_counts  # noqa: E402

TABLE = pd.DataFrame({
    "Attendance":  ["High", "High", "High", "High", "Low", "Low", "Low", "Low"],
    "Assignments": ["Done", "Done", "Missed", "Missed", "Done", "Done", "Missed", "Missed"],
    "Internal":    ["Good", "Poor", "Good", "Poor", "Good", "Poor", "Good", "Poor"],
    "Result":      ["Pass", "Pass", "Pass", "Fail", "Pass", "Fail", "Fail", "Fail"]},
    index=range(1, 9))


def run(table, criterion="entropy", trace=True, **kw):
    X, y = table.drop(columns="Result"), table["Result"]
    t = DecisionTree(criterion=criterion, trace=trace, **kw).fit(X, y)
    return t, X, y


def main():
    print("=" * 78)
    print("TASK 2.4  DESK-CHECK TABLE  (8 students)")
    print(TABLE.to_string())
    print("=" * 78)
    print("H(Result) = H(4 Pass, 4 Fail) = %.4f bits" % entropy_from_counts([4, 4]))
    print("\n-- Full ID3 trace produced by the code --")
    t, X, y = run(TABLE)
    print("\n".join(t.trace_lines))
    print("\nFinal tree:\n" + t.to_text())
    print("depth = %d, leaves = %d, training accuracy = %.3f" % (t.depth(), t.n_leaves(), (t.predict(X) == y.values).mean()))

    # ------------------------------------------------------------ TASK 2.7  modified case
    print("\n" + "=" * 78)
    print("TASK 2.7  CHALLENGE: change ONE value  (row 6: Low, Done, Poor  Fail -> Pass)")
    print("PREDICTION (made before tracing): Assignments=Done becomes pure (rows 1,2,5,6 all Pass),")
    print("so Assignments should overtake Attendance as the root attribute.")
    mod = TABLE.copy()
    mod.loc[6, "Result"] = "Pass"
    print(mod.to_string())
    t2, X2, y2 = run(mod)
    print("\n".join(l for l in t2.trace_lines if l.startswith("[root]")))
    print("ROOT ATTRIBUTE AFTER CHANGE =", t2.root.feature, "  (before change:", t.root.feature + ")")
    print("Prediction correct:", t2.root.feature == "Assignments")

    # ------------------------------------------------------------ identifier-like attribute
    print("\n" + "=" * 78)
    print("TASK 2.7  IDENTIFIER-LIKE ATTRIBUTE (unique StudentID) - bias of information gain")
    with_id = TABLE.copy()
    with_id.insert(0, "StudentID", ["S%03d" % i for i in with_id.index])
    for crit in ["entropy", "gain_ratio"]:
        tt, _, _ = run(with_id, criterion=crit)
        print("\nCriterion = %-10s  root = %s" % (crit, tt.root.feature))
        print("\n".join(l for l in tt.trace_lines if l.startswith("[root]")))
    print("\nRepair 1: forbid branches smaller than 2 examples (min_samples_leaf = 2), criterion = entropy")
    tr, _, _ = run(with_id, criterion="entropy", min_samples_leaf=2)
    print("  root =", tr.root.feature)
    print("Repair 2: remove the identifier column (data hygiene) -> same as original tree, root =", t.root.feature)
    print("Note: on this 8-row table gain ratio is NOT enough (ID: 1.0/3.0 = 0.333 > 0.189), because the real")
    print("attributes carry little gain. On larger data it works - see task2_experiments.py (section D).")


if __name__ == "__main__":
    main()
