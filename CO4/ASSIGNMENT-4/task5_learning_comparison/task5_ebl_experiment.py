"""
TASK 5.3  -  Explanation-Based Learning of a reusable hazard rule from ONE explained case.

Run:  python task5_learning_comparison/task5_ebl_experiment.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from ebl import Rule, KnowledgeBase, fmt, tree_text, count_nodes, ebg  # noqa: E402

# ------------------------------------------------------------------------------------------ domain theory (8 first-order rules)
DOMAIN_THEORY = [
    Rule("R1", ("Overheating", "?s"),     [("HighLoad", "?s"), ("CoolingFailed", "?s")]),
    Rule("R2", ("Overheating", "?s"),     [("HighTemp", "?s"), ("LowOilLevel", "?s")]),
    Rule("R3", ("InsulationRisk", "?s"),  [("Overheating", "?s"), ("InsulationAged", "?s")]),
    Rule("R4", ("InsulationRisk", "?s"),  [("InsulationWet", "?s"), ("HarmonicsHigh", "?s")]),
    Rule("R5", ("ProtectionWeak", "?s"),  [("BreakerOld", "?s"), ("RelayUntested", "?s")]),
    Rule("R6", ("Hazardous", "?s"),       [("InsulationRisk", "?s"), ("ProtectionWeak", "?s")]),
    Rule("R7", ("Critical", "?s"),        [("Feeds", "?s", "?h"), ("Hospital", "?h")]),
    Rule("R8", ("Hazardous", "?s"),       [("Overheating", "?s"), ("Critical", "?s")]),
]
OPERATIONAL = {"HighLoad", "CoolingFailed", "HighTemp", "LowOilLevel", "InsulationAged", "InsulationWet",
               "HarmonicsHigh", "BreakerOld", "RelayUntested", "Feeds", "Hospital", "NearRiver"}


def facts_for(sub, atoms):
    out = []
    for a in atoms:
        if isinstance(a, tuple):
            out.append(a)
        else:
            out.append((a, sub))
    return out


def case(name, *atoms):
    return name, list(atoms)


F = lambda p, *a: (p,) + a
CASES = [
    # --- hazardous, same reasoning path as the training example (covered by the learned rule)
    case("c01 exact five premises", "HighLoad", "CoolingFailed", "InsulationAged", "BreakerOld", "RelayUntested"),
    case("c02 + irrelevant facts", "HighLoad", "CoolingFailed", "InsulationAged", "BreakerOld", "RelayUntested", "HighTemp", "NearRiver"),
    case("c03 + feeds hospital", "HighLoad", "CoolingFailed", "InsulationAged", "BreakerOld", "RelayUntested", F("Feeds", "SUB", "Hosp1"), F("Hospital", "Hosp1")),
    case("c04 + wet insulation", "HighLoad", "CoolingFailed", "InsulationAged", "BreakerOld", "RelayUntested", "InsulationWet"),
    case("c05 + low oil", "HighLoad", "CoolingFailed", "InsulationAged", "BreakerOld", "RelayUntested", "LowOilLevel", "HighTemp"),
    # --- hazardous through OTHER proofs (not covered by the learned rule)
    case("c06 hot+low oil path", "HighTemp", "LowOilLevel", "InsulationAged", "BreakerOld", "RelayUntested"),
    case("c07 wet+harmonics path", "InsulationWet", "HarmonicsHigh", "BreakerOld", "RelayUntested"),
    case("c08 overheating+hospital", "HighLoad", "CoolingFailed", F("Feeds", "SUB", "Hosp1"), F("Hospital", "Hosp1")),
    case("c09 hot/oil+hospital", "HighTemp", "LowOilLevel", F("Feeds", "SUB", "Hosp2"), F("Hospital", "Hosp2")),
    case("c10 wet+harmonics+river", "InsulationWet", "HarmonicsHigh", "BreakerOld", "RelayUntested", "NearRiver"),
    # --- NOT hazardous
    case("c11 relay tested", "HighLoad", "CoolingFailed", "InsulationAged", "BreakerOld"),
    case("c12 breaker new", "HighLoad", "CoolingFailed", "InsulationAged", "RelayUntested"),
    case("c13 cooling ok", "HighLoad", "InsulationAged", "BreakerOld", "RelayUntested"),
    case("c14 load normal", "CoolingFailed", "InsulationAged", "BreakerOld", "RelayUntested"),
    case("c15 insulation young", "HighLoad", "CoolingFailed", "BreakerOld", "RelayUntested"),
    case("c16 only high temp", "HighTemp"),
    case("c17 feeds school only", "HighLoad", "CoolingFailed", F("Feeds", "SUB", "School1")),
    case("c18 wet, no harmonics", "InsulationWet", "BreakerOld", "RelayUntested"),
    case("c19 healthy substation"),
    case("c20 wet+harmonics only", "HarmonicsHigh", "InsulationWet"),
]


def main():
    print("=" * 84 + "\nTASK 5.3 (Design)  DOMAIN THEORY (8 first-order rules)\n" + "=" * 84)
    for r in DOMAIN_THEORY:
        print("  " + str(r))
    print("Operational predicates (directly observable):", sorted(OPERATIONAL))

    # ---------------------------------------------------------------- the single explained training example
    train_facts = facts_for("S1", ["HighLoad", "CoolingFailed", "InsulationAged", "BreakerOld", "RelayUntested", "NearRiver", F("Feeds", "S1", "School1")])
    kb = KnowledgeBase(DOMAIN_THEORY, train_facts)
    goal = ("Hazardous", "S1")
    print("\n" + "=" * 84 + "\nTRAINING EXAMPLE: substation S1\n" + "=" * 84)
    print("Facts:", ", ".join(fmt(f) for f in train_facts))
    tree = kb.explain(goal)
    print("\nEXPLANATION TREE for %s (%d nodes):" % (fmt(goal), count_nodes(tree)))
    print(tree_text(tree))
    learned = ebg(tree, OPERATIONAL)
    print("\nGENERALISED (operational) RULE learned by EBG:")
    print("  " + str(learned))
    print("  -> constants S1 replaced by a variable; irrelevant facts (NearRiver, Feeds School1) dropped; every condition is operational.")

    # ---------------------------------------------------------------- 20 test cases
    print("\n" + "=" * 84 + "\nTASK 5.3 (Implement)  TEST ON 20 DESIGNED CASES\n" + "=" * 84)
    print("%-26s %-7s | %-9s %-8s | %-9s %-8s | %-12s" % ("case", "truth", "learned", "steps", "theory", "steps", "hybrid steps"))
    tot = dict(theory=0, learned=0, hybrid=0)
    cov_theory = cov_learned = 0
    tp = fp = fn = 0
    for name, atoms in CASES:
        facts = facts_for("SUB", atoms)
        kb_theory = KnowledgeBase(DOMAIN_THEORY, facts)
        truth, s_theory = kb_theory.holds(("Hazardous", "SUB"))
        kb_learned = KnowledgeBase([Rule("LEARNED", learned.head, learned.body)], facts)
        ans_l, s_learned = kb_learned.holds(("Hazardous", "SUB"))
        kb_hybrid = KnowledgeBase([Rule("LEARNED", learned.head, learned.body)] + DOMAIN_THEORY, facts)
        ans_h, s_hybrid = kb_hybrid.holds(("Hazardous", "SUB"))
        assert ans_h == truth, "hybrid system must stay correct"
        tot["theory"] += s_theory; tot["learned"] += s_learned; tot["hybrid"] += s_hybrid
        tp += int(ans_l and truth); fp += int(ans_l and not truth); fn += int((not ans_l) and truth)
        if truth and ans_l:
            cov_theory += s_theory; cov_learned += s_learned
        print("%-26s %-7s | %-9s %-8d | %-9s %-8d | %-12d" % (name, truth, ans_l, s_learned, truth, s_theory, s_hybrid))
    n_haz = tp + fn
    print("\nLearned rule ALONE : true positives = %d, false positives = %d, false negatives = %d  -> precision = %.2f, recall = %.2f" % (
        tp, fp, fn, tp / max(tp + fp, 1), tp / max(n_haz, 1)))
    print("  (sound, never wrong when it fires; but incomplete: it only encodes ONE of the several proofs)")
    print("Inference steps on the %d hazardous cases the learned rule covers:  theory = %d   learned rule = %d   saving = %.1f%%" % (
        tp, cov_theory, cov_learned, 100 * (1 - cov_learned / cov_theory)))
    print("Total steps over all 20 cases:  theory only = %d   hybrid (learned rule tried first, theory as fall-back) = %d" % (tot["theory"], tot["hybrid"]))
    print("  -> average steps/case: theory %.1f vs hybrid %.1f %s" % (tot["theory"] / 20, tot["hybrid"] / 20,
          "(hybrid cheaper)" if tot["hybrid"] < tot["theory"] else "(hybrid COSTLIER: the extra rule is tried and fails on uncovered cases - the 'utility problem')"))

    print("\nWHAT WAS LEARNED THAT THE DOMAIN THEORY DID NOT ALREADY ENTAIL?")
    print("  Nothing new in terms of truth: DOMAIN_THEORY |= LEARNED rule (every hazardous verdict above agrees with the theory, no false positives).")
    print("  What EBL adds is OPERATIONALITY and EFFICIENCY: a single-step shortcut over five observable facts that skips the intermediate")
    print("  concepts Overheating / InsulationRisk / ProtectionWeak, plus the knowledge of WHICH facts of the example mattered (NearRiver and")
    print("  Feeds(S1,School1) were irrelevant to the explanation and were dropped). It does not extend what the system can conclude.")


if __name__ == "__main__":
    main()
