"""
ebl.py  -  A small first-order reasoner and Explanation-Based Generalisation (EBG).

Terms:    variables start with '?'  (e.g. '?s'),  everything else is a constant.
Literals: tuples  (predicate, arg1, arg2, ...)   e.g.  ("Overheating", "?s")
Rules:    Rule(name, head, body)   -  head <- body[0] AND body[1] AND ...

Cost model (used to compare inference effort):
    * one step per fact lookup of a goal whose predicate has stored facts
    * one step per attempt to unify a goal with a rule head
"""
import itertools

_counter = itertools.count()


class Rule:
    def __init__(self, name, head, body):
        self.name, self.head, self.body = name, head, list(body)

    def __str__(self):
        return "%s: %s <- %s" % (self.name, fmt(self.head), " ^ ".join(fmt(b) for b in self.body))


def fmt(lit):
    return "%s(%s)" % (lit[0], ", ".join(lit[1:]))


def is_var(t):
    return isinstance(t, str) and t.startswith("?")


def walk(t, theta):
    while is_var(t) and t in theta:
        t = theta[t]
    return t


def apply_theta(lit, theta):
    return (lit[0],) + tuple(walk(a, theta) for a in lit[1:])


def unify(a, b, theta):
    """Unify two literals under substitution theta; return the extended substitution or None."""
    if a[0] != b[0] or len(a) != len(b):
        return None
    theta = dict(theta)
    for x, y in zip(a[1:], b[1:]):
        x, y = walk(x, theta), walk(y, theta)
        if x == y:
            continue
        if is_var(x):
            theta[x] = y
        elif is_var(y):
            theta[y] = x
        else:
            return None
    return theta


def standardize_apart(rule):
    k = next(_counter)
    ren = lambda lit: (lit[0],) + tuple((a + "_" + str(k)) if is_var(a) else a for a in lit[1:])
    return Rule(rule.name, ren(rule.head), [ren(b) for b in rule.body])


class KnowledgeBase:
    def __init__(self, rules, facts=()):
        self.rules = list(rules)
        self.facts = []
        self.steps = 0
        self.add_facts(facts)

    def add_facts(self, facts):
        self.facts.extend(facts)
        self.facts_by_pred = {}
        for f in self.facts:
            self.facts_by_pred.setdefault(f[0], []).append(f)

    # ---------------------------------------------------------------- proving (counts steps)
    def prove_conj(self, goals, theta):
        """Generator of (theta, [proof trees]) for a conjunction of goals."""
        if not goals:
            yield theta, []
            return
        first, rest = goals[0], goals[1:]
        for th1, tree1 in self.prove_one(first, theta):
            for th2, trees in self.prove_conj(rest, th1):
                yield th2, [tree1] + trees

    def prove_one(self, goal, theta):
        goal = apply_theta(goal, theta)
        if goal[0] in self.facts_by_pred:
            self.steps += 1                                           # fact lookup
            for fact in self.facts_by_pred[goal[0]]:
                th = unify(goal, fact, theta)
                if th is not None:
                    yield th, ("FACT", apply_theta(goal, th), [])
        for rule in self.rules:
            if rule.head[0] != goal[0]:
                continue
            self.steps += 1                                           # rule-head unification attempt
            r = standardize_apart(rule)
            th = unify(goal, r.head, theta)
            if th is None:
                continue
            for th2, subtrees in self.prove_conj(r.body, th):
                yield th2, (rule.name, apply_theta(goal, th2), subtrees, r)   # r keeps the renamed rule instance

    def holds(self, goal):
        """Return (True/False, steps used) for a ground goal."""
        before = self.steps
        for _ in self.prove_one(goal, {}):
            return True, self.steps - before
        return False, self.steps - before

    def explain(self, goal):
        """First proof tree of a ground goal (the explanation) or None."""
        for _, tree in self.prove_one(goal, {}):
            return tree
        return None


# ------------------------------------------------------------------------------------------------ explanation helpers
def tree_text(tree, indent=0):
    if tree[0] == "FACT":
        return "  " * indent + "FACT  " + fmt(tree[1])
    out = ["  " * indent + "%s  =>  %s" % (tree[0], fmt(tree[1]))]
    for sub in tree[2]:
        out.append(tree_text(sub, indent + 1))
    return "\n".join(out)


def count_nodes(tree):
    return 1 + sum(count_nodes(s) for s in tree[2]) if tree[0] != "FACT" else 1


def ebg(tree, operational):
    """
    Explanation-Based Generalisation.
    Walk the explanation structure, but use the RULES' variables instead of the example's constants:
        1. unify the general goal with the (renamed) rule head
        2. recurse into each body literal with the substitution gathered so far
        3. a leaf is an operational literal: collect it, in generalised form
    Returns the generalised (head, [operational conditions]) as a new rule.
    """
    root_var = ("%s" % tree[1][0],) + tuple("?g%d" % i for i, _ in enumerate(tree[1][1:]))   # most general goal pattern
    leaves, theta = [], {}

    def rec(node, goal, theta):
        if node[0] == "FACT":
            assert node[1][0] in operational, "leaf %s is not operational" % node[1][0]
            leaves.append(goal)
            return theta
        rule_inst = node[3]
        # fresh copy of the rule so that variables are not shared with the explanation of the example
        std = standardize_apart(Rule(rule_inst.name, rule_inst.head, rule_inst.body))
        th = unify(goal, std.head, theta)
        for sub, body_lit in zip(node[2], std.body):
            th = rec(sub, apply_theta(body_lit, th), th)
        return th

    final = rec(tree, root_var, theta)
    head = apply_theta(root_var, final)
    body = [apply_theta(l, final) for l in leaves]
    # canonical variable names: ?x1, ?x2, ...
    names, canon = {}, lambda a: names.setdefault(a, "?x%d" % (len(names) + 1)) if is_var(a) else a
    head = (head[0],) + tuple(canon(a) for a in head[1:])
    body = [(l[0],) + tuple(canon(a) for a in l[1:]) for l in body]
    return Rule("LEARNED", head, body)
