"""
fol.py  --  First-order logic core shared by Tasks 3, 4, 5, 6 and 7
CSA17 Artificial Intelligence - Assignment 3 (CO3)

Representation (plain Python, no external libraries)
    variable   : string starting with '?'            e.g. '?x'
    constant   : any other string                    e.g. 'John'
    compound   : tuple (functor, arg1, ..., argk)    e.g. ('Mother', '?y')
    atom       : tuple (Predicate, arg1, ..., argk)  e.g. ('Knows', 'John', '?x')
    literal in a rule body may also be ('not', atom) (negation as failure) or a builtin
    ('lt'|'gt'|'le'|'ge'|'neq', a, b) evaluated on ground numbers / constants.
"""
import itertools
import sys
import time
from collections import Counter, defaultdict

sys.setrecursionlimit(50000)


# ------------------------------------------------------------------- terms
def is_var(t): return isinstance(t, str) and t[:1] == '?'


def walk(t, th):
    while is_var(t) and t in th:
        t = th[t]
    return t


def occurs_in(v, t, th):
    t = walk(t, th)
    if t == v: return True
    if isinstance(t, tuple): return any(occurs_in(v, a, th) for a in t[1:])
    return False


def _bind(v, t, th, oc):
    if oc and occurs_in(v, t, th): return None
    n = dict(th)
    n[v] = t
    return n


def unify(x, y, th, oc=True):
    """Most general unifier of x and y extending substitution th (dict), or None.  oc = occurs check."""
    if th is None: return None
    x, y = walk(x, th), walk(y, th)
    if x == y: return th
    if is_var(x): return _bind(x, y, th, oc)
    if is_var(y): return _bind(y, x, th, oc)
    if isinstance(x, tuple) and isinstance(y, tuple):
        if x[0] != y[0] or len(x) != len(y): return None
        for a, b in zip(x[1:], y[1:]):
            th = unify(a, b, th, oc)
            if th is None: return None
        return th
    return None


def subst(th, t):
    t = walk(t, th)
    if isinstance(t, tuple): return (t[0],) + tuple(subst(th, a) for a in t[1:])
    return t


def mgu_dict(th):
    """Idempotent form of a (triangular) substitution."""
    return {v: subst(th, v) for v in th}


def compose(s1, s2):
    """(s1 ∘ s2): apply s1 first, then s2.   compose(s1,s2)(t) == s2(s1(t))"""
    out = {v: subst(s2, t) for v, t in s1.items()}
    for v, t in s2.items():
        if v not in out: out[v] = t
    return {v: t for v, t in out.items() if v != t}


def variables(t, acc=None):
    acc = [] if acc is None else acc
    if is_var(t):
        if t not in acc: acc.append(t)
    elif isinstance(t, tuple):
        for a in t[1:]: variables(a, acc)
    return acc


_ctr = itertools.count(1)


def rename(t, suf):
    if is_var(t): return t + suf
    if isinstance(t, tuple): return (t[0],) + tuple(rename(a, suf) for a in t[1:])
    return t


def fresh(): return f'_{next(_ctr)}'


def variant_key(t):
    """Canonical key of a term up to renaming of variables (used for loop checks / tabling)."""
    m = {}
    def go(x):
        if is_var(x):
            if x not in m: m[x] = f'?{len(m)}'
            return m[x]
        if isinstance(x, tuple): return (x[0],) + tuple(go(a) for a in x[1:])
        return x
    return go(t)


def term_str(t):
    if isinstance(t, tuple):
        return f'{t[0]}(' + ','.join(term_str(a) for a in t[1:]) + ')' if len(t) > 1 else str(t[0])
    return str(t)


def sub_str(th):
    return '{' + ', '.join(f'{v[1:] if v.startswith("?") else v}/{term_str(t)}' for v, t in sorted(th.items())) + '}'


# ------------------------------------------------------------------- lifted Modus Ponens
def lifted_modus_ponens(premises, conclusion, known_facts):
    """Generalised Modus Ponens.  premises: list of atoms p_i (may contain variables),
    known_facts: list of atoms p_i' ; returns list of (theta, subst(conclusion))."""
    out = []
    suf = fresh()
    prem = [rename(p, suf) for p in premises]
    concl = rename(conclusion, suf)
    def go(i, th):
        if i == len(prem):
            out.append((mgu_dict(th), subst(th, concl)))
            return
        for f in known_facts:
            th2 = unify(prem[i], rename(f, fresh()), th)
            if th2 is not None: go(i + 1, th2)
    go(0, {})
    return out


# ------------------------------------------------------------------- engine
class CapExceeded(Exception):
    pass


class Stats:
    def __init__(self):
        self.goals = 0           # goals expanded
        self.unifs = 0           # unification attempts
        self.max_depth = 0
        self.fact_matches = 0
        self.rule_apps = 0
        self.repeated = Counter()  # variant key -> times the (sub)goal was expanded
        self.pruned = 0
        self.memo_hits = 0
        self.tab_lookups = 0

    @property
    def repeated_subgoals(self):
        return sum(c - 1 for c in self.repeated.values() if c > 1)

    def as_dict(self):
        return dict(goals=self.goals, unifs=self.unifs, max_depth=self.max_depth, rule_apps=self.rule_apps,
                    repeated=self.repeated_subgoals, pruned=self.pruned, memo_hits=self.memo_hits)


BUILTINS = {'lt', 'gt', 'le', 'ge', 'neq', 'eq'}


def _num(x):
    try: return float(x)
    except (TypeError, ValueError): return x


def eval_builtin(b):
    op, a, c = b[0], _num(b[1]), _num(b[2])
    return {'lt': lambda: a < c, 'gt': lambda: a > c, 'le': lambda: a <= c, 'ge': lambda: a >= c,
            'neq': lambda: a != c, 'eq': lambda: a == c}[op]()


class KB:
    """Definite-clause knowledge base with indexed facts and rules, SLD backward chaining, NAF,
    loop check, depth bound, ground memoisation and (for Datalog) tabling."""

    def __init__(self):
        self.facts = defaultdict(list)      # pred -> [atom]
        self.fidx = defaultdict(list)       # (pred, first arg) -> [atom]
        self.factset = set()
        self.rules = defaultdict(list)      # pred -> [rule]
        self.rkey = defaultdict(list)       # (pred, constant first arg of head) -> [rule]
        self.rgen = defaultdict(list)       # pred -> rules whose head first arg is not a constant
        self.all_rules = []
        self.names = {}

    # --- construction
    def add_fact(self, atom):
        if atom in self.factset: return
        self.factset.add(atom)
        self.facts[atom[0]].append(atom)
        if len(atom) > 1: self.fidx[(atom[0], atom[1])].append(atom)

    def add_rule(self, head, body, name=None):
        r = (head, tuple(body), name or f'R{len(self.all_rules) + 1}')
        self.all_rules.append(r)
        self.rules[head[0]].append(r)
        if len(head) > 1 and not is_var(head[1]) and not isinstance(head[1], tuple):
            self.rkey[(head[0], head[1])].append(r)
        else:
            self.rgen[head[0]].append(r)

    def size(self):
        return len(self.factset) + len(self.all_rules)

    # --- candidate retrieval
    def _facts_for(self, g, th):
        if len(g) > 1:
            a0 = walk(g[1], th)
            if not is_var(a0) and not isinstance(a0, tuple):
                return self.fidx.get((g[0], a0), ())
        return self.facts.get(g[0], ())

    def _rules_for(self, g, th, index_rules):
        if not index_rules: return self.all_rules
        if len(g) > 1:
            a0 = walk(g[1], th)
            if not is_var(a0) and not isinstance(a0, tuple):
                return self.rkey.get((g[0], a0), []) + self.rgen.get(g[0], [])
        return self.rules.get(g[0], ())

    # --- SLD backward chaining
    def solve(self, goals, th=None, st=None, depth_limit=None, loop_check=False, cap=None,
              index_rules=True, memo=None, proof=False):
        """Generator of (theta, proof) for the conjunction `goals`. Raises CapExceeded when cap goals are expanded."""
        st = st or Stats()
        opts = dict(depth_limit=depth_limit, loop_check=loop_check, cap=cap, index_rules=index_rules,
                    memo=memo, proof=proof, st=st)
        yield from self._solve(list(goals), th or {}, 0, (), opts, None)

    def _solve(self, goals, th, depth, anc, o, prf):
        if not goals:
            yield th, prf
            return
        g, rest = goals[0], goals[1:]
        st = o['st']
        if g[0] == 'not':
            a = subst(th, g[1])
            found = False
            for _ in self._solve([a], {}, depth + 1, (), o, None):
                found = True
                break
            if not found:
                yield from self._solve(rest, th, depth, anc, o, ('N', a, prf) if o['proof'] else None)
            return
        if g[0] in BUILTINS:
            gs = subst(th, g)
            if eval_builtin(gs):
                yield from self._solve(rest, th, depth, anc, o, ('B', gs, prf) if o['proof'] else None)
            return
        if o['depth_limit'] is not None and depth > o['depth_limit']:
            st.pruned += 1
            return
        gs = subst(th, g)
        key = variant_key(gs)
        if o['loop_check'] and key in anc:
            st.pruned += 1
            return
        memo = o['memo']
        ground = not variables(gs)
        if memo is not None and ground and key in memo:
            st.memo_hits += 1
            if memo[key]:
                yield from self._solve(rest, th, depth, anc, o, prf)
            return
        st.goals += 1
        st.repeated[key] += 1
        st.max_depth = max(st.max_depth, depth)
        if o['cap'] is not None and st.goals > o['cap']:
            raise CapExceeded(f'goal-expansion cap {o["cap"]} reached')
        success = False
        for f in self._facts_for(g, th):
            st.unifs += 1
            th2 = unify(g, f, th)
            if th2 is not None:
                st.fact_matches += 1
                success = True
                yield from self._solve(rest, th2, depth, anc, o, ('F', f, prf) if o['proof'] else None)
                if memo is not None and ground: break
        for r in self._rules_for(g, th, o['index_rules']):
            head, body, name = r
            if head[0] != g[0]: continue
            suf = fresh()
            h2 = rename(head, suf)
            st.unifs += 1
            th2 = unify(g, h2, th)
            if th2 is None: continue
            st.rule_apps += 1
            b2 = [rename(b, suf) for b in body]
            p2 = ('R', name, h2, tuple(b2), prf) if o['proof'] else None
            for th3, pr3 in self._solve(b2, th2, depth + 1, anc + (key,), o, p2):
                success = True
                yield from self._solve(rest, th3, depth, anc, o, pr3)
                if memo is not None and ground: break
            if memo is not None and ground and success: break
        if memo is not None and ground and key not in memo:
            memo[key] = success

    def ask(self, goal, **kw):
        """All solutions of a single goal: list of dict var->value (only the goal's own variables)."""
        vs = variables(goal)
        out, seen = [], set()
        for th, _ in self.solve([goal], **kw):
            ans = tuple(term_str(subst(th, v)) for v in vs)
            if ans not in seen:
                seen.add(ans)
                out.append({v: subst(th, v) for v in vs})
        return out

    def prove_once(self, goal, **kw):
        for th, prf in self.solve([goal], **kw):
            return th, prf
        return None, None

    def holds(self, goal, **kw):
        return self.prove_once(goal, **kw)[0] is not None

    @staticmethod
    def explain(th, prf, indent='  '):
        """Turn the linked proof record into readable lines (instantiated with the final substitution)."""
        items = []
        while prf is not None:
            items.append(prf[:-1])
            prf = prf[-1]
        items.reverse()
        lines = []
        for it in items:
            if it[0] == 'F': lines.append(f'{indent}fact   {term_str(subst(th, it[1]))}')
            elif it[0] == 'N': lines.append(f'{indent}not    {term_str(subst(th, it[1]))}   (negation as failure)')
            elif it[0] == 'B': lines.append(f'{indent}check  {term_str(subst(th, it[1]))}')
            else:
                body = ' ∧ '.join(term_str(subst(th, b)) for b in it[3])
                lines.append(f'{indent}rule {it[1]}: {term_str(subst(th, it[2]))}  <=  {body}')
        return lines

    # --- tabling (SLG-style, naive re-evaluation) for Datalog programs: terminates on left recursion
    def tabled(self, goal, st=None, cap=None):
        st = st or Stats()
        self._tab = dict(table={}, done={}, prog=set(), st=st, cap=cap)
        ans = self._tab_goal(goal)
        return sorted(ans), st

    def _tab_goal(self, goal):
        T = self._tab
        st = T['st']
        key = variant_key(goal)
        if key in T['done']: return T['done'][key]
        if key in T['prog']:
            st.tab_lookups += 1
            self._deps.add(key)
            return set(T['table'][key])
        saved_deps = getattr(self, '_deps', set())
        self._deps = set()
        T['prog'].add(key)
        T['table'][key] = set()
        while True:
            before = len(T['table'][key])
            st.goals += 1
            st.repeated[key] += 1
            if T['cap'] is not None and st.goals > T['cap']: raise CapExceeded('tabling cap reached')
            for f in self._facts_for(goal, {}):
                st.unifs += 1
                if unify(goal, f, {}) is not None: T['table'][key].add(f)
            for head, body, name in self._rules_for(goal, {}, True):
                if head[0] != goal[0]: continue
                suf = fresh()
                h2 = rename(head, suf)
                st.unifs += 1
                th = unify(goal, h2, {})
                if th is None: continue
                st.rule_apps += 1
                for th2 in self._tab_body([rename(b, suf) for b in body], th):
                    T['table'][key].add(subst(th2, goal))
            if len(T['table'][key]) == before: break
        T['prog'].discard(key)
        mine = self._deps - {key}
        if not mine:
            T['done'][key] = set(T['table'][key])
        self._deps = saved_deps | mine
        return T['table'][key]

    def _tab_body(self, body, th):
        if not body:
            yield th
            return
        g, rest = body[0], body[1:]
        if g[0] == 'not':
            a = subst(th, g[1])
            if not self._tab_goal(a): yield from self._tab_body(rest, th)
            return
        if g[0] in BUILTINS:
            if eval_builtin(subst(th, g)): yield from self._tab_body(rest, th)
            return
        gs = subst(th, g)
        for a in list(self._tab_goal(gs)):
            self._tab['st'].unifs += 1
            th2 = unify(gs, a, th)
            if th2 is not None: yield from self._tab_body(rest, th2)


# ------------------------------------------------------------------- propositional (ground) Horn engine
def ground_horn_fc(rules, facts):
    """Forward chaining over ground definite clauses (AIMA PL-FC-ENTAILS) with counters.
    rules: list of (head, [body atoms]); facts: iterable of atoms.  Returns (inferred set, steps dict)."""
    count = [len(b) for h, b in rules]
    by_prem = defaultdict(list)
    for i, (h, b) in enumerate(rules):
        for p in b: by_prem[p].append(i)
    inferred, agenda = set(), list(facts)
    steps = dict(pops=0, decrements=0, firings=0)
    while agenda:
        p = agenda.pop()
        steps['pops'] += 1
        if p in inferred: continue
        inferred.add(p)
        for i in by_prem[p]:
            count[i] -= 1
            steps['decrements'] += 1
            if count[i] == 0:
                steps['firings'] += 1
                agenda.append(rules[i][0])
    return inferred, steps


def ground_horn_index(rules):
    """Index ground definite clauses by head (built once per KB, like a database index)."""
    heads = defaultdict(list)
    for h, b in rules: heads[h].append(b)
    return heads


def ground_horn_bc(heads, facts, goal):
    """Backward chaining over indexed ground definite clauses (loop-checked). Returns (result, stats)."""
    st = dict(goals=0)
    fs = facts if isinstance(facts, (set, frozenset)) else set(facts)
    def go(g, stack):
        st['goals'] += 1
        if g in fs: return True
        if g in stack: return False
        for b in heads.get(g, ()):
            if all(go(x, stack | {g}) for x in b): return True
        return False
    return go(goal, frozenset()), st
