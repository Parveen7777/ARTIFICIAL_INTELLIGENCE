"""
logic.py  --  Propositional-logic core shared by Task 1, Task 2 and Task 5
CSA17 Artificial Intelligence - Assignment 3 (CO3)

Sentences are nested tuples:
    ('sym', 'P_1_1')  ('not', a)  ('and', a, b)  ('or', a, b)  ('imp', a, b)  ('iff', a, b)
Clauses (CNF) are frozensets of literal strings:  'A'  or  '~A'
"""
import heapq
import itertools
import time
from functools import reduce


# ----------------------------------------------------------------- constructors
def Sym(n): return ('sym', n)
def Not(a): return ('not', a)
def And(*a): return reduce(lambda x, y: ('and', x, y), a)
def Or(*a): return reduce(lambda x, y: ('or', x, y), a)
def Imp(a, b): return ('imp', a, b)
def Iff(a, b): return ('iff', a, b)


def symbols(e, acc=None):
    acc = set() if acc is None else acc
    if e[0] == 'sym':
        acc.add(e[1])
    else:
        for x in e[1:]:
            symbols(x, acc)
    return acc


def show(e):
    """Pretty printer using logical symbols."""
    t = e[0]
    if t == 'sym': return e[1]
    if t == 'not': return '¬' + show(e[1]) if e[1][0] == 'sym' else '¬(' + show(e[1]) + ')'
    op = {'and': ' ∧ ', 'or': ' ∨ ', 'imp': ' ⇒ ', 'iff': ' ⇔ '}[t]
    return '(' + show(e[1]) + op + show(e[2]) + ')'


# ------------------------------------------------------------ truth-table checker
class Infeasible(Exception):
    pass


def _py(e, idx):
    t = e[0]
    if t == 'sym': return f'v[{idx[e[1]]}]'
    if t == 'not': return f'(not {_py(e[1], idx)})'
    a, b = _py(e[1], idx), _py(e[2], idx)
    if t == 'and': return f'({a} and {b})'
    if t == 'or': return f'({a} or {b})'
    if t == 'imp': return f'((not {a}) or {b})'
    if t == 'iff': return f'({a} == {b})'
    raise ValueError(t)


def compile_sentence(e, syms):
    idx = {s: i for i, s in enumerate(syms)}
    return eval('lambda v: ' + _py(e, idx))


def tt_entails(kb, alpha, syms=None, max_models=None):
    """TT-ENTAILS?(KB, alpha).  kb: list of sentences.
    Returns (result, stats) with stats = models enumerated, KB evaluations, query evaluations, time."""
    kb_all = reduce(lambda x, y: ('and', x, y), kb)
    if syms is None:
        syms = sorted(symbols(kb_all) | symbols(alpha))
    m = len(syms)
    if max_models is not None and 2 ** m > max_models:
        raise Infeasible(f'2^{m} = {2 ** m:.3e} models exceeds cap {max_models:.1e}')
    f_kb, f_a = compile_sentence(kb_all, syms), compile_sentence(alpha, syms)
    models = kb_evals = q_evals = 0
    t0 = time.perf_counter()
    result = True
    for v in itertools.product((False, True), repeat=m):
        models += 1
        kb_evals += 1
        if f_kb(v):
            q_evals += 1
            if not f_a(v):
                result = False
                break
    return result, dict(symbols=m, models=models, kb_evals=kb_evals, query_evals=q_evals,
                        time=time.perf_counter() - t0)


# --------------------------------------------------------------------- CNF
def elim_iff(e):
    t = e[0]
    if t == 'sym': return e
    if t == 'iff':
        a, b = elim_iff(e[1]), elim_iff(e[2])
        return ('and', ('imp', a, b), ('imp', b, a))
    return (t,) + tuple(elim_iff(x) for x in e[1:])


def elim_imp(e):
    t = e[0]
    if t == 'sym': return e
    if t == 'imp': return ('or', ('not', elim_imp(e[1])), elim_imp(e[2]))
    return (t,) + tuple(elim_imp(x) for x in e[1:])


def nnf(e):
    t = e[0]
    if t == 'sym': return e
    if t == 'not':
        x = e[1]
        if x[0] == 'sym': return e
        if x[0] == 'not': return nnf(x[1])
        if x[0] == 'and': return ('or', nnf(('not', x[1])), nnf(('not', x[2])))
        if x[0] == 'or': return ('and', nnf(('not', x[1])), nnf(('not', x[2])))
    return (t,) + tuple(nnf(x) for x in e[1:])


def distribute(e):
    """Input in NNF; returns a list of clauses (frozenset of literal strings)."""
    t = e[0]
    if t == 'sym': return [frozenset([e[1]])]
    if t == 'not': return [frozenset(['~' + e[1][1]])]
    if t == 'and': return distribute(e[1]) + distribute(e[2])
    if t == 'or':
        return [a | b for a in distribute(e[1]) for b in distribute(e[2])]
    raise ValueError(t)


def is_taut(c):
    return any(('~' + l) in c for l in c if not l.startswith('~'))


def to_cnf(e, drop_taut=True):
    cl = distribute(nnf(elim_imp(elim_iff(e))))
    out, seen = [], set()
    for c in cl:
        if drop_taut and is_taut(c): continue
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def cnf_steps(e):
    """Return the list of (step name, formula-text) used in the hand-conversion table."""
    s1 = elim_iff(e)
    s2 = elim_imp(s1)
    s3 = nnf(s2)
    cl = distribute(s3)
    return [('original', show(e)), ('1. eliminate ⇔', show(s1)), ('2. eliminate ⇒', show(s2)),
            ('3. push ¬ inward (NNF)', show(s3)),
            ('4. distribute ∨ over ∧ -> clauses', '  ∧  '.join(clause_str(c) for c in cl))]


def neg(l): return l[1:] if l.startswith('~') else '~' + l


def clause_str(c):
    return '{' + ', '.join(sorted(c, key=lambda x: (x.lstrip('~'), x))) + '}' if c else '{}'


# -------------------------------------------------------------- resolution prover
def resolve_prove(clauses, max_clauses=300000, subsumption=True, unit_pref=True, sos=None):
    """Given-clause propositional resolution refutation.
    clauses : list of frozensets (KB clauses + negated query).
    sos     : optional list of clauses forming the set of support (None = all clauses).
    Returns (proved, stats, parents)."""
    st = dict(generated=0, attempted=0, tautologies=0, subsumed=0, duplicates=0, kept=0,
              time=0.0, capped=False)
    t0 = time.perf_counter()
    parents = {}
    alive = {}                       # clause -> True while kept
    lit_index = {}                   # literal -> set(clauses) (all kept clauses)
    proc_index = {}                  # literal -> list(clauses) processed (usable)
    agenda, counter = [], itertools.count()

    def add_index(c):
        for l in c: lit_index.setdefault(l, set()).add(c)

    def del_index(c):
        for l in c: lit_index[l].discard(c)

    def subsumed_by_existing(c):
        for l in c:
            for d in lit_index.get(l, ()):
                if d <= c and d != c: return True
        return False

    def remove_subsumed_by(c):
        if not c: return
        l0 = next(iter(c))
        for d in list(lit_index.get(l0, ())):
            if c < d and alive.get(d):
                alive[d] = False
                del_index(d)
                st['subsumed'] += 1

    sos_set = set(sos) if sos is not None else None
    in_sos = {}

    def push(c, from_sos):
        alive[c] = True
        add_index(c)
        st['kept'] += 1
        in_sos[c] = from_sos
        key = (len(c) if unit_pref else 0, next(counter))
        heapq.heappush(agenda, (key, id(c), c))

    seen = set()
    for c in clauses:
        if is_taut(c):
            st['tautologies'] += 1
            continue
        if c in seen: continue
        seen.add(c)
        if not c:
            st['time'] = time.perf_counter() - t0
            return True, st, parents
        if sos_set is not None and c not in sos_set:
            # usable (non-SoS) clause: pre-loaded as a partner, never selected as the given clause
            alive[c] = True
            add_index(c)
            st['kept'] += 1
            in_sos[c] = False
            for l in c: proc_index.setdefault(l, []).append(c)
        else:
            push(c, True)
    usable = []
    while agenda:
        _, _, g = heapq.heappop(agenda)
        if not alive.get(g): continue
        usable.append(g)
        for l in g: proc_index.setdefault(l, []).append(g)
        for l in list(g):
            for p in list(proc_index.get(neg(l), ())):
                if not alive.get(p): continue
                st['attempted'] += 1
                r = (g - {l}) | (p - {neg(l)})
                if is_taut(r):
                    st['tautologies'] += 1
                    continue
                if r in seen:
                    st['duplicates'] += 1
                    continue
                if subsumption and r and subsumed_by_existing(r):
                    st['subsumed'] += 1
                    seen.add(r)
                    continue
                seen.add(r)
                st['generated'] += 1
                parents[r] = (g, p, l)
                if not r:
                    st['time'] = time.perf_counter() - t0
                    return True, st, parents
                if subsumption: remove_subsumed_by(r)
                push(r, True)
                if st['generated'] >= max_clauses:
                    st['capped'] = True
                    st['time'] = time.perf_counter() - t0
                    return None, st, parents
    st['time'] = time.perf_counter() - t0
    return False, st, parents


def pl_entails(kb_sentences, alpha, **kw):
    """KB |= alpha  via resolution refutation of KB ∧ ¬alpha."""
    clauses = []
    for s in kb_sentences: clauses += to_cnf(s)
    q = to_cnf(('not', alpha))
    allc = clauses + q
    if kw.pop('use_sos', False):
        proved, st, par = resolve_prove(allc, sos=q, **kw)
    else:
        proved, st, par = resolve_prove(allc, **kw)
    return proved, st, par


def proof_lines(parents, root=frozenset()):
    """Linearised proof (post-order) of the empty clause."""
    order, seen = [], set()

    def go(c):
        if c in seen: return
        seen.add(c)
        if c in parents:
            a, b, l = parents[c]
            go(a); go(b)
        order.append(c)

    go(root)
    ids = {c: i + 1 for i, c in enumerate(order)}
    out = []
    for c in order:
        if c in parents:
            a, b, l = parents[c]
            out.append(f'  [{ids[c]:>2}] {clause_str(c):<34} <- resolve [{ids[a]}] , [{ids[b]}] on {l.lstrip("~")}')
        else:
            out.append(f'  [{ids[c]:>2}] {clause_str(c):<34} (input clause)')
    return out, order, ids


# ----------------------------------------------------------------------- DPLL
def dpll(clauses, assignment=None):
    """Tiny DPLL SAT solver (used only to cross-check entailment when truth tables are infeasible)."""
    assignment = dict(assignment or {})
    cl = [set(c) for c in clauses]
    while True:
        changed = False
        new = []
        for c in cl:
            if any(assignment.get(l.lstrip('~')) == (not l.startswith('~')) for l in c if l.lstrip('~') in assignment):
                continue
            c2 = {l for l in c if l.lstrip('~') not in assignment}
            if not c2: return None
            new.append(c2)
        cl = new
        for c in cl:
            if len(c) == 1:
                l = next(iter(c))
                assignment[l.lstrip('~')] = not l.startswith('~')
                changed = True
                break
        if not changed: break
    if not cl: return assignment
    l = next(iter(cl[0]))
    v = l.lstrip('~')
    for val in (not l.startswith('~'), l.startswith('~')):
        r = dpll(cl, {**assignment, v: val})
        if r is not None: return r
    return None


def dpll_entails(kb, alpha):
    cl = []
    for s in kb: cl += to_cnf(s)
    cl += to_cnf(('not', alpha))
    return dpll(cl) is None
