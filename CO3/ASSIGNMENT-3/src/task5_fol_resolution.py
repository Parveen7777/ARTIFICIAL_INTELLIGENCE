"""
Task 5 - Upgrading Resolution from Propositional to First-Order Logic
 * English -> FOL -> CNF (every step printed, Skolemisation)
 * first-order resolution prover built on the Task 4 unifier, strategies: none / unit-preference / set-of-support
 * entailed query (exists an authorised employee) and a non-entailed query with step cap
 * comparison with the propositional prover of Task 2 on the equivalent finite domain
Run: python src/task5_fol_resolution.py
"""
import itertools
import json
import re
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from fol import unify, subst, rename, variables, term_str, is_var, mgu_dict, sub_str
from logic import resolve_prove, Infeasible
from viz_util import Tee, draw_dag

OUT = os.path.join(os.path.dirname(__file__), '..', 'outputs')
PLOTS = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(OUT, exist_ok=True); os.makedirs(PLOTS, exist_ok=True)
sys.stdout = Tee(os.path.join(OUT, 'task5_output.txt'))

CONN = {'forall', 'exists', 'and', 'or', 'not', 'imp', 'iff'}


def A(p, *a): return (p,) + a
def And(*f):
    r = f[0]
    for x in f[1:]: r = ('and', r, x)
    return r
def Or(*f):
    r = f[0]
    for x in f[1:]: r = ('or', r, x)
    return r
def Not(f): return ('not', f)
def Imp(a, b): return ('imp', a, b)
def All(v, f): return ('forall', v, f)
def Ex(v, f): return ('exists', v, f)


def fstr(f):
    op = f[0]
    if op not in CONN: return term_str(f).replace('?', '')
    if op == 'not': return '¬' + fstr(f[1])
    if op in ('forall', 'exists'): return ('∀' if op == 'forall' else '∃') + f[1].lstrip('?') + ' ' + fstr(f[2])
    sym = {'and': ' ∧ ', 'or': ' ∨ ', 'imp': ' ⇒ ', 'iff': ' ⇔ '}[op]
    return '(' + fstr(f[1]) + sym + fstr(f[2]) + ')'


# ------------------------------------------------------------- CNF pipeline
def elim_imp(f):
    op = f[0]
    if op == 'imp': return ('or', ('not', elim_imp(f[1])), elim_imp(f[2]))
    if op == 'iff': return elim_imp(And(Imp(f[1], f[2]), Imp(f[2], f[1])))
    if op in ('and', 'or'): return (op, elim_imp(f[1]), elim_imp(f[2]))
    if op == 'not': return ('not', elim_imp(f[1]))
    if op in ('forall', 'exists'): return (op, f[1], elim_imp(f[2]))
    return f


def nnf(f):
    op = f[0]
    if op in ('and', 'or'): return (op, nnf(f[1]), nnf(f[2]))
    if op in ('forall', 'exists'): return (op, f[1], nnf(f[2]))
    if op == 'not':
        g = f[1]; gop = g[0]
        if gop == 'not': return nnf(g[1])
        if gop == 'and': return ('or', nnf(('not', g[1])), nnf(('not', g[2])))
        if gop == 'or': return ('and', nnf(('not', g[1])), nnf(('not', g[2])))
        if gop == 'forall': return ('exists', g[1], nnf(('not', g[2])))
        if gop == 'exists': return ('forall', g[1], nnf(('not', g[2])))
        return f
    return f


_vc = itertools.count(1)
_sk = itertools.count(1)


def sub_formula(f, v, t):
    op = f[0]
    if op in ('forall', 'exists'):
        return f if f[1] == v else (op, f[1], sub_formula(f[2], v, t))
    if op == 'not': return ('not', sub_formula(f[1], v, t))
    if op in ('and', 'or'): return (op, sub_formula(f[1], v, t), sub_formula(f[2], v, t))
    return subst({v: t}, f)


def standardize(f):
    op = f[0]
    if op in ('forall', 'exists'):
        nv = f'{f[1]}{next(_vc)}'
        return (op, nv, standardize(sub_formula(f[2], f[1], nv)))
    if op == 'not': return ('not', standardize(f[1]))
    if op in ('and', 'or'): return (op, standardize(f[1]), standardize(f[2]))
    return f


def skolemize(f, univ=(), log=None):
    op = f[0]
    if op == 'forall': return ('forall', f[1], skolemize(f[2], univ + (f[1],), log))
    if op == 'exists':
        n = next(_sk)
        if univ:
            t = (f'sk{n}',) + tuple(univ)
            if log is not None: log.append(f'∃{f[1].lstrip("?")} is inside the scope of ∀{",".join(u.lstrip("?") for u in univ)}  ->  Skolem FUNCTION {term_str(t).replace("?", "")}')
        else:
            t = f'SK{n}'
            if log is not None: log.append(f'∃{f[1].lstrip("?")} has NO enclosing ∀  ->  Skolem CONSTANT {t}')
        return skolemize(sub_formula(f[2], f[1], t), univ, log)
    if op in ('and', 'or'): return (op, skolemize(f[1], univ, log), skolemize(f[2], univ, log))
    return f


def drop_forall(f):
    """Remove every ∀ (variables are standardised apart, so pulling quantifiers out of ∧/∨ is safe)."""
    if f[0] == 'forall': return drop_forall(f[2])
    if f[0] in ('and', 'or'): return (f[0], drop_forall(f[1]), drop_forall(f[2]))
    return f


def distribute(f):
    op = f[0]
    if op == 'and': return distribute(f[1]) + distribute(f[2])
    if op == 'or':
        return [a + b for a in distribute(f[1]) for b in distribute(f[2])]
    if op == 'not': return [((False, f[1]),)]
    return [((True, f),)]


def canon(clause):
    """variant-invariant key of a clause (variables renamed in order of appearance after sorting)."""
    m = {}
    def go(t):
        if is_var(t):
            m.setdefault(t, f'?{len(m)}')
            return m[t]
        if isinstance(t, tuple): return (t[0],) + tuple(go(a) for a in t[1:])
        return t
    lits = sorted(clause, key=lambda l: (l[1][0], l[0], str(l[1])))
    return tuple((s, go(a)) for s, a in lits)


def _pretty_vars(c):
    names = iter('xyzuvwabcdefgh')
    m = {}
    def go(t):
        if is_var(t):
            if t not in m: m[t] = '?' + next(names)
            return m[t]
        if isinstance(t, tuple): return (t[0],) + tuple(go(a) for a in t[1:])
        return t
    return tuple((s_, go(a)) for s_, a in c)


def cl_str(c):
    if not c: return '□'
    return ' ∨ '.join(('' if s else '¬') + term_str(a).replace('?', '') for s, a in _pretty_vars(c))


def to_cnf(f, name, verbose=True):
    log = []
    s1 = elim_imp(f); s2 = nnf(s1); s3 = standardize(s2); s4 = skolemize(s3, (), log)
    s5 = drop_forall(s4); cl = distribute(s5)
    out, seen = [], set()
    for c in cl:
        # remove duplicate literals / tautologies
        c = tuple(dict.fromkeys(c))
        if any((not s, a) in c for s, a in c): continue
        k = canon(c)
        if k in seen: continue
        seen.add(k); out.append(c)
    if verbose:
        print(f'\n  {name}')
        print(f'    FOL                    : {fstr(f)}')
        print(f'    1. eliminate ⇒/⇔       : {fstr(s1)}')
        print(f'    2. negations inward    : {fstr(s2)}')
        print(f'    3. standardise vars    : {fstr(s3)}')
        print(f'    4. Skolemise           : {fstr(s4)}')
        for l in log: print(f'         · {l}')
        print(f'    5. drop ∀ (prenex)     : {fstr(s5)}')
        print(f'    6. distribute ∨ over ∧ : ' + '  ∧  '.join('(' + cl_str(c) + ')' for c in out))
    return out


# ------------------------------------------------------------- the prover
class Result:
    pass


def resolvents(c1, c2, st):
    out = []
    suf = f'_{next(_vc)}'
    c2r = tuple((s, rename(a, suf)) for s, a in c2)
    for i, (s1, a1) in enumerate(c1):
        for j, (s2, a2) in enumerate(c2r):
            if s1 == s2 or a1[0] != a2[0] or len(a1) != len(a2): continue
            st['unifs'] += 1
            th = unify(a1, a2, {})
            if th is None: continue
            rest = [l for k, l in enumerate(c1) if k != i] + [l for k, l in enumerate(c2r) if k != j]
            new = tuple(dict.fromkeys((s, subst(th, a)) for s, a in rest))
            out.append((new, th))
    return out


def prove(kb_clauses, goal_clauses, strategy='none', max_steps=5000, max_clauses=4000):
    """Given-clause resolution.  strategy in {'none','unit','sos','sos+unit'}."""
    st = dict(unifs=0, steps=0, generated=0, time=0.0)
    t0 = time.perf_counter()
    nodes = {}                   # id -> (clause, parents(ids), theta)
    keys = {}
    usable, sos = [], []
    def add(c, par, th, to_sos):
        k = canon(c)
        if k in keys: return None
        cid = len(nodes) + 1
        keys[k] = cid; nodes[cid] = (c, par, th)
        (sos if to_sos else usable).append(cid)
        return cid
    use_sos = strategy.startswith('sos')
    for c in kb_clauses: add(c, (), None, not use_sos)
    for c in goal_clauses: add(c, (), None, True)
    given_cnt = 0
    while sos:
        if strategy.endswith('unit'):
            gid = min(sos, key=lambda i: (len(nodes[i][0]), i))
            sos.remove(gid)
        else:
            gid = sos.pop(0)
        usable.append(gid)
        g = nodes[gid][0]
        given_cnt += 1
        for pid in list(usable):
            for new, th in resolvents(g, nodes[pid][0], st):
                st['steps'] += 1
                if any((not s, a) in new for s, a in new):
                    continue
                if st['steps'] > max_steps or len(nodes) > max_clauses:
                    st['time'] = time.perf_counter() - t0
                    st['generated'] = len(nodes) - len(kb_clauses) - len(goal_clauses)
                    return 'CAP', st, nodes, given_cnt
                nid = add(new, (gid, pid), th, True)
                if nid is not None and not new:
                    st['time'] = time.perf_counter() - t0
                    st['generated'] = len(nodes) - len(kb_clauses) - len(goal_clauses)
                    return 'PROVED', st, nodes, given_cnt
    st['time'] = time.perf_counter() - t0
    st['generated'] = len(nodes) - len(kb_clauses) - len(goal_clauses)
    return 'SATURATED', st, nodes, given_cnt


def proof_nodes(nodes, root):
    need, stack = set(), [root]
    while stack:
        i = stack.pop()
        if i in need: continue
        need.add(i); stack += list(nodes[i][1])
    return sorted(need)


# ------------------------------------------------------------- the problem
x, d = '?x', '?d'
S1 = All(x, All(d, Imp(And(A('Employee', x), A('Document', d), A('Restricted', d), A('Handled', x, d)), A('Authorised', x))))
S2 = All(d, Imp(And(A('Document', d), A('ContainsPatientData', d)), A('Restricted', d)))
S3 = Ex(x, And(A('Employee', x), All(d, Imp(And(A('Document', d), A('Audited', d)), A('Handled', x, d)))))
S4 = And(A('ContainsPatientData', 'R1'), A('Audited', 'R1'), A('Report', 'R1'))
S5 = All(d, Imp(A('Report', d), A('Document', d)))        # background knowledge: every report is a document
S6 = All(d, Imp(A('Document', d), Ex('?r', And(A('Revision', '?r', d), A('Document', '?r')))))   # extension with a function symbol
Q1 = Ex(x, A('Authorised', x))
Q2 = Ex(x, And(A('Employee', x), Not(A('Authorised', x))))

STATEMENTS = [('S1  Every employee who handled a restricted document is authorised', S1),
              ('S2  Every document containing patient data is restricted', S2),
              ('S3  Some employee handled every document that was audited', S3),
              ('S4  Report R1 contains patient data and R1 was audited', S4),
              ('S5  (background) every report is a document', S5)]

if __name__ == '__main__':
    R = {}
    print('=' * 100); print('TASK 5  -  First-order resolution'); print('=' * 100)
    print('\n[1] English -> FOL -> CNF  (every conversion step)')
    kb = []
    for name, f in STATEMENTS:
        kb += to_cnf(f, name)
    print('\n  Negated query  Q1: ∃x Authorised(x)   ->   negate: ∀x ¬Authorised(x)')
    goal1 = to_cnf(Not(Q1), 'NEG-Q1  ¬(∃x Authorised(x))')
    print('\n  KB clause list (numbered):')
    for i, c in enumerate(kb + goal1, 1): print(f'    C{i:<2} {cl_str(c)}')
    print('\n  Skolem justification: S3 = ∃x (Employee(x) ∧ ∀d …). The ∃ is not inside any ∀, so the witness does not depend on d:')
    print('  a Skolem CONSTANT SK1 suffices.  (If the sentence were ∀d ∃x Handled(x,d), the witness could differ per document -> Skolem function sk(d).)')

    print('\n[2] Resolution results for Q1 (entailed) with three strategies')
    print(f'  {"strategy":<12} {"outcome":<10} {"clauses gen":>11} {"res. steps":>10} {"unifications":>12} {"given clauses":>13} {"proof length":>12} {"time ms":>8}')
    R['q1'] = {}
    best = None
    for strat in ('none', 'unit', 'sos', 'sos+unit'):
        out, st, nodes, gc = prove(kb, goal1, strat)
        root = max(nodes) if out == 'PROVED' else None
        plen = sum(1 for i in proof_nodes(nodes, root) if nodes[i][1]) if root else None
        print(f'  {strat:<12} {out:<10} {st["generated"]:>11} {st["steps"]:>10} {st["unifs"]:>12} {gc:>13} {str(plen):>12} {st["time"] * 1000:>8.2f}')
        R['q1'][strat] = dict(outcome=out, generated=st['generated'], steps=st['steps'], unifs=st['unifs'], proof_len=plen)
        if strat == 'sos+unit': best = (nodes, root)
    nodes, root = best
    pn = proof_nodes(nodes, root)
    print('\n[3] Proof found with set-of-support + unit preference')
    lab = {}
    for k, i in enumerate(pn, 1): lab[i] = k
    dag = []
    for i in pn:
        c, par, th = nodes[i]
        if par:
            print(f'   [{lab[i]:>2}] {cl_str(c):<44} <- resolve [{lab[par[0]]}],[{lab[par[1]]}]   θ = {re.sub(r'_\d+', '', sub_str(mgu_dict(th)))}')
        else:
            print(f'   [{lab[i]:>2}] {cl_str(c):<44} (input)')
        dag.append(dict(id=lab[i], label=cl_str(c), parents=[lab[p] for p in par], kind='empty' if not c else ('derived' if par else 'input')))
    draw_dag(dag, os.path.join(PLOTS, 'task5_proof_tree.png'), 'Task 5 - first-order resolution refutation of ¬∃x Authorised(x)  (set-of-support + unit preference)')
    print('   conclusion: an authorised employee exists; the binding found is x = SK1 (the employee who handled every audited document).')

    print('\n[4] Second query Q2: "some employee is NOT authorised" (not entailed)  -- step cap 3000 resolution steps')
    goal2 = to_cnf(Not(Q2), 'NEG-Q2  ¬(∃x Employee(x) ∧ ¬Authorised(x))', verbose=False)
    print('   negated Q2 clause:', ' ; '.join(cl_str(c) for c in goal2))
    R['q2'] = {}
    ext = to_cnf(S6, 'S6 (extension) every document has a revision that is also a document')
    kb_ext = kb + ext
    print(f'\n  {"KB":<28} {"strategy":<9} {"outcome":<10} {"clauses gen":>11} {"res. steps":>10} {"unifs":>8}')
    for label, K in (('finite (S1-S5)', kb), ('with function symbol (+S6)', kb_ext)):
        for strat in ('none', 'sos'):
            out, st, nodes2, gc = prove(K, goal2, strat, max_steps=3000, max_clauses=3000)
            print(f'  {label:<28} {strat:<9} {out:<10} {st["generated"]:>11} {st["steps"]:>10} {st["unifs"]:>8}')
            R['q2'][f'{label}|{strat}'] = dict(outcome=out, generated=st['generated'], steps=st['steps'])
    print('   Finite Herbrand universe {SK1, R1}: saturation ends -> "not entailed" (decided).')
    print('   With the Skolem function sk2(d): Document(sk2(R1)), Document(sk2(sk2(R1))), ... never ends -> cap reached, no verdict.')
    print('   => first-order entailment is only SEMI-decidable: a proof is found if KB |= α, but there is no guarantee of halting if KB ⊭ α.')

    print('\n[5] Equivalent small domain: propositional resolution (Task 2 prover) on the Herbrand instances over {SK1, R1}')
    consts = ['SK1', 'R1']
    pclauses = set()
    for c in kb + goal1:
        vs = []
        for s, a in c: variables(a, vs)
        for combo in itertools.product(consts, repeat=len(vs)):
            th = dict(zip(vs, combo))
            pclauses.add(frozenset(('' if s else '~') + term_str(subst(th, a)).replace('?', '') for s, a in c))
    pclauses = list(pclauses)
    print(f'   ground clauses over 2 constants = {len(pclauses)}   (first-order clauses = {len(kb) + len(goal1)})')
    R['prop'] = {}
    for label, kw in (('FIFO, no subsumption', dict(unit_pref=False, subsumption=False)), ('unit-pref + subsumption', dict())):
        proved, st, _ = resolve_prove(pclauses, **kw)
        print(f'   {label:<26} proved={proved}  clauses generated={st["generated"]}  attempts={st["attempted"]}  time={st["time"] * 1000:.2f} ms')
        R['prop'][label] = dict(proved=proved, generated=st['generated'], attempted=st['attempted'])
    goalc = [frozenset(['~Authorised(SK1)']), frozenset(['~Authorised(R1)'])]
    gset = set(pclauses)
    proved, st, _ = resolve_prove(pclauses, sos=[c for c in pclauses if all(l.startswith('~Authorised') for l in c)], unit_pref=True)
    print(f'   set-of-support (propositional)  proved={proved}  clauses generated={st["generated"]}  attempts={st["attempted"]}')
    fo = R['q1']
    print(f'   first-order (no strategy): {fo["none"]["generated"]} clauses, {fo["none"]["steps"]} steps;  first-order (SoS+unit): {fo["sos+unit"]["generated"]} clauses, {fo["sos+unit"]["steps"]} steps.')
    print('   Lifting keeps the clause set independent of the number of constants (propositional ground set grows k^vars) but each step costs a unification.')

    with open(os.path.join(OUT, 'task5_results.json'), 'w') as f: json.dump(R, f, indent=1, default=str)
    print('\nplots written: plots/task5_proof_tree.png')
