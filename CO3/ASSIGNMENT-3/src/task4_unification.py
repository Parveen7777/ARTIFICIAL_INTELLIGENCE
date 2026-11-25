"""
Task 4 - The Reality of Unification and Lifted Inference
 * UNIFY (occurs check optional, substitution composition), lifted Modus Ponens
 * propositionalisation of a FOL KB and growth of ground clauses (constants, arity, function symbols)
 * lifted vs propositional inference on the same query
Run: python src/task4_unification.py
"""
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from fol import *
from viz_util import Tee

OUT = os.path.join(os.path.dirname(__file__), '..', 'outputs')
PLOTS = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(OUT, exist_ok=True); os.makedirs(PLOTS, exist_ok=True)
sys.stdout = Tee(os.path.join(OUT, 'task4_output.txt'))


# ---------------------------------------------------------------- propositionalisation
def ground_terms(consts, funcs, depth):
    """All ground terms up to nesting depth `depth`.  funcs: list of (name, arity)."""
    terms = list(consts)
    for _ in range(depth):
        new = set(terms)
        for f, a in funcs:
            for args in itertools.product(terms, repeat=a):
                new.add((f,) + args)
        terms = sorted(new, key=term_str)
    return terms


def count_terms(n_consts, funcs, depth):
    """Closed recurrence for |terms of depth<=d|:  N_{d+1} = N_0 + sum_f (N_d)^arity(f)  (nested counts)."""
    N = n_consts
    for _ in range(depth):
        N = n_consts + sum(N ** a for f, a in funcs)
    return N


def propositionalise(rules, facts, terms):
    """Instantiate every universally quantified rule with all ground terms; returns ground Horn clauses."""
    ground = []
    for head, body in rules:
        vs = []
        for a in [head] + body: variables(a, vs)
        for combo in itertools.product(terms, repeat=len(vs)):
            th = dict(zip(vs, combo))
            ground.append((subst(th, head), [subst(th, b) for b in body]))
    return ground


def timed(f, *a, **k):
    t0 = time.perf_counter()
    r = f(*a, **k)
    return r, time.perf_counter() - t0


if __name__ == '__main__':
    R = {}
    print('=' * 100); print('TASK 4  -  Unification and lifted inference'); print('=' * 100)

    # ------------------------------------------------------------- 1. unifier tests
    print('\n[1] UNIFY tests (hand-derived MGUs are in the report; code must reproduce them)')
    pairs = [
        ('P1', ('Knows', 'John', '?x'), ('Knows', 'John', 'Jane'), {'?x': 'Jane'}),
        ('P2', ('Knows', 'John', '?x'), ('Knows', '?y', ('Mother', '?y')), {'?y': 'John', '?x': ('Mother', 'John')}),
        ('P3', ('Knows', 'John', '?x'), ('Knows', '?x', 'Elizabeth'), None),                      # clash without standardising apart
        ('P3*', ('Knows', 'John', '?x'), rename(('Knows', '?x', 'Elizabeth'), '17'), {'?x': 'Elizabeth', '?x17': 'John'}),
        ('P4', ('Q', '?x', ('f', '?x')), ('Q', '?y', '?y'), None),                               # occurs check matters
        ('P5', ('R', ('f', '?x'), ('g', '?y'), '?z'), ('R', ('f', 'a'), ('g', ('g', 'b')), ('h', '?x')),
         {'?x': 'a', '?y': ('g', 'b'), '?z': ('h', 'a')}),
        ('P6', ('P', 'a', '?x'), ('P', 'b', '?y'), None),                                       # constant clash
    ]
    ok_all = True
    print(f'  {"id":<4} {"term 1":<30} {"term 2":<34} {"MGU (occurs check ON)":<38} {"occurs check OFF":<28} ok')
    for pid, t1, t2, exp in pairs:
        th_on = unify(t1, t2, {}, True)
        th_off = unify(t1, t2, {}, False)
        got = mgu_dict(th_on) if th_on is not None else None
        ok = (got == exp)
        ok_all &= ok
        try:
            off = sub_str(mgu_dict(th_off)) if th_off is not None else 'FAIL'
        except RecursionError:                      # circular binding: x -> y -> f(x) -> ...
            off = '{x/y, y/f(x)} CIRCULAR'
        print(f'  {pid:<4} {term_str(t1):<30} {term_str(t2):<34} {(sub_str(got) if got is not None else "FAIL"):<38} {off:<28} {"ok" if ok else "WRONG"}')
    print('  all expectations met:', ok_all)
    th = unify(('Q', '?x', ('f', '?x')), ('Q', '?y', '?y'), {}, False)
    print('  Without the occurs check P4 returns', th, '-> the unifier claims success although no finite term satisfies x = f(x): UNSOUND.')

    print('\n  Substitution composition  (s1 ∘ s2)(t) = s2(s1(t)):')
    s1 = {'?x': ('f', '?y')}; s2 = {'?y': 'a', '?z': 'b'}
    t = ('P', '?x', '?y', '?z')
    c = compose(s1, s2)
    print(f'    s1 = {sub_str(s1)}  s2 = {sub_str(s2)}  compose = {sub_str(c)}')
    print(f'    s2(s1(P(x,y,z))) = {term_str(subst(s2, subst(s1, t)))}   compose applied = {term_str(subst(c, t))}   equal: {subst(s2, subst(s1, t)) == subst(c, t)}')

    # ------------------------------------------------------------- 2. lifted modus ponens
    print('\n[2] Lifted (generalised) Modus Ponens')
    out = lifted_modus_ponens([('King', '?x'), ('Greedy', '?x')], ('Evil', '?x'),
                              [('King', 'John'), ('Greedy', '?y'), ('King', 'Richard')])
    for th_, c_ in out: print(f'    θ = {sub_str(th_)}   =>   {term_str(c_)}')
    print('    Rule King(x) ∧ Greedy(x) ⇒ Evil(x) fires ONCE, with the single unifier θ, instead of one ground instance per constant.')

    # ------------------------------------------------------------- 3. ground-clause growth
    rules = [([('P', '?x')], ('Q', '?x')),
             ([('Q', '?x'), ('R', '?x', '?y')], ('S', '?x', '?y')),
             ([('S', '?x', '?y'), ('R', '?y', '?z')], ('T', '?x', '?y', '?z'))]
    rules_g = [(h, b) for b, h in rules]
    print('\n[3a] Ground clauses vs number of constants k (3 rules with 1, 2 and 3 variables)')
    print(f'  {"k":>3} {"rule1 k":>8} {"rule2 k^2":>10} {"rule3 k^3":>11} {"total clauses":>14} {"measured":>10} {"time(ms)":>9}')
    R['constants'] = []
    for k in list(range(3, 31, 3)) + [30]:
        if k in [r['k'] for r in R['constants']]: continue
        consts = [f'c{i}' for i in range(k)]
        g, dt = timed(propositionalise, rules_g, [], consts)
        total = k + k ** 2 + k ** 3
        assert len(g) == total
        print(f'  {k:>3} {k:>8} {k * k:>10} {k ** 3:>11} {total:>14,} {len(g):>10,} {dt * 1000:>9.1f}')
        R['constants'].append(dict(k=k, total=total, time=dt))

    print('\n[3b] Ground ATOMS of one predicate vs arity a (k^a)')
    print(f'  {"k":>3} ' + ' '.join(f'{"a=" + str(a):>12}' for a in (1, 2, 3, 4, 5)))
    R['arity'] = []
    for k in (3, 6, 10, 20, 30):
        row = [k ** a for a in (1, 2, 3, 4, 5)]
        print(f'  {k:>3} ' + ' '.join(f'{x:>12,}' for x in row))
        R['arity'].append(dict(k=k, atoms=row))
    for a in (1, 2, 3):
        assert len(list(itertools.product(range(5), repeat=a))) == 5 ** a

    print('\n[3c] Function symbols: ground terms vs depth bound d  (verified by explicit enumeration for small d)')
    print(f'  {"d":>2} {"1 const, f/1":>14} {"1 const, g/2":>14} {"2 const, g/2":>14} {"2 const,f/1,g/2":>16} {"2-var rule inst. (2c,g/2)":>26}')
    R['functions'] = []
    for d in range(0, 6):
        a = count_terms(1, [('f', 1)], d); b = count_terms(1, [('g', 2)], d)
        c_ = count_terms(2, [('g', 2)], d); e = count_terms(2, [('f', 1), ('g', 2)], d)
        if d <= 2:
            assert len(ground_terms(['a'], [('f', 1)], d)) == a
            assert len(ground_terms(['a'], [('g', 2)], d)) == b
            assert len(ground_terms(['a', 'b'], [('g', 2)], d)) == c_
        print(f'  {d:>2} {a:>14,} {b:>14,} {c_:>14,} {e:>16,} {c_ ** 2:>26,}')
        R['functions'].append(dict(d=d, f1=a, g2=b, g2_2c=c_, mixed=e))
    print('  (d = 0..2 verified by explicit enumeration; larger d by the recurrence N_{d+1} = N_0 + Σ N_d^arity)')
    print('  With ANY function symbol the Herbrand universe is infinite: propositionalisation never finishes without a depth bound.')

    # ------------------------------------------------------------- 4. lifted vs propositional
    print('\n[4] Lifted vs propositional inference on one query  Ancestor(c0, ck)')
    print('    KB: Parent(c_i, c_i+1) for i<k ; Parent(x,y) ⇒ Ancestor(x,y) ; Parent(x,y) ∧ Ancestor(y,z) ⇒ Ancestor(x,z)')
    hdr = f'  {"k":>3} {"lifted goals":>12} {"lifted unifs":>12} {"lifted ms":>10} | {"ground clauses":>14} {"ground ms":>10} {"FC pops":>8} {"FC ms":>8} {"BC goals":>9} {"BC ms":>8} | {"break-even #queries":>20}'
    print(hdr)
    R['lifted'] = []
    for k in (5, 10, 20, 40, 60):
        consts = [f'c{i}' for i in range(k + 1)]
        facts = [('Parent', f'c{i}', f'c{i + 1}') for i in range(k)]
        kb = KB()
        for f in facts: kb.add_fact(f)
        kb.add_rule(('Ancestor', '?x', '?y'), [('Parent', '?x', '?y')], 'A1')
        kb.add_rule(('Ancestor', '?x', '?z'), [('Parent', '?x', '?y'), ('Ancestor', '?y', '?z')], 'A2')
        goal = ('Ancestor', 'c0', f'c{k}')
        st = Stats()
        reps = 20
        t0 = time.perf_counter()
        for _ in range(reps):
            st = Stats()
            ans = kb.holds(goal, st=st)
        t_l = (time.perf_counter() - t0) / reps
        assert ans
        rules_p = [([('Parent', '?x', '?y')], ('Ancestor', '?x', '?y')),
                   ([('Parent', '?x', '?y'), ('Ancestor', '?y', '?z')], ('Ancestor', '?x', '?z'))]
        (gcl, t_g) = timed(propositionalise, [(h, b) for b, h in rules_p], facts, consts)
        (inf, fcst), t_fc = timed(ground_horn_fc, [(h, b) for h, b in gcl], facts)
        assert goal in inf
        heads, t_idx = timed(ground_horn_index, [(h, b) for h, b in gcl])
        t_g += t_idx                                   # index construction is part of the propositionalisation cost
        fset = set(facts)
        reps2 = 20
        t0 = time.perf_counter()
        for _ in range(reps2):
            resbc, bcst = ground_horn_bc(heads, fset, goal)
        t_bc = (time.perf_counter() - t0) / reps2
        assert resbc
        # break-even: grounding cost is paid once; per-query cost lifted vs ground-BC
        diff = t_l - t_bc
        be = (t_g / diff) if diff > 0 else float('inf')
        print(f'  {k:>3} {st.goals:>12} {st.unifs:>12} {t_l * 1000:>10.3f} | {len(gcl):>14,} {t_g * 1000:>10.1f} {fcst["pops"]:>8} {t_fc * 1000:>8.1f} {bcst["goals"]:>9} {t_bc * 1000:>8.3f} | {("%.0f" % be) if be != float("inf") else "never":>20}')
        R['lifted'].append(dict(k=k, lifted_goals=st.goals, lifted_unifs=st.unifs, lifted_time=t_l, ground=len(gcl),
                                ground_time=t_g, fc_pops=fcst['pops'], fc_time=t_fc, bc_goals=bcst['goals'],
                                bc_time=t_bc, breakeven=None if be == float('inf') else be))

    with open(os.path.join(OUT, 'task4_results.json'), 'w') as f: json.dump(R, f, indent=1, default=str)

    # ------------------------------------------------------------- plots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ks = [r['k'] for r in R['constants']]
    ax[0].plot(ks, [r['total'] for r in R['constants']], 'o-', label='total ground clauses (k+k²+k³)')
    ax[0].plot(ks, [k for k in ks], 's--', label='lifted rules (constant: 3)')
    ax[0].set_yscale('log'); ax[0].set_xlabel('constants k'); ax[0].set_title('(a) Ground clauses vs constants'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    for i, a in enumerate((1, 2, 3, 4, 5)):
        ax[1].plot([r['k'] for r in R['arity']], [r['atoms'][i] for r in R['arity']], 'o-', label=f'arity {a}')
    ax[1].set_yscale('log'); ax[1].set_xlabel('constants k'); ax[1].set_title('(b) Ground atoms k^arity'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ds = [r['d'] for r in R['functions']]
    for key, lab in (('f1', '1 const, f/1'), ('g2', '1 const, g/2'), ('g2_2c', '2 const, g/2'), ('mixed', '2 const, f/1 + g/2')):
        ax[2].plot(ds, [r[key] for r in R['functions']], 'o-', label=lab)
    ax[2].set_yscale('log'); ax[2].set_xlabel('term-depth bound d'); ax[2].set_title('(c) Ground terms vs depth bound'); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, 'task4_growth.png'), dpi=150); plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ks = [r['k'] for r in R['lifted']]
    ax[0].plot(ks, [r['lifted_goals'] for r in R['lifted']], 'o-', label='lifted BC goals')
    ax[0].plot(ks, [r['ground'] for r in R['lifted']], 's-', label='ground clauses generated')
    ax[0].plot(ks, [r['fc_pops'] for r in R['lifted']], '^-', label='propositional FC agenda pops')
    ax[0].set_yscale('log'); ax[0].set_xlabel('chain length k'); ax[0].set_title('Inference work (operation counts)'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(ks, [r['lifted_time'] * 1000 for r in R['lifted']], 'o-', label='lifted query')
    ax[1].plot(ks, [(r['ground_time'] + r['bc_time']) * 1000 for r in R['lifted']], 's-', label='ground + 1 query')
    ax[1].plot(ks, [r['bc_time'] * 1000 for r in R['lifted']], '^--', label='ground query only (after grounding)')
    ax[1].set_yscale('log'); ax[1].set_xlabel('chain length k'); ax[1].set_ylabel('ms'); ax[1].set_title('Time'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, 'task4_lifted_vs_prop.png'), dpi=150); plt.close()
    print('\nplots written: plots/task4_growth.png, plots/task4_lifted_vs_prop.png')
