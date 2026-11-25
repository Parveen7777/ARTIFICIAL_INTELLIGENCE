"""
Task 6 - Advanced Inference Mini-Investigations: forward chaining vs backward chaining
KB   : organisation hierarchy, 10 definite-clause rules, 100 / 1,000 / 10,000 facts
Run  : python src/task6_chaining.py
"""
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from fol import *
from viz_util import Tee

OUT = os.path.join(os.path.dirname(__file__), '..', 'outputs')
PLOTS = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(OUT, exist_ok=True); os.makedirs(PLOTS, exist_ok=True)

M, E_, Y_, R_, X_, D_ = '?m', '?e', '?y', '?r', '?x', '?d'
RULES = [
    (('Superior', M, E_), [('ReportsTo', E_, M)], 'R1  Superior base'),
    (('Superior', M, E_), [('ReportsTo', E_, Y_), ('Superior', M, Y_)], 'R2  Superior (right-recursive)'),
    (('Peer', X_, Y_), [('ReportsTo', X_, M), ('ReportsTo', Y_, M), ('neq', X_, Y_)], 'R3  Peer'),
    (('SameDept', X_, Y_), [('InDept', X_, D_), ('InDept', Y_, D_), ('neq', X_, Y_)], 'R4  SameDept'),
    (('Approver', M, R_), [('RequestBy', R_, E_), ('ReportsTo', E_, M), ('Small', R_)], 'R5  Approver (small request: direct manager)'),
    (('Approver', M, R_), [('RequestBy', R_, E_), ('Superior', M, E_), ('Large', R_), ('Director', M)], 'R6  Approver (large request: a director above)'),
    (('Authorised', M, R_), [('Approver', M, R_), ('Active', M)], 'R7  Authorised'),
    (('Escalate', R_, M), [('RequestBy', R_, E_), ('Superior', M, E_)], 'R8  Escalate'),
    (('Delegate', X_, Y_), [('Peer', X_, Y_), ('SameDept', X_, Y_), ('Active', Y_)], 'R9  Delegate'),
    (('Chain', E_, M), [('Superior', M, E_), ('Director', M)], 'R10 Chain-to-director'),
]
LEFT_REC = (('Superior', M, E_), [('Superior', Y_, E_), ('ReportsTo', Y_, M)], 'R2L Superior (LEFT-recursive)')


def gen_facts(F):
    E = max(4, F // 4)
    R = E // 2
    facts = []
    for i in range(1, E): facts.append(('ReportsTo', f'e{i}', f'e{(i - 1) // 3}'))
    for i in range(E): facts.append(('InDept', f'e{i}', f'd{i // 5}'))
    for i in range(E):
        if i % 10 != 7: facts.append(('Active', f'e{i}'))
    for i in range(0, E, 20): facts.append(('Director', f'e{i}'))
    for k in range(R):
        facts.append(('RequestBy', f'r{k}', f'e{(k * 7 + 3) % E}'))
        facts.append(('Small' if k % 2 == 0 else 'Large', f'r{k}'))
    facts = facts[:F]
    k = 0
    while len(facts) < F:                         # pad with unrelated facts so the size is exactly F
        facts.append(('Badge', f'e{k % E}', f'b{k}')); k += 1
    return facts


def depth_of(i):
    d = 0
    while i > 0: i = (i - 1) // 3; d += 1
    return d


# ---------------------------------------------------------------- forward chaining
class Store:
    def __init__(self):
        self.by = defaultdict(set); self.idx = defaultdict(list)

    def add(self, a):
        if a in self.by[a[0]]: return False
        self.by[a[0]].add(a)
        for p, v in enumerate(a[1:], 1): self.idx[(a[0], p, v)].append(a)
        return True

    def cands(self, pat, th):
        for p, a in enumerate(pat[1:], 1):
            a = walk(a, th)
            if not is_var(a): return self.idx.get((pat[0], p, a), ())
        return self.by[pat[0]]


def fc(rules, facts, seminaive=False):
    st = Store()
    for f in facts: st.add(f)
    base = set(facts)
    stats = dict(firings=0, redundant=0, iterations=0, inferred=0, unifs=0)
    delta = set(facts)
    old = set()
    t0 = time.perf_counter()
    while True:
        stats['iterations'] += 1
        new = set()
        for head, body, name in rules:
            # (position restricted to delta, others) combos for semi-naive
            positions = [None] if not seminaive else [i for i, b in enumerate(body) if b[0] not in BUILTINS]

            def join(i, th, k):
                if i == len(body):
                    stats['firings'] += 1
                    h = subst(th, head)
                    if h in st.by[h[0]] or h in new: stats['redundant'] += 1
                    else: new.add(h)
                    return
                b = body[i]
                if b[0] in BUILTINS:
                    if eval_builtin(subst(th, b)): join(i + 1, th, k)
                    return
                for f in st.cands(b, th):
                    if seminaive and k is not None:
                        if i == k and f not in delta: continue
                        if i < k and f in delta: continue
                    stats['unifs'] += 1
                    th2 = unify(b, f, th)
                    if th2 is not None: join(i + 1, th2, k)
            for k in positions: join(0, {}, k)
        added = [h for h in new if st.add(h)]
        stats['inferred'] += len(added)
        if not added: break
        delta = set(added)
    stats['time'] = time.perf_counter() - t0
    return st, stats


# ---------------------------------------------------------------- helpers
def build_kb(facts, left=False):
    kb = KB()
    for f in facts: kb.add_fact(f)
    for h, b, n in RULES:
        if left and n.startswith('R2 '): continue
        kb.add_rule(h, b, n)
    if left: kb.add_rule(*LEFT_REC)
    return kb


def relevant_derived(kb, goal):
    """derived facts that occur in SOME proof of the goal (all SLD proofs enumerated)."""
    rel = set()
    for th, prf in kb.solve([goal], proof=True):
        p = prf
        while p is not None:
            if p[0] == 'R': rel.add(subst(th, p[2]))
            p = p[-1]
    return rel


if __name__ == '__main__':
    sys.stdout = Tee(os.path.join(OUT, 'task6_output.txt'))
    R = {'fc': [], 'bc': [], 'lr': []}
    print('=' * 100); print('TASK 6  -  Forward vs backward chaining on the organisation KB'); print('=' * 100)
    print('\n[1] The 10 definite-clause rules')
    for h, b, n in RULES:
        print(f'   {n:<48} {term_str(h)}  <=  ' + ' ∧ '.join(term_str(x) for x in b))
    print(f'   {LEFT_REC[2]:<48} {term_str(LEFT_REC[0])}  <=  ' + ' ∧ '.join(term_str(x) for x in LEFT_REC[1]) + '    (replaces R2 in Part B)')

    print('\n[2] Student A  -  forward chaining to a fixed point')
    print('   query used for the "relevance" column: Authorised(?m, r2)   (who may approve request r2?)')
    hdr = f'   {"facts":>6} {"mode":<11} {"inferred":>9} {"firings":>9} {"redundant":>10} {"redund %":>8} {"iterations":>10} {"time s":>8} {"irrelevant % of derived":>24}'
    print(hdr)
    for F in (100, 1000, 10000):
        facts = gen_facts(F)
        kb = build_kb(facts)
        rel = relevant_derived(kb, ('Authorised', '?m', 'r2'))
        for mode in ('naive', 'semi-naive'):
            st, s = fc(list(RULES), facts, seminaive=(mode == 'semi-naive'))
            derived = {a for p in st.by for a in st.by[p]} - set(facts)
            irr = 100.0 * (1 - len(rel & derived) / max(1, len(derived)))
            print(f'   {len(facts):>6} {mode:<11} {s["inferred"]:>9} {s["firings"]:>9} {s["redundant"]:>10} {100 * s["redundant"] / max(1, s["firings"]):>8.1f} {s["iterations"]:>10} {s["time"]:>8.2f} {irr:>23.2f}%')
            R['fc'].append(dict(F=len(facts), mode=mode, inferred=s['inferred'], firings=s['firings'], redundant=s['redundant'],
                                iterations=s['iterations'], time=s['time'], irrelevant_pct=irr, relevant=len(rel & derived)))
    print('   When the query touches a tiny part of the KB, forward chaining still derives EVERY consequence of EVERY rule:')
    print('   the work (firings) is identical for a query about one request and for a query about all requests.')

    print('\n[3] Student B  -  backward chaining with substitutions (right-recursive Superior, query Authorised(?m, r2))')
    print(f'   {"facts":>6} {"goals expanded":>14} {"unifications":>12} {"max depth":>9} {"repeated subgoals":>17} {"answers":>8} {"time ms":>8} | {"FC firings":>10} {"FC/BC unif ratio":>16}')
    for F in (100, 1000, 10000):
        facts = gen_facts(F)
        kb = build_kb(facts)
        st = Stats()
        t0 = time.perf_counter()
        ans = kb.ask(('Authorised', '?m', 'r2'), st=st)
        dt = time.perf_counter() - t0
        fcrow = [r for r in R['fc'] if r['F'] == len(facts) and r['mode'] == 'naive'][0]
        print(f'   {len(facts):>6} {st.goals:>14} {st.unifs:>12} {st.max_depth:>9} {st.repeated_subgoals:>17} {len(ans):>8} {dt * 1000:>8.2f} | {fcrow["firings"]:>10} {fcrow["firings"] / max(1, st.unifs):>16.0f}')
        R['bc'].append(dict(F=len(facts), goals=st.goals, unifs=st.unifs, depth=st.max_depth, rep=st.repeated_subgoals,
                            answers=len(ans), time=dt, fc_firings=fcrow['firings']))
    print('   Answer for F=10000:', ', '.join(term_str(a['?m']) for a in ans) or '(none)')
    print('   BC cost depends on the length of the proof (≤ hierarchy depth), NOT on how many unrelated facts the KB holds.')

    print('\n[4] Left-recursive rule  Superior(m,e) <= Superior(y,e) ∧ ReportsTo(y,m)  -- naive BC never terminates')
    print('   query Superior(?m, e_deep): correct answers come from the right-recursive KB; cap = 20,000 goal expansions')
    print(f'   {"facts":>6} {"strategy":<22} {"outcome":<22} {"answers":>7} {"correct":>7} {"goals":>9} {"unifs":>9} {"max depth":>9} {"repeated":>9} {"time ms":>8}')
    for F in (100, 1000, 10000):
        facts = gen_facts(F)
        E = max(4, F // 4)
        deep = max(range(E), key=lambda i: (depth_of(i), i))
        goal = ('Superior', '?m', f'e{deep}')
        truth = {term_str(a['?m']) for a in build_kb(facts).ask(goal)}
        kbL = build_kb(facts, left=True)
        for strat in ('naive', 'loop check', 'depth bound (d=12)', 'tabling'):
            st = Stats()
            t0 = time.perf_counter()
            try:
                if strat == 'naive':
                    ans = kbL.ask(goal, st=st, cap=20000); outcome = 'terminated'
                elif strat == 'loop check':
                    ans = kbL.ask(goal, st=st, loop_check=True); outcome = 'terminated'
                elif strat == 'depth bound (d=12)':
                    ans = kbL.ask(goal, st=st, depth_limit=12); outcome = 'terminated'
                else:
                    a_, st = kbL.tabled(goal); ans = [{'?m': x[1]} for x in a_]; outcome = 'terminated'
            except CapExceeded:
                ans, outcome = [], 'NON-TERMINATING (cap)'
            except RecursionError:
                ans, outcome = [], 'NON-TERMINATING (stack)'
            dt = time.perf_counter() - t0
            got = {term_str(a['?m']) for a in ans}
            corr = 'n/a' if outcome.startswith('NON') else ('yes' if got == truth else f'NO ({len(got)}/{len(truth)})')
            print(f'   {len(facts):>6} {strat:<22} {outcome:<22} {len(got):>7} {corr:>7} {st.goals:>9} {st.unifs:>9} {st.max_depth:>9} {st.repeated_subgoals:>9} {dt * 1000:>8.2f}')
            R['lr'].append(dict(F=len(facts), strategy=strat, outcome=outcome, answers=len(got), truth=len(truth), goals=st.goals,
                                unifs=st.unifs, depth=st.max_depth, repeated=st.repeated_subgoals, time=dt))
    print('   loop check: sound but INCOMPLETE (it prunes the recursive branch that carries the 2nd, 3rd, ... superior).')
    print('   depth bound: complete only if the bound ≥ hierarchy depth; wastes work re-deriving the same subgoals.')
    print('   tabling (memoise answers per variant subgoal, iterate to fixpoint): complete, terminates, cost ≈ depth × answers.')

    with open(os.path.join(OUT, 'task6_results.json'), 'w') as f: json.dump(R, f, indent=1, default=str)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    Fs = [100, 1000, 10000]
    fcn = [r for r in R['fc'] if r['mode'] == 'naive']; fcs = [r for r in R['fc'] if r['mode'] == 'semi-naive']
    ax[0].plot(Fs, [r['firings'] for r in fcn], 'o-', label='FC naive firings')
    ax[0].plot(Fs, [r['firings'] for r in fcs], 's-', label='FC semi-naive firings')
    ax[0].plot(Fs, [r['unifs'] for r in R['bc']], '^-', label='BC unifications (one query)')
    ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel('facts'); ax[0].set_title('Work for ONE query'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].bar([str(f) for f in Fs], [r['irrelevant_pct'] for r in fcn], color='#d97706')
    ax[1].set_ylabel('% derived facts irrelevant to query'); ax[1].set_xlabel('facts'); ax[1].set_title('Forward chaining: irrelevant derivations'); ax[1].set_ylim(0, 100)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, 'task6_fc_vs_bc.png'), dpi=150); plt.close()
    print('\nplots written: plots/task6_fc_vs_bc.png')
