"""
Task 2 - The Quality of Reasoning Method: Model Checking versus Resolution
 * smart-building access-control policy (14 rules: implication, biconditional, disjunction)
 * manual/automatic CNF conversion, resolution proof tree
 * KB families that make each method look strong or weak
Run: python src/task2_resolution.py
"""
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from logic import *
from viz_util import draw_proof_tree, Tee

OUT = os.path.join(os.path.dirname(__file__), '..', 'outputs')
PLOTS = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(OUT, exist_ok=True); os.makedirs(PLOTS, exist_ok=True)
sys.stdout = Tee(os.path.join(OUT, 'task2_output.txt'))

# ---------------------------------------------------------------- policy KB
names = dict(E='Employee', V='Visitor', BV='BadgeValid', BX='BadgeExpired', BH='BusinessHours', ESC='Escorted',
             EM='Emergency', LD='Lockdown', RA='RestrictedArea', CL='Clearance', AL='Alarm', ACC='AccessGranted',
             DU='DoorUnlocked', LOG='AuditLogged')
s = {k: Sym(k) for k in names}
E, V, BV, BX, BH, ESC, EM, LD, RA, CL, AL, ACC, DU, LOG = [s[k] for k in names]
POLICY = [
    ('R1', Iff(E, Not(V)), 'A person is an employee iff not a visitor'),
    ('R2', Iff(BV, Not(BX)), 'A badge is valid iff it is not expired'),
    ('R3', Imp(And(E, BV, BH, Not(LD)), ACC), 'Employee + valid badge + business hours + no lockdown => access'),
    ('R4', Imp(And(V, ESC, BH, Not(LD)), ACC), 'Escorted visitor in business hours, no lockdown => access'),
    ('R5', Imp(EM, ACC), 'Emergency override grants access (evacuation)'),
    ('R6', Iff(ACC, DU), 'Access is granted iff the door unlocks'),
    ('R7', Imp(And(V, Not(ESC), Not(EM)), Not(ACC)), 'Unescorted visitors never enter (no emergency)'),
    ('R8', Imp(And(LD, Not(EM)), Not(ACC)), 'Lockdown blocks access unless emergency'),
    ('R9', Imp(And(RA, Not(CL), Not(EM)), Not(ACC)), 'Restricted area needs clearance'),
    ('R10', Imp(AL, Or(LD, EM)), 'An alarm implies lockdown or emergency'),
    ('R11', Imp(And(BX, Not(EM)), Not(ACC)), 'Expired badge blocks access'),
    ('R12', Imp(ACC, LOG), 'Every granted access is logged'),
    ('R13', Imp(Not(BH), Or(EM, Not(ACC), And(E, CL))), 'Out of hours: only emergency or cleared employees'),
    ('R14', Imp(And(E, BX), Not(DU)), 'Employee with expired badge: door stays locked'),
]
KB = [r[1] for r in POLICY]

FACTS = {
    'Q1 (entailed)': ([E, BV, BH, Not(LD), Not(EM)], ACC, 'Employee, valid badge, office hours, no lockdown -> access?'),
    'Q2 (negation entailed)': ([V, Not(ESC), Not(EM)], ACC, 'Unescorted visitor, no emergency -> access?'),
    'Q3 (not entailed)': ([V, ESC, BH, Not(EM)], ACC, 'Escorted visitor in hours; lockdown status unknown -> access?'),
}


def mc(kb, q, syms=None, cap=2 ** 26):
    try:
        r, st = tt_entails(kb, q, syms, max_models=cap)
        return r, st
    except Infeasible:
        allsyms = syms or sorted(set().union(*[symbols(x) for x in kb]) | symbols(q))
        return None, dict(symbols=len(allsyms), models=2 ** len(allsyms), time=None, kb_evals=None)


def lit_sent(l): return Not(Sym(l[1:])) if l.startswith('~') else Sym(l)


def rand3(m, k, seed):
    r = random.Random(seed)
    cl = []
    for _ in range(k):
        vs = r.sample(range(m), 3)
        cl.append(frozenset(('~' if r.random() < .5 else '') + f'X{v}' for v in vs))
    return cl


def php(p, h):
    X = lambda i, j: Sym(f'X{i}_{j}')
    kb = [Or(*[X(i, j) for j in range(h)]) for i in range(p)]
    for j in range(h):
        for a in range(p):
            for b in range(a + 1, p):
                kb.append(Or(Not(X(a, j)), Not(X(b, j))))
    return kb


def chain(n, extra=0):
    A = [Sym(f'A{i}') for i in range(n + 1)]
    kb = [A[0]] + [Imp(A[i], A[i + 1]) for i in range(n)]
    kb += [Sym(f'Z{i}') for i in range(extra)]          # irrelevant facts
    return kb, A


def run_res(kb, q, **kw):
    t0 = time.perf_counter()
    r, st, par = pl_entails(kb, q, **kw)
    st['time'] = time.perf_counter() - t0
    return r, st, par


if __name__ == '__main__':
    res = {}
    print('=' * 100); print('TASK 2  -  Model checking versus resolution'); print('=' * 100)
    print('\n[1] Smart-building access-control policy (14 rules)')
    for rid, f, txt in POLICY:
        print(f'  {rid:<4} {show(f):<78}')
        print(f'       {txt}')
    allsyms = sorted(set().union(*[symbols(x) for x in KB]))
    print(f'\n  symbols = {len(allsyms)}, rules = {len(KB)}, rules with ⇒ = {sum(1 for r in POLICY if "⇒" in show(r[1]))}, '
          f'with ⇔ = {sum(1 for r in POLICY if "⇔" in show(r[1]))}, with ∨ = {sum(1 for r in POLICY if "∨" in show(r[1]))}')

    print('\n[2] Manual CNF conversion (three sentences) -- machine check of the hand derivation')
    for rid in ('R1', 'R3', 'R10'):
        f = dict((a, b) for a, b, c in POLICY)[rid]
        print(f'\n  {rid}:')
        for name, txt in cnf_steps(f): print(f'    {name:<38} {txt}')

    print('\n[3] Queries on the policy: resolution vs truth table (same KB + facts)')
    hdr = f'  {"query":<24} {"method":<16} {"verdict":<9} {"models":>9} {"clauses gen":>11} {"attempts":>9} {"taut":>5} {"subsumed":>8} {"time(ms)":>9}'
    print(hdr)
    res['policy'] = []
    for name, (facts, q, txt) in FACTS.items():
        kb = KB + facts
        r1, st1 = mc(kb, q)
        r2, st2, par2 = run_res(kb, q)
        print(f'  {name:<24} {"truth table":<16} {str(r1):<9} {st1["models"]:>9,} {"-":>11} {"-":>9} {"-":>5} {"-":>8} {st1["time"] * 1000:>9.2f}')
        print(f'  {"":<24} {"resolution":<16} {str(r2):<9} {"-":>9} {st2["generated"]:>11} {st2["attempted"]:>9} {st2["tautologies"]:>5} {st2["subsumed"]:>8} {st2["time"] * 1000:>9.2f}')
        # negation of the query for the 'negation entailed' case
        r3, st3, _ = run_res(kb, Not(q))
        print(f'  {"":<24} {"res. on ¬query":<16} {str(r3):<9} {"-":>9} {st3["generated"]:>11} {st3["attempted"]:>9} {st3["tautologies"]:>5} {st3["subsumed"]:>8} {st3["time"] * 1000:>9.2f}')
        res['policy'].append(dict(query=name, tt=r1, tt_models=st1['models'], res=r2, gen=st2['generated'],
                                  att=st2['attempted'], neg_res=r3))
        # mc-verdict must equal resolution verdict
        assert r1 == r2
    # proof for Q1
    facts, q, _ = FACTS['Q1 (entailed)']
    kb = KB + facts
    r, st, par = run_res(kb, q)
    lines, order, ids = proof_lines(par)
    print('\n[4] Resolution proof of Q1  (KB ∧ ¬AccessGranted  |-  {})')
    for l in lines: print(l)
    draw_proof_tree(par, os.path.join(PLOTS, 'task2_proof_tree.png'), 'Task 2 - resolution refutation of Q1: KB ⊨ AccessGranted')
    print('  proof length (resolution steps in the refutation) =', sum(1 for c in order if c in par))

    # non-entailed query termination
    facts, q, _ = FACTS['Q3 (not entailed)']
    r, st, par = run_res(KB + facts, q)
    print('\n[5] Non-entailed query Q3: resolution terminates with NO empty clause.')
    print(f'    clauses generated = {st["generated"]}, attempted = {st["attempted"]}, result entailed = {r}')
    print('    Reason: the set of clauses over 14 symbols is finite (at most 3^14), every resolvent is new or a duplicate,')
    print('    so the saturation loop must reach a fixed point; the empty clause is absent => KB ∧ ¬α satisfiable => not entailed.')

    # ------------------------------------------------------------ scaling: implication chains
    print('\n[6] Implication chains  A0, A0⇒A1, ..., A(n-1)⇒An ;  query An   (entailed)   and   A(n+1) (not entailed)')
    hdr = f'  {"n":>3} {"variant":<26} {"verdict":<8} {"clauses gen":>11} {"attempted":>9} {"taut":>5} {"subsumed":>8} {"time(ms)":>9}'
    print(hdr)
    res['chain'] = []
    for n in (10, 20, 40, 80):
        kb, A = chain(n)
        row = dict(n=n)
        for label, kw in (('unit-pref + subsumption', dict()), ('FIFO, no subsumption', dict(unit_pref=False, subsumption=False))):
            r, st, _ = run_res(kb, A[n], **kw)
            print(f'  {n:>3} {label:<26} {str(r):<8} {st["generated"]:>11} {st["attempted"]:>9} {st["tautologies"]:>5} {st["subsumed"]:>8} {st["time"] * 1000:>9.2f}')
            row[label] = dict(gen=st['generated'], att=st['attempted'], sub=st['subsumed'], time=st['time'])
        r, st, _ = run_res(kb, Sym(f'A{n + 1}'))
        print(f'  {n:>3} {"not entailed A(n+1)":<26} {str(r):<8} {st["generated"]:>11} {st["attempted"]:>9} {st["tautologies"]:>5} {st["subsumed"]:>8} {st["time"] * 1000:>9.2f}')
        row['not_entailed'] = dict(result=r, gen=st['generated'])
        r2, st2 = mc(kb, A[n])
        row['mc'] = dict(models=st2['models'], time=st2['time'], result=r2)
        if r2 is None:
            print(f'  {n:>3} {"truth table":<26} {"INFEAS.":<8} 2^{n + 1} = {st2["models"]:.3e} models')
        else:
            print(f'  {n:>3} {"truth table":<26} {str(r2):<8} {st2["models"]:>11,} models  {st2["time"] * 1000:>9.2f} ms')
        res['chain'].append(row)

    # ------------------------------------------------------------ irrelevant symbols
    print('\n[7] Chain n=5 plus z irrelevant facts (symbols grow, relevant structure does not)')
    print(f'  {"z":>3} {"symbols":>8} {"models(TT)":>14} {"TT time(ms)":>12} {"res gen":>8} {"res time(ms)":>13}')
    res['irrelevant'] = []
    for z in (0, 5, 10, 15, 18, 22):
        kb, A = chain(5, z)
        r1, st1 = mc(kb, A[5], cap=2 ** 24)
        r2, st2, _ = run_res(kb, A[5], use_sos=True)
        tt = f'{st1["time"] * 1000:>12.1f}' if st1['time'] is not None else f'{"INFEAS":>12}'
        print(f'  {z:>3} {st1["symbols"]:>8} {st1["models"]:>14,} {tt} {st2["generated"]:>8} {st2["time"] * 1000:>13.3f}')
        res['irrelevant'].append(dict(z=z, models=st1['models'], tt_time=st1['time'], res_gen=st2['generated'], res_time=st2['time']))

    # ------------------------------------------------------------ pigeonhole
    print('\n[8] Pigeonhole PHP(p+1 pigeons, p holes) -- unsatisfiable, so KB entails every query  (query X0_0)')
    print(f'  {"p":>2} {"symbols":>8} {"clauses":>8} {"TT models":>12} {"TT ms":>9} {"res gen":>9} {"res attempted":>13} {"res ms":>9}  winner')
    res['php'] = []
    for p in (2, 3, 4, 5):
        kb = php(p + 1, p)
        q = Sym('X0_0')
        r1, st1 = mc(kb, q, cap=2 ** 22)
        r2, st2, _ = run_res(kb, q, max_clauses=20000)
        nm = len(symbols(reduce(lambda a, b: ('and', a, b), kb)))
        tt = f'{st1["time"] * 1000:>9.1f}' if st1['time'] is not None else f'{"INFEAS":>9}'
        resv = '(capped)' if st2['capped'] else ''
        win = 'TT' if (st1['time'] is not None and st1['time'] < st2['time']) else 'resolution'
        if st1['time'] is None and st2['capped']: win = 'neither'
        print(f'  {p:>2} {nm:>8} {len(kb):>8} {st1["models"]:>12,} {tt} {st2["generated"]:>9} {st2["attempted"]:>13} {st2["time"] * 1000:>9.1f}  {win} {resv}')
        res['php'].append(dict(p=p, m=nm, tt_models=st1['models'], tt_time=st1['time'], res_gen=st2['generated'],
                               res_time=st2['time'], capped=st2['capped'], winner=win))

    # ------------------------------------------------------------ random dense 3-CNF
    print('\n[9] Dense random 3-CNF, m=14 symbols, 80 clauses, query Q (a fresh symbol) -- three seeds')
    print(f'  {"seed":>4} {"KB sat?":>8} {"TT models":>10} {"TT ms":>8} {"res gen":>8} {"res no-subsump. gen":>20} {"res ms":>8}')
    res['rand'] = []
    for seed in range(3):
        cl = rand3(14, 80, seed)
        kb = [Or(*[lit_sent(l) for l in c]) for c in cl]
        syms = sorted({l.lstrip('~') for c in cl for l in c} | {'Q'})
        r1, st1 = mc(kb, Sym('Q'), syms)
        r2, st2, _ = run_res(kb, Sym('Q'), max_clauses=30000)
        r3, st3, _ = run_res(kb, Sym('Q'), max_clauses=30000, subsumption=False)
        print(f'  {seed:>4} {"unsat" if r1 else "sat":>8} {st1["models"]:>10,} {st1["time"] * 1000:>8.1f} {st2["generated"]:>8} {st3["generated"]:>20}{"+" if st3["capped"] else " "} {st2["time"] * 1000:>8.1f}')
        res['rand'].append(dict(seed=seed, tt_models=st1['models'], tt_time=st1['time'], res_gen=st2['generated'],
                                res_gen_nosub=st3['generated'], capped=st3['capped'], res_time=st2['time']))

    with open(os.path.join(OUT, 'task2_results.json'), 'w') as f:
        json.dump(res, f, indent=1, default=str)

    # ------------------------------------------------------------ plots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ns = [r['n'] for r in res['chain']]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ns, [r['unit-pref + subsumption']['gen'] for r in res['chain']], 'o-', label='resolution (unit-pref + subsumption)')
    ax[0].plot(ns, [r['FIFO, no subsumption']['gen'] for r in res['chain']], 's-', label='resolution (FIFO, no subsumption)')
    ax[0].plot(ns, [2 ** (n + 1) for n in ns], '^--', label='truth table models 2^(n+1)')
    ax[0].set_yscale('log'); ax[0].set_xlabel('chain length n'); ax[0].set_ylabel('clauses generated / models enumerated')
    ax[0].set_title('Implication chains: work vs KB size'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    zs = [r['z'] for r in res['irrelevant']]
    ax[1].plot(zs, [r['models'] for r in res['irrelevant']], '^--', label='truth-table models')
    ax[1].plot(zs, [max(r['res_gen'], 1) for r in res['irrelevant']], 'o-', label='resolution clauses (SoS)')
    ax[1].set_yscale('log'); ax[1].set_xlabel('irrelevant symbols z'); ax[1].set_title('Irrelevant symbols hurt model checking only')
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, 'task2_scaling.png'), dpi=150); plt.close()
    ps = [r['p'] for r in res['php']]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ps, [r['tt_models'] for r in res['php']], '^--', label='truth-table models')
    ax.plot(ps, [r['res_gen'] for r in res['php']], 'o-', label='resolution clauses generated')
    ax.set_yscale('log'); ax.set_xlabel('p (p+1 pigeons, p holes)'); ax.set_title('Pigeonhole: both methods blow up')
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, 'task2_php.png'), dpi=150); plt.close()
    print('\nplots written to plots/task2_scaling.png, plots/task2_php.png, plots/task2_proof_tree.png')
