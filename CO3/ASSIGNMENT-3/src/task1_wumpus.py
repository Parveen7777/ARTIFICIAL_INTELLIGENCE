"""
Task 1 - Logical Agents: Knowledge Bases, Entailment and Reasoning Patterns
Part A : n x n Wumpus-world propositional KB + truth-table entailment checker (instrumented)
Part B : derivations with Modus Ponens / And-Elimination / Unit Resolution (step counted)
Run    : python src/task1_wumpus.py
"""
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from logic import *

OUT = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUT, exist_ok=True)


class Tee:
    def __init__(self, path):
        self.f = open(path, 'w', encoding='utf-8')
    def write(self, s):
        sys.__stdout__.write(s); self.f.write(s)
    def flush(self):
        sys.__stdout__.flush(); self.f.flush()


sys.stdout = Tee(os.path.join(OUT, 'task1_output.txt'))


# ---------------------------------------------------------------- KB construction
def nb(i, j, n):
    return [(a, b) for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)) if 1 <= a <= n and 1 <= b <= n]


def S(kind, i, j): return Sym(f'{kind}_{i}_{j}')


def count_formulae(n):
    """Closed forms derived in the report."""
    sym = 4 * n * n
    sent = 2 + 2 * n * n + 1 + n * n * (n * n - 1) // 2
    return sym, sent


def build_full_kb(n):
    kb = [Not(S('P', 1, 1)), Not(S('W', 1, 1))]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            kb.append(Iff(S('B', i, j), Or(*[S('P', a, b) for a, b in nb(i, j, n)])))
            kb.append(Iff(S('S', i, j), Or(*[S('W', a, b) for a, b in nb(i, j, n)])))
    sq = [(i, j) for i in range(1, n + 1) for j in range(1, n + 1)]
    kb.append(Or(*[S('W', i, j) for i, j in sq]))                       # at least one wumpus
    for (a, b), (c, d) in itertools.combinations(sq, 2):                # at most one wumpus
        kb.append(Or(Not(S('W', a, b)), Not(S('W', c, d))))
    return kb


def percept_sentences(percepts):
    """percepts: list of ((i,j), breeze, stench) for squares visited by the (living) agent."""
    out = []
    for (i, j), b, s in percepts:
        out.append(S('B', i, j) if b else Not(S('B', i, j)))
        out.append(S('S', i, j) if s else Not(S('S', i, j)))
        out += [Not(S('P', i, j)), Not(S('W', i, j))]
    return out


def build_restricted_kb(n, percepts):
    """Relevant-sentence restriction: keep the percept rules of VISITED squares only.
    The biconditionals of unvisited squares only define an otherwise-unconstrained percept symbol,
    so dropping them cannot change any entailment about pits/wumpus (conservative extension)."""
    kb = [Not(S('P', 1, 1)), Not(S('W', 1, 1))]
    for (i, j), b, s in percepts:
        kb.append(Iff(S('B', i, j), Or(*[S('P', a, c) for a, c in nb(i, j, n)])))
        kb.append(Iff(S('S', i, j), Or(*[S('W', a, c) for a, c in nb(i, j, n)])))
    ws = sorted({a for a in itertools.chain(*[[(x, y)] + nb(x, y, n) for (x, y), _, _ in percepts] )})
    for (a, b), (c, d) in itertools.combinations(ws, 2):
        kb.append(Or(Not(S('W', a, b)), Not(S('W', c, d))))
    return kb + percept_sentences(percepts)


def safe(i, j): return And(Not(S('P', i, j)), Not(S('W', i, j)))
def unsafe(i, j): return Not(safe(i, j))


def scenarios(n):
    """Returns (dict of named percept sequences, list of (query type, scenario name, square))."""
    if n == 2:
        # on a 2x2 board the 'at least one wumpus' axiom is very strong, so each query type gets its own percepts
        scs = {'S1': [((1, 1), False, False)],
               'S2': [((1, 1), False, False), ((2, 1), True, True)],
               'S3': [((1, 1), False, True)]}
        return scs, [('ENTAILED', 'S1', (1, 2)), ('NEGATION ENTAILED', 'S2', (2, 2)), ('NEITHER', 'S3', (2, 2))]
    scs = {'S1': [((1, 1), False, False)],
           'S2': [((1, 1), False, False), ((2, 1), True, False), ((1, 2), False, False)]}
    return scs, [('ENTAILED', 'S1', (1, 2)), ('NEGATION ENTAILED', 'S2', (3, 1)), ('NEITHER', 'S1', (2, 2))]


def classify(kb, i, j, syms=None, cap=None):
    r1, s1 = tt_entails(kb, safe(i, j), syms, cap)
    r2, s2 = tt_entails(kb, unsafe(i, j), syms, cap)
    label = 'safe entailed' if r1 else ('negation entailed (unsafe)' if r2 else 'neither entailed')
    agg = {k: s1[k] + s2[k] if k != 'symbols' else s1[k] for k in s1}
    return label, agg


def pct(a, b): return 100.0 * a / b


if __name__ == '__main__':
    results = {'counts': [], 'runs': [], 'part_b': {}}
    print('=' * 100)
    print('TASK 1  PART A  -  Wumpus-world KB and truth-table model checking')
    print('=' * 100)
    print('\n[1] Symbol / sentence / model counts as a function of n   (symbols m = 4n^2, models = 2^m)')
    print(f'{"n":>3} {"symbols m":>10} {"sentences":>11} {"models 2^m":>24}  {"pit-only symbols":>17}')
    for n in range(2, 6):
        m, s = count_formulae(n)
        print(f'{n:>3} {m:>10} {s:>11} {2 ** m:>24,}  {2 * n * n:>17}')
        results['counts'].append(dict(n=n, symbols=m, sentences=s, models=2 ** m))
    for n in (2, 3, 4):
        assert len(build_full_kb(n)) == count_formulae(n)[1], 'sentence-count formula mismatch'
    print('  (formula 2 + 2n^2 + 1 + n^2(n^2-1)/2 verified against the generated KB for n = 2, 3, 4)')

    print('\n[2] Truth-table entailment experiments  (full symbol set vs relevant-symbol restriction)')
    hdr = f'{"n":>2} {"query type":<18} {"query":<14} {"mode":<10} {"m":>3} {"models":>14} {"KB evals":>14} {"time(s)":>9}  verdict'
    print(hdr); print('-' * len(hdr))
    FULL_CAP = 2 ** 20
    rate = None
    for n in (2, 3, 4):
        scs, queries = scenarios(n)
        for qtype, sc, (i, j) in queries:
            pcs = scs[sc]
            full_kb = build_full_kb(n) + percept_sentences(pcs)
            res_kb = build_restricted_kb(n, pcs)
            # --- restricted (relevant symbols)
            label_r, st_r = classify(res_kb, i, j)
            print(f'{n:>2} {qtype:<18} {"safe("+str(i)+","+str(j)+")":<14} {"relevant":<10} {st_r["symbols"]:>3} '
                  f'{st_r["models"]:>14,} {st_r["kb_evals"]:>14,} {st_r["time"]:>9.4f}  {label_r}')
            # --- full
            all_syms = sorted(symbols(reduce(lambda x, y: ('and', x, y), full_kb)))
            try:
                label_f, st_f = classify(full_kb, i, j, cap=FULL_CAP)
                print(f'{n:>2} {qtype:<18} {"safe("+str(i)+","+str(j)+")":<14} {"full":<10} {st_f["symbols"]:>3} '
                      f'{st_f["models"]:>14,} {st_f["kb_evals"]:>14,} {st_f["time"]:>9.4f}  {label_f}')
                rate = st_f['time'] / st_f['kb_evals']
                assert label_f == label_r, 'restriction changed the answer!'
                results['runs'].append(dict(n=n, qtype=qtype, mode='full', m=st_f['symbols'], models=st_f['models'],
                                            time=st_f['time'], verdict=label_f))
            except Infeasible:
                est = 2 * 2 ** len(all_syms) * (rate or 2e-6)
                print(f'{n:>2} {qtype:<18} {"safe("+str(i)+","+str(j)+")":<14} {"full":<10} {len(all_syms):>3} '
                      f'{2 ** len(all_syms):>14.3e} {"-":>14} {"INFEASIBLE":>9}  est. {est:.2e} s '
                      f'(= {est / 3.15e7:.2e} years)')
                # cross-check the restricted answer with DPLL on the FULL KB (no truth table needed)
                a1 = dpll_entails(full_kb, safe(i, j)); a2 = dpll_entails(full_kb, unsafe(i, j))
                lab = 'safe entailed' if a1 else ('negation entailed (unsafe)' if a2 else 'neither entailed')
                print(f'{"":>2} {"":<18} {"":<14} {"DPLL full":<10} cross-check on full KB -> {lab}  '
                      f'({"MATCH" if lab == label_r else "MISMATCH"})')
                results['runs'].append(dict(n=n, qtype=qtype, mode='full', m=len(all_syms), models=2 ** len(all_syms),
                                            time=None, verdict='infeasible', est_seconds=est))
            results['runs'].append(dict(n=n, qtype=qtype, mode='relevant', m=st_r['symbols'], models=st_r['models'],
                                        time=st_r['time'], verdict=label_r))
        print()

    print('[3] Minimal relevant symbol set for the query safe(2,2) with S1 percepts (n = 4):')
    k = build_restricted_kb(4, scenarios(4)[0]['S1'])
    ks = sorted(symbols(reduce(lambda x, y: ('and', x, y), k)) | symbols(safe(2, 2)))
    print('   ', ks, f'-> m = {len(ks)} (instead of 64), 2^{len(ks)} = {2 ** len(ks)} models')

    # ------------------------------------------------------------------ PART B
    print('\n' + '=' * 100)
    print('TASK 1  PART B  -  Reasoning patterns: Modus Ponens, And-Elimination, Unit Resolution')
    print('=' * 100)

    class Deriver:
        """Forward derivation with logged, dependency-tracked inference steps (only sound rules are used)."""
        def __init__(self, n, percepts):
            self.n, self.percepts, self.steps, self.lit_step = n, percepts, [], {}
            self.lits = set()
            self.counts = {'Modus Ponens': 0, 'And-Elimination': 0, 'Unit Resolution': 0,
                           'Biconditional-Elim': 0, 'And-Introduction': 0}

        def log(self, rule, prem, concl, deps=()):
            self.counts[rule] += 1
            sid = len(self.steps) + 1
            self.steps.append((sid, rule, prem, concl, tuple(deps)))
            return sid

        def run(self):
            n = self.n
            for (i, j), b, s in self.percepts:
                B = f'B_{i}_{j}' if b else f'~B_{i}_{j}'
                St = f'S_{i}_{j}' if s else f'~S_{i}_{j}'
                conj = f'{B} ∧ {St} ∧ ~P_{i}_{j} ∧ ~W_{i}_{j}'
                for lit in (B, St, f'~P_{i}_{j}', f'~W_{i}_{j}'):
                    self.lit_step[lit] = self.log('And-Elimination', conj, lit)
                    self.lits.add(lit)
            # pass 1: negative percepts (every neighbour is clear); pass 2: positive percepts (case reduction)
            for want_positive in (False, True):
                for (i, j), b, s in self.percepts:
                    for kind, per, present in (('P', 'B', b), ('W', 'S', s)):
                        if present != want_positive: continue
                        nbrs = [f'{kind}_{a}_{c}' for a, c in nb(i, j, n)]
                        pv = f'{per}_{i}_{j}'
                        bic = self.log('Biconditional-Elim', f'{pv} ⇔ ({" ∨ ".join(nbrs)})',
                                       f'({" ∨ ".join(nbrs)}) ⇒ {pv}  and  {pv} ⇒ ({" ∨ ".join(nbrs)})')
                        if present:
                            disj = list(nbrs)
                            cur = self.log('Modus Ponens', f'{pv} , {pv} ⇒ ({" ∨ ".join(nbrs)})', ' ∨ '.join(nbrs),
                                           [self.lit_step[pv], bic])
                            for x in list(disj):
                                if f'~{x}' in self.lits:
                                    before = ' ∨ '.join(disj); disj.remove(x)
                                    cur = self.log('Unit Resolution', f'({before}) , ~{x}', ' ∨ '.join(disj),
                                                   [cur, self.lit_step[f'~{x}']])
                            if len(disj) == 1:
                                self.lits.add(disj[0]); self.lit_step[disj[0]] = cur
                        else:
                            for x in nbrs:
                                cl = self.log('And-Elimination', ' ∧ '.join(f'(~{y} ∨ {pv})' for y in nbrs),
                                              f'~{x} ∨ {pv}', [bic])
                                self.lit_step[f'~{x}'] = self.log('Unit Resolution', f'(~{x} ∨ {pv}) , ~{pv}', f'~{x}',
                                                                  [cl, self.lit_step[f'~{pv}']])
                                self.lits.add(f'~{x}')

        def derive_safe(self, i, j):
            if f'~P_{i}_{j}' in self.lits and f'~W_{i}_{j}' in self.lits:
                return self.log('And-Introduction', f'~P_{i}_{j} , ~W_{i}_{j}', f'~P_{i}_{j} ∧ ~W_{i}_{j}   (= safe[{i},{j}])',
                                [self.lit_step[f'~P_{i}_{j}'], self.lit_step[f'~W_{i}_{j}']])
            return None

        def slice_for(self, sid):
            need, stack = set(), [sid]
            while stack:
                x = stack.pop()
                if x in need: continue
                need.add(x); stack += list(self.steps[x - 1][4])
            return need

    n = 3
    pcs = scenarios(n)[0]['S2']
    print(f'\nScenario: n = {n}, percepts {pcs}')
    d = Deriver(n, pcs); d.run()
    sid = {}
    for (i, j) in [(2, 2), (1, 3), (3, 1)]:
        sid[(i, j)] = d.derive_safe(i, j)
        print(f'  safe[{i},{j}] derived by rules?  {sid[(i, j)] is not None}')
    print('\nStep-by-step derivation (complete forward derivation, all percepts):')
    for k_, r, p, c, dp in d.steps:
        print(f'  step {k_:>3} {r:<19} | {p[:58]:<58} => {c[:56]}')
    print('\nRule usage count (whole forward run):', d.counts, ' TOTAL steps =', len(d.steps))
    forced = sorted(x for x in d.lits if not x.startswith('~') and x[0] in 'PW')
    print('Positive hazards forced by the rules:', forced, ' -> square [3,1] holds a pit, hence it is NOT safe')
    slices = {}
    for t in [(2, 2), (1, 3)]:
        slices[t] = d.slice_for(sid[t])
        used = [d.steps[k - 1][1] for k in sorted(slices[t])]
        print(f'  steps needed (backward slice) for safe{list(t)} = {len(slices[t])}  '
              f'[MP={used.count("Modus Ponens")}, AE={used.count("And-Elimination")}, UR={used.count("Unit Resolution")}, '
              f'other={len(used) - used.count("Modus Ponens") - used.count("And-Elimination") - used.count("Unit Resolution")}]')
    both = slices[(2, 2)] | slices[(1, 3)]
    print(f'  steps needed for both queries together (shared steps counted once) = {len(both)}')

    full_kb = build_full_kb(n) + percept_sentences(pcs)
    res_kb = build_restricted_kb(n, pcs)
    la, sa = classify(res_kb, 2, 2)
    print(f'\nComparison for the query safe[2,2]  (n = 3):')
    print(f'  inference rules          : {len(slices[(2, 2)])} inference steps (sliced), {len(d.steps)} steps for the full forward run')
    print(f'  model checking, relevant : {sa["models"]:,} models enumerated ({sa["symbols"]} symbols), {sa["time"]:.4f} s, verdict {la}')
    print(f'  model checking, full KB  : 2^36 = {2 ** 36:,} models  (INFEASIBLE)')
    print(f'  ratio relevant-MC models / rule steps = {sa["models"] / len(slices[(2, 2)]):,.0f} : 1')
    results['part_b'] = dict(steps_total=len(d.steps), steps_22=len(slices[(2, 2)]), steps_13=len(slices[(1, 3)]),
                             steps_both=len(both), counts=d.counts, mc_relevant_models=sa['models'],
                             mc_full_models=2 ** 36)

    # not derived != false
    print('\n--- "Not derived" is not the same as "false" ---')
    scs1 = scenarios(3)[0]['S1']
    dd = Deriver(3, scs1); dd.run()
    print(f'  scenario S1 (only [1,1] visited): safe[2,2] derivable by rules? {dd.derive_safe(2, 2) is not None};'
          f'  model checking says: {classify(build_restricted_kb(3, scs1), 2, 2)[0]}')
    A, Bq, C = Sym('A'), Sym('B'), Sym('C')
    kbx = [Or(A, Bq), Imp(A, C), Imp(Bq, C)]
    print(f'  KB = {{A ∨ B, A ⇒ C, B ⇒ C}} : truth table says KB |= C ? {tt_entails(kbx, C)[0]}  '
          f'but MP + AE + UR alone cannot derive C (no unit literal available) -> rule set incomplete,')
    print('  therefore "not derived" means "unknown", never "false"; only a complete procedure (resolution) can derive C:',
          pl_entails(kbx, C)[0])

    # ------------------------------------------------------- soundness truth tables
    print('\n--- Soundness truth tables (premises true => conclusion true in every row) ---')
    def table(name, prem_fn, concl_fn, nvars, labels):
        print(f'\n{name}')
        print('   ' + '  '.join(f'{l:^5}' for l in labels) + '   premises  conclusion  sound-row')
        ok = True
        for v in itertools.product((True, False), repeat=nvars):
            p, c = prem_fn(*v), concl_fn(*v)
            row_ok = (not p) or c
            ok &= row_ok
            print('   ' + '  '.join(f'{"T" if x else "F":^5}' for x in v) + f'   {"T" if p else "F":^8}  {"T" if c else "F":^10}  {"ok" if row_ok else "VIOLATED"}')
        print(f'   => {name.split(":")[0]} is SOUND' if ok else '   => UNSOUND')
        return ok
    table('Modus Ponens: a, a⇒b |= b', lambda a, b: a and ((not a) or b), lambda a, b: b, 2, ['a', 'b'])
    table('And-Elimination: a∧b |= a', lambda a, b: a and b, lambda a, b: a, 2, ['a', 'b'])
    table('Unit Resolution: (a∨b), ¬a |= b', lambda a, b: (a or b) and (not a), lambda a, b: b, 2, ['a', 'b'])
    table('And-Introduction: a, b |= a∧b', lambda a, b: a and b, lambda a, b: a and b, 2, ['a', 'b'])
    table('Biconditional-Elim: a⇔b |= (a⇒b)∧(b⇒a)', lambda a, b: a == b, lambda a, b: ((not a) or b) and ((not b) or a), 2, ['a', 'b'])

    with open(os.path.join(OUT, 'task1_results.json'), 'w') as f:
        json.dump(results, f, indent=1, default=str)
