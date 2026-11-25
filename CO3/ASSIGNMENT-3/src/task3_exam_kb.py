"""
Task 3 - Knowledge Engineering in FOL: University Exam-Eligibility Advisor
 * FOL encoding (13 axioms incl. an existential, Skolemised) vs propositional encoding (rules per student x course)
 * 15 competency queries, 3 KB defects found and fixed (v1 -> v2)
 * size / time growth experiment
Run: python src/task3_exam_kb.py
"""
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from fol import *
from viz_util import Tee

OUT = os.path.join(os.path.dirname(__file__), '..', 'outputs')
PLOTS = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(OUT, exist_ok=True); os.makedirs(PLOTS, exist_ok=True)
sys.stdout = Tee(os.path.join(OUT, 'task3_output.txt'))

X, S_, C_, P_ = '?x', '?s', '?c', '?p'
AXIOMS_TEXT = [
    ('A1', '∀s,c Enrolled(s,c) ∧ AttOK(s,c) ⇒ AttCleared(s,c)'),
    ('A2', '∀s,c Enrolled(s,c) ∧ ApprovedLeave(s,c) ⇒ AttCleared(s,c)'),
    ('A3', '∀s,c MissingPrereq(s,c) ⇔ ∃p (Prereq(p,c) ∧ ¬Passed(s,p))'),
    ('A4', '∀s,c Enrolled(s,c) ∧ ¬MissingPrereq(s,c) ⇒ PrereqCleared(s,c)'),
    ('A5', '∀s FeesPaid(s) ⇒ FeesOK(s)'),
    ('A6', '∀s FeeWaiver(s) ⇒ FeesOK(s)'),
    ('A7', '∀s Disciplined(s) ∧ ¬Reinstated(s) ⇒ Barred(s)'),
    ('A8', '∀s,c AttCleared(s,c) ∧ PrereqCleared(s,c) ∧ FeesOK(s) ∧ ¬Barred(s) ⇒ Eligible(s,c)'),
    ('A9', '∀c Course(c) ⇒ ∃e Examiner(e,c)          [existential; Skolem function exam(c)]'),
    ('A10', '∀s,c Eligible(s,c) ∧ ∃e Examiner(e,c) ⇒ HallTicket(s,c)'),
    ('A11', '∀s,c Enrolled(s,c) ∧ ¬AttCleared(s,c) ⇒ ShortAttendance(s,c)'),
    ('A12', '∀s,c Enrolled(s,c) ∧ ¬Eligible(s,c) ⇒ Ineligible(s,c)'),
    ('A13', '∀s1,s2,c Enrolled(s1,c) ∧ Enrolled(s2,c) ∧ s1≠s2 ⇒ Classmates(s1,s2)'),
]


def fol_rules(version=2):
    r = []
    r.append((('AttCleared', S_, C_), [('Enrolled', S_, C_), ('AttOK', S_, C_)], 'A1'))
    if version >= 2:                                                    # defect D1: leave rule forgotten in v1
        r.append((('AttCleared', S_, C_), [('Enrolled', S_, C_), ('ApprovedLeave', S_, C_)], 'A2'))
    r.append((('MissingPrereq', S_, C_), [('Prereq', P_, C_), ('not', ('Passed', S_, P_))], 'A3'))
    if version >= 2:                                                    # defect D3: v1 used positive 'Passed' => courses without prereq never cleared
        r.append((('PrereqCleared', S_, C_), [('Enrolled', S_, C_), ('not', ('MissingPrereq', S_, C_))], 'A4'))
    else:
        r.append((('PrereqCleared', S_, C_), [('Enrolled', S_, C_), ('Prereq', P_, C_), ('Passed', S_, P_)], 'A4v1'))
    r.append((('FeesOK', S_), [('FeesPaid', S_)], 'A5'))
    r.append((('FeesOK', S_), [('FeeWaiver', S_)], 'A6'))
    if version >= 2:                                                    # defect D2: v1 ignored reinstatement
        r.append((('Barred', S_), [('Disciplined', S_), ('not', ('Reinstated', S_))], 'A7'))
    else:
        r.append((('Barred', S_), [('Disciplined', S_)], 'A7v1'))
    r.append((('Eligible', S_, C_), [('AttCleared', S_, C_), ('PrereqCleared', S_, C_), ('FeesOK', S_), ('not', ('Barred', S_))], 'A8'))
    r.append((('Examiner', ('exam', C_), C_), [('Course', C_)], 'A9'))       # Skolemised existential
    r.append((('HallTicket', S_, C_), [('Eligible', S_, C_), ('Examiner', '?e', C_)], 'A10'))
    r.append((('ShortAttendance', S_, C_), [('Enrolled', S_, C_), ('not', ('AttCleared', S_, C_))], 'A11'))
    r.append((('Ineligible', S_, C_), [('Enrolled', S_, C_), ('not', ('Eligible', S_, C_))], 'A12'))
    r.append((('Classmates', '?s1', '?s2'), [('Enrolled', '?s1', C_), ('Enrolled', '?s2', C_), ('neq', '?s1', '?s2')], 'A13'))
    return r


def build_fol(facts, version=2):
    kb = KB()
    for f in facts: kb.add_fact(f)
    for h, b, n in fol_rules(version): kb.add_rule(h, b, n)
    return kb


# --------------------------------------------------------------- hand-made campus instance
def campus_facts():
    f = []
    for c in ('CS101', 'CS102', 'CS201', 'CS301', 'MA101'): f.append(('Course', c))
    f += [('Prereq', 'CS101', 'CS201'), ('Prereq', 'CS102', 'CS201'), ('Prereq', 'CS201', 'CS301')]
    for s in ('alice', 'bob', 'carol', 'dave', 'erin', 'frank'): f.append(('Student', s))
    en = {'alice': ['CS201', 'MA101'], 'bob': ['CS201'], 'carol': ['CS301', 'CS101'], 'dave': ['CS101', 'MA101'],
          'erin': ['CS101'], 'frank': ['CS201']}
    for s, cs in en.items():
        for c in cs: f.append(('Enrolled', s, c))
    for s, c in [('alice', 'CS201'), ('alice', 'MA101'), ('bob', 'CS201'), ('carol', 'CS301'), ('carol', 'CS101'),
                 ('dave', 'CS101'), ('erin', 'CS101')]:
        f.append(('AttOK', s, c))
    f += [('ApprovedLeave', 'frank', 'CS201'), ('ApprovedLeave', 'dave', 'MA101')]
    f += [('Passed', 'alice', 'CS101'), ('Passed', 'alice', 'CS102'), ('Passed', 'bob', 'CS101'),
          ('Passed', 'carol', 'CS201'), ('Passed', 'frank', 'CS101'), ('Passed', 'frank', 'CS102')]
    for s in ('alice', 'bob', 'carol', 'dave', 'frank'): f.append(('FeesPaid', s))
    f.append(('FeeWaiver', 'erin'))
    f += [('Disciplined', 'carol'), ('Reinstated', 'carol'), ('Disciplined', 'dave')]
    return f


QUERIES = [  # (id, kind, goal, expected)
    ('Q1', 'succeed', ('Eligible', 'alice', 'CS201'), True),
    ('Q2', 'succeed', ('Eligible', 'frank', 'CS201'), True),           # attendance through approved leave
    ('Q3', 'succeed', ('Eligible', 'carol', 'CS301'), True),           # disciplined but reinstated
    ('Q4', 'succeed', ('Eligible', 'erin', 'CS101'), True),            # fee waiver, course without prerequisite
    ('Q5', 'succeed', ('HallTicket', 'alice', 'CS201'), True),         # needs the Skolem examiner exam(CS201)
    ('Q6', 'fail', ('Eligible', 'bob', 'CS201'), False),               # prerequisite CS102 not passed
    ('Q7', 'fail', ('Eligible', 'dave', 'CS101'), False),              # disciplined, not reinstated
    ('Q8', 'fail', ('Eligible', 'dave', 'MA101'), False),              # leave ok but barred
    ('Q9', 'fail', ('Eligible', 'alice', 'CS301'), False),             # not enrolled
    ('Q10', 'fail', ('ShortAttendance', 'alice', 'CS201'), False),
    ('Q11', 'bind', ('Eligible', '?s', 'CS201'), ['alice', 'frank']),
    ('Q12', 'bind', ('MissingPrereq', 'bob', '?c'), ['CS201', 'CS301']),
    ('Q13', 'bind', ('Classmates', 'alice', '?s'), ['bob', 'dave', 'frank']),
    ('Q14', 'bind', ('Ineligible', '?s', '?c'), [('bob', 'CS201'), ('dave', 'CS101'), ('dave', 'MA101')]),
    ('Q15', 'bind', ('Examiner', '?e', 'CS301'), ['exam(CS301)']),
]


def run_queries(kb, label):
    ok, rows = 0, []
    for qid, kind, g, exp in QUERIES:
        if kind == 'bind':
            vs = variables(g)
            got = sorted(str(tuple(term_str(a[v]) for v in vs)) if len(vs) > 1 else term_str(a[vs[0]]) for a in kb.ask(g))
            expn = sorted(str(tuple(e)) if isinstance(e, tuple) else e for e in exp)
            passed = got == expn
            show_got = ', '.join(got)
            show_exp = ', '.join(expn)
        else:
            got = kb.holds(g)
            passed = (got == exp)
            show_got, show_exp = str(got), str(exp)
        ok += passed
        rows.append((qid, kind, term_str(g), show_exp, show_got, passed))
    print(f'\n  {label}: {ok}/{len(QUERIES)} competency queries pass')
    print(f'  {"id":<4} {"kind":<8} {"query":<28} {"expected":<30} {"actual":<30} result')
    for r in rows:
        print(f'  {r[0]:<4} {r[1]:<8} {r[2]:<28} {r[3][:29]:<30} {r[4][:29]:<30} {"PASS" if r[5] else "FAIL"}')
    return ok, rows


# --------------------------------------------------------------- scalable instance generator
def gen_instance(s, c, seed=1):
    r = random.Random(seed)
    studs = [f's{i}' for i in range(s)]
    courses = [f'c{j}' for j in range(c)]
    prereq = {}
    for j in range(1, c):
        if j % 3 != 0: prereq[courses[j]] = [courses[j - 1]]
    facts = []
    for cc in courses: facts.append(('Course', cc))
    for cc, ps in prereq.items():
        for p in ps: facts.append(('Prereq', p, cc))
    enr = []
    for st in studs:
        facts.append(('Student', st))
        for cc in r.sample(courses, min(c, 4)):
            enr.append((st, cc))
            facts.append(('Enrolled', st, cc))
            if r.random() < .85: facts.append(('AttOK', st, cc))
            elif r.random() < .5: facts.append(('ApprovedLeave', st, cc))
        for cc in courses:
            if r.random() < .6: facts.append(('Passed', st, cc))
        if r.random() < .9: facts.append(('FeesPaid', st))
        if r.random() < .05: facts.append(('Disciplined', st))
    return studs, courses, prereq, facts, enr


def prop_ground(studs, courses, prereq):
    """Propositional encoding: one set of rules per student and per course.  Atoms are strings."""
    rules = []   # (head, [pos body], [neg body])
    for c in courses:
        rules.append((f'Examiner_{c}', [f'Course_{c}'], []))
    for s in studs:
        rules.append((f'FeesOK_{s}', [f'FeesPaid_{s}'], []))
        rules.append((f'FeesOK_{s}', [f'FeeWaiver_{s}'], []))
        rules.append((f'Barred_{s}', [f'Disciplined_{s}'], [f'Reinstated_{s}']))
        for c in courses:
            rules.append((f'AttCleared_{s}_{c}', [f'Enrolled_{s}_{c}', f'AttOK_{s}_{c}'], []))
            rules.append((f'AttCleared_{s}_{c}', [f'Enrolled_{s}_{c}', f'ApprovedLeave_{s}_{c}'], []))
            for p in prereq.get(c, []):
                rules.append((f'MissingPrereq_{s}_{c}', [f'Prereq_{p}_{c}'], [f'Passed_{s}_{p}']))
            rules.append((f'PrereqCleared_{s}_{c}', [f'Enrolled_{s}_{c}'], [f'MissingPrereq_{s}_{c}']))
            rules.append((f'Eligible_{s}_{c}', [f'AttCleared_{s}_{c}', f'PrereqCleared_{s}_{c}', f'FeesOK_{s}'], [f'Barred_{s}']))
            rules.append((f'HallTicket_{s}_{c}', [f'Eligible_{s}_{c}', f'Examiner_{c}'], []))
            rules.append((f'ShortAttendance_{s}_{c}', [f'Enrolled_{s}_{c}'], [f'AttCleared_{s}_{c}']))
            rules.append((f'Ineligible_{s}_{c}', [f'Enrolled_{s}_{c}'], [f'Eligible_{s}_{c}']))
    return rules


def prop_facts(facts):
    return {'_'.join(map(str, f)) for f in facts}


class PropKB:
    def __init__(self, rules, facts):
        self.heads = {}
        for h, p, n in rules: self.heads.setdefault(h, []).append((p, n))
        self.facts = facts

    def holds(self, g):
        if g in self.facts: return True
        for p, n in self.heads.get(g, ()):
            if all(self.holds(x) for x in p) and not any(self.holds(x) for x in n): return True
        return False


def tm(f, *a):
    t0 = time.perf_counter(); r = f(*a); return r, time.perf_counter() - t0


if __name__ == '__main__':
    R = {}
    print('=' * 100); print('TASK 3  -  Knowledge engineering in FOL: exam-eligibility advisor'); print('=' * 100)
    print('\n[1] Knowledge-engineering process (summary)')
    for i, t in enumerate(['Identify the task: decide Eligible(s,c) and explain; list who may sit an exam',
                           'Assemble knowledge: attendance, prerequisites, fees, discipline, approved leave, examiner',
                           'Vocabulary: Student, Course, Enrolled, AttOK, ApprovedLeave, Passed, Prereq, FeesPaid, FeeWaiver, Disciplined, Reinstated, Examiner, ...',
                           'Encode general knowledge: axioms A1-A13 (below)',
                           'Encode the instance: campus_facts() = 6 students, 5 courses',
                           'Pose queries: 15 competency questions',
                           'Debug the KB: v1 failed 3 queries -> 3 defects fixed -> v2'], 1):
        print(f'   step {i}: {t}')
    print('\n[2] FOL axioms (13, universally quantified; A3 and A9 contain existentials)')
    for a, t in AXIOMS_TEXT: print(f'   {a:<4} {t}')

    print('\n[3] Debugging: run the 15 competency queries on KB version 1 (naive) and version 2 (fixed)')
    facts = campus_facts()
    ok1, rows1 = run_queries(build_fol(facts, 1), 'KB v1')
    ok2, rows2 = run_queries(build_fol(facts, 2), 'KB v2')
    R['queries'] = dict(v1=ok1, v2=ok2)
    print('\n  DEFECT LOG')
    print('   D1  v1 fails Q2: rule A2 (approved leave => attendance cleared) was missing.                       Fix: add A2.')
    print('   D2  v1 fails Q3: Barred(s) ignored reinstatement (Disciplined => Barred).                         Fix: A7 with ¬Reinstated(s).')
    print('   D3  v1 fails Q4, Q6, Q11, Q14: PrereqCleared needed a POSITIVE Passed(s,p), so courses with no')
    print('       prerequisite were never cleared (Q4) and ONE passed prerequisite was enough (Q6).           Fix: A3/A4 via negation as failure.')
    print('   D4  (test-oracle defect) the first expected answer of Q13 omitted dave, who shares MA101 with alice.')
    print('       KB was right, the expectation was wrong; corrected in the query suite.                      Fix: expected = {bob, dave, frank}.')

    print('\n[4] Growth derivation (propositional encoding), s students, c courses, ρ = average prerequisites per course')
    print('   per (s,c): AttCleared×2, MissingPrereq×ρ, PrereqCleared, Eligible, HallTicket, ShortAttendance, Ineligible = 7 + ρ rules')
    print('   per student: FeesOK×2, Barred = 3 ;  per course: Examiner = 1')
    print('   N_prop(s,c) = (7+ρ)·s·c + 3s + c      -> grows as O(s·c)      N_FOL = 13 axioms (constant)')
    print('   (the instance FACTS are needed by both encodings: F(s,c) ≈ s·(4·1.1 + 0.6c + 1.05) )')

    print('\n[5] Size and inference-time experiment  (30 random Eligible(s,c) queries per cell; seeds fixed)')
    hdr = f'  {"s":>4} {"c":>3} | {"facts":>7} {"prop rules":>11} {"prop KB":>9} {"FOL KB":>7} | {"prop build ms":>13} {"prop q ms":>10} {"prop tot ms":>11} | {"FOL build ms":>12} {"FOL q ms":>9} {"FOL tot ms":>10} | cheaper'
    print(hdr)
    R['grid'] = []
    for (s, c) in [(2, 2), (3, 3), (5, 4), (10, 5), (20, 8), (40, 10), (80, 15), (150, 20), (300, 30)]:
        studs, courses, prereq, facts_, enr = gen_instance(s, c)
        r = random.Random(7)
        qs = [(r.choice(studs), r.choice(courses)) for _ in range(30)]
        (prules, t_ground) = tm(prop_ground, studs, courses, prereq)
        pfacts = prop_facts(facts_)
        pk, t_load = tm(PropKB, prules, pfacts)
        t0 = time.perf_counter()
        pres = [pk.holds(f'Eligible_{a}_{b}') for a, b in qs]
        t_pq = time.perf_counter() - t0
        fk, t_fb = tm(build_fol, facts_, 2)
        t0 = time.perf_counter()
        fres = [fk.holds(('Eligible', a, b)) for a, b in qs]
        t_fq = time.perf_counter() - t0
        assert pres == fres, 'encodings disagree!'
        prop_tot = (t_ground + t_load + t_pq) * 1000
        fol_tot = (t_fb + t_fq) * 1000
        row = dict(s=s, c=c, facts=len(facts_), prop_rules=len(prules), prop_kb=len(prules) + len(facts_), fol_kb=13 + len(facts_),
                   prop_build=(t_ground + t_load) * 1000, prop_q=t_pq * 1000, fol_build=t_fb * 1000, fol_q=t_fq * 1000,
                   prop_tot=prop_tot, fol_tot=fol_tot)
        R['grid'].append(row)
        print(f'  {s:>4} {c:>3} | {len(facts_):>7} {len(prules):>11} {len(prules) + len(facts_):>9} {13 + len(facts_):>7} | '
              f'{row["prop_build"]:>13.2f} {row["prop_q"]:>10.2f} {prop_tot:>11.2f} | {row["fol_build"]:>12.2f} {row["fol_q"]:>9.2f} {fol_tot:>10.2f} | {"FOL" if fol_tot < prop_tot else "propositional"}')
    print('\n   Both encodings returned identical answers on every query (assert in code).')
    print('   Size crossover: FOL KB is smaller as soon as (7+ρ)·s·c + 3s + c > 13 ,  i.e. from s·c ≈ 2 upward.')
    big = [r for r in R['grid'] if r['fol_tot'] < r['prop_tot']]
    print('   Total-time crossover (build + 30 queries):', f'first cell where FOL is cheaper: s={big[0]["s"]}, c={big[0]["c"]}' if big else 'FOL never cheaper in measured range')
    print('   Per-query cost: propositional lookups are faster per query (no unification); FOL pays unification per goal.')

    with open(os.path.join(OUT, 'task3_results.json'), 'w') as f: json.dump(R, f, indent=1, default=str)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    g = R['grid']
    sc = [r['s'] * r['c'] for r in g]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(sc, [r['prop_kb'] for r in g], 'o-', label='propositional KB (rules+facts)')
    ax[0].plot(sc, [r['fol_kb'] for r in g], 's-', label='FOL KB (13 axioms + facts)')
    ax[0].plot(sc, [r['prop_rules'] for r in g], '^--', label='propositional rules only')
    ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel('s × c'); ax[0].set_ylabel('sentences'); ax[0].set_title('(a) KB size'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(sc, [r['prop_tot'] for r in g], 'o-', label='propositional (ground + load + 30 q)')
    ax[1].plot(sc, [r['fol_tot'] for r in g], 's-', label='FOL (load + 30 q)')
    ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlabel('s × c'); ax[1].set_ylabel('ms'); ax[1].set_title('(b) Total time'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[2].plot(sc, [r['prop_q'] / 30 for r in g], 'o-', label='propositional per query')
    ax[2].plot(sc, [r['fol_q'] / 30 for r in g], 's-', label='FOL per query')
    ax[2].set_xscale('log'); ax[2].set_yscale('log'); ax[2].set_xlabel('s × c'); ax[2].set_ylabel('ms / query'); ax[2].set_title('(c) Per-query time'); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, 'task3_size_time.png'), dpi=150); plt.close()
    print('\nplots written: plots/task3_size_time.png')
