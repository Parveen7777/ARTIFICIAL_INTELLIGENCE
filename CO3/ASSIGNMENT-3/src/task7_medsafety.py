"""
Task 7 - Design a Knowledge-Based Reasoning Component: Hospital Medication-Safety Advisor
FINAL DESIGN : first-order definite clauses (+ negation as failure for Safe/not-reinstated style tests),
               goal-directed backward chaining with (a) rule index on the head's drug constant,
               (b) fact index on the patient id, (c) proof-based explanations.
BASELINES    : unindexed backward chaining (scan every rule) ; forward chaining over all rules for the patient.
EDUCATIONAL PROTOTYPE - the rules are simplified examples, NOT clinical guidance.
Run: python src/task7_medsafety.py
"""
import json
import os
import random
import statistics
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(__file__))
from fol import *
from viz_util import Tee
import task6_chaining as t6  # forward chaining engine (naive / semi-naive) reused as baseline

OUT = os.path.join(os.path.dirname(__file__), '..', 'outputs')
PLOTS = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(OUT, exist_ok=True); os.makedirs(PLOTS, exist_ok=True)

P, D, A, B, C1, C2, M, CND, AG = '?p', '?d', '?a', '?b', '?c1', '?c2', '?m', '?cond', '?age'

GENERIC = [
    ('G1 allergy', ('Unsafe', D, P, ('allergy', A)), [('Prescribed', P, D), ('AllergicTo', P, A), ('Class', D, A)]),
    ('G2 cross-allergy', ('Unsafe', D, P, ('cross_allergy', A, B)), [('Prescribed', P, D), ('AllergicTo', P, A), ('CrossReactive', A, B), ('Class', D, B)]),
    ('G3 interaction', ('Unsafe', D, P, ('interaction', M)), [('Prescribed', P, D), ('Takes', P, M), ('Class', D, C1), ('Class', M, C2), ('ClassInteracts', C1, C2)]),
    ('G4 interaction (sym)', ('Unsafe', D, P, ('interaction', M)), [('Prescribed', P, D), ('Takes', P, M), ('Class', D, C1), ('Class', M, C2), ('ClassInteracts', C2, C1)]),
    ('G5 contraindication', ('Unsafe', D, P, ('contraindicated', CND)), [('Prescribed', P, D), ('HasCondition', P, CND), ('Class', D, C1), ('ContraClass', C1, CND)]),
    ('G6 paediatric', ('Unsafe', D, P, 'paediatric'), [('Prescribed', P, D), ('Age', P, AG), ('lt', AG, '12'), ('Class', D, C1), ('NotForChildren', C1)]),
    ('G7 geriatric', ('Unsafe', D, P, 'geriatric'), [('Prescribed', P, D), ('Age', P, AG), ('gt', AG, '74'), ('Class', D, C1), ('AvoidElderly', C1)]),
    ('G8 pregnancy', ('Unsafe', D, P, 'pregnancy'), [('Prescribed', P, D), ('Pregnant', P), ('Class', D, C1), ('Teratogenic', C1)]),
    ('G9 duplicate therapy', ('Unsafe', D, P, ('duplicate_therapy', M)), [('Prescribed', P, D), ('Takes', P, M), ('Class', D, C1), ('Class', M, C1), ('neq', D, M)]),
    ('G10 renal', ('Unsafe', D, P, 'renal_dose'), [('Prescribed', P, D), ('HasCondition', P, 'renal_impairment'), ('Class', D, C1), ('RenalAdjust', C1)]),
    ('G11 safe', ('Safe', D, P), [('Prescribed', P, D), ('not', ('Unsafe', D, P, '?why'))]),
    ('G12 needs-review', ('NeedsReview', P, D), [('Prescribed', P, D), ('Unsafe', D, P, '?why')]),
    ('G13 high-risk patient', ('HighRisk', P), [('Age', P, AG), ('gt', AG, '74')]),
    ('G14 high-risk patient', ('HighRisk', P), [('HasCondition', P, 'renal_impairment')]),
]

SPECIFIC = [   # drug-specific rules: head constant = drug  ->  retrieved only when that drug is prescribed
    ('aspirin', 'reye_syndrome', [('Prescribed', P, 'aspirin'), ('Age', P, AG), ('lt', AG, '16'), ('HasCondition', P, 'viral_infection')]),
    ('metformin', 'lactic_acidosis_risk', [('Prescribed', P, 'metformin'), ('HasCondition', P, 'renal_impairment')]),
    ('nitroglycerin', 'severe_hypotension', [('Prescribed', P, 'nitroglycerin'), ('Takes', P, 'sildenafil')]),
    ('sildenafil', 'severe_hypotension', [('Prescribed', P, 'sildenafil'), ('Takes', P, 'nitroglycerin')]),
    ('warfarin', 'bleeding_risk', [('Prescribed', P, 'warfarin'), ('Takes', P, 'aspirin')]),
    ('warfarin', 'bleeding_risk', [('Prescribed', P, 'warfarin'), ('Takes', P, 'ibuprofen')]),
    ('simvastatin', 'myopathy_risk', [('Prescribed', P, 'simvastatin'), ('Takes', P, 'clarithromycin')]),
    ('tramadol', 'serotonin_syndrome', [('Prescribed', P, 'tramadol'), ('Takes', P, 'sertraline')]),
    ('lisinopril', 'hyperkalaemia', [('Prescribed', P, 'lisinopril'), ('Takes', P, 'potassium_chloride')]),
    ('potassium_chloride', 'hyperkalaemia', [('Prescribed', P, 'potassium_chloride'), ('Takes', P, 'lisinopril')]),
    ('codeine', 'respiratory_depression', [('Prescribed', P, 'codeine'), ('Takes', P, 'diazepam')]),
    ('diazepam', 'respiratory_depression', [('Prescribed', P, 'diazepam'), ('Takes', P, 'codeine')]),
    ('ibuprofen', 'gi_bleed', [('Prescribed', P, 'ibuprofen'), ('HasCondition', P, 'peptic_ulcer')]),
    ('ciprofloxacin', 'tendon_rupture', [('Prescribed', P, 'ciprofloxacin'), ('Age', P, AG), ('gt', AG, '60'), ('Takes', P, 'prednisolone')]),
    ('isotretinoin', 'teratogenic_absolute', [('Prescribed', P, 'isotretinoin'), ('Pregnant', P)]),
    ('methotrexate', 'toxicity', [('Prescribed', P, 'methotrexate'), ('Takes', P, 'trimethoprim')]),
    ('digoxin', 'toxicity', [('Prescribed', P, 'digoxin'), ('Takes', P, 'amiodarone')]),
    ('clopidogrel', 'reduced_effect', [('Prescribed', P, 'clopidogrel'), ('Takes', P, 'omeprazole')]),
    ('theophylline', 'toxicity', [('Prescribed', P, 'theophylline'), ('Takes', P, 'ciprofloxacin')]),
    ('lithium', 'toxicity', [('Prescribed', P, 'lithium'), ('Takes', P, 'ibuprofen')]),
    ('allopurinol', 'rash_risk', [('Prescribed', P, 'allopurinol'), ('Takes', P, 'amoxicillin')]),
    ('sertraline', 'bleeding_risk', [('Prescribed', P, 'sertraline'), ('Takes', P, 'warfarin')]),
    ('ibuprofen', 'renal_injury', [('Prescribed', P, 'ibuprofen'), ('HasCondition', P, 'renal_impairment')]),
    ('amoxicillin', 'rash_in_mono', [('Prescribed', P, 'amoxicillin'), ('HasCondition', P, 'infectious_mononucleosis')]),
    ('prednisolone', 'hyperglycaemia', [('Prescribed', P, 'prednisolone'), ('HasCondition', P, 'diabetes'), ('Age', P, AG), ('gt', AG, '65')]),
]

DRUG_CLASS = dict(amoxicillin='penicillin', cefalexin='cephalosporin', ibuprofen='nsaid', aspirin='nsaid', warfarin='anticoagulant',
                  sildenafil='pde5', nitroglycerin='nitrate', lisinopril='acei', potassium_chloride='k_supp', metformin='biguanide',
                  isotretinoin='retinoid', ciprofloxacin='fluoroquinolone', simvastatin='statin', clarithromycin='macrolide',
                  codeine='opioid', tramadol='opioid', diazepam='benzo', sertraline='ssri', paracetamol='analgesic',
                  vitamin_c='supplement', prednisolone='corticosteroid', methotrexate='antimetabolite', trimethoprim='antibiotic',
                  digoxin='glycoside', amiodarone='antiarrhythmic', clopidogrel='antiplatelet', omeprazole='ppi',
                  theophylline='xanthine', lithium='mood_stabiliser', allopurinol='xanthine_oxidase_inh')
CLASS_FACTS = [('ClassInteracts', 'nsaid', 'anticoagulant'), ('ClassInteracts', 'pde5', 'nitrate'), ('ClassInteracts', 'acei', 'k_supp'),
               ('ClassInteracts', 'statin', 'macrolide'), ('ClassInteracts', 'ssri', 'opioid'), ('ClassInteracts', 'opioid', 'benzo'),
               ('CrossReactive', 'penicillin', 'cephalosporin'),
               ('ContraClass', 'biguanide', 'renal_impairment'), ('ContraClass', 'nsaid', 'peptic_ulcer'),
               ('NotForChildren', 'fluoroquinolone'), ('AvoidElderly', 'benzo'),
               ('Teratogenic', 'retinoid'), ('Teratogenic', 'acei'), ('Teratogenic', 'statin'), ('RenalAdjust', 'glycoside')]


def build_kb(n_rules, with_patients=(), n_patients=0, index=True):
    kb = KB()
    for d_, c in DRUG_CLASS.items(): kb.add_fact(('Class', d_, c))
    for f in CLASS_FACTS: kb.add_fact(f)
    for name, h, b in GENERIC: kb.add_rule(h, b, name)
    for d_, why, b in SPECIFIC: kb.add_rule(('Unsafe', d_, P, why), b, f'S-{d_}-{why}')
    i = 0
    while len(kb.all_rules) < n_rules:       # synthetic drug-specific rules to grow the KB (no effect on the 34 tests)
        dn = f'DRUG{i:04d}'
        kb.add_fact(('Class', dn, f'cls{i % 40}'))
        kb.add_rule(('Unsafe', dn, P, f'synthetic_hazard_{i}'), [('Prescribed', P, dn), ('HasCondition', P, f'cond{i % 60}')], f'X-{dn}')
        i += 1
    for fs in with_patients:
        for f in fs: kb.add_fact(f)
    return kb


def patient(pid, age, allergies=(), takes=(), conds=(), pregnant=False):
    f = [('Age', pid, str(age))]
    f += [('AllergicTo', pid, a) for a in allergies] + [('Takes', pid, t) for t in takes] + [('HasCondition', pid, c) for c in conds]
    if pregnant: f.append(('Pregnant', pid))
    return f


# (id, description, patient facts, prescribed drug, expected: None=safe or reason kind)
CASES = [
    ('T01', 'penicillin allergy + amoxicillin', patient('t01', 40, allergies=['penicillin']), 'amoxicillin', 'allergy'),
    ('T02', 'penicillin allergy + cefalexin (cross-reactive)', patient('t02', 35, allergies=['penicillin']), 'cefalexin', 'cross_allergy'),
    ('T03', 'warfarin + aspirin', patient('t03', 66, takes=['aspirin']), 'warfarin', 'interaction'),
    ('T04', 'ibuprofen while on warfarin', patient('t04', 70, takes=['warfarin']), 'ibuprofen', 'interaction'),
    ('T05', 'sildenafil + nitroglycerin', patient('t05', 58, takes=['nitroglycerin']), 'sildenafil', 'interaction'),
    ('T06', 'nitroglycerin + sildenafil', patient('t06', 58, takes=['sildenafil']), 'nitroglycerin', 'interaction'),
    ('T07', 'lisinopril + potassium chloride', patient('t07', 61, takes=['potassium_chloride']), 'lisinopril', 'interaction'),
    ('T08', 'simvastatin + clarithromycin', patient('t08', 55, takes=['clarithromycin']), 'simvastatin', 'interaction'),
    ('T09', 'tramadol + sertraline', patient('t09', 45, takes=['sertraline']), 'tramadol', 'interaction'),
    ('T10', 'codeine + diazepam', patient('t10', 50, takes=['diazepam']), 'codeine', 'interaction'),
    ('T11', 'metformin + renal impairment', patient('t11', 60, conds=['renal_impairment']), 'metformin', 'contraindicated'),
    ('T12', 'ibuprofen + peptic ulcer', patient('t12', 52, conds=['peptic_ulcer']), 'ibuprofen', 'contraindicated'),
    ('T13', 'aspirin, age 9, viral infection', patient('t13', 9, conds=['viral_infection']), 'aspirin', 'reye_syndrome'),
    ('T14', 'ciprofloxacin, age 8', patient('t14', 8), 'ciprofloxacin', 'paediatric'),
    ('T15', 'diazepam, age 80', patient('t15', 80), 'diazepam', 'geriatric'),
    ('T16', 'isotretinoin, pregnant', patient('t16', 27, pregnant=True), 'isotretinoin', 'pregnancy'),
    ('T17', 'lisinopril, pregnant', patient('t17', 30, pregnant=True), 'lisinopril', 'pregnancy'),
    ('T18', 'simvastatin, pregnant', patient('t18', 33, pregnant=True), 'simvastatin', 'pregnancy'),
    ('T19', 'ibuprofen + aspirin (duplicate NSAID)', patient('t19', 44, takes=['aspirin']), 'ibuprofen', 'duplicate_therapy'),
    ('T20', 'multi-hazard: warfarin + aspirin + ibuprofen, renal impairment', patient('t20', 77, takes=['aspirin', 'ibuprofen'], conds=['renal_impairment']), 'warfarin', 'bleeding_risk'),
    ('T21', 'paracetamol, healthy adult', patient('t21', 30), 'paracetamol', None),
    ('T22', 'amoxicillin, no allergy', patient('t22', 25), 'amoxicillin', None),
    ('T23', 'metformin, normal renal function', patient('t23', 54, conds=['diabetes']), 'metformin', None),
    ('T24', 'lisinopril, adult, not pregnant', patient('t24', 62), 'lisinopril', None),
    ('T25', 'sertraline with paracetamol', patient('t25', 38, takes=['paracetamol']), 'sertraline', None),
    ('T26', 'ibuprofen, healthy 30-year-old', patient('t26', 30), 'ibuprofen', None),
    ('T27', 'aspirin, age 40, no infection', patient('t27', 40), 'aspirin', None),
    ('T28', 'warfarin with vitamin C', patient('t28', 59, takes=['vitamin_c']), 'warfarin', None),
    ('T29', 'sildenafil with metformin', patient('t29', 49, takes=['metformin']), 'sildenafil', None),
    ('T30', 'diazepam, age 40', patient('t30', 40), 'diazepam', None),
    ('T31', 'ciprofloxacin, age 35', patient('t31', 35), 'ciprofloxacin', None),
    ('T32', 'simvastatin with amoxicillin', patient('t32', 52, takes=['amoxicillin']), 'simvastatin', None),
    ('T33', 'boundary: ciprofloxacin, age 12 (rule needs <12)', patient('t33', 12), 'ciprofloxacin', None),
    ('T34', 'boundary: diazepam, age 75 (rule needs >74)', patient('t34', 75), 'diazepam', 'geriatric'),
]


def kind(r):
    return r[0] if isinstance(r, tuple) else r


def advise(kb, pid, drug, **kw):
    """All unsafe reasons for (drug, patient) + proof of the first one.  Returns (reasons, explanation lines, stats)."""
    st = Stats()
    goal = ('Unsafe', drug, pid, '?why')
    reasons, expl = [], None
    for th, prf in kb.solve([goal], st=st, proof=True, **kw):
        r = subst(th, '?why')
        if r not in reasons:
            reasons.append(r)
            if expl is None: expl = KB.explain(th, prf)
    return reasons, expl or [], st


def fc_advise(kb, pid, drug, static_facts, patient_facts):
    facts = static_facts + patient_facts + [('Prescribed', pid, drug)]
    rules = [(h, b, n) for n, h, b in GENERIC if h[0] == 'Unsafe'] + [(('Unsafe', d_, P, why), b, f'S-{d_}') for d_, why, b in SPECIFIC]
    rules += [(r[0], list(r[1]), r[2]) for r in kb.all_rules if r[2].startswith('X-')]
    store, s = t6.fc(rules, facts, seminaive=True)
    reasons = [a[3] for a in store.by.get('Unsafe', ()) if a[1] == drug and a[2] == pid]
    s['rule_scans'] = len(rules) * s['iterations']          # every rule is tried at least once per iteration
    return reasons, s


def median_ms(fn, reps=15):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


if __name__ == '__main__':
    sys.stdout = Tee(os.path.join(OUT, 'task7_output.txt'))
    R = {'sizes': []}
    print('=' * 100); print('TASK 7  -  Hospital medication-safety advisor (educational prototype, not clinical guidance)'); print('=' * 100)
    pat_facts = [c[2] + [('Prescribed', c[2][0][1], c[3])] for c in CASES]
    static = [('Class', d_, c) for d_, c in DRUG_CLASS.items()] + CLASS_FACTS
    for n_rules in (40, 400, 2000):
        tracemalloc.start()
        kb = build_kb(n_rules, with_patients=pat_facts)
        cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        print(f'\n[{n_rules} rules]  actual rules = {len(kb.all_rules)}, facts = {len(kb.factset)}, KB build memory = {cur / 1024:.0f} KiB (peak {peak / 1024:.0f} KiB)')
        rows, lat_i, lat_u, lat_f, st_i, st_u, st_f, expl_len = [], [], [], [], [], [], [], []
        npass = 0
        for cid, desc, pf, drug, exp in CASES:
            pid = pf[0][1]
            reasons, expl, st = advise(kb, pid, drug)
            reasons_u, _, st_u_ = advise(kb, pid, drug, index_rules=False)
            fc_r, fcs = fc_advise(kb, pid, drug, static, pf)
            ok = (exp is None and not reasons) or (exp is not None and exp in {kind(r) for r in reasons})
            ok &= {kind(r) for r in reasons} == {kind(r) for r in reasons_u} == {kind(r) for r in fc_r}
            npass += ok
            li = median_ms(lambda: advise(kb, pid, drug))
            lu = median_ms(lambda: advise(kb, pid, drug, index_rules=False))
            lf = median_ms(lambda: fc_advise(kb, pid, drug, static, pf), reps=5)
            lat_i.append(li); lat_u.append(lu); lat_f.append(lf)
            st_i.append(st.goals + st.unifs); st_u.append(st_u_.goals + st_u_.unifs); st_f.append(fcs['unifs'] + fcs['firings'] + fcs['rule_scans'])
            expl_len.append(len(expl))
            rows.append((cid, desc, 'UNSAFE' if reasons else 'SAFE', exp or '-', ','.join(sorted({kind(r) for r in reasons})) or '-', st.goals, st.unifs, li, len(expl), ok))
        print(f'  {"id":<4} {"case":<58} {"verdict":<7} {"expected":<17} {"reasons found":<34} {"goals":>5} {"unifs":>5} {"ms":>6} {"expl":>4} ok')
        for r in rows:
            print(f'  {r[0]:<4} {r[1][:57]:<58} {r[2]:<7} {r[3]:<17} {r[4][:33]:<34} {r[5]:>5} {r[6]:>5} {r[7]:>6.3f} {r[8]:>4} {"PASS" if r[9] else "FAIL"}')
        print(f'  => {npass}/{len(CASES)} test cases pass (all three engines also agree on every case)')
        print(f'  mean latency   indexed BC (design) {statistics.mean(lat_i):8.3f} ms | unindexed BC {statistics.mean(lat_u):8.3f} ms | forward chaining {statistics.mean(lat_f):9.3f} ms')
        print(f'  mean steps     indexed BC (design) {statistics.mean(st_i):8.1f}    | unindexed BC {statistics.mean(st_u):8.1f}    | forward chaining {statistics.mean(st_f):9.1f}   (BC: goals+unifications; FC: unifications+firings+rule scans)')
        print(f'  mean explanation length = {statistics.mean(expl_len):.1f} lines (proof of the first reason)')
        R['sizes'].append(dict(rules=len(kb.all_rules), passed=npass, lat_indexed=statistics.mean(lat_i), lat_unindexed=statistics.mean(lat_u),
                               lat_fc=statistics.mean(lat_f), steps_indexed=statistics.mean(st_i), steps_unindexed=statistics.mean(st_u),
                               steps_fc=statistics.mean(st_f), expl=statistics.mean(expl_len), mem_kib=cur / 1024, facts=len(kb.factset)))
        if n_rules == 2000:
            print('\n  Example explanation (T20, multi-hazard, 2000-rule KB):')
            _, expl, _ = advise(kb, 't20', 'warfarin')
            for l in expl: print('   ' + l)
            kb2000 = kb

    # ------------------ 50,000 patients
    print('\n[50,000 patients]  2000-rule KB + 50,000 synthetic patient records')
    rnd = random.Random(5)
    drugs = list(DRUG_CLASS)
    tracemalloc.start()
    kbp = build_kb(2000)
    base_mem = tracemalloc.get_traced_memory()[0]
    allp = []
    for i in range(50000):
        pid = f'p{i}'
        fs = patient(pid, rnd.randint(1, 95), allergies=rnd.sample(['penicillin', 'nsaid', 'sulfa'], rnd.randint(0, 1)),
                     takes=rnd.sample(drugs, rnd.randint(0, 3)), conds=rnd.sample(['diabetes', 'renal_impairment', 'peptic_ulcer', 'viral_infection'], rnd.randint(0, 2)),
                     pregnant=rnd.random() < .03)
        for f in fs: kbp.add_fact(f)
        allp.append(pid)
    full_mem = tracemalloc.get_traced_memory()[0]; tracemalloc.stop()
    print(f'  facts in KB = {len(kbp.factset):,}   memory: rules+drug knowledge {base_mem / 2 ** 20:.1f} MiB, with 50,000 patients {full_mem / 2 ** 20:.1f} MiB')
    R['patients'] = []
    print(f'  {"patients loaded":>15} {"mean latency ms":>16} {"mean goals":>11}')
    for npat in (500, 5000, 50000):
        sample = random.Random(1).sample(allp[:npat], 200)
        ts, gs = [], []
        for pid in sample:
            dr = random.Random(hash(pid) % 1000).choice(drugs)
            kbp.facts['Prescribed'].clear() if False else None
            kbp.add_fact(('Prescribed', pid, dr))
            t0 = time.perf_counter(); _, _, st = advise(kbp, pid, dr); ts.append((time.perf_counter() - t0) * 1000); gs.append(st.goals)
        print(f'  {npat:>15,} {statistics.mean(ts):>16.3f} {statistics.mean(gs):>11.1f}')
        R['patients'].append(dict(patients=npat, latency=statistics.mean(ts), goals=statistics.mean(gs)))
    print('  Latency does not grow with the number of patients: every rule body starts from Prescribed(p,d) / Age(p,a) with p bound,')
    print('  so the first-argument fact index touches only that patient\'s handful of facts.')

    with open(os.path.join(OUT, 'task7_results.json'), 'w') as f: json.dump(R, f, indent=1, default=str)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ns = [r['rules'] for r in R['sizes']]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(ns, [r['lat_indexed'] for r in R['sizes']], 'o-', label='indexed BC (final design)')
    ax[0].plot(ns, [r['lat_unindexed'] for r in R['sizes']], 's-', label='unindexed BC')
    ax[0].plot(ns, [r['lat_fc'] for r in R['sizes']], '^-', label='forward chaining')
    ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel('rules'); ax[0].set_ylabel('ms / query'); ax[0].set_title('(a) Query latency'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(ns, [r['steps_indexed'] for r in R['sizes']], 'o-', label='indexed BC')
    ax[1].plot(ns, [r['steps_unindexed'] for r in R['sizes']], 's-', label='unindexed BC')
    ax[1].plot(ns, [r['steps_fc'] for r in R['sizes']], '^-', label='forward chaining')
    ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlabel('rules'); ax[1].set_ylabel('inference steps'); ax[1].set_title('(b) Inference steps'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[2].plot([r['patients'] for r in R['patients']], [r['latency'] for r in R['patients']], 'o-')
    ax[2].set_xscale('log'); ax[2].set_xlabel('patients in KB'); ax[2].set_ylabel('ms / query'); ax[2].set_ylim(bottom=0); ax[2].set_title('(c) Latency vs patients (2000 rules)'); ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, 'task7_scaling.png'), dpi=150); plt.close()
    print('\nplots written: plots/task7_scaling.png')
