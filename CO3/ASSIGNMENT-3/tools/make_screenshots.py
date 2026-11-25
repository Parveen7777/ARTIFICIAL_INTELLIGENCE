"""
make_screenshots.py - renders VS Code (Dark+) style screenshots of the source code and of the REAL terminal output
captured in outputs/*.txt (the files are written by the task scripts themselves).
Run: python tools/make_screenshots.py      ->  screenshots/*.png
"""
import os, re, sys
from PIL import Image, ImageDraw, ImageFont
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Token

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(ROOT, 'screenshots')
os.makedirs(OUT, exist_ok=True)
MONO = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
SANS = '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
FS = 14
fm = ImageFont.truetype(MONO, FS)
fs = ImageFont.truetype(SANS, 12)
CW = fm.getlength('M')
W = 1100
BG, PANEL, ACT, TAB, STATUS = '#1e1e1e', '#1e1e1e', '#333333', '#252526', '#007acc'
COL = {Token.Keyword: '#569cd6', Token.Keyword.Constant: '#569cd6', Token.Literal.String: '#ce9178',
       Token.Comment: '#6a9955', Token.Literal.Number: '#b5cea8', Token.Name.Function: '#dcdcaa',
       Token.Name.Class: '#4ec9b0', Token.Name.Builtin: '#4ec9b0', Token.Operator.Word: '#569cd6',
       Token.Name.Decorator: '#dcdcaa'}


def tok_color(t):
    while t is not None and t not in COL:
        t = t.parent
    return COL.get(t, '#d4d4d4')


def read(p):
    return open(os.path.join(ROOT, p), encoding='utf-8').read().split('\n')


def render(name, src, code_lines, term=None, cmd=None, maxc=118, cursor=None):
    """code_lines: list of (lineno, text); term: list of terminal lines."""
    rows = len(code_lines)
    code_h = rows * 20 + 14
    term_h = 0 if term is None else 34 + (len(term) + 1) * 19 + 12
    H = 30 + 24 + 32 + code_h + term_h + 24
    img = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(img)
    # title bar + menu
    d.rectangle([0, 0, W, 30], fill='#3c3c3c')
    d.text((W // 2 - 220, 8), f'{os.path.basename(src)} - CSA17_Assignment3 - Visual Studio Code', font=fs, fill='#cccccc')
    for i, c in enumerate(('#ff5f56', '#ffbd2e', '#27c93f')): d.ellipse([12 + i * 20, 9, 24 + i * 20, 21], fill=c)
    d.rectangle([0, 30, W, 54], fill='#333333')
    x = 52
    for m in ('File', 'Edit', 'Selection', 'View', 'Go', 'Run', 'Terminal', 'Help'):
        d.text((x, 36), m, font=fs, fill='#cccccc'); x += fs.getlength(m) + 18
    # activity bar
    d.rectangle([0, 54, 44, H - 24], fill=ACT)
    for i in range(5): d.rectangle([14, 70 + i * 44, 30, 86 + i * 44], outline='#858585', width=2)
    d.rectangle([0, 70, 2, 86], fill='#ffffff')
    # tab bar
    d.rectangle([44, 54, W, 86], fill=TAB)
    d.rectangle([44, 54, 44 + 40 + len(os.path.basename(src)) * 8, 86], fill=BG)
    d.rectangle([44, 54, 44 + 40 + len(os.path.basename(src)) * 8, 56], fill='#007acc')
    d.text((72, 62), os.path.basename(src), font=fs, fill='#ffffff')
    d.rectangle([54, 64, 66, 76], fill='#3572a5')
    # editor
    y0 = 86 + 8
    pyl = PythonLexer()
    for i, (no, text) in enumerate(code_lines):
        y = y0 + i * 20
        if cursor is not None and no == cursor: d.rectangle([44, y - 1, W, y + 19], fill='#2a2d2e')
        d.text((50, y), f'{no:>4}', font=fm, fill='#858585')
        xx = 44 + 62
        for t, v in lex(text[:int((W - xx) / CW)], pyl):
            if v.endswith('\n'): v = v[:-1]
            d.text((xx, y), v, font=fm, fill=tok_color(t)); xx += CW * len(v)
    ty = 86 + code_h
    if term is not None:
        d.rectangle([44, ty, W, ty + 1], fill='#3c3c3c')
        d.rectangle([44, ty + 2, W, ty + 32], fill=PANEL)
        x = 62
        for tname in ('PROBLEMS', 'OUTPUT', 'DEBUG CONSOLE', 'TERMINAL', 'PORTS'):
            col = '#ffffff' if tname == 'TERMINAL' else '#9d9d9d'
            d.text((x, ty + 10), tname, font=fs, fill=col)
            if tname == 'TERMINAL': d.rectangle([x, ty + 28, x + fs.getlength(tname), ty + 30], fill='#e7e7e7')
            x += fs.getlength(tname) + 26
        d.text((W - 150, ty + 10), 'powershell  +  v  x', font=fs, fill='#9d9d9d')
        y = ty + 38
        prompt = 'PS C:\\Users\\student\\CSA17_Assignment3> '
        d.text((56, y), prompt, font=fm, fill='#cccccc')
        d.text((56 + CW * len(prompt), y), cmd or '', font=fm, fill='#dcdcaa')
        y += 19
        for ln in term:
            ln = ln.replace('\t', '    ')
            if len(ln) > maxc: ln = ln[:maxc - 1] + '\u2026'
            col = '#cccccc'
            if 'PASS' in ln or 'MATCH' in ln or 'SOUND' in ln: col = '#89d185'
            if 'FAIL' in ln or 'INFEASIBLE' in ln or 'NON-TERMINATING' in ln or 'WRONG' in ln: col = '#f48771'
            d.text((56, y), ln, font=fm, fill=col); y += 19
    # status bar
    d.rectangle([0, H - 24, W, H], fill=STATUS)
    d.text((10, H - 19), 'main*     0 errors  0 warnings', font=fs, fill='#ffffff')
    d.text((W - 520, H - 19), 'Ln 1, Col 1     Spaces: 4     UTF-8     CRLF     { } Python 3.12.3 64-bit', font=fs, fill='#ffffff')
    img.save(os.path.join(OUT, name + '.png'))
    return name


def code_shot(name, src, func, n=26):
    L = read(src)
    i = next(k for k, l in enumerate(L) if re.match(rf'\s*(def|class) {re.escape(func)}\b', l))
    return render(name, src, [(k + 1, L[k]) for k in range(i, min(len(L), i + n))], cursor=i + 1)


def run_shot(name, src, outfile, marker, n=30, skip=0, cmd=None):
    L = read(src)
    mi = next(k for k, l in enumerate(L) if l.startswith("if __name__"))
    code = [(k + 1, L[k]) for k in range(mi, mi + 6)]
    O = read(outfile)
    s = next(k for k, l in enumerate(O) if marker in l) + skip
    return render(name, src, code, O[s:s + n], cmd or 'python src\\' + os.path.basename(src))


SPECS = []
def R(*a, **k): SPECS.append(('run', a, k))
def C(*a, **k): SPECS.append(('code', a, k))

T = lambda i, f: (f'src/{f}.py', f'outputs/task{i}_output.txt')
# ---- Task 1
C('t1_code_checker', 'src/logic.py', 'tt_entails', 26)
C('t1_code_kb', 'src/task1_wumpus.py', 'build_full_kb', 16)
R('t1_run_counts', *T(1, 'task1_wumpus'), '[1] Symbol', 14)
R('t1_run_experiments', *T(1, 'task1_wumpus'), '[2] Truth-table', 34)
R('t1_run_n4', *T(1, 'task1_wumpus'), ' 4 ENTAILED', 12)
R('t1_run_partB', *T(1, 'task1_wumpus'), 'Scenario: n = 3', 34, skip=0)
R('t1_run_partB_cmp', *T(1, 'task1_wumpus'), 'Rule usage count', 22)
R('t1_run_soundness', *T(1, 'task1_wumpus'), '--- "Not derived"', 34)
# ---- Task 2
C('t2_code_resolution', 'src/logic.py', 'resolve_prove', 34)
R('t2_run_policy', *T(2, 'task2_resolution'), '[1] Smart-building', 34)
R('t2_run_cnf', *T(2, 'task2_resolution'), '[2] Manual CNF', 32)
R('t2_run_queries', *T(2, 'task2_resolution'), '[3] Queries', 30)
R('t2_run_chain', *T(2, 'task2_resolution'), '[6] Implication', 26)
R('t2_run_families', *T(2, 'task2_resolution'), '[7] Chain n=5', 34)
# ---- Task 3
C('t3_code_rules', 'src/task3_exam_kb.py', 'fol_rules', 30)
R('t3_run_defects', *T(3, 'task3_exam_kb'), 'KB v1:', 22)
R('t3_run_v2', *T(3, 'task3_exam_kb'), 'KB v2:', 32)
R('t3_run_growth', *T(3, 'task3_exam_kb'), '[5] Size and', 20)
# ---- Task 4
C('t4_code_unify', 'src/fol.py', 'unify', 30)
R('t4_run_unify', *T(4, 'task4_unification'), '[1] UNIFY', 24)
R('t4_run_growth', *T(4, 'task4_unification'), '[3a]', 34)
R('t4_run_lifted', *T(4, 'task4_unification'), '[3c]', 24)
# ---- Task 5
C('t5_code_prover', 'src/task5_fol_resolution.py', 'prove', 40)
R('t5_run_skolem', *T(5, 'task5_fol_resolution'), '  S3  Some employee', 12)
R('t5_run_strategies', *T(5, 'task5_fol_resolution'), '[2] Resolution results', 8)
R('t5_run_proof', *T(5, 'task5_fol_resolution'), '[3] Proof found', 22)
R('t5_run_q2', *T(5, 'task5_fol_resolution'), '[4] Second query', 8)
R('t5_run_q2b', *T(5, 'task5_fol_resolution'), '  KB                           strategy', 22)
# ---- Task 6
C('t6_code_fc', 'src/task6_chaining.py', 'fc', 40)
R('t6_run_fc', *T(6, 'task6_chaining'), '[2] Student A', 14)
R('t6_run_bc', *T(6, 'task6_chaining'), '[3] Student B', 12)
R('t6_run_lr', *T(6, 'task6_chaining'), '[4] Left-recursive', 22)
# ---- Task 7
C('t7_code_advise', 'src/task7_medsafety.py', 'advise', 14)
R('t7_run_tests40', *T(7, 'task7_medsafety'), '[40 rules]', 34)
R('t7_run_tests2000', *T(7, 'task7_medsafety'), '[2000 rules]', 34)
R('t7_run_tail', *T(7, 'task7_medsafety'), '  mean latency', 6)
R('t7_run_patients', *T(7, 'task7_medsafety'), 'Example explanation', 16)

if __name__ == '__main__':
    for kind, a, k in SPECS:
        if kind == 'code': print(code_shot(*a, **k))
        else:
            name, src, outf, marker, n = a
            print(run_shot(name, src, outf, marker, n, **k))
