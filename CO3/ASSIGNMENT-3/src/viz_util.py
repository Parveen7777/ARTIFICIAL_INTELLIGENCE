"""viz_util.py - small helpers: Tee (console + file logging) and proof-tree drawing (matplotlib)."""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Tee:
    """Write everything printed to the console AND to an output file (used as evidence for the report)."""
    def __init__(self, path):
        self.f = open(path, 'w', encoding='utf-8')

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


def draw_dag(nodes, path, title, width=13, wrap=30):
    """nodes: list of dict(id, label, parents=[ids], kind='input'|'derived'|'empty').
    Staircase layout: derived clauses form the spine, input clauses sit beside the step that consumes them."""
    import textwrap
    byid = {n['id']: n for n in nodes}
    children = {n['id']: [] for n in nodes}
    for n in nodes:
        for p in n['parents']: children[p].append(n['id'])
    level = {}
    def lv(i):
        if i in level: return level[i]
        n = byid[i]
        ds = [lv(p) for p in n['parents'] if byid[p]['parents']]
        level[i] = 1 + max(ds) if n['parents'] and ds else (1 if n['parents'] else None)
        return level[i]
    for n in nodes:
        if n['parents']: lv(n['id'])
    for n in nodes:                                   # inputs sit one row above their lowest consumer
        if not n['parents']:
            level[n['id']] = min(level[c] for c in children[n['id']]) - 1 if children[n['id']] else 0
    mn = min(level.values())
    for k in level: level[k] -= mn
    L = max(level.values()) + 1
    rows = {}
    for n in nodes: rows.setdefault(level[n['id']], []).append(n)
    pos = {}
    for l, ns in rows.items():
        spine = [n for n in ns if n['parents']]
        ins = [n for n in ns if not n['parents']]
        for k, n in enumerate(spine): pos[n['id']] = (0.0 + 0.0 * k, -l)
        left = right = 0
        for k, n in enumerate(ins):
            if k % 2 == 0:
                left += 1; pos[n['id']] = (-3.6 * left, -l + 0.0)
            else:
                right += 1; pos[n['id']] = (3.6 * right, -l + 0.0)
    xs = [p[0] for p in pos.values()]
    fig, ax = plt.subplots(figsize=(width, max(4.5, 1.05 * L + 1.2)))
    colors = {'input': '#dbeafe', 'derived': '#fef3c7', 'empty': '#bbf7d0'}
    for n in nodes:
        for p in n['parents']:
            (x1, y1), (x2, y2) = pos[p], pos[n['id']]
            ax.plot([x1, x2], [y1 - 0.18, y2 + 0.18] if y1 != y2 else [y1, y2], color='#6b7280', lw=1.0, zorder=1)
    for n in nodes:
        x, y = pos[n['id']]
        txt = '\n'.join(textwrap.wrap(n['label'], wrap)) or '\u25a1 (empty clause)'
        ax.text(x, y, txt, ha='center', va='center', fontsize=7.6, zorder=3,
                bbox=dict(boxstyle='round,pad=0.3', fc=colors[n.get('kind', 'derived')], ec='#374151', lw=.8))
    ax.set_xlim(min(xs) - 2.6, max(xs) + 2.6); ax.set_ylim(-L + 0.4, 0.6); ax.axis('off')
    ax.set_title(title, fontsize=10.5, fontweight='bold')
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def draw_proof_tree(parents, path, title):
    """Propositional proof (parents: clause -> (c1, c2, literal)); draws only clauses used in the refutation."""
    from logic import clause_str
    root = frozenset()
    ids, nodes = {}, []
    def go(c):
        if c in ids: return ids[c]
        pars = []
        if c in parents:
            a, b, _ = parents[c]
            pars = [go(a), go(b)]
        ids[c] = len(ids) + 1
        kind = 'empty' if not c else ('derived' if c in parents else 'input')
        nodes.append(dict(id=ids[c], label=clause_str(c), parents=pars, kind=kind))
        return ids[c]
    go(root)
    draw_dag(nodes, path, title)
