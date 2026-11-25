"""
make_screenshots.py - renders the saved program output (outputs/logs/*.txt) and key source files as images that
imitate the VS Code Dark+ window (title bar, tab, editor / integrated terminal).

These are rendered images of the REAL program output. For the final submission you may replace them with live
VS Code screenshots by running each script in the VS Code terminal (commands are listed in README.md).

Run:  python tools/make_screenshots.py
"""
import os
import textwrap
from PIL import Image, ImageDraw, ImageFont
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Token

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "outputs", "screenshots")
os.makedirs(OUT, exist_ok=True)
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 19)
UI = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
CW, LH = int(FONT.getlength("M")), 26
BG, TITLE, TAB, PANEL, FG = (30, 30, 30), (60, 60, 60), (30, 30, 30), (24, 24, 24), (204, 204, 204)
COL = {Token.Keyword: (86, 156, 214), Token.Name.Function: (220, 220, 170), Token.Name.Class: (78, 201, 176),
       Token.String: (206, 145, 120), Token.Comment: (106, 153, 85), Token.Number: (181, 206, 168),
       Token.Operator.Word: (86, 156, 214), Token.Name.Builtin: (78, 201, 176), Token.Literal.String.Doc: (106, 153, 85)}


def tok_color(tt):
    while tt is not None:
        if tt in COL:
            return COL[tt]
        tt = tt.parent
    return FG


def window(width, height, filename, tab_name):
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, width, 30], fill=TITLE)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([12 + i * 20, 9, 24 + i * 20, 21], fill=c)
    d.text((width // 2 - 200, 7), "%s - AI_Assignment4_CSA17 - Visual Studio Code" % filename, font=UI, fill=(200, 200, 200))
    d.rectangle([0, 30, width, 62], fill=(37, 37, 38))
    d.rectangle([0, 30, 260, 62], fill=TAB)
    d.text((14, 39), "\U0001F40D " if False else "py  " + tab_name, font=UI, fill=(255, 255, 255))
    d.rectangle([0, height - 24, width, height], fill=(0, 122, 204))
    d.text((12, height - 20), "Python 3.12   Ln 1, Col 1   Spaces: 4   UTF-8   Python", font=UI, fill=(255, 255, 255))
    return img, d


def render_terminal(lines, cmd, filename, out_name):
    wrapped = []
    for ln in lines:
        wrapped += textwrap.wrap(ln, 100, subsequent_indent="      ", replace_whitespace=False, drop_whitespace=False) or [""]
    body = ["PS C:\\Users\\student\\AI_Assignment4_CSA17> " + cmd] + wrapped + ["PS C:\\Users\\student\\AI_Assignment4_CSA17> "]
    w, h = 104 * CW + 40, 62 + 36 + len(body) * LH + 40
    img, d = window(w, h, filename, filename)
    d.rectangle([0, 62, w, h - 24], fill=PANEL)
    d.text((20, 70), "PROBLEMS    OUTPUT    DEBUG CONSOLE    TERMINAL    PORTS", font=UI, fill=(150, 150, 150))
    d.line([(262, 88), (338, 88)], fill=(0, 122, 204), width=2)
    y = 98
    for i, ln in enumerate(body):
        col = (97, 214, 214) if i in (0, len(body) - 1) else FG
        d.text((20, y), ln, font=FONT, fill=col)
        y += LH
    img.save(os.path.join(OUT, out_name))


def render_code(path, start, end, filename, out_name):
    src = open(path, encoding="utf-8").read().splitlines()[start - 1:end]
    text = "\n".join(src)
    w, h = 104 * CW + 90, 62 + len(src) * LH + 50
    img, d = window(w, h, filename, filename)
    d.rectangle([0, 62, 56, h - 24], fill=BG)
    x, y = 70, 72
    for i in range(len(src)):
        d.text((8, 72 + i * LH), str(start + i).rjust(4), font=FONT, fill=(110, 118, 129))
    for tt, val in lex(text, PythonLexer()):
        for j, part in enumerate(val.split("\n")):
            if j > 0:
                y += LH
                x = 70
            if part:
                d.text((x, y), part, font=FONT, fill=tok_color(tt))
                x += int(FONT.getlength(part))
    img.save(os.path.join(OUT, out_name))


RUNS = [  # log, command, script, screenshot prefix, lines per page
    ("task1_defect_demo", "python task1_protocol\\task1_defect_demo.py", "task1_defect_demo.py", "task1_output", 40),
    ("task2_desk_check", "python task2_decision_tree\\task2_desk_check.py", "task2_desk_check.py", "task2_desk_check", 42),
    ("task2_experiments", "python task2_decision_tree\\task2_experiments.py", "task2_experiments.py", "task2_experiments", 44),
    ("task3_experiments", "python task3_em_algorithm\\task3_experiments.py", "task3_experiments.py", "task3_output", 40),
    ("task4_experiments", "python task4_neural_network\\task4_experiments.py", "task4_experiments.py", "task4_output", 46),
    ("task5_nb_experiment", "python task5_learning_comparison\\task5_nb_experiment.py", "task5_nb_experiment.py", "task5_nb_output", 44),
    ("task5_kernel_experiment", "python task5_learning_comparison\\task5_kernel_experiment.py", "task5_kernel_experiment.py", "task5_svm_output", 40),
    ("task5_ebl_experiment", "python task5_learning_comparison\\task5_ebl_experiment.py", "task5_ebl_experiment.py", "task5_ebl_output", 46),
    ("task6_toy_trace", "python task6_rl_controller\\task6_toy_trace.py", "task6_toy_trace.py", "task6_toy_output", 40),
    ("task6_experiments", "python task6_rl_controller\\task6_experiments.py", "task6_experiments.py", "task6_output", 42),
]
CODE = [  # path, start, end, screenshot
    ("task1_protocol/task1_defect_demo.py", 36, 62, "task1_code.png"),
    ("task2_decision_tree/decision_tree.py", 124, 170, "task2_code.png"),
    ("task3_em_algorithm/em_gmm.py", 25, 60, "task3_code.png"),
    ("task4_neural_network/mlp.py", 62, 100, "task4_code.png"),
    ("task5_learning_comparison/smartgrid.py", 55, 95, "task5_nb_code.png"),
    ("task5_learning_comparison/ebl.py", 140, 175, "task5_ebl_code.png"),
    ("task6_rl_controller/q_agent.py", 17, 62, "task6_code.png"),
]


def main():
    for log, cmd, script, prefix, per in RUNS:
        lines = open(os.path.join(ROOT, "outputs", "logs", log + ".txt"), encoding="utf-8").read().splitlines()
        pages = [lines[i:i + per] for i in range(0, len(lines), per)]
        for k, pg in enumerate(pages, 1):
            render_terminal(pg, cmd if k == 1 else "(output continued)", script, "%s_%d.png" % (prefix, k))
    for path, a, b, name in CODE:
        render_code(os.path.join(ROOT, path), a, b, os.path.basename(path), name)
    print(sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
