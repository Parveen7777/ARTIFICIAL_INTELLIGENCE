# CSA17 – Artificial Intelligence · Assignment 3 (CO3) · Code, outputs, plots and screenshots

SIMATS Engineering · Pair work · Knowledge representation and reasoning
(propositional logic, first-order logic, resolution, forward / backward chaining, knowledge-based design).

## Folder layout
| Path | Content |
|---|---|
| `src/logic.py` | propositional AST, truth-table entailment checker (instrumented), CNF conversion, given-clause resolution prover (tautology deletion, subsumption, unit preference, set-of-support), DPLL cross-checker |
| `src/fol.py` | UNIFY (occurs check optional), substitution composition, standardising apart, lifted Modus Ponens, definite-clause backward chainer (fact/rule indexing, negation as failure, loop check, depth bound, memoisation, tabling), ground Horn forward/backward chaining |
| `src/task1_wumpus.py` | Task 1 – Wumpus-world KB, model checking, MP / And-Elim / Unit-Resolution derivations, soundness tables |
| `src/task2_resolution.py` | Task 2 – 14-rule access-control policy, manual CNF, resolution proof tree, chain / irrelevant-symbol / pigeonhole / random-3-CNF families |
| `src/task3_exam_kb.py` | Task 3 – exam-eligibility advisor, FOL (13 axioms) vs propositional encoding, 15 competency queries, defect log, size/time experiment |
| `src/task4_unification.py` | Task 4 – unifier tests, lifted MP, propositionalisation growth, lifted vs propositional inference |
| `src/task5_fol_resolution.py` | Task 5 – English → FOL → CNF (Skolemisation), first-order resolution prover with strategies, non-entailed query with step cap |
| `src/task6_chaining.py` | Task 6 – forward chaining (naive / semi-naive) and backward chaining incl. left-recursion remedies on the organisation KB |
| `src/task7_medsafety.py` | Task 7 – hospital medication-safety advisor (40 / 400 / 2,000 rules, 34 tests, 50,000 patients, baselines) |
| `src/viz_util.py` | proof-tree drawing, console + file logging |
| `outputs/` | raw terminal output (`taskN_output.txt`) and raw test data (`taskN_results.json`) of every run |
| `plots/` | all graphs and proof trees used in the report |
| `screenshots/` | VS Code-style screenshots of code and real terminal output (made by `tools/make_screenshots.py`) |
| `tools/make_screenshots.py` | renders the screenshots from `outputs/*.txt` and the sources |

## How to run (VS Code terminal)
```bash
python -m venv .venv            # optional
pip install -r requirements.txt # matplotlib, pillow, pygments (the logic code itself uses only the standard library)
python src/task1_wumpus.py      # ... one script per task, each prints its tables and writes outputs/taskN_output.txt
python src/task7_medsafety.py
python tools/make_screenshots.py
```
`run_all.bat` (Windows) / `run_all.sh` (Linux, macOS) run all seven tasks. Total run time ≈ 3 minutes.
Operation counts (models, clauses, inference steps, unifications) are the primary evidence; wall-clock times are secondary and machine-dependent.

## Notes
* Task 7 is an **educational prototype**; the medical rules are simplified examples and are not clinical guidance.
* Screenshots are rendered images of the real captured output in a VS Code Dark+ layout. Replace them with your own VS Code screenshots (run the scripts and capture the terminal) if your evaluator requires native captures.
