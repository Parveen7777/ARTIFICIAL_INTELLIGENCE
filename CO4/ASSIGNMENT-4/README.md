# CSA17 – Artificial Intelligence · Assignment 4 (CO4)

**Learning: Algorithm Design, Implementation, Test, Debug and Evaluate** – SIMATS Engineering

Every task follows the workflow **Design → Implement → Instrument → Test → Break → Debug → Validate → Defend**.
All random seeds are fixed, so every number in the report can be reproduced.

## Repository layout

| Folder | Task | What it contains |
|---|---|---|
| `task1_protocol/` | Task 1 (5) | Defect demonstration (train-on-test optimism, baseline, seeds) and the repaired protocol |
| `task2_decision_tree/` | Task 2 (9) | From-scratch ID3/C4.5 learner (`decision_tree.py`), desk-check, sklearn verification, pruning, identifier-bias test |
| `task3_em_algorithm/` | Task 3 (6) | Faulty and repaired EM for a 1-D Gaussian mixture, traces, 20 random initialisations |
| `task4_neural_network/` | Task 4 (9) | NumPy back-propagation MLP, gradient check, XOR, 8×8 digits grid, failure diagnosis |
| `task5_learning_comparison/` | Task 5 (8) | Naive Bayes (Laplace), linear/RBF SVM, tree comparison, Explanation-Based Learning |
| `task6_rl_controller/` | Task 6 (13) | HVAC simulator, Q-learning/SARSA, toy desk-check, 20-seed study, two adversarial tests |
| `outputs/logs/` | – | Captured console output of every script |
| `outputs/figures/` | – | Plots used in the report |
| `outputs/screenshots/` | – | VS Code-style screenshots of code and output |
| `tools/` | – | Screenshot renderer |

## Setup (VS Code)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Run

```bash
python run_all.py                                   # everything, refreshes outputs/
# or task by task
python task1_protocol/task1_defect_demo.py
python task2_decision_tree/task2_desk_check.py
python task2_decision_tree/task2_experiments.py
python task3_em_algorithm/task3_experiments.py
python task4_neural_network/task4_experiments.py
python task5_learning_comparison/task5_nb_experiment.py
python task5_learning_comparison/task5_kernel_experiment.py
python task5_learning_comparison/task5_ebl_experiment.py
python task6_rl_controller/task6_toy_trace.py
python task6_rl_controller/task6_experiments.py
```

Total run time is a few minutes on a laptop CPU. Python ≥ 3.9 is required.

## Headline results

| Task | Result |
|---|---|
| 1 | Unpruned tree: train accuracy 1.000 vs held-out 0.804 (optimism ≈ +0.20); majority baseline scores 0.816 accuracy with recall 0 |
| 2 | Scratch tree has the same root split as scikit-learn; test accuracy 0.953 (entropy); unique-ID attribute wins the root under plain information gain |
| 3 | Faulty EM: 0/20 runs monotone, Σπ collapses to ~1e-15. Repaired (one normalisation line): 20/20 monotone, Σπ = 1 |
| 4 | Max relative gradient error 1.7e-8; perceptron ≤ 0.75 on XOR, MLP solves it; digits best test accuracy 0.975 |
| 5 | RBF-SVM F1 0.963 vs linear SVM 0.707 on a non-linearly-separable grid; EBL rule is sound (precision 1.0) but covers 5 of 10 hazardous cases |
| 6 | Q-learning beats the thermostat on 20/20 seeds: comfort violations −58 %, reward +36 % |

## Notes on honesty

Where a prediction made before running turned out wrong (Task 4 `P4`, Task 6 attack 1 magnitude) the scripts print the
refutation rather than hiding it. Datasets are public (scikit-learn Breast Cancer, Digits) or synthetic and fully described
in the code (student at-risk, smart grid, HVAC simulator).
