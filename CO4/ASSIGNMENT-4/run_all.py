"""Runs every task script in order and refreshes outputs/logs and outputs/figures. Usage: python run_all.py"""
import subprocess, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [("task1_protocol/task1_defect_demo.py", "task1_defect_demo"),
           ("task2_decision_tree/task2_desk_check.py", "task2_desk_check"),
           ("task2_decision_tree/task2_experiments.py", "task2_experiments"),
           ("task3_em_algorithm/task3_experiments.py", "task3_experiments"),
           ("task4_neural_network/task4_experiments.py", "task4_experiments"),
           ("task5_learning_comparison/task5_nb_experiment.py", "task5_nb_experiment"),
           ("task5_learning_comparison/task5_kernel_experiment.py", "task5_kernel_experiment"),
           ("task5_learning_comparison/task5_ebl_experiment.py", "task5_ebl_experiment"),
           ("task6_rl_controller/task6_toy_trace.py", "task6_toy_trace"),
           ("task6_rl_controller/task6_experiments.py", "task6_experiments")]
os.makedirs(os.path.join(HERE, "outputs", "logs"), exist_ok=True)
for script, name in SCRIPTS:
    print("running", script, "...", flush=True)
    out = subprocess.run([sys.executable, os.path.join(HERE, script)], capture_output=True, text=True, cwd=HERE)
    open(os.path.join(HERE, "outputs", "logs", name + ".txt"), "w", encoding="utf-8").write(out.stdout + out.stderr)
    print("   exit code", out.returncode)
subprocess.run([sys.executable, os.path.join(HERE, "tools", "make_screenshots.py")], cwd=HERE)
print("done - see outputs/")
