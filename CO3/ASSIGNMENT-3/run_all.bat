@echo off
REM Windows / VS Code terminal version of run_all.sh
for %%t in (task1_wumpus task2_resolution task3_exam_kb task4_unification task5_fol_resolution task6_chaining task7_medsafety) do (
  echo === running %%t ===
  python src\%%t.py > NUL
)
echo All tasks finished. See outputs\ and plots\.
