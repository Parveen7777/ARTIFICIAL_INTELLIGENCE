#!/usr/bin/env bash
# Runs every task and regenerates outputs/*.txt, outputs/*.json and plots/*.png
set -e
cd "$(dirname "$0")"
for t in task1_wumpus task2_resolution task3_exam_kb task4_unification task5_fol_resolution task6_chaining task7_medsafety; do
  echo "=== running $t ==="
  python3 src/$t.py > /dev/null
done
echo "All tasks finished. See outputs/ and plots/."
