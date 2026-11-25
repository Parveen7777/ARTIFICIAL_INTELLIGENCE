"""
TASK 6 / STAGE 5  -  Desk-check on a toy environment (3 states, 2 actions).

States   : S0 = Comfortable, S1 = Warm, S2 = Hot            (room occupied throughout)
Actions  : Off, Cool
Reward   : -(energy cost) - (comfort penalty of the NEW state);  energy: Off 0, Cool 2;  comfort: S0 0, S1 1, S2 4
alpha = 0.5, gamma = 0.9, Q initialised to 0.  The code re-uses the SAME td_update() as the full agent, so the hand
calculation in the report is checked against the production update rule.

Run:  python task6_rl_controller/task6_toy_trace.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from q_agent import td_update  # noqa: E402

STATES = ["S0 Comfortable", "S1 Warm", "S2 Hot"]
ACTIONS = ["Off", "Cool"]
# deterministic transition table: (state, action) -> (next_state, reward)
MODEL = {(0, 0): (1, -1), (0, 1): (0, -2),
         (1, 0): (2, -4), (1, 1): (0, -2),
         (2, 0): (2, -4), (2, 1): (1, -3)}
ALPHA, GAMMA = 0.5, 0.9
TRAJECTORY = [(2, 1), (1, 1), (0, 0), (1, 0), (2, 1), (1, 1)]          # (state, action) pairs, includes exploratory moves


def main():
    print("=" * 96)
    print("TASK 6 STAGE 5  TOY ENVIRONMENT  (alpha=%.1f, gamma=%.1f, Q initialised to 0)" % (ALPHA, GAMMA))
    print("Transition / reward table  (state, action) -> (next state, reward):")
    for (s, a), (s2, r) in MODEL.items():
        print("   (%-14s, %-4s) -> (%-14s, %+d)" % (STATES[s], ACTIONS[a], STATES[s2], r))
    print("=" * 96)
    Q = [[0.0, 0.0] for _ in range(3)]
    print("%-3s %-15s %-5s %-6s %-15s %-34s %-9s %-9s" % ("#", "state", "act", "reward", "next state", "TD target = r + gamma*max Q(s',.)", "old Q", "new Q"))
    for i, (s, a) in enumerate(TRAJECTORY, 1):
        s2, r = MODEL[(s, a)]
        old = Q[s][a]
        max_next = max(Q[s2])                                   # read BEFORE the update
        target = r + GAMMA * max_next
        new = td_update(Q, s, a, r, s2, None, ALPHA, GAMMA, "q")
        print("%-3d %-15s %-5s %-6d %-15s %-34s %-9.4f %-9.4f" % (i, STATES[s], ACTIONS[a], r, STATES[s2],
                                                                "%d + 0.9*%.4f = %.4f" % (r, max_next, target), old, new))
        print("      Q(%s,%s) <- %.4f + %.1f*(%.4f - %.4f) = %.4f" % (STATES[s][:2], ACTIONS[a], old, ALPHA, target, old, new))
    print("\nFinal Q-table after 6 updates (rows = states, columns = Off, Cool):")
    for s in range(3):
        print("   %-15s Off = %8.4f   Cool = %8.4f   greedy action = %s" % (STATES[s], Q[s][0], Q[s][1], ACTIONS[0 if Q[s][0] > Q[s][1] else 1]))
    print("\nObservation: (1) Q(S2,Off) = 0 has never been tried, so the greedy action in S2 is 'Off' only because of OPTIMISTIC")
    print("initialisation (all true values are negative) - this is what drives early exploration. (2) After 6 updates the values are still far")
    print("from converged; repeated sweeps (the training loop) are needed to push Q(S2,Off) down to its true, very negative value.")

if __name__ == "__main__":
    main()
