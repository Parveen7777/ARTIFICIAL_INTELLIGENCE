"""
q_agent.py  -  Tabular temporal-difference control (Q-learning and SARSA) with epsilon-greedy exploration.

    Q-learning :  Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
    SARSA      :  Q(s,a) <- Q(s,a) + alpha * [ r + gamma * Q(s',a')        - Q(s,a) ]     (a' = the action actually taken next)

Terminal transitions (end of the teaching day) bootstrap from 0.
Algorithm logic is separate from the environment and from any printing / plotting.
"""
import random
import numpy as np
from hvac_env import HVACEnv, N_STATES, thermostat_action, run_day

N_ACTIONS = 3


def td_update(Q, s, a, r, s_next, a_next, alpha, gamma, algo="q", done=False):
    """One TD update; returns the new value Q(s,a). Shared by the toy desk-check and the full agent."""
    if done:
        target = r
    elif algo == "q":
        target = r + gamma * max(Q[s_next])
    else:
        target = r + gamma * Q[s_next][a_next]
    Q[s][a] += alpha * (target - Q[s][a])
    return Q[s][a]


def epsilon_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return rng.randrange(N_ACTIONS)
    q = Q[s]
    best = max(q)
    return rng.choice([a for a in range(N_ACTIONS) if q[a] == best])          # random tie-break


def train(env, episodes, seed, algo="q", alpha=0.1, gamma=0.95, eps_start=1.0, eps_end=0.05, eps_decay_frac=0.6, q_init=0.0):
    """Train for `episodes` days. Returns Q-table, visit counts and the per-episode training return."""
    rng = random.Random(seed)
    Q = [[q_init] * N_ACTIONS for _ in range(N_STATES)]
    visits = np.zeros(N_STATES, dtype=int)
    returns = []
    decay_eps = max(1, int(episodes * eps_decay_frac))
    for ep in range(episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / decay_eps)
        s = env.reset(day_seed=seed * 1_000_003 + ep)
        a = epsilon_greedy(Q, s, eps, rng)
        total, done = 0.0, False
        while not done:
            s_next, r, done, _ = env.step(a)
            visits[s] += 1
            if done:
                td_update(Q, s, a, r, None, None, alpha, gamma, algo, done=True)
            else:
                a_next = epsilon_greedy(Q, s_next, eps, rng)
                td_update(Q, s, a, r, s_next, a_next, alpha, gamma, algo)
                s, a = s_next, a_next
            total += r
        returns.append(total)
    return Q, visits, np.array(returns)


def greedy_chooser(Q, visits):
    """Greedy policy. Unvisited states fall back to the thermostat rule (documented safe default)."""
    def choose(env, s):
        if visits[s] == 0:
            return thermostat_action(env.observe())
        q = Q[s]
        return int(np.argmax(q))
    return choose


def thermostat_chooser():
    return lambda env, s: thermostat_action(env.observe())


def evaluate(env, chooser, n_days=200, seed0=10_000_000):
    """Average per-day metrics of a policy on a FIXED set of evaluation days (same days for every policy)."""
    rows = [run_day(env, seed0 + i, chooser) for i in range(n_days)]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def episodes_to_convergence(returns, window=100, tol=0.05, tail=200):
    """First episode at which the moving average of the training return is within `tol` (relative) of its final level."""
    returns = np.asarray(returns, float)
    final = returns[-tail:].mean()
    ma = np.convolve(returns, np.ones(window) / window, mode="valid")
    ok = np.where(ma >= final - tol * abs(final))[0]
    return int(ok[0] + window) if len(ok) else len(returns)
