"""
TASK 6 / STAGE 6  -  Train, instrument and ATTACK the RL controller.

  A. 20 seeds of Q-learning (and SARSA) vs the fixed-rule thermostat on the same 200 evaluation days.
  B. Learning curves, a sample day, and the extracted policy.
  C. Adversarial test 1: price spike during lectures + weak comfort weight  -> reward-misspecification failure.
  D. Adversarial test 2: myopic agent (gamma = 0) -> never pre-cools; "switch off before occupants arrive" trap.

Run:  python task6_rl_controller/task6_experiments.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from hvac_env import HVACEnv, NOMINAL_PRICE, ACTIONS, BAND, run_day, N_STATES  # noqa: E402
from q_agent import train, greedy_chooser, thermostat_chooser, evaluate, episodes_to_convergence  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
EPISODES, SEEDS, EVAL_DAYS = 4000, 20, 200


def ms(x):
    return "%.2f +/- %.2f" % (np.mean(x), np.std(x))


def run_seeds(env, algo="q", gamma=0.95, episodes=EPISODES, seeds=SEEDS):
    out = []
    for seed in range(seeds):
        Q, visits, ret = train(env, episodes, seed=seed, algo=algo, gamma=gamma)
        out.append(dict(Q=Q, visits=visits, ret=ret, **evaluate(env, greedy_chooser(Q, visits), EVAL_DAYS)))
        out[-1]["conv"] = episodes_to_convergence(ret)
    return out


def table(title, rows):
    print("\n" + title)
    print("%-34s %-18s %-18s %-18s %-18s %-14s" % ("Policy", "reward/day", "energy kWh/day", "cost Rs/day", "comfort viol. steps", "pre-cool steps"))
    for name, d in rows:
        print("%-34s %-18s %-18s %-18s %-18s %-14s" % (name, d["reward"], d["kwh"], d["cost"], d["violations"], d["precool"]))


def summarise(res):
    f = lambda k: ms([r[k] for r in res])
    return dict(reward=f("reward"), kwh=f("kwh"), cost=f("cost"), violations=f("violations"), precool=f("precool_steps"))


def single(d):
    f = lambda k: "%.2f" % d[k]
    return dict(reward=f("reward"), kwh=f("kwh"), cost=f("cost"), violations=f("violations"), precool=f("precool_steps"))


def main():
    env = HVACEnv()
    print("=" * 100)
    print("TASK 6 STAGE 6  SETTINGS: %d training episodes/seed, %d seeds, %d fixed evaluation days, alpha=0.1, gamma=0.95, eps 1.0->0.05 over 60%% of training" % (EPISODES, SEEDS, EVAL_DAYS))
    print("State space = %d discrete states x 3 actions; comfort band %s C (enforced when occupied); price tiers from Rs/kWh by hour." % (N_STATES, BAND))
    print("=" * 100)

    # ---------------------------------------------------------------- A: main comparison
    base = evaluate(env, thermostat_chooser(), EVAL_DAYS)
    q_runs = run_seeds(env, "q")
    s_runs = run_seeds(env, "sarsa")
    table("A. NOMINAL ENVIRONMENT - held-out evaluation days (mean +/- std over 20 seeds; baseline is deterministic)",
          [("Fixed-rule thermostat", single(base)), ("Q-learning (gamma=0.95)", summarise(q_runs)), ("SARSA (gamma=0.95)", summarise(s_runs))])
    qr = np.array([r["reward"] for r in q_runs])
    print("\nQ-learning beats the thermostat in reward on %d / 20 seeds (baseline reward %.2f; worst seed %.2f, best seed %.2f)" % ((qr > base["reward"]).sum(), base["reward"], qr.min(), qr.max()))
    print("Q-learning has fewer comfort violations than the thermostat on %d / 20 seeds" % (np.array([r["violations"] for r in q_runs]) < base["violations"]).sum())
    sr = np.array([r["reward"] for r in s_runs])
    print("SARSA beats the thermostat in reward on %d / 20 seeds" % (sr > base["reward"]).sum())
    print("Episodes to convergence (moving-average return within 5%% of its final level): Q-learning %s,  SARSA %s" % (ms([r["conv"] for r in q_runs]), ms([r["conv"] for r in s_runs])))
    print("Reduction vs thermostat:  comfort violations %.1f%%   reward improvement %.1f%%   energy change %+.1f%%" % (
        100 * (1 - np.mean([r["violations"] for r in q_runs]) / base["violations"]),
        100 * (np.mean(qr) - base["reward"]) / abs(base["reward"]),
        100 * (np.mean([r["kwh"] for r in q_runs]) / base["kwh"] - 1)))
    print("Note: the agent may use MORE kWh than the thermostat - it spends energy at cheap off-peak prices (pre-cooling) to avoid discomfort")
    print("      and expensive peak-hour cooling; the objective is cost + comfort, not kWh alone.")

    # ---------------------------------------------------------------- B: learning curves + sample day + policy
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    w = 100
    for runs, label, col in ((q_runs, "Q-learning", "#1f77b4"), (s_runs, "SARSA", "#ff7f0e")):
        curves = np.array([np.convolve(r["ret"], np.ones(w) / w, mode="valid") for r in runs])
        xs = np.arange(w, w + curves.shape[1])
        ax[0].plot(xs, curves.mean(0), color=col, label=label)
        ax[0].fill_between(xs, curves.mean(0) - curves.std(0), curves.mean(0) + curves.std(0), color=col, alpha=.2)
    ax[0].axhline(base["reward"], color="r", ls="--", label="thermostat (evaluation)")
    ax[0].set_xlabel("training episode (day)"); ax[0].set_ylabel("reward per day (100-episode moving average)")
    ax[0].set_title("Learning curves, mean +/- std over 20 seeds"); ax[0].legend(); ax[0].grid(alpha=.3)
    labels = ["thermostat", "Q-learning", "SARSA"]
    viol = [base["violations"], np.mean([r["violations"] for r in q_runs]), np.mean([r["violations"] for r in s_runs])]
    cost = [base["cost"], np.mean([r["cost"] for r in q_runs]), np.mean([r["cost"] for r in s_runs])]
    xb = np.arange(3)
    ax[1].bar(xb - .2, viol, .4, color="#d62728", label="comfort-violation steps/day")
    ax2 = ax[1].twinx(); ax2.bar(xb + .2, cost, .4, color="#2ca02c", label="energy cost Rs/day")
    ax[1].set_xticks(xb); ax[1].set_xticklabels(labels); ax[1].set_ylabel("violation steps/day"); ax2.set_ylabel("Rs/day")
    ax[1].set_title("Comfort and cost on evaluation days"); ax[1].legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "task6_learning_curves.png"), dpi=160); plt.close(fig)

    # one hot day, side by side
    best = q_runs[0]
    chooser = greedy_chooser(best["Q"], best["visits"])
    day_seed = None
    for cand in range(10_000_000, 10_000_050):
        e = HVACEnv(); e.reset(cand)
        if e.t_out[20] > 36.5:
            day_seed = cand; break
    day_seed = day_seed or 10_000_000
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    hours = 8 + np.arange(41) * 0.25
    for name, ch, col in (("Thermostat", thermostat_chooser(), "#d62728"), ("Q-learning agent (seed 0)", chooser, "#1f77b4")):
        e = HVACEnv(); s = e.reset(day_seed); temps, acts, done = [e.t_in], [], False
        while not done:
            a = ch(e, s); acts.append(a); s, _, done, _ = e.step(a); temps.append(e.t_in)
        axes[0].plot(hours, temps, color=col, label=name)
        axes[1].step(hours[:-1], np.array(acts) + (0.06 if col == "#1f77b4" else -0.06), where="post", color=col, label=name)
    e = HVACEnv(); e.reset(day_seed)
    for h in range(10):
        if e.occ[h] > 0:
            for a_ in axes:
                a_.axvspan(8 + h, 9 + h, color="gray", alpha=.10 * e.occ[h] + .02)
    axes[0].axhspan(*BAND, color="green", alpha=.12, label="comfort band")
    axes[0].set_ylabel("indoor temperature (C)"); axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
    axes[0].set_title("One hot evaluation day (shaded = occupied, darker = fuller)")
    axes[1].set_yticks([0, 1, 2]); axes[1].set_yticklabels(ACTIONS); axes[1].set_xlabel("hour of day"); axes[1].set_ylabel("action"); axes[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "task6_sample_day.png"), dpi=160); plt.close(fig)

    print("\nLEARNED POLICY (seed 0) at 08:00-08:45, room EMPTY, first class at 09:00 (expected pre-cooling), by indoor-temperature bin / outdoor bin:")
    from hvac_env import N_OCC, N_TIN, N_TOUT, N_PRICE, price_tier
    pt = price_tier(NOMINAL_PRICE[0])
    for outb in (0, 1):
        row = []
        for tb in range(5):
            s = (((0 * N_OCC + 0) * N_TIN + tb) * N_TOUT + outb) * N_PRICE + pt
            row.append(ACTIONS[int(np.argmax(best["Q"][s]))] if best["visits"][s] > 0 else "(unvisited)")
        print("   outdoor %-10s indoor <22:%-6s 22-24:%-6s 24-26:%-6s 26-28:%-6s >28:%-6s" % (("<33C" if outb == 0 else ">=33C",) + tuple(row)))

    # ---------------------------------------------------------------- C: adversarial price spike
    print("\n" + "=" * 100)
    print("C. ADVERSARIAL TEST 1 - DEMAND-RESPONSE PRICE SPIKE DURING PEAK OCCUPANCY (Rs 45/kWh from 11:00 to 13:00) WITH THE ORIGINAL WEAK COMFORT WEIGHT")
    print("=" * 100)
    print("PREDICTION (written before running): with lambda = 8 Rs per degC per occupant-level, two hours of High cooling at Rs 45/kWh")
    print("costs Rs 45 per step, which equals the penalty of only ~2.8 degC of violation at full occupancy. A reward-maximising agent will therefore")
    print("SWITCH OFF during the spike. Locally the reward improves (cheap steps), globally comfort collapses: comfort-violation steps of the spike-trained")
    print("agent should be at least TWICE those of the nominal agent, even though its energy cost is lower. The failing assumption is the reward weight")
    print("(comfort is priced, not protected), not the learning algorithm.")
    spike_price = list(NOMINAL_PRICE); spike_price[3] = spike_price[4] = 45
    env_spike = HVACEnv(price_by_hour=spike_price)
    base_spike = evaluate(env_spike, thermostat_chooser(), EVAL_DAYS)
    spike_runs = run_seeds(env_spike, "q")
    nominal_in_spike = [evaluate(env_spike, greedy_chooser(r["Q"], r["visits"]), EVAL_DAYS) for r in q_runs]
    repaired_env = HVACEnv(price_by_hour=spike_price, lam=60.0)
    repair_runs = run_seeds(repaired_env, "q")
    repair_eval = [evaluate(env_spike, greedy_chooser(r["Q"], r["visits"]), EVAL_DAYS) for r in repair_runs]     # judged on TRUE comfort metrics in the spike world
    table("Evaluation in the spike world (same 200 days; reward column uses the ORIGINAL lambda = 8 so rows are comparable)",
          [("Thermostat", single(base_spike)),
           ("Nominal-trained agent (never saw spike)", summarise(nominal_in_spike)),
           ("Spike-trained agent, lambda = 8 (FAILS)", summarise(spike_runs)),
           ("Spike-trained agent, lambda = 60 (repair)", summarise(repair_eval))])
    v_nom, v_spk = np.mean([r["violations"] for r in q_runs]), np.mean([r["violations"] for r in spike_runs])
    v_rep = np.mean([r["violations"] for r in repair_eval])
    print("\nRESULT: spike-trained agent violations/day = %.2f vs nominal agent (nominal world) = %.2f  -> ratio %.2fx  (prediction %s)" % (
        v_spk, v_nom, v_spk / v_nom, "CONFIRMED" if v_spk >= 2 * v_nom else "NOT CONFIRMED"))
    print("        its cost Rs/day = %.1f (thermostat %.1f); violations of the repaired lambda=60 agent = %.2f" % (np.mean([r["cost"] for r in spike_runs]), base_spike["cost"], v_rep))
    print("DIAGNOSIS: the learner did exactly what the reward asked. The comfort term is a soft price that a spike can outbid; repair = raise lambda")
    print("(or add a hard constraint / safety layer that overrides 'Off' when an occupied room leaves the band).")

    # ---------------------------------------------------------------- D: myopic agent
    print("\n" + "=" * 100)
    print("D. ADVERSARIAL TEST 2 - MYOPIC AGENT (gamma = 0): 'switch off before the occupants arrive'")
    print("=" * 100)
    print("PREDICTION (written before running): with gamma = 0 the empty-room steps at 08:00-08:45 have no immediate comfort penalty, only an energy cost,")
    print("so the agent will choose Off there (pre-cool steps ~ 0) and occupants will walk into a hot room: violations should be clearly higher than for")
    print("gamma = 0.95. Missing assumption: the discount factor must be large enough for the agent to see the delayed discomfort caused by a warm start.")
    rows, myop = [], {}
    for g in (0.0, 0.5, 0.95):
        runs = q_runs if g == 0.95 else run_seeds(env, "q", gamma=g)
        myop[g] = runs
        rows.append(("Q-learning gamma = %.2f" % g, summarise(runs)))
    table("Nominal world, evaluation on 200 held-out days (mean +/- std over 20 seeds)", rows)
    v0, v95 = np.mean([r["violations"] for r in myop[0.0]]), np.mean([r["violations"] for r in myop[0.95]])
    p0, p95 = np.mean([r["precool_steps"] for r in myop[0.0]]), np.mean([r["precool_steps"] for r in myop[0.95]])
    print("\nRESULT: pre-cool steps/day gamma=0: %.2f vs gamma=0.95: %.2f ; violations gamma=0: %.2f vs gamma=0.95: %.2f -> prediction %s" % (
        p0, p95, v0, v95, "CONFIRMED" if (p0 < p95 and v0 > v95) else "NOT CONFIRMED"))
    print("DIAGNOSIS: failing assumption = horizon. The Markov state is fine; the TD target ignores the future (gamma = 0), so the value of cooling an")
    print("empty room (an investment whose pay-off arrives 15-60 minutes later) is never learned.")

    # figure for attacks
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    names = ["thermostat", "nominal\nagent", "spike-trained\nlambda=8", "spike-trained\nlambda=60"]
    vals = [base_spike["violations"], np.mean([r["violations"] for r in nominal_in_spike]), v_spk, v_rep]
    ax[0].bar(names, vals, color=["#7f7f7f", "#1f77b4", "#d62728", "#2ca02c"])
    ax[0].set_ylabel("comfort-violation steps / day"); ax[0].set_title("Attack 1: price spike (Rs 45/kWh, 11:00-13:00)"); ax[0].grid(alpha=.3, axis="y")
    gs = [0.0, 0.5, 0.95]
    ax[1].bar(["gamma=%.2f" % g for g in gs], [np.mean([r["violations"] for r in myop[g]]) for g in gs], color=["#d62728", "#ff7f0e", "#1f77b4"])
    for i, g in enumerate(gs):
        ax[1].text(i, np.mean([r["violations"] for r in myop[g]]) + .15, "pre-cool %.1f" % np.mean([r["precool_steps"] for r in myop[g]]), ha="center", fontsize=8)
    ax[1].set_ylabel("comfort-violation steps / day"); ax[1].set_title("Attack 2: myopic discount factor"); ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "task6_attacks.png"), dpi=160); plt.close(fig)
    print("\nSaved figures: task6_learning_curves.png, task6_sample_day.png, task6_attacks.png")


if __name__ == "__main__":
    main()
