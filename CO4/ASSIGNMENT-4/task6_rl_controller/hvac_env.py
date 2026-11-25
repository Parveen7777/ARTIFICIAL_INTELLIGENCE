"""
hvac_env.py  -  Simulator for the energy-aware lecture-hall air-conditioning controller.

Episode      : one teaching day, 08:00 -> 18:00, 40 decisions of 15 minutes (starts with the room warm after a night off).
Actions      : 0 = Off, 1 = Low, 2 = High
Observation  : (hour block, occupancy level, indoor-temperature bin, outdoor bin, price tier)  ->  one integer state
Reward       : - price * kWh  -  LAMBDA * occupancy * comfort_violation(deg C beyond the 22-26 C band)

Physical model (all assumptions are explicit constants):
    T' = T + K_LEAK*(T_out - T) + HEAT_PER_PERSON*occupancy - COOLING[a] + N(0, 0.1^2)
"""
import math
import random

STEPS_PER_DAY = 40                 # 10 hours x 4 quarter-hours
DT_H = 0.25                        # hours per step
BAND = (22.0, 26.0)                # comfort band, deg C (enforced only when the room is occupied)
POWER_KW = (0.0, 1.6, 4.0)         # Off / Low / High electrical power
COOLING_C = (0.0, 0.8, 1.6)        # cooling effect per step, deg C
K_LEAK = 0.10                      # heat leak from outdoors per step
HEAT_PER_PERSON = 0.12             # deg C per occupancy level (0,1,2) per step
LAMBDA = 8.0                       # Rs per (deg C violation x occupancy level) per step
ACTIONS = ("Off", "Low", "High")

NOMINAL_PRICE = (5, 5, 8, 8, 8, 8, 12, 12, 12, 8)      # Rs/kWh for hour blocks 08..17
MEAN_OCC = (0.2, 1.6, 1.8, 1.8, 1.5, 0.4, 1.6, 1.7, 1.4, 0.3)     # typical occupancy by hour block

N_HOUR, N_OCC, N_TIN, N_TOUT, N_PRICE = 10, 3, 5, 2, 4
N_STATES = N_HOUR * N_OCC * N_TIN * N_TOUT * N_PRICE


def tin_bin(t):
    return 0 if t < 22 else 1 if t < 24 else 2 if t < 26 else 3 if t < 28 else 4


def price_tier(p):
    return 0 if p < 6.5 else 1 if p < 10 else 2 if p < 20 else 3


class HVACEnv:
    def __init__(self, price_by_hour=NOMINAL_PRICE, lam=LAMBDA):
        self.price_by_hour, self.lam = tuple(price_by_hour), lam

    # ------------------------------------------------------------------ episode start
    def reset(self, day_seed):
        self.rng = random.Random(day_seed)
        base = self.rng.choice((30.0, 33.0, 36.0))                              # mild / warm / hot day
        self.t_out = [base - 2.0 + 5.0 * math.sin(math.pi * (t / 4.0) / 10.0) for t in range(STEPS_PER_DAY + 1)]
        self.occ = [min(2, max(0, int(round(MEAN_OCC[h] + self.rng.gauss(0, 0.6))))) for h in range(10)]
        self.t = 0
        self.t_in = self.t_out[0] - 1.5 + self.rng.gauss(0, 0.5)
        return self._state()

    def _state(self):
        h = min(self.t // 4, 9)
        occ = self.occ[h]
        price = self.price_by_hour[h]
        outb = 0 if self.t_out[self.t] < 33.0 else 1
        s = (((h * N_OCC + occ) * N_TIN + tin_bin(self.t_in)) * N_TOUT + outb) * N_PRICE + price_tier(price)
        return s

    def observe(self):
        """Raw (human-readable) observation, used by the baseline thermostat."""
        h = min(self.t // 4, 9)
        return dict(hour=h, occ=self.occ[h], t_in=self.t_in, t_out=self.t_out[self.t], price=self.price_by_hour[h])

    # ------------------------------------------------------------------ one decision
    def step(self, action):
        h = min(self.t // 4, 9)
        occ, price = self.occ[h], self.price_by_hour[h]
        kwh = POWER_KW[action] * DT_H
        t_next = (self.t_in + K_LEAK * (self.t_out[self.t] - self.t_in) + HEAT_PER_PERSON * occ
                  - COOLING_C[action] + self.rng.gauss(0, 0.1))
        violation = max(0.0, t_next - BAND[1]) + max(0.0, BAND[0] - t_next)
        reward = -price * kwh - self.lam * occ * violation
        self.t_in = t_next
        self.t += 1
        done = self.t >= STEPS_PER_DAY
        info = dict(kwh=kwh, cost=price * kwh, violation_steps=int(occ > 0 and violation > 0), violation_deg=occ * violation)
        return (None if done else self._state()), reward, done, info


def thermostat_action(obs):
    """Fixed-rule baseline: purely reactive thermostat, no scheduling and no price awareness."""
    if obs["occ"] == 0:
        return 0
    if obs["t_in"] > 26.0:
        return 2
    if obs["t_in"] > 24.5:
        return 1
    return 0


def run_day(env, day_seed, choose):
    """Run one day with policy `choose(env, state) -> action`; return per-day metrics."""
    s = env.reset(day_seed)
    tot = dict(reward=0.0, kwh=0.0, cost=0.0, violations=0, violation_deg=0.0, precool_steps=0)
    done = False
    while not done:
        a = choose(env, s)
        if env.t // 4 == 0 and env.occ[0] == 0 and a != 0:
            tot["precool_steps"] += 1                                  # cooling before the first class starts
        s, r, done, info = env.step(a)
        tot["reward"] += r
        tot["kwh"] += info["kwh"]
        tot["cost"] += info["cost"]
        tot["violations"] += info["violation_steps"]
        tot["violation_deg"] += info["violation_deg"]
    return tot
