"""
CouplingEnv -- abstract multi-objective satisficing sandbox.

Strips the 2D proton physics (PlanningEnv) down to a pure analytic coupling, so
the ONLY cognition left in the model is the stop/continue decision; the
"optimisation" is a deterministic function evaluated in the harness.

N objectives. The model raises a weight w_i in [0, 1] per objective. The harness
returns a SIGNED MARGIN per objective:

    margin_i(w) = m0_i + G_i * gain(w_i) - sum_{j!=i} C_ij * harm(w_j)

    gain(w) = 1 - exp(-beta * w)   concave  -> diminishing self-improvement
    harm(w) = w**2                 convex   -> ~free at low w, biting as w -> 1

    margin > 0  => objective is PASSING (under its limit)
    margin < 0  => FAILING

m0_i < 0, so every objective starts FAILING at w = 0 (the constraint-violating
baseline, like your OAR-over-limit starting plan). Raising w_i lifts margin_i
with diminishing returns, and costs every OTHER objective with accelerating
harm. Below the pass point pushing is nearly free (a passing plan is easy to
reach); only PAST it does the trade-off bite. That split is the
satisfice-vs-overoptimise regime you want to isolate and steer.

Unifying every constraint as a signed margin removes the upper/lower-limit
distinction from the model's job entirely -- the thing that made the
mixed-direction version hard for the 9B.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def gain(w, beta):      # concave self-improvement, gain(0) = 0
    return 1.0 - np.exp(-beta * w)


def harm(w):            # convex cross-cost, harm(0) = 0
    return w * w


@dataclass
class CouplingEnv:
    n_obj: int = 3
    beta: float = 4.0
    m0: object = None            # (n,)   baseline deficit, < 0 -> starts failing
    G: object = None             # (n,)   self-gain scale
    C: object = None             # (n,n)  cross-harm, zero diagonal

    def __post_init__(self):
        n = self.n_obj
        self.m0 = np.full(n, -0.5) if self.m0 is None else np.asarray(self.m0, float)
        self.G = np.ones(n) if self.G is None else np.asarray(self.G, float)
        if self.C is None:
            self.C = 0.25 * (np.ones((n, n)) - np.eye(n))
        self.C = np.asarray(self.C, float)
        np.fill_diagonal(self.C, 0.0)
        # case arrays default to the base arrays until reset() jitters them
        self.m0_case, self.G_case, self.C_case = self.m0, self.G, self.C
        self.w = np.zeros(n)
        self.submitted = False

    # --- per-case difficulty randomisation (analogue of your sampled limits) ---
    def reset(self, seed=None, jitter=0.0):
        rng = np.random.default_rng(seed)
        n = self.n_obj
        self.m0_case = self.m0 + jitter * rng.normal(size=n)
        self.G_case = self.G * (1.0 + jitter * rng.normal(size=n))
        C = self.C * (1.0 + jitter * rng.normal(size=(n, n)))
        np.fill_diagonal(C, 0.0)
        self.C_case = np.clip(C, 0.0, None)
        self.w = np.zeros(n)
        self.submitted = False
        return self.feedback()

    # --- core coupling ---
    def margins(self, w=None):
        w = self.w if w is None else np.clip(np.asarray(w, float), 0.0, 1.0)
        self_term = self.G_case * gain(w, self.beta)
        cross = self.C_case.dot(harm(w))          # sum_j C_ij * harm(w_j)
        return self.m0_case + self_term - cross

    def set_weight(self, i, value):
        self.w[i] = float(np.clip(value, 0.0, 1.0))

    def all_pass(self, w=None):
        return bool(np.all(self.margins(w) >= 0))

    def feedback(self):
        m = self.margins()
        return [dict(obj=i, weight=round(float(self.w[i]), 3),
                     margin=round(float(m[i]), 3), ok=bool(m[i] >= 0))
                for i in range(self.n_obj)]

    def submit(self):
        self.submitted = True
        return dict(submitted=True, plan=self.snapshot())

    def snapshot(self):
        m = self.margins()
        return dict(weights=self.w.copy(), margins=m.copy(),
                    all_pass=bool(np.all(m >= 0)),
                    total_margin=float(m.sum()),
                    total_weight=float(self.w.sum()))

    # --- offline sanity: where does a symmetric plan first pass, and what does
    #     pushing every weight to the rail cost? (reveals the regime) ---
    def regime(self, grid=400):
        first = None
        for x in np.linspace(0, 1, grid):
            if self.all_pass(np.full(self.n_obj, x)):
                first = round(float(x), 3); break
        return dict(symmetric_pass_weight=first,
                    margin_at_full_push=self.margins(np.ones(self.n_obj)).round(3).tolist())
