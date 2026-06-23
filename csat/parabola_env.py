"""
ParabolaEnv -- convex single-basin sandbox. Strips the experiment to its core:
a fixed, fully-known bowl with ONE global optimum every rollout can reach from
the 0.5/0.5 start. No case difficulty, no luck, no local minima. Effort = how
far the model pushes toward the known optimum; the flat floor (a small) is where
effort can vary at near-matched final quality.

    z(w) = a * r2 + b * r2**2          r2 = sum_i (w_i - c_i)**2
    margin = z_pass - z                (positive = passing; higher = better)

Steep quartic walls (b) standardise every rollout to the floor fast; the floor
tilt is set by a (a=0 -> near-flat floor, submitting early is ~rational;
a>0 -> faint central gradient, pushing always repays a little). Optimum c is
sampled once per RUN inside [0.2,0.8]^n (reachable, off-centre, not at 0.5),
fixed across all repeats so only sampling varies.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ParabolaEnv:
    n_obj: int = 2
    a: float = 0.15           # floor tilt: 0 = flat floor, >0 = gentle central gradient
    b: float = 1.0            # wall steepness (quartic)
    z_pass_frac: float = 0.4  # pass threshold as a fraction of z at the 0.5 start
    grid: float = 0.0
    # filled by reset():
    c: object = None          # optimum coordinates (n,)
    z_pass: float = None
    z_start: float = None

    def __post_init__(self):
        n = self.n_obj
        self.w = np.full(n, 0.5)
        self.priority = 0     # kept for interface compatibility; single margin here
        self.submitted = False
        if self.c is None:
            self.c = np.full(n, 0.7)   # placeholder until reset

    def _z(self, w):
        r2 = float(np.sum((np.asarray(w, float) - self.c) ** 2))
        return self.a * r2 + self.b * r2 * r2

    def reset(self, seed=None, wide=True, w_init=0.5, **kw):
        rng = np.random.default_rng(seed)
        n = self.n_obj
        # optimum sampled inside [0.2,0.8]^n: reachable from 0.5, off-centre, not at a special point
        self.c = rng.uniform(0.2, 0.8, size=n)
        self.w = np.full(n, float(w_init))
        if self.grid:
            self.w = np.round(self.w / self.grid) * self.grid
        # pass threshold defined relative to the starting loss, so the model
        # passes early (out on the wall) and the floor descent is the real task
        self.z_start = self._z(np.full(n, float(w_init)))
        self.z_pass = self.z_pass_frac * self.z_start
        self.submitted = False
        return self.feedback()

    def margins(self, w=None):
        w = self.w if w is None else np.clip(np.asarray(w, float), 0.0, 1.0)
        # single scalar margin, returned as an (n,)-broadcast so feedback/all_pass
        # keep the same shape contract as CouplingEnv
        m = self.z_pass - self._z(w)
        return np.full(self.n_obj, m)

    def set_weight(self, i, value):
        v = float(np.clip(value, 0.0, 1.0))
        if self.grid:
            v = float(np.clip(round(v / self.grid) * self.grid, 0.0, 1.0))
        self.w[i] = v

    def all_pass(self, w=None):
        return bool(self.margins(w)[0] >= 0)

    def feedback(self):
        m = float(self.margins()[0])
        # one row per coordinate (weights are the coords); margin is shared/global
        return [dict(obj=i, weight=round(float(self.w[i]), 3),
                     margin=round(m, 3), ok=bool(m >= 0))
                for i in range(self.n_obj)]

    def submit(self):
        self.submitted = True
        return dict(submitted=True, plan=self.snapshot())

    def snapshot(self):
        m = float(self.margins()[0])
        return dict(weights=self.w.copy(),
                    margins=self.margins().copy(),
                    all_pass=bool(m >= 0),
                    total_margin=m,
                    total_weight=float(self.w.sum()),
                    priority=int(self.priority),
                    margin_priority=m)

    def optimum(self, **kw):
        # KNOWN exactly: the bowl's centre. margin there is z_pass - 0 = z_pass.
        return dict(feasible=True, priority=int(self.priority),
                    weights=self.c.round(4).tolist(),
                    margins=[float(self.z_pass)] * self.n_obj,
                    margin_priority=float(self.z_pass))
    def case_dict(self):
        return dict(n_obj=self.n_obj, env_kind="parabola",
                    a=self.a, b=self.b, z_pass=self.z_pass, z_start=self.z_start,
                    c=self.c.tolist(), priority=int(self.priority))