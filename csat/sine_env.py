"""sine_env.py -- 1D sinusoidal single-optimum sandbox.

ONE weight w in [0,1]. A swappable loss landscape (default: a single full-period
sine, phase-shifted per case so its unique minimum sits inside (0,1), off-centre
from the 0.5 start). The model searches for the w that MINIMISES the loss ==
MAXIMISES the margin. Same single-margin contract as ParabolaEnv, so it reuses
the parabola prompt/render dispatch -- which is the point: a genuinely different
task wearing the same SET/SUBMIT interface, to test whether the steering
direction is task-agnostic.

    loss(w) = sin(2*pi*w + phase)       in [-1, 1]; one minimum (=-1) per period
    q(w)    = loss(w) + 1               in [0, 2]; 0 at the optimum (lower=better)
    margin  = z_pass - q(w)             >0 = PASS; max (= z_pass) at the optimum

phase is set per case so the loss minimum lands at a sampled w_opt in (0,1),
>= opt_gap away from 0.5 (reachable, non-trivial climb). The start (w=0.5) is
below threshold (fails), exactly like ParabolaEnv, so "pass then keep optimising"
is the same regime. Swap `loss_1d` (e.g. add a 2nd harmonic) to make the search
harder without touching the env.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def loss_1d(w, phase, **kw):
    """Default landscape: one full sine period over w in [0,1]. Range [-1,1],
    unique minimum -1. To add local minima (if a single sine is too easy), e.g.:
        return np.sin(2*np.pi*w + phase) + 0.3*np.sin(6*np.pi*w + kw.get('p2',0))
    (keep the global min reachable and inside (0,1))."""
    return float(np.sin(2.0 * np.pi * float(w) + phase))


@dataclass
class Sine1DEnv:
    n_obj: int = 1                 # one weight; kept as the interface field
    z_pass_frac: float = 0.4       # pass threshold as a fraction of q at the 0.5 start
    grid: float = 0.0
    opt_lo: float = 0.15           # keep w_opt inside [opt_lo, 1-opt_lo]
    opt_gap: float = 0.12          # min |w_opt - 0.5| (non-trivial climb)
    # filled by reset():
    phase: float = None
    w_opt: float = None
    z_pass: float = None
    q_start: float = None

    def __post_init__(self):
        self.n_obj = 1                                   # force 1D
        self.w = np.full(1, 0.5)
        self.priority = 0                                # interface compatibility
        self.submitted = False
        if self.phase is None:
            self.phase = (-np.pi / 2 - 2 * np.pi * 0.7) % (2 * np.pi)

    def _q(self, w):
        return loss_1d(w, self.phase) + 1.0              # 0 at optimum

    def reset(self, seed=None, wide=True, w_init=0.5, **kw):
        rng = np.random.default_rng(seed)
        span = 0.5 - self.opt_lo
        mag  = rng.uniform(self.opt_gap, max(self.opt_gap, span))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        self.w_opt = float(np.clip(0.5 + sign*mag, self.opt_lo, 1.0 - self.opt_lo))   # interior optimum
        self.phase = float((-np.pi/2 - 2*np.pi*self.w_opt) % (2*np.pi))               # phase derived from it
        self.w = np.full(1, float(w_init))
        if self.grid:
            self.w = np.round(self.w / self.grid) * self.grid
        self.q_start = self._q(0.5)
        self.z_pass = self.z_pass_frac * self.q_start
        self.submitted = False
        return self.feedback()

    def margins(self, w=None):
        if w is None:
            w = self.w[0]
        else:
            w = float(np.clip(np.asarray(w, float).reshape(-1)[0], 0.0, 1.0))
        return np.full(1, self.z_pass - self._q(w))

    def set_weight(self, i, value):
        v = float(np.clip(value, 0.0, 1.0))
        if self.grid:
            v = float(np.clip(round(v / self.grid) * self.grid, 0.0, 1.0))
        self.w[i] = v

    def all_pass(self, w=None):
        return bool(self.margins(w)[0] >= 0)

    def feedback(self):
        m = float(self.margins()[0])
        return [dict(obj=i, weight=round(float(self.w[i]), 6),
                    margin=round(m, 6), ok=bool(m >= 0))
                for i in range(self.n_obj)]

    def submit(self):
        self.submitted = True
        return dict(submitted=True, plan=self.snapshot())

    def snapshot(self):
        m = float(self.margins()[0])
        return dict(weights=self.w.copy(), margins=self.margins().copy(),
                    all_pass=bool(m >= 0), total_margin=m,
                    total_weight=float(self.w.sum()), priority=0, margin_priority=m)

    def optimum(self, **kw):
        # known exactly: margin at w_opt is z_pass - 0 = z_pass
        return dict(feasible=True, priority=0, weights=[round(self.w_opt, 4)],
                    margins=[float(self.z_pass)], margin_priority=float(self.z_pass))

    def case_dict(self):
        return dict(n_obj=1, env_kind="sine", z_pass=self.z_pass,
                    z_start=self.q_start, w_opt=self.w_opt, phase=float(self.phase),
                    z_pass_frac=self.z_pass_frac)