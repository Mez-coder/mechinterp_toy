"""Pencil-weight optimisation and global beam weighting.

Two modes (env config `opt_mode`):

  SFO (default, single-field uniform dose)
    Each beam is optimised INDEPENDENTLY so that, on its own, it paints uniform
    dose (= Rx = 1.0) across the target. Because every beam is individually
    uniform on the target, any simplex combination  sum_i g_i * dose_i  (with
    sum g_i = 1) is ALSO uniform there -- so the model's global beam weights g_i
    are a genuine extra degree of freedom that trade OAR dose (purely geometric)
    without breaking target coverage.

  MFO (multi-field optimisation)
    All pencils from all beams are optimised JOINTLY to a uniform target. Global
    weights are then applied on top and the plan renormalised; note this
    perturbs the coverage the joint solve achieved (kept only as an option).

The inner solver is projected-gradient descent (non-negativity) on the masked
MSE -- the "internal SGD" of the spec. No OAR term ever enters the loss: OAR
sparing is achieved only by which beams/weights the model chooses.
"""
from __future__ import annotations
import numpy as np


def _lipschitz(A, iters=30):
    """Largest eigenvalue of A^T A via power iteration (for the GD step size)."""
    v = np.random.default_rng(0).standard_normal(A.shape[1])
    v /= np.linalg.norm(v) + 1e-12
    for _ in range(iters):
        w = A.T @ (A @ v)
        n = np.linalg.norm(w)
        if n < 1e-30:
            return 1.0
        v = w / n
    return float(n)


def solve_uniform(D_target, target_val=1.0, iters=400, w0=None):
    """min_{w>=0} || D_target w - target_val ||^2  by projected GD.

    D_target : (n_target_vox, n_rays) dense or sparse influence on target voxels.
    Returns non-negative pencil weights w.
    """
    A = D_target
    m, n = A.shape
    if n == 0:
        return np.zeros(0)
    b = np.full(m, float(target_val))
    w = np.ones(n) if w0 is None else np.clip(w0.astype(float), 0, None)
    L = _lipschitz(np.asarray(A.todense()) if hasattr(A, "todense") else A)
    lr = 1.0 / max(L, 1e-9)
    v = w.copy(); mom = 0.9
    upd = np.zeros(n)
    for _ in range(iters):
        r = A @ w - b
        g = 2.0 * (A.T @ r)
        upd = mom * upd - lr * g
        w = np.clip(w + upd, 0.0, None)
    return w


class PlanOptimiser:
    def __init__(self, density, opt_mode="sfo", inner_iters=400, Rx=1.0):
        self.den = density
        self.opt_mode = opt_mode
        self.inner_iters = inner_iters
        self.Rx = Rx

    def _target_rows(self, D, target_flat):
        return D[target_flat, :]

    def solve(self, beams, target_flat):
        """beams: list of dicts {angle, D (csc n_vox x n_rays), meta}.
        Returns per-beam dose vectors (each uniform-normalised on target) and the
        raw pencil weights. Global g_i weighting is applied separately.
        """
        if self.opt_mode == "mfo":
            return self._solve_mfo(beams, target_flat)
        return self._solve_sfo(beams, target_flat)

    def _solve_sfo(self, beams, target_flat):
        per_beam_dose = []
        per_beam_w = []
        for bm in beams:
            D = bm["D"]
            if D.shape[1] == 0:
                per_beam_dose.append(np.zeros(self.den.n_voxels))
                per_beam_w.append(np.zeros(0)); continue
            At = np.asarray(self._target_rows(D, target_flat).todense())
            w = solve_uniform(At, self.Rx, self.inner_iters)
            dose = np.asarray(D @ w).ravel()
            # normalise so this beam's MEAN target dose == Rx (true SFUD field)
            mt = dose[target_flat].mean()
            if mt > 1e-9:
                scale = self.Rx / mt
                dose *= scale; w = w * scale
            per_beam_dose.append(dose); per_beam_w.append(w)
        return per_beam_dose, per_beam_w

    def _solve_mfo(self, beams, target_flat):
        import scipy.sparse as sp
        mats = [bm["D"] for bm in beams if bm["D"].shape[1] > 0]
        if not mats:
            return [np.zeros(self.den.n_voxels) for _ in beams], [np.zeros(0) for _ in beams]
        sizes = [m.shape[1] for m in mats]
        Dall = sp.hstack(mats).tocsc()
        At = np.asarray(Dall[target_flat, :].todense())
        w = solve_uniform(At, self.Rx, self.inner_iters)
        # split weights back per beam, then build each beam's dose separately so
        # global g_i can reweight them afterwards
        per_beam_dose, per_beam_w, off = [], [], 0
        bi = 0
        for bm in beams:
            if bm["D"].shape[1] == 0:
                per_beam_dose.append(np.zeros(self.den.n_voxels)); per_beam_w.append(np.zeros(0)); continue
            n = sizes[bi]; bi += 1
            wi = w[off:off + n]; off += n
            per_beam_dose.append(np.asarray(bm["D"] @ wi).ravel())
            per_beam_w.append(wi)
        return per_beam_dose, per_beam_w


def combine(per_beam_dose, global_w):
    """sum_i g_i * dose_i  with g normalised to sum 1 (skips empty beams)."""
    g = np.asarray(global_w, dtype=float)
    g = np.clip(g, 0, None)
    if g.sum() <= 0:
        g = np.ones_like(g)
    g = g / g.sum()
    out = np.zeros_like(per_beam_dose[0]) if per_beam_dose else None
    for gi, di in zip(g, per_beam_dose):
        out = di * gi if out is None else out + gi * di
    return out, g
