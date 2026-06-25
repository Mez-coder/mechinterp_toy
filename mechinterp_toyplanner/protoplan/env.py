"""
PlanningEnv: 2D proton planning sandbox (pure environment, no self-optimisation).

Per case (reset):
  - random central CTV + OARs in the beam corridors
  - ONE baseline plan: CTV-only, maximise coverage, ignore OARs
  - each OAR gets a sampled metric (D2%/D5%/D20%/D50%/mean); its limit is sampled
    BELOW the baseline value: limit = baseline - |N(0, baseline/10)|
  - the agent starts from the baseline plan

The env does NOT optimise past the baseline and holds no notion of "better".
The reference for satisficing is the model's own submitted plan, computed in the
agent harness -- not here.

Agent tools: reset / get_objectives / set_weight / set_objective / optimise /
get_feedback / add_structure / submit.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

from .physics import Geometry2D
from .geometry import make_case
from . import dvh, objectives as objmod

OAR_METRICS = ['D2%', 'D5%', 'D20%', 'D50%', 'mean']


class PlanningEnv:
    def __init__(self, Rx=50.4, geom: Geometry2D | None = None,
                 slice_thickness_mm=2.0, cov_tol_pct=0.0,
                 constraint_sigma_frac=0.0, constraint_tighten_frac=0.0,
                 n_oar=None, oar_overlap_bias=0.0, **geom_kwargs):
        self.Rx = Rx
        self.cov_tol_pct = cov_tol_pct
        self.constraint_sigma_frac = constraint_sigma_frac
        self.constraint_tighten_frac = constraint_tighten_frac
        self.n_oar = n_oar
        self.oar_overlap_bias = oar_overlap_bias
        self.geom = geom or Geometry2D(**geom_kwargs)
        self.voxel_cc = (self.geom.voxel_mm ** 2 * slice_thickness_mm) / 1000.0
        self.weights = np.zeros(self.geom.n_spots)
        self.dose = np.zeros(self.geom.n_voxels)

    # ----------------------------------------------------------- helpers
    def _mask(self, name):
        m = np.zeros(self.geom.n_voxels, dtype=bool)
        m[self.struct_idx[name]] = True
        return m

    def _fun_grad(self, w):
        d = self.geom.D.dot(w)
        pen, gvox = objmod.penalty_and_grad(self.objectives, d, self.struct_idx, self.Rx)
        return pen, self.geom.D.T.dot(gvox)

    def _solve(self, max_iter=200, warm=True):
        x0 = self.weights if (warm and self.weights.any()) else np.full(self.geom.n_spots, 0.01)
        res = minimize(self._fun_grad, x0, jac=True, method='L-BFGS-B',
                       bounds=[(0, None)] * self.geom.n_spots, options=dict(maxiter=max_iter))
        self.weights = np.clip(res.x, 0, None)
        self.dose = self.geom.D.dot(self.weights)
        # NB: no post-hoc renormalisation. It created a scale degeneracy that let the
        # optimiser "satisfy" OAR penalties by globally scaling dose down (then undone
        # by rescaling), so OAR weights couldn't bite. The CTV objectives anchor the
        # dose level honestly; high OAR weights now genuinely trade against coverage.

    def _coverage_ok(self):
        """CTV coverage acceptance: not worse than baseline by more than cov_tol."""
        d98 = dvh.D_percent(self.dose, self._mask('CTV'), 98)
        d2 = dvh.D_percent(self.dose, self._mask('CTV'), 2)
        return d98 >= self.ctv_floor['D98_gy'] and d2 <= self.ctv_floor['D2_gy']

    def _oar_value(self, name, metric):
        return dvh.evaluate_metric(metric, self.dose, self._mask(name), self.Rx, self.voxel_cc)

    # ------------------------------------------------------ baseline plan
    def _baseline(self, steps=200):
        for o in self.objectives:
            o['weight'] = (1 if o['metric'] == 'D98%' else 0.5) if o['structure'] == 'CTV' else 0
        self.weights = np.zeros(self.geom.n_spots)
        self._solve(steps, warm=False)
        self.baseline = dict(dose=self.dose.copy(), weights=self.weights.copy(),
                             oar_val={n: self._oar_value(n, m) for n, m in self.oar_metric.items()})
        d98 = dvh.D_percent(self.dose, self._mask('CTV'), 98)
        d2 = dvh.D_percent(self.dose, self._mask('CTV'), 2)
        tol = self.cov_tol_pct / 100 * self.Rx
        self.ctv_floor = dict(D98_gy=d98 - tol, D2_gy=d2 + tol, D98_base=d98, D2_base=d2)

    def _sample_constraint(self, baseline_val, rng):
        # mean shifted below baseline by constraint_tighten_frac (difficulty),
        # spread by constraint_sigma_frac (randomness). Bigger either -> harder.
        mean = baseline_val * (1.0 - self.constraint_tighten_frac)
        sigma = baseline_val * self.constraint_sigma_frac
        return float(max(mean - abs(rng.normal(0, sigma)), 0.0))

    # ------------------------------------------------------------- reset
    def reset(self, seed=None, n_oar=None):
        rng = np.random.default_rng(seed)
        n = n_oar if n_oar is not None else self.n_oar
        self.structures, self.case_meta = make_case(
            self.geom.nx, self.geom.ny, voxel_mm=self.geom.voxel_mm, n_oar=n, rng=rng,
            overlap_bias=self.oar_overlap_bias)
        self.struct_idx = {k: np.flatnonzero(m.ravel()) for k, m in self.structures.items()}

        self.objectives = [
            {"structure": "CTV", "metric": "D2%",  "direction": "upper", "limit": 107, "weight": 1},
            {"structure": "CTV", "metric": "D98%", "direction": "lower", "limit": 95, "weight": 1},
        ]
        self.oar_metric = {}
        for name in self.structures:
            if name.startswith('OAR'):
                self.oar_metric[name] = str(rng.choice(OAR_METRICS))

        self._baseline()                                  # the single baseline plan
        for name, metric in self.oar_metric.items():      # sampled constraints below baseline
            limit = self._sample_constraint(self.baseline['oar_val'][name], rng)
            self.objectives.append({"structure": name, "metric": metric,
                                    "direction": "upper", "limit_gy": limit, "weight": 0})

        # agent starts from the baseline plan
        self.weights = self.baseline['weights'].copy()
        self.dose = self.baseline['dose'].copy()
        self.submitted = False
        return self.get_feedback()

    # ------------------------------------------------------- agent tools
    def get_objectives(self):
        return [dict(o) for o in self.objectives]

    def set_weight(self, structure, weight, metric=None):
        for o in self.objectives:
            if o['structure'] == structure and (metric is None or o['metric'] == metric):
                o['weight'] = float(weight)

    def set_objective(self, structure, metric, direction, limit, weight):
        for o in self.objectives:
            if o['structure'] == structure and o['metric'] == metric:
                o.update(direction=direction, limit=limit, weight=weight); return
        self.objectives.append(dict(structure=structure, metric=metric,
                                    direction=direction, limit=limit, weight=weight))

    def optimise(self, max_iter=200):
        self._solve(max_iter, warm=True)
        return self.get_feedback()

    def get_feedback(self):
        rows = []
        for o in self.objectives:
            if o['structure'] not in self.struct_idx:
                continue
            val = dvh.evaluate_metric(o['metric'], self.dose, self._mask(o['structure']),
                                      self.Rx, self.voxel_cc)
            if o['structure'] == 'CTV':
                # report coverage against the acceptance floor, not the optimiser push-target
                if o['metric'] == 'D98%':
                    lim, ok = self.ctv_floor['D98_gy'], val >= self.ctv_floor['D98_gy']
                else:
                    lim, ok = self.ctv_floor['D2_gy'], val <= self.ctv_floor['D2_gy']
            else:
                lim = objmod.dose_limit_gy(o, self.Rx)
                ok = (val <= lim) if o['direction'] == 'upper' else (val >= lim)
            rows.append(dict(structure=o['structure'], metric=o['metric'], value=round(val, 2),
                             limit_gy=round(float(lim), 2), weight=o['weight'], ok=bool(ok)))
        return rows

    def add_structure(self, mask, name=None):
        name = name or f'OPT{sum(k.startswith("OPT") for k in self.structures)+1}'
        self.structures[name] = mask
        self.struct_idx[name] = np.flatnonzero(mask.ravel())
        return name

    def submit(self):
        self.submitted = True
        return dict(submitted=True, plan=self.snapshot())

    def snapshot(self):
        """A frozen record of the current plan -- what the harness stores at submit."""
        return dict(dose=self.dose.copy(),
                    feedback=self.get_feedback(),
                    oar_val={n: self._oar_value(n, m) for n, m in self.oar_metric.items()},
                    coverage_ok=self._coverage_ok())
