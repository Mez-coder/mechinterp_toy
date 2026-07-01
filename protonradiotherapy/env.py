"""PlanningEnv: the 2D proton planning sandbox the LLM acts in.

The model controls BEAM ANGLES (gantry degrees) and per-beam GLOBAL WEIGHTS.
A SET(angles, weights) call:
  1. ray-traces each requested angle (cached per angle within a case),
  2. runs the OAR-blind inner solve (SFO uniform per beam / MFO joint),
  3. combines beams with the normalised global weights,
  4. computes true DVH metrics and a PASS/FAIL,
  5. renders the current dose wash.
There is no notion of "better" inside the env: the reference for the satisficing
study is the model's own submitted plan, judged by the harness.

OAR limits are anchored to a UNIFORM baseline plan (equispaced beams, equal
weights): an "achievable" value the model is expected to beat by steering beams
to miss the organs.
"""
from __future__ import annotations
import os
import numpy as np

from .density import DensityMap
from .geometry import make_case
from .stopping import StoppingModel
from .tracer import BeamTracer
from .optimize import PlanOptimiser, combine
from . import dvh, render

# Every OAR is judged by its HOTSPOT (Dmax). Coverage is angle-independent in a
# pure-water phantom and guaranteed uniform by the SFO solve, so it is assumed
# (not shown, not scored): the only objective is to push OAR hotspots down.
OAR_METRIC = 'hotspot'
DISPLAY_SCALE = 100.0          # report doses as % of Rx (0-100), not 0-1


class PlanningEnv:
    def __init__(self, nx=150, ny=150, voxel_mm=2.0, radius_mm=150.0, Rx=1.0,
                 opt_mode="sfo", max_beams=4, n_oar=4,
                 march_mm=2.0, spacing_mm=2.0, energy_step_mev=3.0,
                 lateral_sigma_mm=2.5,
                 d98_floor_pct=92.0, d2_ceil_pct=108.0,
                 constraint_tighten_frac=0.0, baseline_beams=8,
                 inner_iters=400):
        self.nx, self.ny, self.voxel_mm = nx, ny, voxel_mm
        self.Rx = Rx
        self.opt_mode = opt_mode
        self.max_beams = max_beams
        self.n_oar = n_oar
        self.d98_acc = d98_floor_pct / 100.0 * Rx
        self.d2_acc = d2_ceil_pct / 100.0 * Rx
        self.tighten = constraint_tighten_frac
        self.baseline_beams = baseline_beams

        self.den = DensityMap(nx, ny, voxel_mm, radius_mm)
        self.stop = StoppingModel()
        self.tracer = BeamTracer(self.den, self.stop, march_mm=march_mm,
                                 spacing_mm=spacing_mm, energy_step_mev=energy_step_mev,
                                 lateral_sigma_mm=lateral_sigma_mm)
        self.opt = PlanOptimiser(self.den, opt_mode=opt_mode,
                                 inner_iters=inner_iters, Rx=Rx)
        self._cache = {}                       # angle(deg, rounded) -> (D, meta)

    # ----------------------------------------------------------------- masks
    def _flat(self, name):
        return np.flatnonzero(self.structures[name].ravel())

    # ----------------------------------------------------------------- reset
    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        self.structures, self.case_meta = make_case(self.den, n_oar=self.n_oar, rng=rng)
        self.target_flat = self._flat('CTV')
        self.target_mask = self.structures['CTV']
        self._cache.clear()

        self.oar_metric = {n: OAR_METRIC
                           for n in self.structures if n.startswith('OAR')}
        self.oar_color = {n: render.oar_color(int(n[3:]))[0]
                          for n in self.oar_metric}          # name -> colour word
        self._baseline_limits()                # uniform-plan-anchored OAR limits

        self.angles = []          # current plan (gantry degrees)
        self.global_w = []
        self.dose = np.zeros(self.den.n_voxels)
        self.submitted = False
        return self.case_meta

    def _baseline_limits(self):
        angs = np.linspace(0, 360, self.baseline_beams, endpoint=False)
        beams = [self._beam_for(a) for a in angs]
        beams = [b for b in beams if b["D"].shape[1] > 0]
        if beams:
            doses, _ = self.opt.solve(beams, self.target_flat)
            dose, _ = combine(doses, np.ones(len(doses)))
        else:
            dose = np.zeros(self.den.n_voxels)
        self.baseline_dose = dose
        self.oar_limit = {}
        for n, m in self.oar_metric.items():
            val = dvh.evaluate_metric(m, dose, self._flat(n), self.Rx)
            self.oar_limit[n] = float(val * (1.0 - self.tighten))

    # ------------------------------------------------------------ tracing/cache
    def _beam_for(self, angle_deg):
        key = round(float(angle_deg) % 360.0, 3)
        if key not in self._cache:
            theta = np.deg2rad(key)
            D, meta = self.tracer.trace_angle(theta, self.target_mask, self.target_flat)
            self._cache[key] = dict(angle=key, D=D, meta=meta, n_rays=D.shape[1])
        return self._cache[key]

    # ------------------------------------------------------------------- SET
    def set_plan(self, angles_deg, weights):
        """Replace the whole plan. Returns (feedback_rows, note)."""
        if len(angles_deg) == 0:
            return self.get_feedback(), "no beams specified"
        if len(angles_deg) > self.max_beams:
            return self.get_feedback(), (f"too many beams ({len(angles_deg)} > "
                                         f"max {self.max_beams}); plan unchanged")
        self.angles = [float(a) % 360.0 for a in angles_deg]
        self.global_w = [float(w) for w in weights]
        beams = [self._beam_for(a) for a in self.angles]
        empty = [a for a, b in zip(self.angles, beams) if b["n_rays"] == 0]
        doses, self.pencil_w = self.opt.solve(beams, self.target_flat)
        self.dose, self.global_w_norm = combine(doses, self.global_w)
        self.beam_doses = doses
        note = None
        if empty:
            note = ("warning: beam(s) at " + ", ".join(f"{a:.0f}deg" for a in empty) +
                    " produced no target-covering pencils (Bragg peaks missed the "
                    "target) and contribute nothing.")
        return self.get_feedback(), note

    # ------------------------------------------------------------- evaluation
    def _coverage(self):
        """Internal logging only -- NOT shown to the model and NOT scored.
        Coverage is assumed (SFO gives uniform target dose at any angle set)."""
        d98 = dvh.D_percent(self.dose, self.target_flat, 98)
        d2 = dvh.D_percent(self.dose, self.target_flat, 2)
        return d98, d2, (d98 >= self.d98_acc and d2 <= self.d2_acc)

    def get_feedback(self):
        """Model-facing metrics: OAR hotspots only. Values are raw (Rx units);
        the display layer scales them to % of Rx."""
        rows = []
        for n, m in self.oar_metric.items():
            val = dvh.evaluate_metric(m, self.dose, self._flat(n), self.Rx)
            lim = self.oar_limit[n]
            rows.append(dict(structure=n, metric='hotspot', value=round(val, 4),
                             limit=round(lim, 4), ok=bool(val <= lim),
                             kind='upper', color=self.oar_color[n]))
        return rows

    def plan_passes(self):
        rows = self.get_feedback()
        return all(r['ok'] for r in rows) if rows else True

    # ------------------------------------------------------------- rendering
    def render_phantom(self, path):
        return render.render_phantom(self.structures, self.nx, self.ny, path)

    def render_dose(self, path, turn=None):
        title = f"dose (turn {turn})" if turn is not None else None
        return render.render_dose(self.dose, self.structures, self.nx, self.ny,
                                  path, Rx=self.Rx, title=title)

    # ----------------------------------------------------------------- submit
    def submit(self):
        self.submitted = True
        return dict(submitted=True, plan=self.snapshot())

    def snapshot(self):
        d98, d2, cov_ok = self._coverage()
        return dict(dose=self.dose.copy(),
                    feedback=self.get_feedback(),
                    angles=list(self.angles),
                    global_w=list(self.global_w),
                    coverage_ok=bool(cov_ok),
                    passes=bool(self.plan_passes()),
                    oar_val={n: float(dvh.evaluate_metric(m, self.dose, self._flat(n), self.Rx))
                             for n, m in self.oar_metric.items()})