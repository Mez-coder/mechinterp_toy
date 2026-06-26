"""Proton stopping power: a simplified but genuine Bethe-Bloch formulation.

Unlike the old environment (which injected an analytic Bragg-Kleeman depth-dose
curve), here the Bragg peak EMERGES from marching the stopping power: as the
proton slows, 1/beta^2 diverges, so dE/dx spikes at end-of-range. That is the
whole physical payoff of the rewrite, and it scales linearly with local density
(so heterogeneous tissue will Just Work).

Bethe (simplified, no shell/density/Barkas corrections):
    -dE/dx = K * rho * (1/beta^2) * [ ln( 2 m_e c^2 beta^2 gamma^2 / I ) - beta^2 ]
with
    gamma = 1 + E/Mp ,  beta^2 = 1 - 1/gamma^2 .

Absolute dose is arbitrary in this sandbox (everything is normalised to the
prescription Rx=1.0), so the only thing that must be physically faithful is the
SHAPE and the RANGE. We therefore calibrate the constant K once at construction
so that R(150 MeV) ~= 157 mm in water (the textbook value), build a range table
by numerically integrating S(E), and expose range<->energy inversion. Energies
for a case are then chosen from the geometry via this self-consistent table, so
marched Bragg peaks land where we expect.
"""
from __future__ import annotations
import numpy as np

MP_C2 = 938.272      # proton rest energy (MeV)
ME_C2 = 0.510999     # electron rest energy (MeV)
I_WATER = 75.0e-6    # mean excitation potential (MeV) ~ 75 eV
_R150_MM = 157.0     # target water range for a 150 MeV proton (calibration anchor)


class StoppingModel:
    def __init__(self, I=I_WATER, e_grid=None):
        self.I = I
        self._egrid = e_grid if e_grid is not None else np.linspace(0.5, 300.0, 6000)
        self._K = 1.0
        self._calibrate()
        self._build_range_table()

    # ---- raw Bethe shape (unit K, unit density) ---------------------------
    def _shape(self, E):
        E = np.asarray(E, dtype=np.float64)
        gamma = 1.0 + E / MP_C2
        beta2 = np.clip(1.0 - 1.0 / gamma ** 2, 1e-6, 0.999999)
        arg = 2.0 * ME_C2 * beta2 * gamma ** 2 / self.I
        ln = np.log(np.clip(arg, 1.0001, None))             # regularise low-E log
        s = (1.0 / beta2) * (ln - beta2)
        return np.clip(s, 1e-6, None)                       # strictly positive

    def stopping_power(self, E, rho):
        """-dE/dx in MeV/mm at energy E (MeV) and relative density rho."""
        return self._K * np.asarray(rho, dtype=np.float64) * self._shape(E)

    # ---- calibration so R(150) ~= 157 mm in water -------------------------
    def _raw_range(self, E0):
        # range = integral_0^E0 dE / shape(E)   (unit K, rho=1)
        Es = np.linspace(1e-3, float(E0), 4000)
        inv = 1.0 / self._shape(Es)
        return float(np.trapezoid(inv, Es))

    def _calibrate(self):
        # R(E) = (1/K) * raw_range(E); choose K so R(150) = _R150_MM
        self._K = self._raw_range(150.0) / _R150_MM

    # ---- range <-> energy table (water, rho=1) ----------------------------
    def _build_range_table(self):
        Es = self._egrid
        inv = 1.0 / (self._K * self._shape(Es))             # dx/dE in mm/MeV
        R = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(Es))])
        self._Etab, self._Rtab = Es, R

    def range_of_energy(self, E):
        return np.interp(E, self._Etab, self._Rtab)

    def energy_of_range(self, R):
        return np.interp(R, self._Rtab, self._Etab)
