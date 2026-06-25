"""True DVH metrics on a dose array (Gy) restricted to a structure mask.

These are what the agent SEES and what we score with. They are NOT used inside
the optimiser (see objectives.py for the differentiable surrogate).
"""
from __future__ import annotations
import numpy as np
import re


def structure_doses(dose, mask):
    return dose[mask]


def D_percent(dose, mask, p):
    """Dose received by at least p% of the volume (= (100-p)th percentile)."""
    d = structure_doses(dose, mask)
    if d.size == 0:
        return 0.0
    return float(np.percentile(d, 100 - p))


def D_cc(dose, mask, v_cc, voxel_cc):
    """Dose to the hottest v_cc of the structure."""
    d = np.sort(structure_doses(dose, mask))[::-1]
    if d.size == 0:
        return 0.0
    n = max(int(round(v_cc / voxel_cc)), 1)
    return float(d[min(n, d.size) - 1])


def V_dose(dose, mask, d_gy):
    """Percent of structure volume receiving >= d_gy."""
    d = structure_doses(dose, mask)
    if d.size == 0:
        return 0.0
    return float(100.0 * np.mean(d >= d_gy))


def mean_dose(dose, mask):
    d = structure_doses(dose, mask)
    return float(d.mean()) if d.size else 0.0


_METRIC = re.compile(r'^([DV])([0-9.]+)(%|cc|Gy)$')


def evaluate_metric(metric, dose, mask, Rx, voxel_cc):
    """Return achieved value (Gy for D-metrics, % for V-metrics)."""
    if metric == 'mean':
        return mean_dose(dose, mask)
    m = _METRIC.match(metric)
    if not m:
        raise ValueError(f'bad metric {metric}')
    kind, num, unit = m.group(1), float(m.group(2)), m.group(3)
    if kind == 'D' and unit == '%':
        return D_percent(dose, mask, num)
    if kind == 'D' and unit == 'cc':
        return D_cc(dose, mask, num, voxel_cc)
    if kind == 'V' and unit == 'Gy':
        return V_dose(dose, mask, num)
    raise ValueError(f'unsupported metric {metric}')
