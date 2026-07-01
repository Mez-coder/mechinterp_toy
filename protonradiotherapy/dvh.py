"""True DVH metrics on a flat dose vector restricted to a structure mask.

These are the SCORING metrics (what the agent sees and what we judge). They are
NOT used inside the optimiser -- the inner solve only ever sees the target MSE.
Dose is relative to the prescription Rx (= 1.0 here); percentages of Rx are the
natural unit.
"""
from __future__ import annotations
import numpy as np
import re

_METRIC = re.compile(r'^([DV])([0-9.]+)(%|Gy)$')


def _doses(dose, flat_mask):
    return np.asarray(dose).ravel()[flat_mask]


def D_percent(dose, flat_mask, p):
    """Dose to at least p% of the volume = (100-p)th percentile."""
    d = _doses(dose, flat_mask)
    return float(np.percentile(d, 100 - p)) if d.size else 0.0


def V_dose(dose, flat_mask, level):
    d = _doses(dose, flat_mask)
    return float(100.0 * np.mean(d >= level)) if d.size else 0.0


def mean_dose(dose, flat_mask):
    d = _doses(dose, flat_mask)
    return float(d.mean()) if d.size else 0.0


def hotspot(dose, flat_mask, near_max_pct=0.0):
    """Hotspot dose in a structure. near_max_pct=0 -> true Dmax (hottest voxel);
    set e.g. 2 for D2% (near-max) if Dmax is too noisy turn-to-turn."""
    d = _doses(dose, flat_mask)
    if d.size == 0:
        return 0.0
    if near_max_pct <= 0:
        return float(d.max())
    return float(np.percentile(d, 100 - near_max_pct))


def evaluate_metric(metric, dose, flat_mask, Rx=1.0):
    """Value for a metric string. D-% -> dose (Rx units); V-Gy -> percent;
    'hotspot' -> Dmax; 'mean' -> mean dose."""
    if metric == 'mean':
        return mean_dose(dose, flat_mask)
    if metric == 'hotspot':
        return hotspot(dose, flat_mask)
    m = _METRIC.match(metric)
    if not m:
        raise ValueError(f'bad metric {metric}')
    kind, num, unit = m.group(1), float(m.group(2)), m.group(3)
    if kind == 'D' and unit == '%':
        return D_percent(dose, flat_mask, num)
    if kind == 'V' and unit == 'Gy':
        return V_dose(dose, flat_mask, num * Rx)
    raise ValueError(f'unsupported metric {metric}')