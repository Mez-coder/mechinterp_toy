"""TPS-style objectives -> differentiable one-sided penalties.

Limit resolution order:
  - explicit obj['limit_gy']  (absolute Gy; used for sampled/anchored OAR limits)
  - else D-metrics: limit is % of Rx -> limit/100*Rx
Metric 'mean' is supported (penalty on mean structure dose).
True DVH values for feedback/scoring come from dvh.py, not here.
"""
from __future__ import annotations
import json, re
import numpy as np

_METRIC = re.compile(r'^([DV])([0-9.]+)(%|cc|Gy)$')


def load_objectives(path_or_list):
    if isinstance(path_or_list, str):
        with open(path_or_list) as f:
            return json.load(f)['objectives']
    return list(path_or_list)


def dose_limit_gy(obj, Rx):
    if 'limit_gy' in obj:
        return obj['limit_gy']
    m = _METRIC.match(obj['metric'])
    if m and m.group(1) == 'D':
        return obj['limit'] / 100.0 * Rx
    if m and m.group(1) == 'V':
        return float(m.group(2))
    if obj['metric'] == 'mean':
        return obj['limit'] / 100.0 * Rx
    raise ValueError(f"cannot resolve limit for {obj}")


def penalty_and_grad(objectives, dose, struct_idx, Rx):
    g = np.zeros_like(dose)
    total = 0.0
    for obj in objectives:
        w = obj.get('weight', 0)
        if w <= 0:
            continue
        idx = struct_idx.get(obj['structure'])
        if idx is None or idx.size == 0:
            continue
        dl = dose_limit_gy(obj, Rx)
        d = dose[idx]
        metric = obj.get('metric')
        upper = obj['direction'] == 'upper'

        if metric == 'mean':
            md = d.mean()
            viol = max(md - dl, 0.0) if upper else min(md - dl, 0.0)
            total += w * viol ** 2
            g[idx] += w * 2.0 * viol / idx.size
            continue

        m = _METRIC.match(metric or '')
        if m and m.group(1) == 'D' and m.group(3) == '%':
            # ---- DVH-aware penalty -------------------------------------------
            # A Dp% metric measures only a tail of the structure, so the gradient
            # belongs ONLY on that tail -- not on every voxel above the limit.
            #   upper Dp%  (e.g. OAR D5% <= L)   -> the hottest p% of the volume
            #   lower Dp%  (e.g. CTV D98% >= L)  -> the coldest (100-p)% of volume
            # The penalty is normalised by the tail size (not the whole
            # structure), so a high weight actually bites on the few voxels that
            # define the metric instead of being diluted across thousands.
            p = float(m.group(2))
            N = idx.size
            if upper:
                n_reg = min(max(int(round(p / 100.0 * N)), 1), N)
                if n_reg >= N:
                    sub = idx
                else:                                   # indices of hottest n_reg
                    sub = idx[np.argpartition(d, N - n_reg)[N - n_reg:]]
            else:
                n_reg = min(max(int(round((100.0 - p) / 100.0 * N)), 1), N)
                if n_reg >= N:
                    sub = idx
                else:                                   # indices of coldest n_reg
                    sub = idx[np.argpartition(d, n_reg)[:n_reg]]
            dd = dose[sub]
            viol = (np.maximum(dd - dl, 0.0) if upper
                    else np.minimum(dd - dl, 0.0))
            scale = w / max(n_reg, 1)
            total += scale * np.sum(viol ** 2)
            g[sub] += scale * 2.0 * viol
            continue

        # ---- fallback: broad one-sided penalty (V-metrics, D-cc, ...) --------
        viol = (np.maximum(d - dl, 0.0) if upper else np.minimum(d - dl, 0.0))
        scale = w / max(idx.size, 1)
        total += scale * np.sum(viol ** 2)
        g[idx] += scale * 2.0 * viol
    return total, g
