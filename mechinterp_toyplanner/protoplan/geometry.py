"""Random case geometry (clinically-scaled, mm-based): central target + N OARs.

target diameter 3-10 cm; OAR diameter 1-20 cm. OARs biased toward the central
beam-corridor region so they actually receive dose (corner OARs = dull rollouts).
"""
from __future__ import annotations
import numpy as np


def _ellipse_mask(nx, ny, cx, cy, rx, ry, angle=0.0):
    ys, xs = np.mgrid[0:ny, 0:nx]
    xs = xs - cx; ys = ys - cy
    ca, sa = np.cos(angle), np.sin(angle)
    xr = xs*ca + ys*sa
    yr = -xs*sa + ys*ca
    return (xr/rx)**2 + (yr/ry)**2 <= 1.0


def make_case(nx=167, ny=167, voxel_mm=2.0, n_oar=None, rng=None, allow_overlap=True,
              overlap_bias=0.0, subtract_target=True, min_oar_vox=30):
    """Returns ({'CTV':mask,'OAR1':mask,...}, meta). Sizes specified in mm.

    If subtract_target is True (default), each stored OAR mask has the CTV
    removed (OAR := OAR \\ CTV). The geometry is still drawn the same way -- OARs
    may be placed over the target -- but the voxels shared with the CTV are not
    part of the OAR for either optimisation or DVH scoring. This stops the CTV
    coverage objective and an OAR sparing objective from fighting over the very
    same voxels (which made high OAR weights unable to bite). meta['ctv_overlap']
    still records the ORIGINAL geometric overlap for reference.
    """
    rng = rng or np.random.default_rng()
    cx, cy = nx/2, ny/2
    mm = lambda v: v / voxel_mm   # mm -> voxels

    if rng.random() < 0.5:                       # circle, diameter 3-10 cm
        r = mm(rng.uniform(15, 50)); rx = ry = r; ang = 0.0; shape = 'circle'
    else:
        rx = mm(rng.uniform(15, 50)); ry = mm(rng.uniform(15, 50)); ang = rng.uniform(0, np.pi); shape = 'oval'
    ctv = _ellipse_mask(nx, ny, cx, cy, rx, ry, ang)
    structures = {'CTV': ctv}
    meta = {'CTV': dict(shape=shape, rx_mm=rx*voxel_mm, ry_mm=ry*voxel_mm)}

    n_oar = int(rng.integers(2, 5)) if n_oar is None else n_oar
    for k in range(1, n_oar+1):
        m = None; overlap = 0.0; orx = ory = 0.0
        for _ in range(60):
            ox = rng.uniform(0.3*nx, 0.7*nx) * (1-overlap_bias) + cx*overlap_bias
            oy = rng.uniform(0.3*ny, 0.7*ny) * (1-overlap_bias) + cy*overlap_bias
            orx = mm(rng.uniform(5, 100)); ory = mm(rng.uniform(5, 100))  # diameter 1-20 cm
            oang = rng.uniform(0, np.pi)
            cand = _ellipse_mask(nx, ny, ox, oy, orx, ory, oang)
            overlap = (cand & ctv).sum() / max(cand.sum(), 1)
            m_eff = (cand & ~ctv) if subtract_target else cand
            if (allow_overlap or overlap < 0.15) and m_eff.sum() >= min_oar_vox:
                m = m_eff; break
        if m is None:                       # never satisfied the guard: take last candidate
            m = m_eff
        structures[f'OAR{k}'] = m
        meta[f'OAR{k}'] = dict(rx_mm=orx*voxel_mm, ry_mm=ory*voxel_mm,
                               ctv_overlap=float(overlap),
                               regime='parallel' if max(orx, ory)*voxel_mm > 60 else 'serial')
    return structures, meta
