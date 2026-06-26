"""Random case geometry for the proton sandbox: a CTV centred at the world
origin (0,0) plus N organs-at-risk that sit INSIDE the water circle but strictly
OUTSIDE the target.

Everything is specified in mm and converted with `voxel_mm`. Masks share the
DensityMap convention: shape (ny, nx), flat index v = iy*nx + ix.
"""
from __future__ import annotations
import numpy as np

from .density import DensityMap


def _ellipse_mask(nx, ny, cx, cy, rx, ry, angle=0.0):
    """Boolean (ny,nx) ellipse centred at voxel (cx,cy)."""
    ys, xs = np.mgrid[0:ny, 0:nx]
    xs = xs - cx; ys = ys - cy
    ca, sa = np.cos(angle), np.sin(angle)
    xr = xs * ca + ys * sa
    yr = -xs * sa + ys * ca
    return (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0


def make_case(density: DensityMap, n_oar=None, rng=None,
              ctv_diam_cm=(3.0, 10.0), oar_diam_cm=(2.0, 8.0),
              min_oar_vox=40, margin_vox=2, max_tries=200):
    """Return ({'CTV':mask, 'OAR1':mask, ...}, meta).

    The CTV is always centred on the world origin. OARs are sampled uniformly
    inside the phantom, rejected if they (a) leave the water circle, (b) touch
    the CTV (with a small margin), or (c) overlap an already-placed OAR. This
    keeps OAR sparing a purely geometric problem: the model must steer beams to
    miss them.
    """
    rng = rng or np.random.default_rng()
    nx, ny, vox = density.nx, density.ny, density.voxel_mm
    R = density.radius_mm
    cx, cy = nx / 2.0, ny / 2.0           # grid-centre in voxel coords
    mm2vox = lambda mm: mm / vox

    # ---- central target ---------------------------------------------------
    if rng.random() < 0.5:                                   # circle
        r = mm2vox(rng.uniform(*ctv_diam_cm) * 10.0 / 2.0)
        rx = ry = r; ang = 0.0; shape = 'circle'
    else:                                                    # oval
        rx = mm2vox(rng.uniform(*ctv_diam_cm) * 10.0 / 2.0)
        ry = mm2vox(rng.uniform(*ctv_diam_cm) * 10.0 / 2.0)
        ang = rng.uniform(0, np.pi); shape = 'oval'
    ctv = _ellipse_mask(nx, ny, cx, cy, rx, ry, ang)
    structures = {'CTV': ctv}
    meta = {'CTV': dict(shape=shape, rx_mm=rx * vox, ry_mm=ry * vox)}

    # voxels available for OAR centres: inside water, outside an enlarged CTV
    ys, xs = np.mgrid[0:ny, 0:nx]
    wx, wy = density.world_of_voxel(xs, ys)
    in_water = (wx ** 2 + wy ** 2) <= (R - oar_diam_cm[0] * 10.0 / 2.0) ** 2
    ctv_grow = _ellipse_mask(nx, ny, cx, cy, rx + margin_vox, ry + margin_vox, ang)

    n_oar = int(rng.integers(2, 5)) if n_oar is None else int(n_oar)
    occupied = ctv_grow.copy()
    for k in range(1, n_oar + 1):
        placed = None
        for _ in range(max_tries):
            orx = mm2vox(rng.uniform(*oar_diam_cm) * 10.0 / 2.0)
            ory = mm2vox(rng.uniform(*oar_diam_cm) * 10.0 / 2.0)
            oang = rng.uniform(0, np.pi)
            # sample a centre voxel that is in water and clear of the CTV
            valid = in_water & ~ctv_grow & ~occupied
            idx = np.flatnonzero(valid.ravel())
            if idx.size == 0:
                break
            pick = idx[rng.integers(idx.size)]
            ocy, ocx = divmod(int(pick), nx)
            cand = _ellipse_mask(nx, ny, ocx, ocy, orx, ory, oang)
            # reject if it spills out of water, hits CTV margin, or overlaps OARs
            spill = cand & ((wx ** 2 + wy ** 2) > R ** 2)
            if spill.any():
                continue
            if (cand & ctv_grow).any() or (cand & occupied).any():
                continue
            if cand.sum() < min_oar_vox:
                continue
            placed = cand
            break
        if placed is None:
            continue                                    # give up on this OAR
        structures[f'OAR{k}'] = placed
        occupied = occupied | placed
        meta[f'OAR{k}'] = dict(rx_mm=orx * vox, ry_mm=ory * vox,
                               centre_mm=tuple(float(c) for c in
                                               density.world_of_voxel(ocx, ocy)),
                               n_vox=int(placed.sum()))
    meta['n_oar'] = sum(k.startswith('OAR') for k in structures)
    return structures, meta
