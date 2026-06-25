"""
Proton physics for the 2D water sandbox (clinically-scaled defaults).

Depth dose: Bragg-Kleeman range relation R = alpha*E^p (water), stopping power
S(E)=E^(1-p)/(alpha*p); end-of-range singularity regularised by Gaussian range
straggling tied to depth (~1.2% of range, physically correct). Lateral penumbra
is Gaussian. mm + MeV throughout; dose arbitrary until normalised to Rx.

Geometry: square water grid, 4 beams at 0/90/180/270 deg, each beam's spots tile
only the target band (centre +/- spot_half_extent) in lateral and depth.
Influence matrix is built once and (optionally) cached to disk.
"""
from __future__ import annotations
import os, json, hashlib
import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter1d

ALPHA = 0.022      # mm / MeV^p   (water)
P = 1.77

def range_of_energy(E):  return ALPHA * np.power(E, P)              # mm
def energy_of_range(R):  return np.power(np.clip(R,1e-6,None)/ALPHA, 1.0/P)  # MeV

def straggle_for_range(R0):                  # physical range straggling (mm)
    return max(1.0, 0.012 * R0)

def depth_dose(depth_mm, R0_mm, straggle_mm):
    resid = R0_mm - depth_mm
    E = np.where(resid > 0, np.power(np.clip(resid,1e-6,None)/ALPHA, 1.0/P), 0.0)
    S = np.where(resid > 0, np.power(np.clip(E,1e-6,None), 1.0-P)/(ALPHA*P), 0.0)
    step = float(np.mean(np.diff(depth_mm))) if depth_mm.size > 1 else 1.0
    S = gaussian_filter1d(S, sigma=max(straggle_mm/step, 0.5))
    m = S.max()
    return S/m if m > 0 else S


class Geometry2D:
    def __init__(self, nx=167, ny=167, voxel_mm=2.0,
                 lateral_spacing_mm=3.0, n_energy_layers=17,
                 lateral_sigma_mm=3.0, spot_half_extent_mm=60.0,
                 cache_dir=None):
        self.nx, self.ny, self.voxel_mm = nx, ny, voxel_mm
        self.lateral_spacing = lateral_spacing_mm
        self.n_layers = n_energy_layers
        self.lateral_sigma = lateral_sigma_mm
        self.spot_half_extent = spot_half_extent_mm
        self.n_voxels = nx * ny
        self.D, self.spot_meta = self._cached_build(cache_dir)

    def vidx(self, ix, iy): return iy * self.nx + ix

    def _params(self):
        return dict(nx=self.nx, ny=self.ny, voxel_mm=self.voxel_mm,
                    lateral_spacing=self.lateral_spacing, n_layers=self.n_layers,
                    lateral_sigma=self.lateral_sigma,
                    spot_half_extent=self.spot_half_extent)

    def _cached_build(self, cache_dir):
        if cache_dir is None:
            return self._build_influence()
        os.makedirs(cache_dir, exist_ok=True)
        key = hashlib.md5(json.dumps(self._params(), sort_keys=True).encode()).hexdigest()[:12]
        npz, meta = os.path.join(cache_dir, f"infl_{key}.npz"), os.path.join(cache_dir, f"infl_{key}.json")
        if os.path.exists(npz) and os.path.exists(meta):
            D = sparse.load_npz(npz)
            with open(meta) as f: sm = json.load(f)
            return D, sm
        D, sm = self._build_influence()
        sparse.save_npz(npz, D)
        with open(meta, "w") as f: json.dump(sm, f)
        return D, sm

    def _ray(self, beam, lateral_idx):
        nx, ny = self.nx, self.ny
        sig = self.lateral_sigma / self.voxel_mm
        spread = int(np.ceil(3*sig))
        cols = []
        if beam in ('L', 'R'):
            depth_vox = np.arange(nx)
            depth_mm = (depth_vox + 0.5) * self.voxel_mm
            for d in depth_vox:
                ix = d if beam == 'L' else (nx-1-d)
                lat = [(ix, lateral_idx+off, np.exp(-0.5*(off/sig)**2))
                       for off in range(-spread, spread+1) if 0 <= lateral_idx+off < ny]
                cols.append([(a, b, w) for (a, b, w) in lat])
        else:
            depth_vox = np.arange(ny)
            depth_mm = (depth_vox + 0.5) * self.voxel_mm
            for d in depth_vox:
                iy = d if beam == 'B' else (ny-1-d)
                lat = [(lateral_idx+off, iy, np.exp(-0.5*(off/sig)**2))
                       for off in range(-spread, spread+1) if 0 <= lateral_idx+off < nx]
                cols.append([(a, b, w) for (a, b, w) in lat])
        return depth_mm, cols

    def _band_voxels(self, axis_len_vox):
        c = axis_len_vox * self.voxel_mm / 2.0
        lo = max(int((c - self.spot_half_extent)/self.voxel_mm), 0)
        hi = min(int((c + self.spot_half_extent)/self.voxel_mm), axis_len_vox-1)
        step = max(int(round(self.lateral_spacing/self.voxel_mm)), 1)
        return range(lo, hi+1, step)

    def _build_influence(self):
        nx, ny = self.nx, self.ny
        c_depth = max(nx, ny) * self.voxel_mm / 2.0
        peak_depths = np.linspace(c_depth - self.spot_half_extent,
                                  c_depth + self.spot_half_extent, self.n_layers)
        energies = energy_of_range(peak_depths)
        rows, cidx, vals, meta, col = [], [], [], [], 0
        for beam in ('L', 'R', 'B', 'T'):
            lat_axis = ny if beam in ('L', 'R') else nx
            for lat in self._band_voxels(lat_axis):
                depth_mm, ray = self._ray(beam, lat)
                for R0, E in zip(peak_depths, energies):
                    dd = depth_dose(depth_mm, R0, straggle_for_range(R0))
                    for d in np.where(dd > 1e-4)[0]:
                        for (ix, iy, lw) in ray[d]:
                            v = dd[d]*lw
                            if v > 1e-5:
                                rows.append(self.vidx(ix, iy)); cidx.append(col); vals.append(v)
                    meta.append(dict(beam=beam, lateral=int(lat), energy=float(E), peak_mm=float(R0)))
                    col += 1
        D = sparse.csr_matrix((vals, (rows, cidx)), shape=(self.n_voxels, col))
        return D, meta

    @property
    def n_spots(self): return self.D.shape[1]
