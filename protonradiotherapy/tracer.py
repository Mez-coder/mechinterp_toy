"""Ray-tracing influence builder for one beam angle.

Pipeline per angle theta (exactly the spec):
  1. entry point on the phantom circle, beam direction toward centre, and the
     perpendicular axis along which we lay a parallel array of pencil beams
     (2 mm spacing by default);
  2. cull pencils whose ray cannot cross the target (lateral projection test);
  3. for each surviving pencil, sweep monoenergetic protons over a geometry-
     derived energy band (Emin..Emax in ~3 MeV steps);
  4. march each ray in 2 mm steps, depositing -dE/dx*step via the Bethe-Bloch
     StoppingModel; density is the mean of the step's endpoints; a step that
     crosses a voxel boundary splits its energy equally between the two voxels;
     marching stops when E reaches 0;
  5. find each ray's Bragg peak (voxel of maximum deposited energy). Keep the ray
     (store its deposition as an influence column) iff that voxel lies INSIDE the
     target mask; otherwise discard it.

The march is vectorised over (pencils x energies); only the depth loop (<=~150
2 mm steps) is in Python. The returned influence matrix is sparse
(n_voxels x n_kept_rays); the SFO/MFO optimiser then solves pencil weights on it.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter1d

from .density import DensityMap
from .stopping import StoppingModel


class BeamTracer:
    def __init__(self, density: DensityMap, stopping: StoppingModel,
                 march_mm=2.0, spacing_mm=2.0, energy_step_mev=3.0,
                 energy_margin_mm=10.0, dep_eps=1e-6,
                 straggle_frac=0.012, lateral_sigma_mm=0.0):
        self.den = density
        self.stop = stopping
        self.march_mm = march_mm
        self.spacing_mm = spacing_mm
        self.e_step = energy_step_mev
        self.e_margin = energy_margin_mm
        self.dep_eps = dep_eps
        # range straggling broadens the sharp monoenergetic Bragg peak (sigma ~
        # straggle_frac of the range, physically ~1.2%); fills the depth gaps the
        # coarse ~3 MeV energy sampling would otherwise leave between peaks.
        self.straggle_frac = straggle_frac
        # optional lateral penumbra (mm); 0 = thin pencil (one voxel column).
        self.lateral_sigma_mm = lateral_sigma_mm
        self.R = density.radius_mm

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _dirs(theta):
        c, s = np.cos(theta), np.sin(theta)
        u = np.array([-c, -s])           # entry (R c, R s) -> centre
        u_perp = np.array([s, -c])       # perpendicular axis (unit)
        return u, u_perp

    def _target_projection(self, target_mask, u, u_perp):
        """Along-beam (s) and lateral (o) coordinate ranges of the target."""
        iy, ix = np.nonzero(target_mask)
        wx, wy = self.den.world_of_voxel(ix, iy)
        s = wx * u[0] + wy * u[1]               # depth coord (centre = 0)
        o = wx * u_perp[0] + wy * u_perp[1]     # lateral coord
        return (float(s.min()), float(s.max())), (float(o.min()), float(o.max()))

    def _energies(self, s_range):
        """Geometry-derived Emin/Emax: ranges to reach the near/far target edge
        on a central pencil (entry depth = R), padded by a margin."""
        s_lo, s_hi = s_range
        r_near = max(self.R + s_lo - self.e_margin, 1.0)
        r_far = self.R + s_hi + self.e_margin
        e_lo = float(self.stop.energy_of_range(r_near))
        e_hi = float(self.stop.energy_of_range(r_far))
        if e_hi <= e_lo:
            e_hi = e_lo + self.e_step
        return np.arange(e_lo, e_hi + self.e_step, self.e_step)

    # ----------------------------------------------------------------- tracing
    def trace_angle(self, theta, target_mask, target_flat):
        """Return (D_angle [n_voxels x n_rays] CSC, ray_meta list). May be empty."""
        u, u_perp = self._dirs(theta)
        s_range, o_range = self._target_projection(target_mask, u, u_perp)

        # --- pencil offsets across the target's lateral shadow (+ margin) ----
        pad = 2 * self.spacing_mm
        o0 = o_range[0] - pad
        o1 = o_range[1] + pad
        offsets = np.arange(o0, o1 + self.spacing_mm, self.spacing_mm)
        offsets = offsets[np.abs(offsets) < self.R - 1e-6]      # must hit phantom
        if offsets.size == 0:
            return sparse.csc_matrix((self.den.n_voxels, 0)), []

        energies = self._energies(s_range)
        J, Ne = offsets.size, energies.size

        # --- per-pencil entry point and depth grid ---------------------------
        half_chord = np.sqrt(self.R ** 2 - offsets ** 2)         # [J]
        # entry = o*u_perp - half_chord*u ; march +u in march_mm steps
        K = int(np.ceil(2 * self.R / self.march_mm)) + 2
        d = np.arange(K) * self.march_mm                         # [K] depth from entry
        # world points P[j,k] = entry_j + d_k * u
        ex = offsets * u_perp[0] - half_chord * u[0]             # [J]
        ey = offsets * u_perp[1] - half_chord * u[1]
        Px = ex[:, None] + d[None, :] * u[0]                     # [J,K]
        Py = ey[:, None] + d[None, :] * u[1]
        rho = self.den.sample(Px, Py)                            # [J,K]
        vox = self.den.flat_of_world(Px, Py)                     # [J,K] flat idx

        # mark steps still inside the phantom (depth < chord)
        inside = d[None, :] <= (2 * half_chord)[:, None]         # [J,K]

        # --- vectorised Bethe-Bloch march over (pencils x energies) ----------
        E = np.repeat(energies[None, :], J, axis=0).astype(np.float64)   # [J,Ne]
        seg_rho = 0.5 * (rho[:, :-1] + rho[:, 1:])               # [J,K-1]
        dep = np.zeros((J, Ne, K - 1), dtype=np.float64)         # energy lost per step
        for k in range(K - 1):
            rho_k = seg_rho[:, k][:, None]                       # [J,1]
            live = inside[:, k][:, None] & (E > 0)
            dEdx = self.stop.stopping_power(E, rho_k)            # [J,Ne]
            dE = np.minimum(dEdx * self.march_mm, E)
            dE = np.where(live, dE, 0.0)
            dep[:, :, k] = dE
            E -= dE

        # --- range straggling: broaden each ray's depth-deposition profile ----
        if self.straggle_frac > 0:
            for e in range(Ne):
                R0 = float(self.stop.range_of_energy(energies[e]))
                sig_steps = max(1.0, self.straggle_frac * R0) / self.march_mm
                dep[:, e, :] = gaussian_filter1d(dep[:, e, :], sig_steps, axis=1,
                                                 mode='constant')

        # --- Bragg peak per ray and target filter ----------------------------
        kstar = np.argmax(dep, axis=2)                           # [J,Ne] peak step
        jj = np.arange(J)[:, None]
        bragg_vox = vox[jj, kstar]                               # [J,Ne] start voxel of peak step
        # voxel boundaries of each step (for equal-split deposition)
        vs0 = vox[:, :-1]                                        # [J,K-1]
        vs1 = vox[:, 1:]
        in_target = np.isin(bragg_vox, target_flat)             # [J,Ne]
        # require a non-trivial peak (avoid all-zero rays)
        peakval = np.max(dep, axis=2)
        keep = in_target & (peakval > self.dep_eps)
        kj, ke = np.nonzero(keep)
        if kj.size == 0:
            return sparse.csc_matrix((self.den.n_voxels, 0)), []

        # --- assemble sparse influence columns for kept rays -----------------
        # lateral penumbra fan (along u_perp); single point if sigma==0
        u_p = u_perp
        if self.lateral_sigma_mm > 0:
            half = int(np.ceil(3 * self.lateral_sigma_mm / self.den.voxel_mm))
            ks = np.arange(-half, half + 1) * self.den.voxel_mm
            gk = np.exp(-0.5 * (ks / self.lateral_sigma_mm) ** 2)
            gk /= gk.sum()
        else:
            ks = np.array([0.0]); gk = np.array([1.0])
        Mx = 0.5 * (Px[:, :-1] + Px[:, 1:])                # step midpoints [J,K-1]
        My = 0.5 * (Py[:, :-1] + Py[:, 1:])

        rows, cols, vals = [], [], []
        ray_meta = []
        for col, (j, e) in enumerate(zip(kj.tolist(), ke.tolist())):
            dk = dep[j, e]                                  # [K-1]
            nz = np.nonzero(dk > self.dep_eps)[0]
            if nz.size == 0:
                continue
            # fan points: [m, nk]  (lateral spread of each marched step)
            Xp = Mx[j, nz][:, None] + ks[None, :] * u_p[0]
            Yp = My[j, nz][:, None] + ks[None, :] * u_p[1]
            Vp = dk[nz][:, None] * gk[None, :]
            r, v = self._bilinear(Xp.ravel(), Yp.ravel(), Vp.ravel())
            rows.append(r); cols.append(np.full(r.size, col)); vals.append(v)
            ray_meta.append(dict(angle=float(theta), offset=float(offsets[j]),
                                 energy=float(energies[e]),
                                 bragg_vox=int(bragg_vox[j, e])))
        if not rows:
            return sparse.csc_matrix((self.den.n_voxels, 0)), []
        rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
        n_rays = len(ray_meta)
        D = sparse.csc_matrix((vals, (rows, cols)),
                              shape=(self.den.n_voxels, n_rays))
        D.sum_duplicates()
        return D, ray_meta

    def _bilinear(self, wx, wy, val):
        """Area-weighted (bilinear) splat of point energies onto the 4 nearest
        voxels. Angle-independent -- avoids the diagonal striping that
        nearest-voxel sampling produces. Out-of-grid weight is dropped (air)."""
        nx, ny, vox = self.den.nx, self.den.ny, self.den.voxel_mm
        cx = wx / vox + nx / 2.0 - 0.5                      # continuous voxel coords
        cy = wy / vox + ny / 2.0 - 0.5
        ix0 = np.floor(cx).astype(np.int64); fx = cx - ix0
        iy0 = np.floor(cy).astype(np.int64); fy = cy - iy0
        rows, vals = [], []
        for dix, wxx in ((0, 1 - fx), (1, fx)):
            for diy, wyy in ((0, 1 - fy), (1, fy)):
                ixx = ix0 + dix; iyy = iy0 + diy
                ok = (ixx >= 0) & (ixx < nx) & (iyy >= 0) & (iyy < ny)
                rows.append((iyy * nx + ixx)[ok]); vals.append((val * wxx * wyy)[ok])
        return np.concatenate(rows), np.concatenate(vals)
