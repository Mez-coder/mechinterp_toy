"""Material / density grid for the 2D phantom.

Kept deliberately separate from the structure masks (CTV/OAR) so heterogeneous
tissue can be dropped in later WITHOUT touching the ray tracer: the tracer only
ever asks `DensityMap.sample(points_mm)` for a relative electron/mass density,
and the Bethe-Bloch stopping power scales linearly with it. Today the only two
materials are air (0.0) outside the phantom circle and water (1.0) inside it.

To add lung/bone later you only edit `build()` (or hand a custom array to the
constructor); nothing downstream changes.
"""
from __future__ import annotations
import numpy as np


class DensityMap:
    """Relative mass density on the voxel grid, sampled in physical mm.

    Coordinate convention (shared across the whole package):
      * world origin (0,0) mm sits at the GEOMETRIC CENTRE of the grid,
      * +x -> increasing column index, +y -> increasing row index,
      * arrays are indexed [iy, ix] with shape (ny, nx) and flattened C-order,
        so the flat voxel index is  v = iy * nx + ix.
    """

    AIR = 0.0
    WATER = 1.0

    def __init__(self, nx: int, ny: int, voxel_mm: float, radius_mm: float,
                 grid: np.ndarray | None = None):
        self.nx, self.ny, self.voxel_mm = nx, ny, voxel_mm
        self.radius_mm = radius_mm
        self.grid = grid if grid is not None else self.build()

    # ---- physical <-> voxel transforms ------------------------------------
    def world_of_voxel(self, ix, iy):
        """Centre of voxel (ix,iy) in mm (origin at grid centre)."""
        wx = (np.asarray(ix) + 0.5 - self.nx / 2.0) * self.voxel_mm
        wy = (np.asarray(iy) + 0.5 - self.ny / 2.0) * self.voxel_mm
        return wx, wy

    def voxel_of_world(self, wx, wy):
        """Nearest voxel (ix,iy) for world points in mm (vectorised, clipped)."""
        ix = np.floor(np.asarray(wx) / self.voxel_mm + self.nx / 2.0).astype(np.int64)
        iy = np.floor(np.asarray(wy) / self.voxel_mm + self.ny / 2.0).astype(np.int64)
        np.clip(ix, 0, self.nx - 1, out=ix)
        np.clip(iy, 0, self.ny - 1, out=iy)
        return ix, iy

    def flat_of_world(self, wx, wy):
        ix, iy = self.voxel_of_world(wx, wy)
        return iy * self.nx + ix

    # ---- material model ----------------------------------------------------
    def build(self) -> np.ndarray:
        """Air outside the centred circle, water inside. Override for tissue."""
        ys, xs = np.mgrid[0:self.ny, 0:self.nx]
        wx, wy = self.world_of_voxel(xs, ys)
        inside = (wx ** 2 + wy ** 2) <= self.radius_mm ** 2
        g = np.full((self.ny, self.nx), self.AIR, dtype=np.float64)
        g[inside] = self.WATER
        return g

    def sample(self, wx, wy):
        """Nearest-neighbour density at world points (mm). 0.0 outside the grid."""
        wx = np.asarray(wx); wy = np.asarray(wy)
        ix = np.floor(wx / self.voxel_mm + self.nx / 2.0).astype(np.int64)
        iy = np.floor(wy / self.voxel_mm + self.ny / 2.0).astype(np.int64)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        out = np.zeros(np.broadcast(wx, wy).shape, dtype=np.float64)
        out[ok] = self.grid[iy[ok], ix[ok]]
        return out

    @property
    def n_voxels(self):
        return self.nx * self.ny
