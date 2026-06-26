"""Phantom and dose rendering for the vision model (and for you to inspect).

Two products:
  * `phantom_png` -- the clean structure map: BLUE target, RED OARs, BLACK
    everywhere else. This is shown to the model at case presentation and saved.
  * `dose_png`    -- the current plan's dose wash (a perceptually-ordered
    colormap) with thin target/OAR contours overlaid, so the model can read its
    own plan each turn. The beam paths are implicit in the dose, so no arrows.

Images use the SAME [iy, ix] orientation as the masks, displayed with the world
+y axis pointing UP (origin = phantom centre).
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def render_phantom(structures, nx, ny, path):
    """Solid BLUE target / RED OAR / BLACK background -> save PNG."""
    img = np.zeros((ny, nx), dtype=np.int32)              # 0 background
    for name, m in structures.items():
        if name.startswith('OAR'):
            img[m] = 2                                     # red
    img[structures['CTV']] = 1                             # blue (drawn last)
    cmap = ListedColormap([(0, 0, 0), (0.15, 0.35, 1.0), (1.0, 0.15, 0.15)])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, cmap=cmap, vmin=0, vmax=2, origin='lower', interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("phantom: target (blue), OARs (red)")
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def render_dose(dose, structures, nx, ny, path, Rx=1.0, title=None):
    """Dose wash + structure contours -> save PNG."""
    d = np.asarray(dose).reshape(ny, nx)
    fig, ax = plt.subplots(figsize=(5, 5))
    vmax = max(float(d.max()), 1e-6)
    im = ax.imshow(d, cmap='turbo', vmin=0, vmax=max(vmax, 1.1 * Rx),
                   origin='lower', interpolation='nearest')
    # contours: target solid white, OARs dashed white
    ax.contour(structures['CTV'].astype(float), levels=[0.5],
               colors='white', linewidths=1.4)
    for name, m in structures.items():
        if name.startswith('OAR'):
            ax.contour(m.astype(float), levels=[0.5],
                       colors='white', linewidths=0.9, linestyles='dashed')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title or "current dose (target solid, OAR dashed)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("dose / Rx")
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
