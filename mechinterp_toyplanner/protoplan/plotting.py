"""Quick visual sanity checks for protoplan.

    from protoplan.plotting import plot_dose, plot_dvh
    plot_dose(env)        # dose map + structure outlines + beam arrows + isodoses
    plot_dvh(env)         # dose-volume histogram for every structure

Both take an env whose .dose is already populated (call env.optimise(), or just
env.reset(), first). Needs matplotlib:  uv run --with matplotlib ...
Dose voxels are raveled iy*nx+ix, so we reshape to (ny, nx).
"""
from __future__ import annotations
import numpy as np

_OAR_COLORS = ['#00d5ff', '#ff4dd2', '#ffe600', '#6cff5b', '#ff8c00', '#b073ff']


def _grids(env):
    ny, nx, vox = env.geom.ny, env.geom.nx, env.geom.voxel_mm
    dose = env.dose.reshape(ny, nx)
    Y, X = np.mgrid[0:ny, 0:nx]
    return ny, nx, vox, dose, (X + 0.5) * vox, (Y + 0.5) * vox


def plot_dose(env, ax=None, title=None, isodose=(0.5, 0.95, 1.05),
              beams=True, cmap='turbo'):
    """Dose heatmap (% of Rx) with structure contours, beam arrows, isodose lines."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ny, nx, vox, dose, Xmm, Ymm = _grids(env)
    dose_pct = 100.0 * dose / env.Rx
    extent = [0, nx * vox, 0, ny * vox]
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6.2, 5.6))

    im = ax.imshow(dose_pct, origin='lower', extent=extent, cmap=cmap,
                   vmin=0, vmax=max(110.0, float(dose_pct.max())))
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('% of prescription')

    handles, oi = [], 0
    for name, mask in env.structures.items():
        if name == 'CTV':
            c, lw = 'white', 2.2
        else:
            c, lw = _OAR_COLORS[oi % len(_OAR_COLORS)], 1.6
            oi += 1
        ax.contour(Xmm, Ymm, mask.astype(float), levels=[0.5], colors=[c], linewidths=lw)
        lbl = name + (f"  ({env.oar_metric[name]})" if name in getattr(env, 'oar_metric', {}) else "")
        handles.append(Line2D([0], [0], color=c, lw=lw, label=lbl))

    for frac in isodose:
        ax.contour(Xmm, Ymm, dose_pct, levels=[frac * 100], colors=['k'],
                   linewidths=0.7, linestyles='--', alpha=0.55)

    if beams:
        L = nx * vox
        cx, cy = nx * vox / 2, ny * vox / 2
        for (x, y, dx, dy) in [(0, cy, 1, 0), (L, cy, -1, 0), (cx, 0, 0, 1), (cx, L, 0, -1)]:
            ax.annotate('', xy=(x + dx * 0.13 * L, y + dy * 0.13 * L), xytext=(x, y),
                        arrowprops=dict(arrowstyle='-|>', color='red', lw=1.8))

    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    ax.set_title(title or 'Dose (dashed = isodose; red = beam entry)')
    ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.75)
    if own:
        fig.tight_layout()
    return ax


def plot_dvh(env, ax=None, bins=200):
    """Cumulative dose-volume histogram for every structure."""
    import matplotlib.pyplot as plt
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6.2, 4.6))

    edges = np.linspace(0, max(env.dose.max(), env.Rx * 1.1), bins)
    oi = 0
    for name, mask in env.structures.items():
        d = env.dose[mask.ravel()]
        if d.size == 0:
            continue
        vol = np.array([(d >= e).mean() for e in edges]) * 100.0
        if name == 'CTV':
            c, lw = 'black', 2.2
        else:
            c, lw = _OAR_COLORS[oi % len(_OAR_COLORS)], 1.6
            oi += 1
        ax.plot(edges, vol, color=c, lw=lw, label=name)

    ax.axvline(env.Rx, color='grey', ls=':', lw=1, label='Rx')
    ax.set_xlabel('Dose (Gy)'); ax.set_ylabel('Volume (%)')
    ax.set_title('DVH'); ax.set_ylim(0, 100.5); ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    if own:
        fig.tight_layout()
    return ax
