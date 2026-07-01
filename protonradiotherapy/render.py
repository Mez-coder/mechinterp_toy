"""Phantom and dose rendering for the vision model (and for you to inspect).

Two products:
  * phantom (case start): BLUE target; each OAR in its OWN colour with a centroid
    label and a legend, so the model can tell OAR1 from OAR2 etc. The colour key
    is established here (the only turn the clean phantom is shown).
  * dose wash (every turn): the current plan's dose as a turbo wash, target drawn
    as a solid white contour and each OAR as a contour IN ITS OWN COLOUR (same
    palette as the phantom), so the mapping stays stable across turns. Dose is
    shown as % of the prescription (Rx).

Images use the [iy, ix] mask orientation with world +y up (origin = phantom
centre).
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# distinct, image-readable OAR colours (avoid blue -> reserved for the target)
OAR_PALETTE = [("green",   (0.20, 0.80, 0.25)),
               ("orange",  (1.00, 0.55, 0.05)),
               ("magenta", (1.00, 0.25, 0.85)),
               ("yellow",  (0.95, 0.90, 0.15)),
               ("cyan",    (0.25, 0.85, 0.95))]
TARGET_RGB = (0.15, 0.35, 1.0)


def oar_color(i):
    """(name, rgb) for OAR number i (1-based)."""
    return OAR_PALETTE[(i - 1) % len(OAR_PALETTE)]


def _oar_index(name):
    return int(name[3:]) if name[3:].isdigit() else 1


def _centroid(mask):
    iy, ix = np.nonzero(mask)
    return float(ix.mean()), float(iy.mean())


def render_phantom(structures, nx, ny, path):
    """Solid BLUE target / per-OAR colour / BLACK background, with labels+legend."""
    img = np.zeros((ny, nx, 3), dtype=float)               # black background
    handles = [Patch(facecolor=TARGET_RGB, label="target (CTV)")]
    for name, m in structures.items():
        if name.startswith('OAR'):
            cname, rgb = oar_color(_oar_index(name))
            img[m] = rgb
            handles.append(Patch(facecolor=rgb, label=f"{name} ({cname})"))
    img[structures['CTV']] = TARGET_RGB                    # target on top

    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.imshow(img, origin='lower', interpolation='nearest')
    for name, m in structures.items():
        if name.startswith('OAR'):
            cx, cy = _centroid(m)
            ax.text(cx, cy, name.replace('OAR', 'O'), color='black',
                    ha='center', va='center', fontsize=8, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("phantom: target (blue) + OARs (see legend)")
    ax.legend(handles=handles, loc='upper left', fontsize=7,
              facecolor='white', framealpha=0.85)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def render_dose(dose, structures, nx, ny, path, Rx=1.0, title=None):
    """Dose wash (% of Rx) + target (white) and per-OAR coloured contours."""
    d = np.asarray(dose).reshape(ny, nx) / Rx * 100.0      # % of Rx
    fig, ax = plt.subplots(figsize=(5.4, 5))
    im = ax.imshow(d, cmap='turbo', vmin=0, vmax=max(float(d.max()), 110.0),
                   origin='lower', interpolation='nearest')
    ax.contour(structures['CTV'].astype(float), levels=[0.5],
               colors='white', linewidths=1.5)
    handles = [Patch(facecolor='white', label="target (CTV)")]
    for name, m in structures.items():
        if name.startswith('OAR'):
            cname, rgb = oar_color(_oar_index(name))
            ax.contour(m.astype(float), levels=[0.5], colors=[rgb], linewidths=1.6)
            handles.append(Patch(facecolor=rgb, label=f"{name} ({cname})"))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title or "current dose (% of Rx)")
    ax.legend(handles=handles, loc='upper left', fontsize=7,
              facecolor='white', framealpha=0.85)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("% of Rx")
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path