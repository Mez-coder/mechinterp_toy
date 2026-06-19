"""Pick rollout seeds whose grid-optimum cells are SPREAD, not clustered.

The optimum is an output of the landscape, not a knob, so we can't place it
directly. Instead: scan candidate seeds, build the case, compute the exact grid
optimum (reuses CouplingEnv.optimum on the 0.1 grid), drop degenerate cases, and
accept a subset that flattens the histogram of optimum cells.

What "cell" we balance on: (priority weight rounded to grid, max non-priority
weight rounded to grid). Priority-w piles at the rail structurally (priority
margin is monotone in its own weight), so we cap how many rail-optimum cases we
keep -- the rest of the budget goes to off-rail (trap) optima, which is also
where DOUBT/converged behaviour is most informative.

Usage in rollout.py:
    from case_spread import pick_spread_seeds
    seeds = pick_spread_seeds(cfg, n_want=cfg.n_rollouts)
    for idx, seed in enumerate(seeds):
        env = build_env(cfg); env.reset(seed=seed, wide=True)
        ...   # run as before, but seed = seeds[idx]
"""
from __future__ import annotations
import numpy as np
from collections import defaultdict


def _opt_cell(env, grid=0.1):
    o = env.optimum()
    if not o.get("feasible") or o.get("margin_priority", 0) <= 1e-6:
        return None                                   # infeasible / trivial
    w = np.round(np.asarray(o["weights"], float) / grid) * grid
    k = env.priority
    return (round(float(w[k]), 2), round(float(np.delete(w, k).max()), 2)), o


def pick_spread_seeds(cfg, n_want, build_env, seed_start=0, scan=None,
                      per_cell_cap=None, rail_frac=0.30, grid=0.1):
    """Return a list of `n_want` seeds whose optimum cells are spread.
       build_env: callable(cfg) -> CouplingEnv  (pass rollout.build_env).
       rail_frac: max fraction of accepted cases whose priority-w optimum == 1.0.
       per_cell_cap: max cases per (priority-w, max-other-w) bin (default auto)."""
    scan = scan or max(20 * n_want, 2000)
    if per_cell_cap is None:
        per_cell_cap = max(2, n_want // 25)           # ~25 distinct cells target
    accepted, by_cell = [], defaultdict(int)
    n_rail = rail_cap = int(rail_frac * n_want)
    rail_used = 0

    for s in range(seed_start, seed_start + scan):
        env = build_env(cfg)
        env.reset(seed=s, wide=True)
        res = _opt_cell(env, grid)
        if res is None:
            continue
        cell, _ = res
        is_rail = abs(cell[0] - 1.0) < 1e-9
        if is_rail and rail_used >= rail_cap:          # cap rail-optimum cases
            continue
        if by_cell[cell] >= per_cell_cap:              # cap any one cell
            continue
        accepted.append(s); by_cell[cell] += 1
        rail_used += int(is_rail)
        if len(accepted) >= n_want:
            break

    if len(accepted) < n_want:
        print(f"[case_spread] only {len(accepted)}/{n_want} after scanning {scan} "
              f"seeds; raise `scan`, per_cell_cap, or rail_frac.")
    # report the spread we got
    cells = sorted(by_cell.items())
    print(f"[case_spread] accepted {len(accepted)} seeds across {len(cells)} optimum "
          f"cells; rail-optimum used {rail_used}/{rail_cap}")
    return accepted