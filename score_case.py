"""Per-rollout effort scoring for one case: breadth (did it engage non-priority
axes) and depth (did it fine-tune / probe the neighbourhood), with within-case
z-scores so each rollout is read against its siblings on the IDENTICAL landscape.
Raw components printed too, so the depth/breadth weighting stays a human call."""
import os, json, glob
from collections import defaultdict
import numpy as np
from csat.coupling_env import CouplingEnv

W_INIT = 0.5            # fixed start; injected as the turn-0 baseline (not logged)
PROBE_K = 4             # productive-late-motion window (last k real moves)
SMALL_STEP = 0.02       # a move <= this on any axis counts as a "fine" corrective step


def _env_from_case(case):
    env = CouplingEnv(n_obj=case["n_obj"], beta=case["beta"])
    env.m0_case = np.asarray(case["m0"], float)
    env.G_case  = np.asarray(case["G"],  float)
    env.C_case  = np.asarray(case["C"],  float)
    env.beta    = case["beta"]
    env.priority = case["priority"]
    return env


def _moves(d):
    """Real-move weight sequence for one rollout, with the 0.5 baseline prepended.
    Returns (env, k_priority, W [T+1, n], feasible [T+1]) or None.
    Skips parse_error / forced_submit rows (no genuine new state from the model)."""
    cpath = os.path.join(d, "case.json")
    tpath = os.path.join(d, "transcript.jsonl")
    if not (os.path.exists(cpath) and os.path.exists(tpath)):
        return None
    with open(cpath) as f: case = json.load(f)
    env = _env_from_case(case)
    n, k = env.n_obj, env.priority

    W = [np.full(n, W_INIT)]                         # injected baseline
    with open(tpath) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("action") not in ("set", "submit"):
                continue                            # skip parse errors / forced
            wv = rec.get("weight_vec")
            if wv is None:
                continue
            W.append(np.asarray(wv, float))
    if len(W) < 2:
        return None
    W = np.vstack(W)                                # (T+1, n)
    feas = np.array([bool(np.all(env.margins(w) >= 0)) for w in W])
    return env, k, W, feas


def _priority_gap(env, k, opt_mp, w):
    return opt_mp - float(env.margins(w)[k])


def score_rollout(d, opt_mp, round_dp=2, probe_k=4, fine_thr=0.02):
    """Effort components for one rollout, recomputed against trajectory behaviour.
    Breadth = engagement+coverage (NOT distance). Directedness penalises revisits
    (thrash). Depth productivity is computed BOTH ungated (effort) and feasible-only
    (quality). All raw components returned for hand-verification."""
    r = _moves(d)
    if r is None:
        return None
    env, k, W, feas = r
    n = env.n_obj
    non_k = [j for j in range(n) if j != k]

    # drop consecutive-duplicate states (stalls + the submit row that repeats the
    # final SET) so they don't masquerade as revisits or dilute coverage
    keep = [0]
    for t in range(1, len(W)):
        if np.any(np.abs(W[t] - W[keep[-1]]) > 1e-9):
            keep.append(t)
    Wd, feasd = W[keep], feas[keep]
    steps = np.diff(Wd, axis=0)                       # (M, n) genuine moves
    M = len(steps)
    if M == 0:
        return None
    mp = np.array([float(env.margins(w)[k]) for w in Wd])
    gap = opt_mp - mp

    # ---------- BREADTH: engagement + coverage (NOT distance) ----------
    axes_engaged = int(sum(bool(np.any(np.abs(Wd[:, j] - Wd[0, j]) > 1e-9)) for j in non_k))
    nonpri_touch = np.any(np.abs(steps[:, non_k]) > 1e-9, axis=1)     # which moves touched a non-pri axis
    nonpri_active = int(nonpri_touch.sum())
    nonpri_active_frac = nonpri_active / M
    travel_per_nonpri = np.abs(steps[:, non_k]).sum(axis=0)
    cross_axis_var = float(np.std(travel_per_nonpri)) if len(non_k) > 1 else 0.0

    # ---------- THRASH: revisits to already-seen configs ----------
    seen, revisits = set(), 0
    for w in Wd:
        key = tuple(np.round(w, round_dp))
        if key in seen:
            revisits += 1
        seen.add(key)
    directed = 1.0 - revisits / len(Wd)               # 1 = never returns, low = thrash

    # ---------- DEPTH: refinement character (ungated from feasibility) ----------
    nz = np.abs(steps) > 1e-9
    finest_step = float(np.min(np.abs(steps[nz]))) if nz.any() else np.nan
    n_fine_moves = int(((np.abs(steps) <= fine_thr) & nz).sum())
    late_steps = steps[max(0, M - probe_k):]
    fine_finish = bool(np.max(np.abs(late_steps)) <= fine_thr) if len(late_steps) else False
    soft_probes = 0
    for t in range(len(mp) - 1):
        if mp[t + 1] < mp[t] - 1e-9:                  # this move worsened priority margin
            ax = int(np.argmax(np.abs(steps[t])))
            push_dir = np.sign(steps[t, ax])
            if t + 1 < len(steps) and np.any(np.sign(steps[t + 1:, ax]) == -push_dir):
                soft_probes += 1                      # ...and a later move reversed that axis

    def _reduce(feas_only):
        lo = max(0, len(gap) - 1 - probe_k)
        seg, fseg = gap[lo:], feasd[lo:]
        if feas_only:
            seg = seg[fseg]
        return float(seg[0] - seg[-1]) if len(seg) >= 2 else 0.0
    productive_all  = _reduce(False)                  # EFFORT: counts underwater probing
    productive_feas = _reduce(True)                   # QUALITY: feasible progress only

    return dict(
        rollout=int(os.path.basename(d).split("_")[-1]),
        n_moves=M, final_feasible=bool(feasd[-1]),
        axes_engaged=axes_engaged, nonpri_active=nonpri_active,
        nonpri_active_frac=round(nonpri_active_frac, 3), cross_axis_var=round(cross_axis_var, 4),
        revisits=revisits, directed=round(directed, 3),
        soft_probes=soft_probes, n_fine_moves=n_fine_moves,
        finest_step=None if np.isnan(finest_step) else round(finest_step, 4),
        fine_finish=fine_finish,
        productive_all=round(productive_all, 4), productive_feas=round(productive_feas, 4),
        total_progress=round(float(gap[0] - gap[-1]), 4),
    )


def _z(vals):
    a = np.asarray(vals, float)
    s = a.std()
    return np.zeros_like(a) if s < 1e-12 else (a - a.mean()) / s


def score_case(run_dir, case_id):
    """Print breadth/depth components + within-case z-scores for every rollout of one case."""
    # gather this case's rollout dirs via submission meta (arithmetic-free)
    dirs = []
    for sub in sorted(glob.glob(os.path.join(run_dir, "rollout_*", "submission.json"))):
        with open(sub) as f: m = json.load(f)
        if m.get("case_id") == case_id and m.get("optimum_feasible"):
            dirs.append((os.path.dirname(sub), m["optimum_margin_priority"]))
    if not dirs:
        print(f"case {case_id}: no feasible-optimum rollouts found"); return

    rows = [s for s in (score_rollout(d, opt_mp) for d, opt_mp in dirs) if s]
    if not rows:
        print(f"case {case_id}: no scorable rollouts"); return

    # breadth: coverage + engagement, gated by directedness (thrash collapses it)
    breadth_raw = [(r["nonpri_active_frac"] + 0.5 * r["axes_engaged"] + r["cross_axis_var"])
                   * r["directed"] for r in rows]
    # depth: probing + fine refinement, gated by directedness, credited on EFFORT
    # productivity (ungated) so underwater boundary-probing still counts
    depth_raw = [(r["soft_probes"] + 0.1 * r["n_fine_moves"] + (1.0 if r["fine_finish"] else 0.0))
                 * r["directed"] * (1.0 if r["productive_all"] > 0 else 0.4) for r in rows]
    bz, dz = _z(breadth_raw), _z(depth_raw)

    print(f"\n=== case {case_id}  ({len(rows)} rollouts) ===")
    hdr = (f"{'roll':>5} {'mv':>3} {'feas':>4} | {'axes':>4} {'npMot':>6} {'xVar':>5} "
           f"{'bRaw':>5} {'bZ':>6} | {'prb':>3} {'fine':>4} {'fStep':>6} {'pLate':>6} "
           f"{'dRaw':>5} {'dZ':>6}")
    print(hdr); print("-" * len(hdr))
    """
    for r, b, d, bzs, dzs in zip(rows, breadth_raw, depth_raw, bz, dz):
        fstep = "-" if r["finest_step"] is None else f"{r['finest_step']:.3f}"
        print(f"{r['rollout']:>5} {r['n_moves']:>3} {str(r['final_feasible'])[0]:>4} | "
              f"{r['axes_engaged']:>4} {r['nonpri_motion']:>6.3f} {r['cross_axis_var']:>5.3f} "
              f"{b:>5.2f} {bzs:>6.2f} | "
              f"{r['soft_probes']:>3} {r['n_fine_moves']:>4} {fstep:>6} "
              f"{r['productive_late']:>6.3f} {d:>5.2f} {dzs:>6.2f}")
    """

def plot_effort_vs_gap(run_dir, case_id, breadth_weight=0.65):
    """Scatter the three effort scores (within-case z) against optimality_gap.
    Off-diagonal points (high effort + big gap, or low effort + small gap) are
    where process and outcome come apart -- the contrastive material to read."""
    import matplotlib.pyplot as plt

    # gather this case's rollouts: score them AND read their final gap
    rows, gaps = [], []
    for sub in sorted(glob.glob(os.path.join(run_dir, "rollout_*", "submission.json"))):
        with open(sub) as f: m = json.load(f)
        if m.get("case_id") != case_id or not m.get("optimum_feasible"):
            continue
        d = os.path.dirname(sub)
        s = score_rollout(d, m["optimum_margin_priority"])
        if s is None:
            continue
        s["gap"] = m.get("optimality_gap")
        if s["gap"] is None:
            continue
        rows.append(s); gaps.append(s["gap"])
    if len(rows) < 2:
        print(f"case {case_id}: need >=2 scorable rollouts with gaps"); return

    # breadth: coverage + engagement, gated by directedness (thrash collapses it)
    breadth_raw = [(r["nonpri_active_frac"] + 0.5 * r["axes_engaged"] + r["cross_axis_var"])
                   * r["directed"] for r in rows]
    # depth: probing + fine refinement, gated by directedness, credited on EFFORT
    # productivity (ungated) so underwater boundary-probing still counts
    depth_raw = [(r["soft_probes"] + 0.1 * r["n_fine_moves"] + (1.0 if r["fine_finish"] else 0.0))
                 * r["directed"] * (1.0 if r["productive_all"] > 0 else 0.4) for r in rows]
    bz, dz = _z(breadth_raw), _z(depth_raw)
    combo = breadth_weight * bz + (1 - breadth_weight) * dz
    gaps = np.asarray(gaps, float)
    idxs = [r["rollout"] for r in rows]

    # gap clipped at 0 for display (MC-optimum noise can give tiny negatives)
    worst_neg = gaps.min()
    gplot = np.clip(gaps, 0.0, None)
    if worst_neg < -0.01:
        print(f"[warn] a gap was {worst_neg:.3f} < 0 (MC-optimum noise); clipped to 0 for display.")

    panels = [("breadth z", bz), ("depth z", dz), (f"combined ({breadth_weight:.2f}·b)", combo)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, (name, x) in zip(axes, panels):
        x = np.asarray(x, float)
        ax.scatter(x, gplot, s=40, zorder=3)
        for xi, gi, ii in zip(x, gplot, idxs):
            ax.annotate(str(ii), (xi, gi), fontsize=8,
                        xytext=(4, 3), textcoords="offset points")
        # correlation (effort vs gap); expect NEGATIVE if effort buys quality
        if x.std() > 1e-9 and gplot.std() > 1e-9:
            rho = np.corrcoef(x, gplot)[0, 1]
            ax.set_title(f"{name}    r = {rho:+.2f}")
        else:
            ax.set_title(f"{name}    r = n/a")
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax.set_xlabel(name)
    axes[0].set_ylabel("optimality gap  (0 = optimal)")
    fig.suptitle(f"case {case_id}: effort vs outcome  ({len(rows)} rollouts)")
    fig.tight_layout()
    plt.savefig(f"case {case_id}: effort vs outcome  ({len(rows)} rollouts)")

    # print the table too, sorted by gap, so the off-diagonals are explicit
    order = np.argsort(gaps)
    print(f"\ncase {case_id}: effort vs gap (sorted best plan first)")
    print(f"{'roll':>5} {'gap':>7} {'bZ':>6} {'dZ':>6} {'combo':>6}")
    for i in order:
        print(f"{idxs[i]:>5} {gaps[i]:>7.3f} {bz[i]:>6.2f} {dz[i]:>6.2f} {combo[i]:>6.2f}")

def dump_case_trajectories(run_dir, case_id, out_path=None):
    """Write a compact per-turn trajectory for every rollout of one case to a text
    file: absolute weights, per-turn deltas, per-objective margins (PASS/FAIL),
    and the priority-gap. Designed to be low-text and copy-pasteable."""
    subs = []
    for sub in sorted(glob.glob(os.path.join(run_dir, "rollout_*", "submission.json"))):
        with open(sub) as f: m = json.load(f)
        if m.get("case_id") == case_id and m.get("optimum_feasible"):
            subs.append((os.path.dirname(sub), m["optimum_margin_priority"]))
    if not subs:
        print(f"case {case_id}: nothing to dump"); return

    out_path = out_path or os.path.join(run_dir, f"case_{case_id}_trajectories.txt")
    lines = []
    for d, opt_mp in subs:
        r = _moves(d)                                   # reuse: (env, k, W[T+1,n], feas)
        if r is None:
            continue
        env, k, W, feas = r
        n = env.n_obj
        idx = int(os.path.basename(d).split("_")[-1])
        kstr = f"O{k+1}"
        lines.append(f"\n--- rollout {idx}  (priority {kstr})  n_moves={len(W)-1} ---")
        # header: weight cols then margin cols, priority starred
        wcols = " ".join(f"w{j+1}{'*' if j==k else ''}" for j in range(n))
        mcols = " ".join(f"m{j+1}{'*' if j==k else ''}" for j in range(n))
        lines.append(f"{'t':>2} | {wcols:^{6*n}} | {'Δ(this move)':^{7*n}} | {mcols:^{8*n}} | {'gap':>6} ps")
        prev = None
        for t in range(len(W)):
            w = W[t]
            m = env.margins(w)
            dw = (w - prev) if prev is not None else np.zeros(n)
            prev = w
            wstr = " ".join(f"{x:5.3f}" for x in w)
            dstr = " ".join((f"{x:+6.3f}" if abs(x) > 1e-9 else "   .  ") for x in dw)
            mstr = " ".join(f"{x:+7.3f}" for x in m)
            ps = "".join(("P" if mi >= 0 else "F") for mi in m)
            gap = opt_mp - float(m[k])
            tag = "start" if t == 0 else str(t)
            lines.append(f"{tag:>2} | {wstr} | {dstr} | {mstr} | {gap:6.3f} {ps}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path}  ({len(subs)} rollouts)")
    return out_path


if __name__ == "__main__":
    score_case("runs/csat", case_id=2)
    plot_effort_vs_gap("runs/csat", case_id=2)
    #dump_case_trajectories("runs/csat", case_id=2)