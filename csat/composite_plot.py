"""composite_plot.py -- read the tidy per-turn data written by the composite study
and draw margin/gap-vs-turn, repeats overlaid per case. Fully decoupled from the
simulation: rerun this to retune the plot with no model and no re-sim.

    python -m csat.composite_plot --run-dir runs/csat_transfer_composite_sine_explore_final_set_L14
    python -m csat.composite_plot --run-dir <dir> --y gap --colour-by alpha --stars all

Data contract (written by transfer_studies.run_composite via write_data):
  composite_turns.csv  one row per (rollout, alpha, turn):
      case_id, rep, idx, seed, alpha, turn, action, margin, gap, all_pass,
      submitted, is_submit, submit_turn, n_turns, branch_turn, branch_kind,
      optimum_margin
  composite_meta.json  run-level: env, vector_type, inject_at, layer, frac, ...
"""
from __future__ import annotations
import os, json, csv, argparse
import numpy as np

FIELDS = ["case_id", "rep", "branch_rep", "idx", "seed", "alpha", "turn", "action",
          "margin", "gap", "proj", "proj_max", "n_steered", "all_pass", "submitted",
          "is_submit", "submit_turn", "n_turns", "branch_turn", "branch_kind",
          "optimum_margin"]


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def write_data(run_dir, rows, meta):
    """Write the tidy per-turn table + run meta. None -> empty cell."""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "composite_turns.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in FIELDS})
    with open(os.path.join(run_dir, "composite_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[data] wrote {path} ({len(rows)} rows) + composite_meta.json")
    return path


def _f(x):
    return float(x) if x not in ("", "None", None) else None


def _i(x):
    return int(float(x)) if x not in ("", "None", None) else None


def _b(x):
    return str(x).lower() in ("true", "1")


def load_turns(run_dir):
    meta = {}
    mp = os.path.join(run_dir, "composite_meta.json")
    if os.path.exists(mp):
        with open(mp) as f:
            meta = json.load(f)
    rows = []
    with open(os.path.join(run_dir, "composite_turns.csv")) as f:
        for d in csv.DictReader(f):
            rows.append(dict(
                case_id=_i(d["case_id"]), rep=_i(d["rep"]),
                branch_rep=(_i(d.get("branch_rep")) or 0),
                idx=_i(d["idx"]),
                seed=_i(d["seed"]), alpha=_f(d["alpha"]), turn=_i(d["turn"]),
                action=(d["action"] or None), margin=_f(d["margin"]), gap=_f(d["gap"]),
                proj=_f(d.get("proj")), proj_max=_f(d.get("proj_max")),
                n_steered=_i(d.get("n_steered")),
                all_pass=_b(d["all_pass"]), submitted=_b(d["submitted"]),
                is_submit=_b(d["is_submit"]), submit_turn=_i(d["submit_turn"]),
                n_turns=_i(d["n_turns"]), branch_turn=_i(d["branch_turn"]),
                branch_kind=(d["branch_kind"] or None),
                optimum_margin=_f(d["optimum_margin"])))
    return rows, meta


# --------------------------------------------------------------------------- #
# rebuild the tidy table from per-rollout dirs (durability / resume / crash)
# --------------------------------------------------------------------------- #
def _meta_from_dirname(run_dir):
    """Recover env / vector / inject / layer from the run-dir name when
    composite_meta.json is missing (e.g. a run that crashed before writing it).
    Dir pattern: <name>_composite_<env>_<vec>_<inject>_L<layer>."""
    import re
    base = os.path.basename(os.path.normpath(run_dir))
    m = re.search(r"_composite_(parabola|sine|coupling)_(set|explore)_"
                  r"(final_set|submit)_L(\d+)$", base)
    if m:
        return dict(env=m.group(1), vector_type=m.group(2),
                    inject_at=m.group(3), layer=int(m.group(4)))
    return {}


def rebuild_from_dirs(run_dir, write=True):
    """Reconstruct the tidy per-turn rows by scanning rollout_* dirs that contain
    composite_summary.json (a completed rollout) + transcript_a*.jsonl. Lets you
    materialise the CSV/plot after a crash WITHOUT re-running the model, and makes
    the CSV reflect every completed rollout (including resumed ones)."""
    import glob, re
    dirs = sorted(glob.glob(os.path.join(run_dir, "rollout_*")))
    rows, id_seed = [], []
    for d in dirs:
        sp = os.path.join(d, "composite_summary.json")
        if not os.path.exists(sp):
            continue                                   # crashed / in-progress rollout
        with open(sp) as f:
            summ = json.load(f)
        idx, seed = summ.get("idx"), summ.get("seed")
        opt_m = summ.get("optimum_margin")
        bt, bk = summ.get("branch_turn"), summ.get("branch_kind")
        per = summ.get("summary", {})
        case_id, rep = summ.get("case_id"), summ.get("rep")
        id_seed.append((idx, seed))
        for tf in sorted(glob.glob(os.path.join(d, "transcript_a*.jsonl"))):
            mm = re.search(r"transcript_a([+-]?\d+\.\d+)(?:_b(\d+))?\.jsonl$",
                           os.path.basename(tf))
            if not mm:
                continue
            a = float(mm.group(1))
            brep = int(mm.group(2)) if mm.group(2) is not None else 0
            sinfo = (per.get(f"{a:+.2f}_b{brep:02d}")    # new keying
                     or per.get(str(a)) or per.get(f"{a:.1f}") or {})  # old keying
            st, submitted, nt = (sinfo.get("submit_turn"),
                                 sinfo.get("submitted"), sinfo.get("n_turns"))
            with open(tf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    mg = rec.get("margin")
                    rows.append(dict(
                        idx=idx, seed=seed, alpha=a, branch_rep=brep, turn=rec.get("turn"),
                        action=rec.get("action"), margin=mg, all_pass=rec.get("all_pass"),
                        submitted=submitted, submit_turn=st, n_turns=nt,
                        is_submit=bool(submitted and st is not None
                                       and rec.get("turn") == st and rec.get("action") == "submit"),
                        branch_turn=bt, branch_kind=bk, optimum_margin=opt_m,
                        gap=((opt_m - mg) if (opt_m is not None and mg is not None) else None),
                        case_id=case_id, rep=rep))
    # derive case_id/rep when older summaries lack them: group by seed in idx order
    if rows and any(r["case_id"] is None or r["rep"] is None for r in rows):
        cof, rctr, cmap, rmap = {}, {}, {}, {}
        for i, s in sorted(set(id_seed)):
            if s not in cof:
                cof[s] = len(cof); rctr[s] = 0
            cmap[i], rmap[i] = cof[s], rctr[s]; rctr[s] += 1
        for r in rows:
            if r["case_id"] is None:
                r["case_id"] = cmap.get(r["idx"])
            if r["rep"] is None:
                r["rep"] = rmap.get(r["idx"])

    meta = {}
    mp = os.path.join(run_dir, "composite_meta.json")
    if os.path.exists(mp):
        with open(mp) as f:
            meta = json.load(f)
    else:
        meta = _meta_from_dirname(run_dir)
        if "env" not in meta and dirs:                 # env fallback from a case.json
            cj = os.path.join(dirs[0], "case.json")
            if os.path.exists(cj):
                with open(cj) as f:
                    meta["env"] = json.load(f).get("env_kind", "?")
    meta.setdefault("kind", "composite")
    if write:
        write_data(run_dir, rows, meta)
    return rows, meta


def aggregate_from_rows(rows):
    """Per-alpha outcome aggregate from the tidy rows, averaging over both whole-
    rollout repeats and branch fan-out continuations. Returns (table, alphas)."""
    fin = {}                                           # (idx, alpha, brep) -> final (turn, margin, gap)
    for r in rows:
        k = (r["idx"], r["alpha"], r.get("branch_rep") or 0)
        cur = fin.get(k)
        if cur is None or (r["turn"] is not None and r["turn"] > cur[0]):
            fin[k] = (r["turn"], r["margin"], r["gap"])
    alphas = sorted({a for (_i, a, _b) in fin if a not in (0.0, None)})
    base = {i: m for (i, a, b), (t, m, g) in fin.items() if a == 0.0 and b == 0}
    out = {}
    for a in [0.0] + alphas:
        # alpha 0.0 row = the UNTOUCHED baseline only (b == 0). Resampled null
        # branches (alpha 0, b >= 1, from --null-branches) are excluded here so
        # this row's meaning is stable whether or not nulls were run; the
        # null-vs-steered comparison lives in composite_null_summary.json.
        items = [(i, v) for (i, aa, _b), v in fin.items()
                 if aa == a and not (a == 0.0 and _b > 0)]
        fm = [v[1] for (_i, v) in items if v[1] is not None]
        gp = [v[2] for (_i, v) in items if v[2] is not None]
        dl = [v[1] - base[i] for (i, v) in items
              if a != 0.0 and v[1] is not None and base.get(i) is not None]
        row = dict(mean_final_margin=(float(np.mean(fm)) if fm else None),
                   mean_gap=(float(np.mean(gp)) if gp else None), n=len(fm))
        if a == 0.0:
            row.update(mean_delta_vs_base=0.0, frac_improved=None)
        else:
            row.update(mean_delta_vs_base=(float(np.mean(dl)) if dl else None),
                       frac_improved=(float(np.mean([d > 1e-9 for d in dl])) if dl else None))
        out[a] = row
    return out, alphas


# --------------------------------------------------------------------------- #
# adapters: turn other studies' outputs into the same tidy schema
# --------------------------------------------------------------------------- #
def rebuild_steer(alpha_dirs, out_dir, write=True):
    """Build tidy rows from a STEER sweep: a list of per-alpha run dirs, each with
    standard rollout_* dirs (transcript.jsonl + submission.json + case.json) and a
    steer_meta.json giving alpha. Per-turn margin is recomputed from the recorded
    weight_vec by replaying the env (rollout transcripts don't store margin)."""
    import glob
    from .config import Config
    from .rollout import build_env
    rows, seeds_seen, env_kind, layer = [], [], None, None
    for ad in alpha_dirs:
        alpha = 0.0
        sm = os.path.join(ad, "steer_meta.json")
        if os.path.exists(sm):
            with open(sm) as f:
                m = json.load(f)
            alpha = float(m.get("alpha", 0.0))
            env_kind = m.get("env_kind", env_kind); layer = m.get("layer", layer)
        for d in sorted(glob.glob(os.path.join(ad, "rollout_*"))):
            tj = os.path.join(d, "transcript.jsonl"); cj = os.path.join(d, "case.json")
            sj = os.path.join(d, "submission.json")
            if not (os.path.exists(tj) and os.path.exists(cj)):
                continue
            with open(cj) as f:
                case = json.load(f)
            seed = int(case.get("seed", 0)); ek = case.get("env_kind", "coupling")
            nobj = int(case.get("n_obj", 1))
            opt_m = submit_turn = case_id = rep = nt = None; submitted = None
            if os.path.exists(sj):
                with open(sj) as f:
                    sub = json.load(f)
                opt_m = sub.get("optimum_margin_priority"); submit_turn = sub.get("submit_turn")
                case_id = sub.get("case_id"); rep = sub.get("rep"); nt = sub.get("n_turns")
                submitted = not bool(sub.get("forced", False))
            # replay env (deterministic from seed) to recompute per-turn margins
            cfg = Config(); cfg.env_kind = ek; cfg.n_obj = nobj
            env = build_env(cfg); env.reset(seed=seed, wide=getattr(cfg, "wide_cases", True))
            prio = env.priority
            with open(tj) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    wv = rec.get("weight_vec")
                    if wv is None:
                        continue
                    mar = env.margins(wv)
                    mg = float(mar[prio]); ap = bool(np.all(mar >= 0)); t = rec.get("turn")
                    rows.append(dict(
                        idx=None, seed=seed, alpha=alpha, branch_rep=0, turn=t,
                        action=rec.get("action"), margin=mg, all_pass=ap, submitted=submitted,
                        submit_turn=submit_turn, n_turns=nt,
                        is_submit=bool(submitted and submit_turn is not None
                                       and t == submit_turn and rec.get("action") == "submit"),
                        branch_turn=None, branch_kind=None, optimum_margin=opt_m,
                        gap=((opt_m - mg) if opt_m is not None else None),
                        case_id=case_id, rep=rep, proj=None))
            seeds_seen.append(seed)
    if rows and any(r["case_id"] is None for r in rows):   # derive case_id by seed order
        cof = {}
        for s in sorted(set(seeds_seen)):
            cof[s] = len(cof)
        for r in rows:
            if r["case_id"] is None:
                r["case_id"] = cof.get(r["seed"])
            if r["rep"] is None:
                r["rep"] = 0
    meta = dict(kind="steer", env=env_kind, layer=layer,
                alphas=sorted({r["alpha"] for r in rows if r["alpha"] != 0.0}))
    if write:
        write_data(out_dir, rows, meta)
    return rows, meta


def rebuild_project(run_dir, directions, layer, pool="before", win=4,
                    model_name=None, write=True):
    """Build tidy rows from a PROJECT run: per-turn projection of the captured
    (before-verb) activations onto the loaded SUBMIT-SET axis (SET_all=-1,
    SUBMIT_all=+1). Reuses direction_extract._turn_proj so it matches extraction."""
    import glob, re
    from . import direction_extract as de
    from .steering import load_direction
    tok = None
    if model_name:
        try:
            from transformers import AutoTokenizer
            try:
                tok = AutoTokenizer.from_pretrained(model_name)
            except Exception:
                tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            tok = None
    set_v, sub_v = load_direction(directions, layer)
    mid = (set_v + sub_v) / 2.0; dirv = (sub_v - set_v); denom = float(dirv @ dirv) + 1e-12
    rows = []
    for d in sorted(glob.glob(os.path.join(run_dir, "rollout_*"))):
        tj = os.path.join(d, "transcript.jsonl")
        if not os.path.exists(tj):
            continue
        seed = 0
        cj = os.path.join(d, "case.json")
        if os.path.exists(cj):
            with open(cj) as f:
                seed = int(json.load(f).get("seed", 0))
        submit_turn = None
        sj = os.path.join(d, "submission.json")
        if os.path.exists(sj):
            with open(sj) as f:
                submit_turn = json.load(f).get("submit_turn")
        mm = re.search(r"rollout_(\d+)", os.path.basename(d))
        idx = int(mm.group(1)) if mm else None
        kinds = de.turn_actions(tj)
        for t in sorted(kinds):
            verb = {"set": "SET", "submit": "SUBMIT"}.get(kinds[t])
            npz = os.path.join(d, "activations", f"turn_{t:02d}.npz")
            if not os.path.exists(npz):
                continue
            p = de._turn_proj(npz, layer, verb, tok, pool, win, mid, dirv, denom)
            if p is None:
                continue
            rows.append(dict(
                idx=idx, seed=seed, alpha=None, branch_rep=0, turn=t,
                action=kinds[t], margin=None, all_pass=None,
                submitted=(submit_turn is not None), submit_turn=submit_turn, n_turns=None,
                is_submit=(kinds[t] == "submit"), branch_turn=None, branch_kind=None,
                optimum_margin=None, gap=None, case_id=None, rep=None, proj=float(p)))
    env = _meta_from_dirname(run_dir).get("env")
    if env is None:
        cjs = glob.glob(os.path.join(run_dir, "rollout_*", "case.json"))
        if cjs:
            with open(cjs[0]) as f:
                env = json.load(f).get("env_kind")
    meta = dict(kind="project", env=env, layer=int(layer), pool=pool, win=win)
    if write:
        write_data(run_dir, rows, meta)
    return rows, meta


def rebuild_trigger(run_dir, write=True):
    """Rebuild the kind='trigger' tidy CSV from per-rollout transcripts (which carry
    per-turn proj_end/proj_max/n_steered + margin). Lets the plot use proj_max (the
    PEAK that crossed the threshold) even for runs whose CSV predates that column.
    optimum_margin is taken from the existing CSV when present."""
    import glob, re
    existing, meta = {}, {}
    try:
        rows0, meta = load_turns(run_dir)
        for r in rows0:
            existing[(r["idx"], r["turn"])] = r
    except Exception:
        pass
    if not meta:
        mp = os.path.join(run_dir, "composite_meta.json")
        if os.path.exists(mp):
            with open(mp) as f:
                meta = json.load(f)
    meta.setdefault("kind", "trigger")
    rows = []
    for d in sorted(glob.glob(os.path.join(run_dir, "rollout_*"))):
        tj = os.path.join(d, "transcript.jsonl")
        if not os.path.exists(tj):
            continue
        mm = re.search(r"rollout_(\d+)", os.path.basename(d))
        idx = int(mm.group(1)) if mm else None
        seed = None
        cj = os.path.join(d, "case.json")
        if os.path.exists(cj):
            with open(cj) as f:
                seed = int(json.load(f).get("seed", 0))
        recs = []
        with open(tj) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        st = next((r["turn"] for r in recs if r.get("action") == "submit"), None)
        submitted = st is not None
        nt = max((r["turn"] for r in recs), default=None)
        for rec in recs:
            t = rec.get("turn"); ti = rec.get("trigger") or {}
            ex = existing.get((idx, t), {})
            optm = ex.get("optimum_margin")
            mg = rec.get("margin", ex.get("margin"))
            rows.append(dict(
                idx=idx, seed=(seed if seed is not None else ex.get("seed")),
                case_id=ex.get("case_id"), rep=ex.get("rep"), branch_rep=0, alpha=None,
                turn=t, action=rec.get("action"),
                margin=mg, all_pass=rec.get("all_pass", ex.get("all_pass")),
                proj=ti.get("proj_end"), proj_max=ti.get("proj_max"),
                n_steered=ti.get("n_steered"), submitted=submitted, submit_turn=st,
                n_turns=nt,
                is_submit=bool(submitted and st is not None and t == st
                               and rec.get("action") == "submit"),
                branch_turn=None, branch_kind=None, optimum_margin=optm,
                gap=((optm - mg) if (optm is not None and mg is not None) else None)))
    if rows and any(r["case_id"] is None for r in rows):
        cof = {}
        for r in rows:
            s = r["seed"]
            if s not in cof:
                cof[s] = len(cof)
        for r in rows:
            if r["case_id"] is None:
                r["case_id"] = cof.get(r["seed"])
            if r["rep"] is None:
                r["rep"] = 0
    if write:
        write_data(run_dir, rows, meta)
    return rows, meta


# --------------------------------------------------------------------------- #
# plot
# --------------------------------------------------------------------------- #
def _draw_traj(ax, pts, color, lstyle, lw, passing="segment", ms=4):
    """Draw one trajectory. pts = [(x, y, all_pass_bool), ...].
    passing='segment': FAILING spans drawn thin + translucent, PASSING spans
        full; markers filled when passing, hollow when failing -- so a line that
        dips while reaching a feasible plan reads as 'now passing', not 'worse'.
    passing='marker': single line style, only the marker fill encodes passing.
    passing='none': no passing encoding."""
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    pa = [bool(p[2]) for p in pts]
    if passing == "segment" and len(pts) >= 2:
        i = 0
        while i < len(pts) - 1:
            cur = pa[i]; j = i
            while j + 1 < len(pts) and pa[j + 1] == cur:
                j += 1
            ax.plot(xs[i:j + 2], ys[i:j + 2], lstyle, color=color,
                    lw=(lw if cur else lw * 0.5),
                    alpha=(0.95 if cur else 0.35), zorder=2)
            i = j + 1
    else:
        ax.plot(xs, ys, lstyle, color=color, lw=lw, alpha=0.9, zorder=2)
    if passing in ("segment", "marker"):
        fx = [x for x, p in zip(xs, pa) if p]; fy = [y for y, p in zip(ys, pa) if p]
        hx = [x for x, p in zip(xs, pa) if not p]; hy = [y for y, p in zip(ys, pa) if not p]
        ax.plot(fx, fy, "o", ms=ms, color=color, zorder=3)
        ax.plot(hx, hy, "o", ms=ms, mfc="white", mec=color, mew=1.0, zorder=3)
    else:
        ax.plot(xs, ys, "o", ms=ms, color=color, zorder=3)


def plot(run_dir, out_dir=None, view="auto", y="margin", stars="steered",
         passing="segment"):
    """Universal entry: read the tidy CSV + meta in run_dir and plot according to
    meta['kind'] ('composite' | 'steer' | 'project'). Writes one or more figures
    into <run_dir>/plots/."""
    rows, meta = load_turns(run_dir)
    if not rows:
        raise SystemExit(f"no rows in {run_dir}/composite_turns.csv")
    out_dir = out_dir or os.path.join(run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    kind = meta.get("kind", "composite")
    if kind == "steer":
        return _plot_steer(rows, meta, out_dir, y=y, stars="all", passing=passing)
    if kind == "project":
        return _plot_project(rows, meta, out_dir)
    if kind == "trigger":
        figs = _plot_trigger(rows, meta, out_dir, passing=passing)
        figs += _plot_trigger_traces(run_dir, meta, out_dir)
        return figs
    return _plot_composite(rows, meta, out_dir, view=view, y=y, stars=stars,
                           passing=passing)


def _plot_composite(rows, meta, out_dir, view="auto", y="margin", stars="steered",
                    passing="segment"):
    """One figure per panel into out_dir.
    view 'repeats' -> panel = case; lines = each (rep, alpha, branch_rep),
        coloured by whole-rollout repeat. (Each repeat re-runs its own baseline.)
    view 'fanout'  -> panel = (case, rep); the single baseline (black) plus the
        branch-fan continuations, coloured by branch_rep.
    view 'auto'    -> 'fanout' if any branch_rep>0 else 'repeats'."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    alphas = sorted({r["alpha"] for r in rows if r["alpha"] not in (0.0, None)})
    conds = [0.0] + alphas
    max_brep = max((r.get("branch_rep") or 0) for r in rows)
    if view == "auto":
        view = "fanout" if max_brep > 0 else "repeats"

    ls_cycle = ["-", "--", ":", "-."]
    cond_ls = {c: ls_cycle[k % len(ls_cycle)] for k, c in enumerate(conds)}

    traj = {}                                          # (case, rep, alpha, brep) -> rows
    for r in rows:
        key = (r["case_id"], r["rep"], r["alpha"], r.get("branch_rep") or 0)
        traj.setdefault(key, []).append(r)
    for k in traj:
        traj[k] = sorted(traj[k], key=lambda r: r["turn"])

    if view == "repeats":
        panels = {}                                    # case -> [keys]
        for k in traj:
            panels.setdefault(k[0], []).append(k)
    else:
        panels = {}                                    # (case, rep) -> [keys]
        for k in traj:
            panels.setdefault((k[0], k[1]), []).append(k)

    env = meta.get("env", "?"); vt = meta.get("vector_type", "?")
    L = meta.get("layer", "?"); inj = meta.get("inject_at", "?")
    written = []
    for pk, keys in sorted(panels.items(), key=lambda kv: str(kv[0])):
        fig, ax = plt.subplots(figsize=(6.2, 4.1))
        if view == "repeats":
            units = sorted({k[1] for k in keys})       # colour by repeat
            ucol = plt.cm.viridis(np.linspace(0.0, 0.9, max(len(units), 1)))
            uidx = {u: j for j, u in enumerate(units)}
            colour_title = "colour = repeat"
        else:
            units = sorted({k[3] for k in keys if k[2] != 0.0})  # colour by branch_rep
            ucol = plt.cm.plasma(np.linspace(0.0, 0.85, max(len(units), 1)))
            uidx = {u: j for j, u in enumerate(units)}
            colour_title = "colour = branch repeat"

        seed = opt = None
        for k in keys:
            _cid, rp, a, bp = k
            tr = traj[k]
            pts = [(r["turn"], r[y], r["all_pass"]) for r in tr if r[y] is not None]
            if not pts:
                continue
            seed = tr[0]["seed"]; opt = tr[0]["optimum_margin"]
            if view == "repeats":
                color = ucol[uidx[rp]]; lstyle = cond_ls.get(a, "--")
            else:
                if a == 0.0 and bp == 0:
                    color, lstyle = "black", "-"       # untouched baseline
                elif a == 0.0:
                    color, lstyle = "0.6", "-"         # resampled null branch
                else:
                    color, lstyle = ucol[uidx[bp]], cond_ls.get(a, "--")
            lw = 1.9 if (a == 0.0 and bp == 0) else (1.1 if a == 0.0 else 1.3)
            _draw_traj(ax, pts, color, lstyle, lw, passing=passing)
            show_star = (stars == "all") or (stars == "steered" and a != 0.0)
            if show_star and tr[0]["submitted"] and tr[0]["submit_turn"] is not None:
                st = tr[0]["submit_turn"]
                yv = next((v for (t, v, _p) in pts if t == st), None)
                if yv is not None:
                    ax.plot([st], [yv], "*", ms=13, color=color, mec="black",
                            mew=0.4, zorder=6)

        if y == "margin" and opt is not None:
            ax.axhline(opt, color="green", ls=":", lw=1.0, alpha=0.7)
        ax.axhline(0, color="gray", ls="--", lw=0.8)

        if view == "repeats":
            title = f"case {pk} (seed {seed})"
            fname = f"case_{pk:03d}_seed{seed}_{y}.png" if isinstance(pk, int) \
                else f"case_{pk}_{y}.png"
        else:
            cid, rp = pk
            title = f"case {cid} rep {rp} (seed {seed})"
            fname = f"case_{cid:03d}_rep{rp:02d}_seed{seed}_{y}.png"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("turn"); ax.set_ylabel(y); ax.grid(alpha=0.3)

        handles = [Line2D([0], [0], color=("black" if c == 0.0 else "gray"),
                          ls=cond_ls[c],
                          label=("baseline" if c == 0.0 else f"\u03b1 {c:+.1f}"))
                   for c in conds]
        if view == "fanout" and any(k[2] == 0.0 and k[3] > 0 for k in keys):
            handles.append(Line2D([0], [0], color="0.6", ls="-",
                                  label="null branch (\u03b1 0, resampled)"))
        if passing in ("segment", "marker"):
            handles += [Line2D([0], [0], marker="o", lw=0, color="gray", label="passing"),
                        Line2D([0], [0], marker="o", lw=0, mfc="white", mec="gray",
                               label="failing")]
        ax.legend(handles=handles, fontsize=7, loc="best", title=colour_title,
                  title_fontsize=7)
        fig.suptitle(f"{env} \u00b7 {vt} vector \u00b7 L{L} \u00b7 inject@{inj}", fontsize=10)
        fig.tight_layout()
        out = os.path.join(out_dir, fname)
        fig.savefig(out, dpi=130); plt.close(fig); written.append(out)

    print(f"[plot] wrote {len(written)} figures to {out_dir}  (view={view}, y={y})")
    return written


def _plot_steer(rows, meta, out_dir, y="margin", stars="all", passing="segment"):
    """One figure per case; the alpha sweep overlaid, colour = alpha (unsteered
    black, steered on a coolwarm scale). Same passing/optimum/star treatment as
    composite."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    alphas = sorted({r["alpha"] for r in rows if r["alpha"] is not None})
    nz = [a for a in alphas if a != 0.0]
    cm = plt.cm.coolwarm(np.linspace(0.0, 1.0, max(len(nz), 1)))
    acol = {0.0: "black"}
    for j, a in enumerate(sorted(nz)):
        acol[a] = cm[j]

    traj = {}
    for r in rows:
        traj.setdefault((r["case_id"], r["alpha"], r.get("rep") or 0), []).append(r)
    for k in traj:
        traj[k] = sorted(traj[k], key=lambda r: r["turn"])
    panels = {}
    for k in traj:
        panels.setdefault(k[0], []).append(k)

    env = meta.get("env", "?"); L = meta.get("layer", "?")
    written = []
    for cid, keys in sorted(panels.items(), key=lambda kv: str(kv[0])):
        fig, ax = plt.subplots(figsize=(6.2, 4.1)); seed = opt = None
        for k in sorted(keys, key=lambda kk: (kk[1], kk[2])):
            _c, a, _rp = k
            tr = traj[k]
            pts = [(r["turn"], r[y], r["all_pass"]) for r in tr if r[y] is not None]
            if not pts:
                continue
            seed = tr[0]["seed"]; opt = tr[0]["optimum_margin"]
            color = acol.get(a, "gray"); lw = 1.9 if a == 0.0 else 1.3
            _draw_traj(ax, pts, color, "-", lw, passing=passing)
            show_star = (stars == "all") or (stars == "steered" and a != 0.0)
            if show_star and tr[0]["submitted"] and tr[0]["submit_turn"] is not None:
                st = tr[0]["submit_turn"]
                yv = next((v for (t, v, _p) in pts if t == st), None)
                if yv is not None:
                    ax.plot([st], [yv], "*", ms=13, color=color, mec="black", mew=0.4, zorder=6)
        if y == "margin" and opt is not None:
            ax.axhline(opt, color="green", ls=":", lw=1.0, alpha=0.7)
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set_title(f"case {cid} (seed {seed})", fontsize=10)
        ax.set_xlabel("turn"); ax.set_ylabel(y); ax.grid(alpha=0.3)
        handles = [Line2D([0], [0], color=acol[a],
                          label=("unsteered" if a == 0.0 else f"\u03b1 {a:+.1f}"))
                   for a in alphas]
        if passing in ("segment", "marker"):
            handles += [Line2D([0], [0], marker="o", lw=0, color="gray", label="passing"),
                        Line2D([0], [0], marker="o", lw=0, mfc="white", mec="gray",
                               label="failing")]
        ax.legend(handles=handles, fontsize=7, loc="best", title="colour = alpha",
                  title_fontsize=7)
        fig.suptitle(f"steer \u00b7 {env} \u00b7 L{L}", fontsize=10)
        fig.tight_layout()
        fname = (f"case_{cid:03d}_seed{seed}_{y}.png" if isinstance(cid, int)
                 else f"case_{cid}_{y}.png")
        out = os.path.join(out_dir, fname)
        fig.savefig(out, dpi=130); plt.close(fig); written.append(out)
    print(f"[plot] wrote {len(written)} steer figures to {out_dir}  (y={y})")
    return written


def _plot_project(rows, meta, out_dir):
    """Single figure: per-turn projection onto the (SUBMIT-SET) axis for every
    rollout, with SET_all=-1 / SUBMIT_all=+1 reference lines and a star at submit."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    L = meta.get("layer", "?"); env = meta.get("env", "?")
    traj = {}
    for r in rows:
        key = r["idx"] if r.get("idx") is not None else r["seed"]
        traj.setdefault(key, []).append(r)
    keys = sorted(traj, key=lambda x: (x is None, x))
    fig, ax = plt.subplots(figsize=(8, 5)); n = 0
    cm = plt.cm.viridis(np.linspace(0.0, 0.9, max(len(keys), 1)))
    for j, k in enumerate(keys):
        tr = sorted(traj[k], key=lambda r: r["turn"])
        pts = [(r["turn"], r["proj"]) for r in tr if r.get("proj") is not None]
        if not pts:
            continue
        xs, ys = zip(*pts)
        line, = ax.plot(xs, ys, "-o", ms=3, alpha=0.85, color=cm[j], label=f"r{k}")
        st = tr[0].get("submit_turn")
        if st is not None:
            yv = next((v for (t, v) in pts if t == st), None)
            if yv is not None:
                ax.plot([st], [yv], "*", ms=14, color=line.get_color(), zorder=6)
        n += 1
    ax.axhline(1, color="g", ls="--", lw=1); ax.axhline(-1, color="b", ls="--", lw=1)
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.text(0.01, 0.98, "SUBMIT_all = +1", color="g", transform=ax.transAxes,
            va="top", fontsize=8)
    ax.text(0.01, 0.02, "SET_all = -1", color="b", transform=ax.transAxes,
            va="bottom", fontsize=8)
    ax.set_xlabel("turn"); ax.set_ylabel("projection onto (SUBMIT-SET)")
    ax.set_title(f"project \u00b7 {env} \u00b7 no steering \u00b7 L{L}")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3); fig.tight_layout()
    out = os.path.join(out_dir, f"projection_L{L}.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[plot] wrote {out}  ({n} rollouts)")
    return [out]


def _plot_trigger(rows, meta, out_dir, passing="segment"):
    """One twin-axis figure per rollout: LEFT = margin vs turn (passing-encoded,
    optimum line, submit star); RIGHT = detector projection vs turn. The PEAK
    running-window projection (proj_max -- the value that actually crossed the
    threshold) is the solid line; the end-of-turn value (proj_end) is a faint line
    showing where it settled after any injection. steer_proj threshold is drawn;
    injected turns are ringed (sized by steered-token count) on the peak line."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    env = meta.get("env", "?"); L = meta.get("layer", "?")
    dv = meta.get("detect_vec", "?"); sv = meta.get("steer_vec", "?")
    alpha = meta.get("alpha"); thr = meta.get("steer_proj", 0.0)
    k = meta.get("k"); sk = meta.get("steer_k"); trig = meta.get("trigger", "above")

    by = {}
    for r in rows:
        by.setdefault((r["case_id"], r["rep"], r["idx"]), []).append(r)
    written = []
    for key, rs in sorted(by.items(), key=lambda kv: str(kv[0])):
        rs = sorted(rs, key=lambda r: r["turn"])
        seed = rs[0]["seed"]; opt = rs[0]["optimum_margin"]
        mpts = [(r["turn"], r["margin"], r["all_pass"]) for r in rs if r["margin"] is not None]
        # peak (what crossed) and end (where it settled); fall back to end if no peak
        peak = [(r["turn"], (r.get("proj_max") if r.get("proj_max") is not None
                             else r.get("proj"))) for r in rs]
        peak = [(t, p) for (t, p) in peak if p is not None]
        endp = [(r["turn"], r.get("proj")) for r in rs if r.get("proj") is not None]
        inj = [(r["turn"], (r.get("proj_max") if r.get("proj_max") is not None
                            else r.get("proj")), (r.get("n_steered") or 0))
               for r in rs if (r.get("n_steered") or 0) > 0]

        fig, axL = plt.subplots(figsize=(7.2, 4.3))
        if mpts:
            _draw_traj(axL, mpts, "tab:blue", "-", 1.9, passing=passing)
        if opt is not None:
            axL.axhline(opt, color="green", ls=":", lw=1.0, alpha=0.7)
        axL.axhline(0, color="gray", ls="--", lw=0.8)
        st = rs[0]["submit_turn"]
        if rs[0]["submitted"] and st is not None:
            yv = next((m for (t, m, _p) in mpts if t == st), None)
            if yv is not None:
                axL.plot([st], [yv], "*", ms=13, color="tab:blue", mec="black",
                         mew=0.4, zorder=6)
        axL.set_xlabel("turn"); axL.set_ylabel("margin", color="tab:blue")
        axL.tick_params(axis="y", labelcolor="tab:blue"); axL.grid(alpha=0.3)

        axR = axL.twinx()
        if peak:
            xs, ys = zip(*peak)
            axR.plot(xs, ys, "-s", ms=3, color="purple", alpha=0.9,
                     label="detector proj (peak)")
        if endp:
            xs, ys = zip(*endp)
            axR.plot(xs, ys, "-", lw=1.0, color="purple", alpha=0.35,
                     label="end of turn")
        axR.axhline(thr, color="red", ls="--", lw=1.0, alpha=0.8)
        axR.axhline(1, color="green", ls=":", lw=0.7, alpha=0.5)
        axR.axhline(-1, color="blue", ls=":", lw=0.7, alpha=0.5)
        if inj:
            it, ip, inn = zip(*inj)
            axR.scatter(it, ip, s=[40 + 10 * n for n in inn], facecolor="none",
                        edgecolor="red", linewidth=1.8, zorder=7)
            for t, p, n in zip(it, ip, inn):
                axR.annotate(f"{n}", (t, p), textcoords="offset points",
                             xytext=(3, 5), fontsize=7, color="red")
        axR.set_ylabel("detector projection  (SET=-1, SUBMIT=+1)", color="purple")
        axR.tick_params(axis="y", labelcolor="purple")

        a_s = f"{alpha:+.1f}" if isinstance(alpha, (int, float)) else "?"
        axL.set_title(f"trigger \u00b7 {env} \u00b7 L{L} \u00b7 case {key[0]} rep {key[1]} "
                      f"(seed {seed})", fontsize=9)
        fig.suptitle(f"detect={dv}  steer={sv} (\u03b1={a_s})  fire {trig} {thr:g}  "
                     f"k={k} steer_k={sk}", fontsize=9)
        handles = [Line2D([0], [0], color="tab:blue", label="margin"),
                   Line2D([0], [0], color="purple", marker="s", lw=1, label="proj (peak)"),
                   Line2D([0], [0], color="purple", lw=1, alpha=0.35, label="proj (end)"),
                   Line2D([0], [0], color="red", ls="--", label=f"threshold {thr:g}"),
                   Line2D([0], [0], marker="o", lw=0, mfc="none", mec="red",
                          label="injected (n=steered)")]
        axR.legend(handles=handles, fontsize=6.5, loc="best")
        fig.tight_layout()
        fname = f"case_{key[0]:03d}_rep{key[1]:02d}_seed{seed}_trigger.png" \
            if isinstance(key[0], int) else f"rollout_{key[2]}_trigger.png"
        out = os.path.join(out_dir, fname)
        fig.savefig(out, dpi=130); plt.close(fig); written.append(out)
    print(f"[plot] wrote {len(written)} trigger figures to {out_dir}")
    return written


def _plot_trigger_traces(run_dir, meta, out_dir):
    """Token-level diagnostic per rollout from trigger_trace.json: projection at
    every generated token (turns concatenated, dividers marked), the threshold
    line, and red shaded spans wherever the steer vector was being injected. This
    is the view that makes a mid-turn spike-then-settle visible."""
    import glob, re, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    thr = meta.get("steer_proj", 0.0); L = meta.get("layer", "?"); env = meta.get("env", "?")
    written = []
    for d in sorted(glob.glob(os.path.join(run_dir, "rollout_*"))):
        tp = os.path.join(d, "trigger_trace.json")
        if not os.path.exists(tp):
            continue
        with open(tp) as f:
            traces = json.load(f)                       # {turn(str): [[tok,proj,steer],...]}
        if not traces:
            continue
        mm = re.search(r"rollout_(\d+)", os.path.basename(d))
        idx = mm.group(1) if mm else "?"
        gx = 0; xs = []; ys = []; spans = []; bounds = []
        for tk in sorted(traces, key=lambda s: int(s)):
            tr = traces[tk] or []
            bounds.append((gx, tk)); span_start = None
            for (tok, proj, steer) in tr:
                if proj is not None:
                    xs.append(gx); ys.append(proj)
                if steer and span_start is None:
                    span_start = gx
                if (not steer) and span_start is not None:
                    spans.append((span_start, gx)); span_start = None
                gx += 1
            if span_start is not None:
                spans.append((span_start, gx))
        if not xs:
            continue
        fig, ax = plt.subplots(figsize=(9.5, 3.8))
        ax.plot(xs, ys, lw=0.8, color="purple")
        for (a, b) in spans:
            ax.axvspan(a, b, color="red", alpha=0.15, lw=0)
        ax.axhline(thr, color="red", ls="--", lw=1.0)
        ax.axhline(1, color="green", ls=":", lw=0.6, alpha=0.5)
        ax.axhline(-1, color="blue", ls=":", lw=0.6, alpha=0.5)
        y1 = ax.get_ylim()[1]
        for (gx0, tk) in bounds:
            ax.axvline(gx0, color="gray", lw=0.5, alpha=0.4)
            ax.text(gx0, y1, f"t{tk}", fontsize=6, va="top", color="gray")
        ax.set_xlabel("generated token  (turns concatenated)")
        ax.set_ylabel("projection (SET=-1, SUBMIT=+1)")
        ax.set_title(f"trigger trace \u00b7 {env} \u00b7 L{L} \u00b7 rollout {idx}  "
                     f"(red span = injecting)", fontsize=9)
        fig.tight_layout()
        out = os.path.join(out_dir, f"rollout_{idx}_trace.png")
        fig.savefig(out, dpi=130); plt.close(fig); written.append(out)
    if written:
        print(f"[plot] wrote {len(written)} token-level trace figures to {out_dir}")
    return written



# =========================================================================== #
# paired per-rollout stats: steered vs resampled-null, one delta per rollout
# =========================================================================== #
def _post_branch_weights(tf_path, branch_turn):
    """Weight vectors of the turns this branch actually generated
    (turn >= branch_turn), in order, from a transcript_a*.jsonl."""
    W = []
    if not os.path.exists(tf_path):
        return W
    with open(tf_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t, wv = rec.get("turn"), rec.get("weight_vec")
            if t is None or wv is None:
                continue
            if branch_turn is None or t >= branch_turn:
                W.append(np.asarray(wv, float))
    return W


def _explore_metrics(W, ndp=3):
    """Exploration of a weight trajectory: distinct points tried, total path
    length, and max displacement from the branch-point weights."""
    if not W:
        return dict(n_points=0, n_unique=0, path_len=0.0, max_disp=0.0)
    uniq = {tuple(np.round(w, ndp)) for w in W}
    path = float(sum(np.linalg.norm(W[i + 1] - W[i]) for i in range(len(W) - 1)))
    disp = float(max(np.linalg.norm(w - W[0]) for w in W))
    return dict(n_points=len(W), n_unique=len(uniq), path_len=path, max_disp=disp)


def paired_rollout_stats(run_dir, ndp=3, opt_eps=1e-6):
    """One record per (rollout, steered alpha): paired deltas of the steered
    branches' means MINUS the null (alpha 0, b>=1) branches' means, for:

      d_best_norm / d_final_norm -- margin delta as a PERCENTAGE OF THE CASE
          OPTIMUM (parabola/sine: optimum_margin = z_pass > 0 known exactly).
          Comparable across cases; well-defined for failing (negative) margins,
          unlike percent-of-baseline which explodes when the null margin ~ 0.
      d_turns        -- extra turns generated past the branch point (persistence;
                        censored at the branch window for non-submitting branches)
      d_submit_rate  -- steered minus null submit rate within the window
      d_unique / d_path_len / d_max_disp -- exploration of the post-branch
                        weight trajectory (distinct points tried, path length,
                        max displacement from the branch point)

    ever_passed marks cases where ANY branch/baseline achieved margin >= 0 --
    split on it to ask 'on impossible cases, did steering at least try harder?'.
    pct_vs_null (percent change vs the null margin) is included in the JSON for
    completeness but is unstable when the null margin is near zero."""
    import glob, re
    out = []
    for d in sorted(glob.glob(os.path.join(run_dir, "rollout_*"))):
        sp = os.path.join(d, "composite_summary.json")
        if not os.path.exists(sp):
            continue
        with open(sp) as f:
            summ = json.load(f)
        bt = summ.get("branch_turn")
        opt = _f(summ.get("optimum_margin"))
        per_alpha = {}
        ever_passed = False
        for key, e in summ.get("summary", {}).items():
            a, b = e.get("alpha"), e.get("branch_rep", 0)
            if a is None:
                m = re.match(r"([+-]?\d+\.\d+)_b(\d+)$", key)
                if not m:
                    continue
                a, b = float(m.group(1)), int(m.group(2))
            a = float(a)
            best = _f(e.get("best_margin"))
            fin = _f(e.get("final_margin"))
            if (best is not None and best >= 0) or (fin is not None and fin >= 0):
                ever_passed = True
            if a == 0.0 and b == 0:
                continue                             # untouched baseline
            tf = os.path.join(d, f"transcript_a{a:+.2f}_b{b:02d}.jsonl")
            ex = _explore_metrics(_post_branch_weights(tf, bt), ndp)
            nt = _i(e.get("n_turns"))
            per_alpha.setdefault(a, []).append(dict(
                best=best, final=fin, submitted=bool(e.get("submitted")),
                turns_past=(None if (nt is None or bt is None)
                            else max(0, nt - bt + 1)), **ex))
        nulls = per_alpha.get(0.0)
        if not nulls:
            continue

        def _m(entries, key):
            vals = [x[key] for x in entries if x.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        for a, steered in per_alpha.items():
            if a == 0.0:
                continue
            rec = dict(rollout=os.path.basename(d), idx=summ.get("idx"),
                       alpha=a, n_steered=len(steered), n_null=len(nulls),
                       optimum_margin=opt, ever_passed=ever_passed,
                       branch_turn=bt)
            for key, name in [("best", "best"), ("final", "final")]:
                sv, nv = _m(steered, key), _m(nulls, key)
                if sv is None or nv is None:
                    continue
                rec[f"d_{name}"] = sv - nv
                if opt is not None and opt > opt_eps:
                    rec[f"d_{name}_norm"] = 100.0 * (sv - nv) / opt
                if abs(nv) > 1e-9:
                    rec[f"pct_{name}_vs_null"] = 100.0 * (sv - nv) / abs(nv)
            for key, name in [("turns_past", "turns_past"), ("n_unique", "unique"),
                              ("path_len", "path_len"), ("max_disp", "max_disp")]:
                sv, nv = _m(steered, key), _m(nulls, key)
                if sv is not None and nv is not None:
                    rec[f"d_{name}"] = sv - nv
            rec["d_submit_rate"] = (float(np.mean([x["submitted"] for x in steered]))
                                    - float(np.mean([x["submitted"] for x in nulls])))
            out.append(rec)
    return out


def _boot_mean_ci(x, n_boot=4000, seed=0):
    x = np.asarray([v for v in x if v is not None], float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = np.mean(x[rng.integers(0, len(x), size=(n_boot, len(x)))], axis=1)
    return dict(n=int(len(x)), mean=float(x.mean()), median=float(np.median(x)),
                frac_positive=float(np.mean(x > 0)),
                ci95=[float(np.percentile(means, 2.5)),
                      float(np.percentile(means, 97.5))])


def _hist_panel(ax, vals, title, xlabel, color="tab:blue", vs_vals=None,
                vs_label="control"):
    vals = np.asarray([v for v in vals if v is not None], float)
    vals = vals[~np.isnan(vals)]
    if not len(vals):
        ax.set_title(title + "  (no data)"); return
    bins = np.histogram_bin_edges(
        np.concatenate([vals] + ([np.asarray([v for v in vs_vals
                                              if v is not None], float)]
                                 if vs_vals else [])), bins="auto")
    ax.hist(vals, bins=bins, color=color, alpha=0.75, label="this run")
    if vs_vals:
        vv = np.asarray([v for v in vs_vals if v is not None], float)
        vv = vv[~np.isnan(vv)]
        if len(vv):
            ax.hist(vv, bins=bins, histtype="step", lw=1.8, color="black",
                    label=vs_label)
    ax.axvline(0, color="tab:red", ls="--", lw=1)
    s = _boot_mean_ci(vals)
    if s:
        ax.set_title(f"{title}\nmean {s['mean']:+.3g} "
                     f"[{s['ci95'][0]:+.3g}, {s['ci95'][1]:+.3g}]  "
                     f"({s['frac_positive']:.0%} > 0, n={s['n']})", fontsize=9)
    else:
        ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("rollouts", fontsize=8)
    ax.grid(alpha=0.3)
    if vs_vals:
        ax.legend(fontsize=7)


def plot_paired(run_dir, out_dir=None, vs_run_dir=None):
    """Per-rollout paired steered-vs-null histograms; one figure per alpha.
    vs_run_dir overlays a second run's deltas (e.g. the random-vector control)
    as a black outline on the same bins."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = out_dir or os.path.join(run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    stats = paired_rollout_stats(run_dir)
    if not stats:
        print("[paired] no paired rollouts found (need null branches: "
              "--null-branches >= 1)."); return []
    vs_stats = paired_rollout_stats(vs_run_dir) if vs_run_dir else []
    with open(os.path.join(run_dir, "paired_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    figs = []
    for a in sorted({r["alpha"] for r in stats}):
        S = [r for r in stats if r["alpha"] == a]
        V = [r for r in vs_stats if r["alpha"] == a] or None
        n_deg = sum(1 for r in S if "d_best_norm" not in r and "d_best" in r)

        def col(rs, k):
            return [r.get(k) for r in rs] if rs else None

        fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
        _hist_panel(axes[0, 0], col(S, "d_best_norm"),
                    "Best margin: steered - null",
                    "delta, % of case optimum", vs_vals=col(V, "d_best_norm"))
        _hist_panel(axes[0, 1], col(S, "d_final_norm"),
                    "Final margin: steered - null",
                    "delta, % of case optimum", vs_vals=col(V, "d_final_norm"))
        _hist_panel(axes[0, 2], col(S, "d_submit_rate"),
                    "Submit rate within window", "delta (steered - null)",
                    color="tab:purple", vs_vals=col(V, "d_submit_rate"))
        # persistence, split by whether the case was ever solvable in practice
        dt_all = col(S, "d_turns_past")
        dt_never = [r.get("d_turns_past") for r in S if not r["ever_passed"]]
        _hist_panel(axes[1, 0], dt_all, "Turns generated past branch",
                    "delta turns (censored at window)", color="tab:orange",
                    vs_vals=col(V, "d_turns_past"))
        if any(v is not None for v in dt_never):
            vv = np.asarray([v for v in dt_never if v is not None], float)
            axes[1, 0].hist(vv, bins=10, histtype="step", lw=1.8,
                            color="tab:red", label="never-passing cases")
            axes[1, 0].legend(fontsize=7)
        _hist_panel(axes[1, 1], col(S, "d_unique"),
                    "Distinct weight points tried", "delta unique points",
                    color="tab:green", vs_vals=col(V, "d_unique"))
        _hist_panel(axes[1, 2], col(S, "d_path_len"),
                    "Weight-space path length", "delta path length",
                    color="tab:green", vs_vals=col(V, "d_path_len"))
        ttl = f"paired steered-vs-null per rollout · alpha {a:+.1f}"
        if vs_run_dir:
            ttl += f"  (black outline = {os.path.basename(vs_run_dir)})"
        if n_deg:
            ttl += f"  [{n_deg} case(s) lack a positive optimum -> no norm]"
        fig.suptitle(ttl, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        p = os.path.join(out_dir, f"paired_a{a:+.1f}.png".replace("+", "p")
                         .replace("-", "m"))
        fig.savefig(p, dpi=130); plt.close(fig)
        figs.append(p)
        # console summary: this run, the overlay run, and the PAIRED contrast
        # between them (arms share seeds -> match rollouts by idx; this is the
        # direction-specificity number: real effect minus matched-norm control)
        keys = [("d_best_norm", "best margin (% of optimum)"),
                ("d_submit_rate", "submit rate"),
                ("d_turns_past", "turns past branch"),
                ("d_unique", "unique points"),
                ("d_path_len", "path length")]
        for key, label in keys:
            s = _boot_mean_ci(col(S, key))
            if s:
                print(f"[paired] a={a:+.1f} {label:28s} mean {s['mean']:+.3g} "
                      f"CI95 [{s['ci95'][0]:+.3g}, {s['ci95'][1]:+.3g}] "
                      f"({s['frac_positive']:.0%} rollouts > 0, n={s['n']})")
        if V:
            print(f"[paired] --- overlay run ({os.path.basename(vs_run_dir)}) ---")
            for key, label in keys:
                s = _boot_mean_ci(col(V, key))
                if s:
                    print(f"[paired] a={a:+.1f} {label:28s} mean {s['mean']:+.3g} "
                          f"CI95 [{s['ci95'][0]:+.3g}, {s['ci95'][1]:+.3g}] "
                          f"({s['frac_positive']:.0%} > 0, n={s['n']})")
            vs_by_idx = {r["idx"]: r for r in V if r.get("idx") is not None}
            contrast = {}
            for key, label in keys:
                dd = [r[key] - vs_by_idx[r["idx"]][key] for r in S
                      if r.get("idx") in vs_by_idx
                      and r.get(key) is not None
                      and vs_by_idx[r["idx"]].get(key) is not None]
                s = _boot_mean_ci(dd)
                if s:
                    contrast[key] = s
                    sig = ("*" if (s["ci95"][0] > 0 or s["ci95"][1] < 0) else " ")
                    print(f"[paired] a={a:+.1f} SPECIFICITY {label:22s} "
                          f"mean {s['mean']:+.3g} "
                          f"CI95 [{s['ci95'][0]:+.3g}, {s['ci95'][1]:+.3g}]{sig} "
                          f"(paired cases={s['n']})")
            print("[paired]    SPECIFICITY = per-case (this-run delta) - "
                  "(overlay delta); * = CI95 excludes 0.")
            with open(os.path.join(run_dir, f"paired_specificity_a{a:+.1f}.json"
                                   .replace("+", "p").replace("-", "m")), "w") as f:
                json.dump(dict(run=run_dir, vs=vs_run_dir, alpha=a,
                               contrast=contrast), f, indent=2)
    print(f"[paired] wrote {len(figs)} figure(s) to {out_dir} + paired_stats.json")
    return figs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="dir with composite_turns.csv")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild composite_turns.csv from the rollout_* dirs first "
                         "(use after a crash, or to refresh from disk). Auto-runs if "
                         "the CSV is missing.")
    ap.add_argument("--view", choices=["auto", "repeats", "fanout"], default="auto")
    ap.add_argument("--y", choices=["margin", "gap"], default="margin")
    ap.add_argument("--stars", choices=["steered", "all", "none"], default="steered")
    ap.add_argument("--passing", choices=["segment", "marker", "none"], default="segment")
    ap.add_argument("--out-dir", default=None, help="default <run-dir>/plots")
    ap.add_argument("--paired", action="store_true",
                    help="per-rollout paired steered-vs-null stats (margin as %% "
                         "of case optimum, persistence, exploration); needs runs "
                         "made with --null-branches >= 1")
    ap.add_argument("--paired-vs", default=None,
                    help="second run dir (e.g. the _ctlrandom0 control) overlaid "
                         "as a black outline on the paired histograms")
    a = ap.parse_args()
    csv_path = os.path.join(a.run_dir, "composite_turns.csv")
    meta_path = os.path.join(a.run_dir, "composite_meta.json")
    kind = "composite"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            kind = json.load(f).get("kind", "composite")
    if a.rebuild or not os.path.exists(csv_path):
        print(f"[rebuild] reconstructing CSV ({kind}) from {a.run_dir}")
        if kind == "trigger":
            rebuild_trigger(a.run_dir, write=True)
        else:
            rebuild_from_dirs(a.run_dir, write=True)
    if a.paired or a.paired_vs:
        plot_paired(a.run_dir, out_dir=a.out_dir, vs_run_dir=a.paired_vs)
    else:
        plot(a.run_dir, out_dir=a.out_dir, view=a.view, y=a.y, stars=a.stars,
             passing=a.passing)


if __name__ == "__main__":
    main()