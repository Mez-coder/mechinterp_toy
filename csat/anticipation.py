"""anticipation.py -- does the SUBMIT-SET projection ANTICIPATE the submit
decision, or merely detect the SUBMIT token about to be emitted?

The confound this rules in/out: the direction was extracted from tokens just
before the action verb, so it might simply encode "the next action word is
SUBMIT" -- a next-token feature, not a decision/intention feature. The test:

  For every SET turn t in a captured (unsteered) run, compute the projection of
  its pooled pre-verb residual onto the SUBMIT-SET axis, and ask how well it
  predicts "the model submits within the next m turns", for m = 1, 2, 3, ...

  * m = 1 only discriminative, m >= 2 at chance  -> the projection is (mostly)
    lexical anticipation of the imminent SUBMIT token in the NEXT turn's text.
    (Even m=1 on SET turns is one full turn of feedback away from the verb, so
    it is already stronger than a pure logit-lens -- but it is the weakest form.)
  * elevated AUROC at m = 2, 3, ...              -> a genuine early signal:
    the residual carries "I am close to stopping" turns before any SUBMIT token
    exists anywhere in the context.

Because SUBMIT happens late, turn index alone predicts it; we therefore also
report AUROC(turn index) and AUROC(projection residualised on turn index) --
the residualised number is the one to quote. 95% CIs are cluster bootstraps
over ROLLOUTS (turns within a rollout are not independent).

Also produced: the event-aligned curve -- mean projection at t = T-1, T-2, ...
relative to the submit turn T, with a cluster-bootstrap band. If the curve only
jumps at the last pre-submit turn, that is the lexical story visualised.

Offline: reads transcript.jsonl + activations/turn_XX.npz written by run.py
(capture on). No model load; the tokenizer is optional (verb-centred pooling).

    python -m csat.anticipation --run-dir runs/csat            # same run the
    # directions came from is fine, but a FRESH capture run (or --study project
    # output) is the cleaner held-out test:
    python -m csat.anticipation --run-dir runs/proj_parabola_nosteer \
        --directions runs/csat/directions.npz

Outputs under <run-dir>: anticipation_L<layer>.json / .png
"""
from __future__ import annotations
import os, json, glob, argparse
import numpy as np

try:
    import ml_dtypes  # noqa: F401  (bf16 activation captures)
except Exception:
    pass

from .config import Config
from . import direction_extract as de


# --------------------------------------------------------------------------- #
# small numerics (no sklearn/scipy dependency)
# --------------------------------------------------------------------------- #
def _midranks(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    sx = x[order]
    i, r = 0, 1
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (r + (r + (j - i)))
        r += (j - i + 1)
        i = j + 1
    return ranks


def auroc(scores, labels):
    """Mann-Whitney AUROC with midranks for ties. nan if one class is empty."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _midranks(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def residualise_on(x, t):
    """x minus its least-squares linear fit on t (turn index)."""
    x = np.asarray(x, float); t = np.asarray(t, float)
    ok = ~(np.isnan(x) | np.isnan(t))
    if ok.sum() < 3 or np.nanstd(t[ok]) == 0:
        return x - np.nanmean(x)
    b, a = np.polyfit(t[ok], x[ok], 1)
    return x - (a + b * t)


def cluster_boot_auroc(scores, labels, groups, n_boot=2000, seed=0):
    """Bootstrap CI on AUROC resampling ROLLOUTS (clusters), not turns."""
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, float); labels = np.asarray(labels, bool)
    groups = np.asarray(groups)
    gids = np.unique(groups)
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(gids, size=len(gids), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in pick])
        stats.append(auroc(scores[idx], labels[idx]))
    stats = np.asarray(stats, float)
    stats = stats[~np.isnan(stats)]
    if not len(stats):
        return (None, None)
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


# --------------------------------------------------------------------------- #
# data collection
# --------------------------------------------------------------------------- #
def _load_direction(directions_path, layer):
    """(set_vec, submit_vec) at hidden-state index `layer` -- local copy of
    steering.load_direction so this module never imports torch via agents."""
    z = np.load(directions_path, allow_pickle=True)
    layers = list(z["layers"].astype(int))
    if layer not in layers:
        raise SystemExit(f"layer {layer} not in saved directions {layers}.")
    li = layers.index(layer)
    set_key = "set_all" if "set_all" in z.files else "set_train"
    sub_key = "submit_all" if "submit_all" in z.files else "submit_train"
    return (z[set_key][li].astype(np.float32), z[sub_key][li].astype(np.float32))


def collect_rows(run_dir, directions, layer, tok, pool, win, include_forced=False,
                 max_rollouts=None):
    """One row per SET turn with a captured npz:
       (rollout, turn, proj, dt) where dt = submit_turn - turn (np.inf if the
       rollout never submitted). Also returns per-rollout submit turns."""
    set_v, sub_v = _load_direction(directions, layer)
    mid, dirv = (set_v + sub_v) / 2.0, (sub_v - set_v)
    denom = float(dirv @ dirv) + 1e-12

    rows, submit_turns = [], {}
    rdirs = sorted(glob.glob(os.path.join(run_dir, "rollout_*")))
    if max_rollouts:
        rdirs = rdirs[:max_rollouts]
    for rd in rdirs:
        rid = os.path.basename(rd)
        kinds = de.turn_actions(os.path.join(rd, "transcript.jsonl"))
        if not kinds:
            continue
        T = next((t for t, k in kinds.items() if k == "submit"), None)
        if T is None and not include_forced:
            continue                                  # forced/never-submitted rollout
        submit_turns[rid] = T
        for t in sorted(kinds):
            if kinds[t] != "set":
                continue
            if T is not None and t >= T:
                continue
            npz = os.path.join(rd, "activations", f"turn_{t:02d}.npz")
            if not os.path.exists(npz):
                continue
            p = de._turn_proj(npz, layer, "SET", tok, pool, win, mid, dirv, denom)
            if p is None:
                continue
            dt = (T - t) if T is not None else np.inf
            rows.append(dict(rollout=rid, turn=t, proj=float(p), dt=float(dt)))
    return rows, submit_turns


# --------------------------------------------------------------------------- #
# analyses
# --------------------------------------------------------------------------- #
def auroc_table(rows, ms=(1, 2, 3, 4), n_boot=2000, seed=0):
    """AUROC of {projection, turn index, projection residualised on turn} for the
    label 'submits within the next m turns', with cluster-bootstrap CIs.
    Turns with dt <= m-1 already inside a smaller horizon are still positives at
    m (label is dt <= m), matching 'within'."""
    proj = np.array([r["proj"] for r in rows], float)
    turn = np.array([r["turn"] for r in rows], float)
    grp = np.array([r["rollout"] for r in rows])
    dt = np.array([r["dt"] for r in rows], float)
    resid = residualise_on(proj, turn)
    out = []
    for m in ms:
        y = dt <= m
        n_pos = int(y.sum())
        if n_pos == 0 or n_pos == len(y):
            out.append(dict(m=m, n=len(y), n_pos=n_pos, auc_proj=None,
                            auc_turn=None, auc_resid=None))
            continue
        row = dict(m=int(m), n=int(len(y)), n_pos=n_pos,
                   auc_proj=auroc(proj, y),
                   auc_turn=auroc(turn, y),
                   auc_resid=auroc(resid, y))
        row["auc_proj_ci95"] = cluster_boot_auroc(proj, y, grp, n_boot, seed)
        row["auc_resid_ci95"] = cluster_boot_auroc(resid, y, grp, n_boot, seed + 1)
        out.append(row)
    return out


def aligned_curve(rows, max_back=8, n_boot=2000, seed=0):
    """Mean projection at offsets -1, -2, ... from the submit turn, cluster
    bootstrap over rollouts. Only rollouts that submitted contribute."""
    rng = np.random.default_rng(seed)
    by_roll = {}
    for r in rows:
        if not np.isfinite(r["dt"]):
            continue
        off = -int(r["dt"])                            # -1 = last SET before submit
        if off < -max_back:
            continue
        by_roll.setdefault(r["rollout"], {})[off] = r["proj"]
    offs = list(range(-max_back, 0))
    rolls = list(by_roll)
    mean, lo, hi, n = [], [], [], []
    for off in offs:
        vals = np.array([by_roll[g][off] for g in rolls if off in by_roll[g]], float)
        n.append(int(len(vals)))
        if len(vals) == 0:
            mean.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            continue
        mean.append(float(vals.mean()))
        if len(vals) >= 3:
            gs = [g for g in rolls if off in by_roll[g]]
            stats = []
            for _ in range(n_boot):
                pick = rng.choice(len(gs), size=len(gs), replace=True)
                stats.append(np.mean(vals[pick]))
            lo.append(float(np.percentile(stats, 2.5)))
            hi.append(float(np.percentile(stats, 97.5)))
        else:
            lo.append(np.nan); hi.append(np.nan)
    return dict(offsets=offs, mean=mean, lo=lo, hi=hi, n=n)


def plot_all(table, curve, out_png, layer, run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # left: event-aligned curve
    xs = curve["offsets"]
    m = np.array(curve["mean"], float)
    lo = np.array(curve["lo"], float); hi = np.array(curve["hi"], float)
    ax1.plot(xs, m, "-o", color="tab:blue")
    ok = ~(np.isnan(lo) | np.isnan(hi))
    ax1.fill_between(np.array(xs)[ok], lo[ok], hi[ok], alpha=0.2, color="tab:blue")
    for x, nn in zip(xs, curve["n"]):
        ax1.annotate(str(nn), (x, ax1.get_ylim()[0]), fontsize=7,
                     ha="center", va="bottom", color="gray",
                     xycoords=("data", "axes fraction"),
                     xytext=(x, 0.02), textcoords=("data", "axes fraction"))
    ax1.axhline(0, color="gray", ls="--", lw=0.8)
    ax1.set_xlabel("SET turn, offset from submit turn (-1 = last SET)")
    ax1.set_ylabel("projection (SET=-1, SUBMIT=+1)")
    ax1.set_title("Event-aligned projection (cluster-boot 95% band; n per offset)")
    ax1.grid(alpha=0.3)

    # right: AUROC vs horizon m
    ms = [r["m"] for r in table if r.get("auc_proj") is not None]
    ap = [r["auc_proj"] for r in table if r.get("auc_proj") is not None]
    at = [r["auc_turn"] for r in table if r.get("auc_proj") is not None]
    ar = [r["auc_resid"] for r in table if r.get("auc_proj") is not None]
    w = 0.25
    x = np.arange(len(ms))
    ax2.bar(x - w, ap, w, label="projection", color="tab:blue", alpha=0.8)
    ax2.bar(x, at, w, label="turn index", color="tab:gray", alpha=0.8)
    ax2.bar(x + w, ar, w, label="proj | turn (residual)", color="tab:green", alpha=0.8)
    for r, xi in zip([r for r in table if r.get("auc_proj") is not None], x):
        ci = r.get("auc_resid_ci95")
        if ci and ci[0] is not None:
            ax2.plot([xi + w, xi + w], ci, color="k", lw=1.2)
    ax2.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels([f"m={m}" for m in ms])
    ax2.set_ylim(0.3, 1.0)
    ax2.set_ylabel("AUROC: submits within next m turns")
    ax2.set_title("Anticipation vs lexical confound\n"
                  "(green ~0.5 for m>=2 => mostly next-token anticipation)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis="y")
    fig.suptitle(f"{os.path.basename(run_dir)}  L{layer}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"[anticipate] wrote {out_png}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    cfg = Config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="a captured (unsteered) run dir: rollout_*/activations/")
    ap.add_argument("--directions", default=None,
                    help="directions.npz (default <run-dir>/directions.npz, else "
                         "pass the source run's file explicitly)")
    ap.add_argument("--layer", type=int, default=None,
                    help="hidden-state index (default: best_layer in directions)")
    ap.add_argument("--pool", choices=["before", "around", "all"], default="before")
    ap.add_argument("--win", type=int, default=4)
    ap.add_argument("--ms", type=int, nargs="+", default=[1, 2, 3, 4],
                    help="horizons m for 'submits within next m turns'")
    ap.add_argument("--max-back", type=int, default=8,
                    help="offsets for the event-aligned curve")
    ap.add_argument("--include-forced", action="store_true",
                    help="include never-submitted rollouts as all-negative turns")
    ap.add_argument("--model", default=cfg.model_name,
                    help="HF id (tokenizer only, for verb-centred pooling)")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    directions = args.directions or os.path.join(args.run_dir, "directions.npz")
    if not os.path.exists(directions):
        raise SystemExit(f"{directions} not found; pass --directions explicitly.")
    layer = args.layer
    if layer is None:
        z = np.load(directions, allow_pickle=True)
        if "best_layer" not in z.files:
            raise SystemExit("no best_layer in directions; pass --layer.")
        layer = int(z["best_layer"])
        print(f"[anticipate] using saved best_layer = {layer}")

    tok = None
    try:
        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained(args.model)
        except Exception:
            tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except Exception as e:
        print(f"[anticipate] no tokenizer ({e}); pooling all captured tokens "
              "(pre-verb exclusion unavailable -- interpret with care).")

    rows, submit_turns = collect_rows(args.run_dir, directions, layer, tok,
                                      args.pool, args.win,
                                      include_forced=args.include_forced)
    n_roll = len({r["rollout"] for r in rows})
    if not rows:
        raise SystemExit("no usable SET turns found (capture on? transcripts?).")
    print(f"[anticipate] {len(rows)} SET turns across {n_roll} rollouts "
          f"({sum(1 for t in submit_turns.values() if t is not None)} submitted)")

    table = auroc_table(rows, ms=args.ms, n_boot=args.n_boot)
    print(f"\n[anticipate] AUROC for 'submits within next m turns' "
          f"(pool={args.pool}, L{layer}):")
    print(f"   {'m':>3s} {'n':>5s} {'pos':>5s} {'proj':>7s} {'turn':>7s} "
          f"{'proj|turn':>10s}  resid 95% CI")
    for r in table:
        if r.get("auc_proj") is None:
            print(f"   {r['m']:>3d} {r['n']:>5d} {r['n_pos']:>5d}   (degenerate label)")
            continue
        ci = r.get("auc_resid_ci95") or (None, None)
        cis = (f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci[0] is not None else "")
        print(f"   {r['m']:>3d} {r['n']:>5d} {r['n_pos']:>5d} "
              f"{r['auc_proj']:7.3f} {r['auc_turn']:7.3f} {r['auc_resid']:10.3f}  {cis}")
    print("\n   reading: proj|turn ~0.5 for m>=2 while m=1 is high  -> mostly "
          "lexical/next-turn anticipation;\n"
          "            proj|turn clearly >0.5 at m=2,3  -> genuine multi-turn "
          "anticipation of the stop decision.")

    curve = aligned_curve(rows, max_back=args.max_back, n_boot=args.n_boot)

    out_json = os.path.join(args.run_dir, f"anticipation_L{layer}.json")
    with open(out_json, "w") as f:
        json.dump(dict(run_dir=args.run_dir, directions=directions, layer=layer,
                       pool=args.pool, win=args.win, n_rows=len(rows),
                       n_rollouts=n_roll, table=table, curve=curve), f, indent=2)
    print(f"[anticipate] wrote {out_json}")
    plot_all(table, curve, os.path.join(args.run_dir, f"anticipation_L{layer}.png"),
             layer, args.run_dir)


if __name__ == "__main__":
    main()