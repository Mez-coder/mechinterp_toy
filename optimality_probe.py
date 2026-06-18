"""Find the 'I have actually reached the optimum' direction -- discrete sandbox only.

Why this works where the continuous version didn't: on a 0.1 grid the constrained
optimum is an exact CELL, so we can label by behaviour instead of coordinates:

  converged (y=1)  : not forced, submitted the optimal cell  (it reached the wall
                     and chose to stop there)
  stopped-short(y=0): not forced, submitted a DIFFERENT feasible cell (it chose to
                     stop while a better cell existed -- your (0.4,0.9) vs (0.4,1.0))

Forced submits are excluded from both: running out of turns is not a belief about
optimality.

The payoff of the grid: one cell short of optimum costs almost no margin
(diminishing returns), so the two classes have NEARLY EQUAL visible priority
margins. Therefore:
  - RAW AUC  = can the activation tell converged from stopped-short at all
  - RESID AUC (regress out visible margin) should be ~ the same as RAW, because
    margin barely differs between the classes. RAW>>chance AND RESID~RAW is the
    clean signature of a genuine optimality belief that is NOT a margin proxy.
    (Contrast with the continuous runs, where RESID collapsed because there the
    classes differed mostly in margin.)

Labelled at the SUBMIT decision (optionally the last `--window` turns before it),
grouped-CV by rollout, difference-of-means direction, random-direction control.
Saves the chosen direction to <run>/optimality_probe.npz for the steering harness.

    python optimality_probe.py --run runs/csat_discrete
    python optimality_probe.py --run runs/csat_discrete --window 3

Needs probe_pilot.py and contrast_sweep.py in the same dir.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from probe_pilot import LOCI, reduce_loci, load_npz_acts
from contrast_sweep import margins_vec, separation, cohen_d


def _cell_eq(a, b, tol=0.05):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and np.all(np.abs(a - b) <= tol)


def build(run_dir, window=1, layer_stride=2, k=5):
    rollouts = sorted(glob.glob(os.path.join(run_dir, "rollout_*")))
    X = {loc: [] for loc in LOCI}
    groups, Mfeat, y = [], [], []
    layer_idx = None
    n_forced = n_nogap = 0

    for gi, rd in enumerate(rollouts):
        cf, sf, tf = (os.path.join(rd, x) for x in
                      ("case.json", "submission.json", "transcript.jsonl"))
        if not all(os.path.exists(p) for p in (cf, sf, tf)):
            continue
        case, sub = json.load(open(cf)), json.load(open(sf))
        if sub.get("forced"):                       # ran out of turns -> not a belief
            n_forced += 1; continue
        opt_w = sub.get("optimum_weights")
        if opt_w is None or not sub.get("optimum_feasible", True):
            n_nogap += 1; continue
        label = 1 if _cell_eq(sub["weights"], opt_w) else 0
        prio = int(case["priority"])

        recs = [json.loads(l) for l in open(tf) if l.strip()]
        sub_turns = [r["turn"] for r in recs if r.get("action") == "submit"]
        if not sub_turns:
            continue
        st = sub_turns[-1]

        for r in recs:
            t = r["turn"]
            if not (st - window < t <= st):
                continue
            actfile = (r.get("meta") or {}).get("activations")
            if not actfile:
                continue
            path = os.path.join(rd, "activations", actfile)
            if not os.path.exists(path):
                continue
            plan_w = sub["weights"] if t == st else r.get("weight_vec")
            m = margins_vec(plan_w, case)
            others = np.delete(m, prio)
            acts = load_npz_acts(path)
            if layer_idx is None:
                layer_idx = list(range(0, acts.shape[0], layer_stride))
            lv = reduce_loci(acts, k=k)
            for loc in LOCI:
                X[loc].append(lv[loc][layer_idx])
            groups.append(gi)
            Mfeat.append([float(m[prio]), float(others.min()),
                          float(others.mean()), float((m >= 0).sum())])
            y.append(label)

    if not y:
        raise SystemExit("no usable submit turns -- check the run has non-forced "
                         "submissions with optimum_weights and captured activations.")
    X = {loc: np.stack(v).astype(np.float32) for loc, v in X.items()}
    return dict(X=X, groups=np.asarray(groups), Mfeat=np.asarray(Mfeat, float),
                y=np.asarray(y), layers=np.asarray(layer_idx),
                n_forced=n_forced, n_nogap=n_nogap)


def sweep(d, n_splits, resid, random_ctrl=False):
    out = {}
    rng = np.random.default_rng(0) if random_ctrl else None
    for loc in LOCI:
        Xloc = d["X"][loc]
        cells = [separation(Xloc[:, j, :], d["Mfeat"], d["y"], d["groups"],
                            resid, n_splits, rng=rng)
                 for j in range(Xloc.shape[1])]
        cells = np.array(cells)
        out[loc] = cells
    return out


def best_nonL0(out, layers):
    best = None
    for loc, cells in out.items():
        for j, l in enumerate(layers):
            if int(l) == 0:
                continue                            # L0 = embeddings, surface artifact
            a = cells[j]
            if not np.isnan(a) and (best is None or a > best[2]):
                best = (loc, int(l), float(a), j)
    return best


def table(title, layers, out):
    print(f"\n{title}")
    print(f"{'locus':12s} " + " ".join(f"L{int(l):02d}" for l in layers))
    for loc in LOCI:
        print(f"{loc:12s} " + " ".join(
            f"{c:.2f}" if not np.isnan(c) else "  - " for c in out[loc]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--window", type=int, default=1, help="turns up to & incl submit")
    ap.add_argument("--layer-stride", type=int, default=2)
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()

    d = build(args.run, window=args.window, layer_stride=args.layer_stride)
    layers, y = d["layers"], d["y"]
    n_groups = len(np.unique(d["groups"]))
    ns = max(2, min(args.splits, n_groups))

    print(f"\nsubmit samples: {len(y)}  rollouts: {n_groups}  window: {args.window}")
    print(f"converged (optimal cell): {int(y.sum())}   stopped-short: {int((y == 0).sum())}")
    print(f"excluded -> forced: {d['n_forced']}   no optimum: {d['n_nogap']}")
    if y.sum() < 5 or (y == 0).sum() < 5:
        print("  !! a class is tiny -- tune case difficulty for ~40-60% converge rate, "
              "or add rollouts, before trusting AUCs.")

    rc = sweep(d, ns, resid=False, random_ctrl=True)
    rand = np.nanmean(np.concatenate([v for v in rc.values()]))
    print(f"random-direction control AUC ~ {rand:.3f}  (chance 0.500)")

    raw = sweep(d, ns, resid=False)
    res = sweep(d, ns, resid=True)
    table("converged - stopped-short  RAW   (steering direction quality)", layers, raw)
    table("converged - stopped-short  RESID (beyond visible margin)", layers, res)

    braw = best_nonL0(raw, layers)
    print(f"\nbest mid-layer RAW : '{braw[0]}' @ L{braw[1]}  AUC {braw[2]:.3f}")
    j = braw[3]
    res_here = res[braw[0]][j]
    print(f"  RESID at that cell : {res_here:.3f}   "
          f"(RESID ~ RAW and both >> {rand:.2f}  => optimality belief, not a margin proxy)")

    # --- save the steering direction at the chosen (locus, layer) over ALL data ---
    loc, lab_layer, jbest = braw[0], braw[1], braw[3]
    Xsel = d["X"][loc][:, jbest, :]
    mu, sd = Xsel.mean(0), Xsel.std(0) + 1e-6
    Z = (Xsel - mu) / sd
    v = Z[y == 1].mean(0) - Z[y == 0].mean(0)
    v = v / (np.linalg.norm(v) + 1e-12)
    out_path = os.path.join(args.run, "optimality_probe.npz")
    np.savez(out_path, direction=v.astype(np.float32),
             feat_mean=mu.astype(np.float32), feat_std=sd.astype(np.float32),
             locus=loc, layer=int(lab_layer),
             auc_raw=float(braw[2]), auc_resid=float(res_here),
             n_pos=int(y.sum()), n_neg=int((y == 0).sum()))
    d_proj = cohen_d(Z @ v, y)
    print(f"\nsaved probe -> {out_path}")
    print(f"  locus={loc}  layer={lab_layer}  in-sample Cohen d {d_proj:+.2f}")
    print("  load 'direction' + 'layer' in the steering harness; add -alpha*direction "
          "at the commit token of a stopped-short rollout and watch for SUBMIT->SET.")


if __name__ == "__main__":
    main()