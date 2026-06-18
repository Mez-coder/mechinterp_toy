"""Contrast-direction sweep -- the follow-up to probe_pilot.py.

The pilot asked "is the continuous gap LINEARLY DECODABLE beyond the visible
margin?" and answered ~no (dR2 ~ 0). This script asks the better-posed question
behind the three-regime hypothesis: are there difference-of-means DIRECTIONS that
separate plan states, and does any of them survive removing the visible margin?

Two contrasts, per (token locus, layer):

  A) pass - fail            all_pass(observed state) vs not.
                            RAW only is meaningful (pass/fail is DEFINED by margin
                            signs, so residualising it is circular -> ~chance; we
                            print it purely as a sanity check that residualisation
                            works). RAW pass-fail is your strongest steering axis
                            and a positive control: if it doesn't separate, the
                            loci/layers are wrong and nothing else will work.

  B) near-optimal - barely-passing   among PASSING turns only, top vs bottom third
                            by per-rollout optimality fraction (priority margin /
                            achievable optimum). Pass/fail is held fixed across both
                            classes, so this isolates "getting better WITHIN the
                            feasible region".
                            RAW  = the "getting better" steering vector (the knob).
                            RESID = same after regressing out the full visible
                                    margin vector. THIS is the headline test: a
                                    surviving separation means the model encodes the
                                    achievable optimum beyond the margin it's shown.
                                    It only has teeth because the optimum varies
                                    across rollouts (margin no longer determines the
                                    fraction) -- the spread line below must be > 0.

Everything is labelled by the OBSERVED state (the table shown that turn = previous
turn's weights), so the captured activation -- the model reading that table -- is
aligned with the label. Directions are difference-of-means in standardised
activation space (the standard steering construction; no PCA). Separation is
held-out AUC under grouped-CV by rollout, with a random-direction control (~0.5).

    python contrast_sweep.py --run runs/csat
    python contrast_sweep.py --run runs/csat --layer-stride 1

Requires probe_pilot.py in the same dir (reuses its loaders).
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from probe_pilot import LOCI, reduce_loci, load_npz_acts   # reuse pilot helpers


# --------------------------------------------------------------------------- #
# ground truth: full margin vector for the observed state                     #
# --------------------------------------------------------------------------- #
def margins_vec(w, case):
    w = np.clip(np.asarray(w, float), 0.0, 1.0)
    m0 = np.asarray(case["m0"], float)
    G = np.asarray(case["G"], float)
    C = np.asarray(case["C"], float)
    beta = float(case["beta"])
    return m0 + G * (1.0 - np.exp(-beta * w)) - C.dot(w * w)        # (n_obj,)


# --------------------------------------------------------------------------- #
# dataset: one row per captured model turn, labelled by the OBSERVED state     #
# --------------------------------------------------------------------------- #
def build(run_dir, layer_stride=2, k=5, max_rollouts=None):
    rollouts = sorted(glob.glob(os.path.join(run_dir, "rollout_*")))
    if max_rollouts:
        rollouts = rollouts[:max_rollouts]

    X = {loc: [] for loc in LOCI}
    groups, Mfeat, allpass, frac, opt_list, prio_margin = [], [], [], [], [], []
    layer_idx = None

    for gi, rd in enumerate(rollouts):
        cf, sf, tf = (os.path.join(rd, x) for x in
                      ("case.json", "submission.json", "transcript.jsonl"))
        if not all(os.path.exists(p) for p in (cf, sf, tf)):
            continue
        case, sub = json.load(open(cf)), json.load(open(sf))
        if not sub.get("optimum_feasible") or not sub.get("optimum_margin_priority"):
            continue
        opt_mp = float(sub["optimum_margin_priority"])
        if opt_mp <= 0:
            continue
        opt_list.append(opt_mp)
        prio = int(case["priority"])

        recs = [json.loads(l) for l in open(tf) if l.strip()]
        prev_w = [0.0] * int(case["n_obj"])
        for rec in recs:
            actfile = (rec.get("meta") or {}).get("activations")
            if actfile:
                path = os.path.join(rd, "activations", actfile)
                if os.path.exists(path):
                    m = margins_vec(prev_w, case)              # observed state
                    pm = float(m[prio])
                    others = np.delete(m, prio)
                    ap = bool(np.all(m >= 0))
                    acts = load_npz_acts(path)
                    if layer_idx is None:
                        layer_idx = list(range(0, acts.shape[0], layer_stride))
                    loc_vecs = reduce_loci(acts, k=k)
                    for loc in LOCI:
                        X[loc].append(loc_vecs[loc][layer_idx])
                    groups.append(gi)
                    # ALIGNED, permutation-invariant visible features. Priority is a
                    # different column per rollout, so we pull the priority margin out
                    # as its own column; otherwise residualising can't remove it.
                    Mfeat.append([pm, float(others.min()), float(others.mean()),
                                  float((m >= 0).sum())])
                    allpass.append(ap)
                    prio_margin.append(pm)
                    frac.append(pm / opt_mp if ap else np.nan)
            prev_w = rec.get("weight_vec", prev_w)

    if not groups:
        raise SystemExit("no captured turns -- capture with cfg.capture_tokens='assistant'")
    X = {loc: np.stack(v).astype(np.float32) for loc, v in X.items()}
    return dict(X=X, groups=np.asarray(groups), Mfeat=np.asarray(Mfeat, float),
                allpass=np.asarray(allpass), frac=np.asarray(frac),
                prio_margin=np.asarray(prio_margin),
                layers=np.asarray(layer_idx), opt=np.asarray(opt_list))
    # Mfeat columns = [priority_margin, min_other_margin, mean_other_margin, n_pass]


# --------------------------------------------------------------------------- #
# held-out separation along a difference-of-means direction (grouped-CV)       #
# --------------------------------------------------------------------------- #
def _resid_fit(Xtr, Mtr):
    M1 = np.hstack([np.ones((len(Mtr), 1)), Mtr])
    beta, *_ = np.linalg.lstsq(M1, Xtr, rcond=None)
    return beta

def _resid_apply(X, M, beta):
    M1 = np.hstack([np.ones((len(M), 1)), M])
    return X - M1 @ beta


def separation(Xll, M, y, groups, resid, n_splits, rng=None):
    """y in {0,1}. Returns held-out AUC for the train-derived (meanA-meanB)
    direction. rng != None -> use a random direction instead (control)."""
    cv = GroupKFold(n_splits=n_splits)
    oof_proj = np.full(len(y), np.nan)
    for tr, te in cv.split(Xll, y, groups):
        mu, sd = Xll[tr].mean(0), Xll[tr].std(0) + 1e-6
        Xtr, Xte = (Xll[tr] - mu) / sd, (Xll[te] - mu) / sd
        if resid:
            beta = _resid_fit(Xtr, M[tr])
            Xtr = Xtr - np.hstack([np.ones((len(tr), 1)), M[tr]]) @ beta
            Xte = _resid_apply(Xte, M[te], beta)
        if rng is not None:
            v = rng.standard_normal(Xtr.shape[1])
        else:
            v = Xtr[y[tr] == 1].mean(0) - Xtr[y[tr] == 0].mean(0)
        v = v / (np.linalg.norm(v) + 1e-12)
        oof_proj[te] = Xte @ v
    ok = ~np.isnan(oof_proj)
    if len(np.unique(y[ok])) < 2:
        return np.nan
    return roc_auc_score(y[ok], oof_proj[ok])


def cohen_d(proj, y):
    a, b = proj[y == 1], proj[y == 0]
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                / max(len(a) + len(b) - 2, 1) + 1e-12)
    return (a.mean() - b.mean()) / s


# --------------------------------------------------------------------------- #
def sweep(data, contrast, n_splits, resid, random_ctrl=False):
    """contrast: 'passfail' or 'nearbarely'. Returns {(locus): (best_layer, best_auc, cells)}."""
    g, ap, fr = data["groups"], data["allpass"], data["frac"]
    if contrast == "passfail":
        mask = np.ones(len(g), bool)
        y = ap.astype(int)
    else:                                     # near vs barely, passing only, tertiles
        pas = ap & ~np.isnan(fr)
        q33, q66 = np.nanquantile(fr[pas], [1/3, 2/3])
        A = pas & (fr >= q66)                 # near-optimal
        B = pas & (fr <= q33)                 # barely passing
        mask = A | B
        y = np.where(A, 1, 0)[mask]
    M, gm = data["Mfeat"][mask], g[mask]
    out = {}
    for loc in LOCI:
        Xloc = data["X"][loc][mask]           # (n, n_layers, d)
        cells = []
        for j in range(Xloc.shape[1]):
            rng = np.random.default_rng(0) if random_ctrl else None
            auc = separation(Xloc[:, j, :], M, y, gm, resid, n_splits, rng=rng)
            cells.append(auc)
        cells = np.array(cells)
        bj = int(np.nanargmax(cells))
        out[loc] = (int(data["layers"][bj]), float(cells[bj]), cells)
    return out


def print_table(title, layers, out):
    print(f"\n{title}")
    print(f"{'locus':12s} " + " ".join(f"L{int(l):02d}" for l in layers))
    for loc in LOCI:
        _, _, cells = out[loc]
        print(f"{loc:12s} " + " ".join(f"{c:.2f}" if not np.isnan(c) else "  - " for c in cells))
    best = max(out.items(), key=lambda kv: (-1 if np.isnan(kv[1][1]) else kv[1][1]))
    print(f"  best: '{best[0]}' @ L{best[1][0]}  AUC {best[1][1]:.3f}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--layer-stride", type=int, default=2)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--max-rollouts", type=int, default=None)
    args = ap.parse_args()

    d = build(args.run, layer_stride=args.layer_stride, k=args.k, max_rollouts=args.max_rollouts)
    n_groups = len(np.unique(d["groups"]))
    ns = max(2, min(args.splits, n_groups))
    layers = d["layers"]

    pas = d["allpass"] & ~np.isnan(d["frac"])
    q33, q66 = np.nanquantile(d["frac"][pas], [1/3, 2/3])
    print(f"\nturns: {len(d['groups'])}   rollouts: {n_groups}   "
          f"passing: {int(pas.sum())}   failing: {int((~d['allpass']).sum())}")
    print(f"optimum_margin_priority across rollouts: std {d['opt'].std():.4f}  "
          f"range [{d['opt'].min():.3f}, {d['opt'].max():.3f}]")
    if d['opt'].std() < 1e-3:
        print("  !! optimum barely varies -> RESID near-barely test has no teeth. raise case_jitter.")
    print(f"near-optimal = frac>={q66:.2f}   barely-passing = frac<={q33:.2f}   "
          f"(frac = priority margin / achievable optimum)")

    # random-direction control (chance reference)
    rc = sweep(d, "passfail", ns, resid=False, random_ctrl=True)
    rand_auc = np.nanmean([v[1] for v in rc.values()])
    print(f"random-direction control AUC ~ {rand_auc:.3f}  (chance = 0.500)")

    # A) pass - fail : RAW (steering control) + RESID (sanity, expect ~chance)
    pf_raw = sweep(d, "passfail", ns, resid=False)
    print_table("[A] pass - fail  RAW   (positive control / strongest steering axis)", layers, pf_raw)
    pf_res = sweep(d, "passfail", ns, resid=True)
    print_table("[A] pass - fail  RESID (sanity: should collapse to ~chance -- pass IS the margin sign)", layers, pf_res)

    # B) near-optimal - barely-passing : RAW (knob) + RESID (headline)
    nb_raw = sweep(d, "nearbarely", ns, resid=False)
    b1 = print_table("[B] near-optimal - barely-passing  RAW   (the 'getting better' steering vector)", layers, nb_raw)
    nb_res = sweep(d, "nearbarely", ns, resid=True)
    b2 = print_table("[B] near-optimal - barely-passing  RESID (HEADLINE: optimality beyond visible margin)", layers, nb_res)

    print("\n" + "=" * 64)
    print("READ:")
    print(f"  pass/fail RAW AUC {pf_raw[max(pf_raw,key=lambda k:pf_raw[k][1])][1]:.3f}"
          f"  -> control; expect well above {rand_auc:.2f}.")
    print(f"  near-barely RAW  AUC {b1[1][1]:.3f}  -> steering-vector quality (knob).")
    print(f"  near-barely RESID AUC {b2[1][1]:.3f}  -> the answer.")
    print(f"     ~{rand_auc:.2f}  => no optimality signal beyond the margin the model is shown")
    print( "             (consistent with the pilot; steer pass/fail, not feasible-region optimality)")
    print( "     >>0.5    => a real beyond-margin optimality direction lives here; steer it & test causally")
    print("=" * 64)


if __name__ == "__main__":
    main()