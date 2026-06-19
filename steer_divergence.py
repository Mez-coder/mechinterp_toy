"""OPTIMAL vs DOUBT steering-vector divergence  (V2: PCA on the per-rollout differences).

Per rollout we take TWO captured turns:
  early  = the turn right after the FIRST SET (first_set_turn + 1) -- the model
           digesting its first feedback. Read even if it parse-errored (the
           residual stream is still processing the SET result); the last-token
           locus is just slightly less "intended" in that case.
  submit = the final SUBMIT turn, kept ONLY if claim_labels.jsonl labels it
           OPTIMAL or DOUBT (drop OTHER and forced submits).

Per (token locus, layer) cell:
  1. pooled z-score over early ∪ submit  (note: the pooled MEAN cancels in the
     submit−early difference, so only the pooled per-dim SCALE bites).
  2. per-rollout steering vector  D = z(submit) − z(early)   in d_model.
  3. reduce to d=PCA via UNCENTERED SVD of D  (origin preserved: D=0 == no
     steering). Centered PCA.transform would force the two class means
     antiparallel (cosine ≡ −1) and is wrong for a cosine-of-means test.
  → ~one steering vector per qualifying rollout, tagged OPTIMAL/DOUBT.

Step 3: cos( mean(OPTIMAL steering vecs), mean(DOUBT steering vecs) ) per cell.
  Separation shows up as cosine BELOW a label-shuffle permutation null (both
  class means ≈ grand mean under random labels → cosine ≈ 1). The cell of
  interest is the ARGMIN cosine; the permutation p says whether it's real.
  Each rollout contributes exactly one steering vector, so a plain label shuffle
  has no within-rollout leakage, and z-score/SVD are label-agnostic (fitting them
  on all rollouts doesn't taint the null).

    python steer_divergence.py --run runs/csat_discrete
    python steer_divergence.py --run runs/csat_discrete --pca 50 --perm 5000

Needs probe_pilot.py in the same dir. Captures should be cfg.capture_tokens=
'assistant' so the full token axis exists for the locus sweep.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np

from probe_pilot import LOCI, reduce_loci, load_npz_acts


# --------------------------------------------------------------------------- #
def load_claim_labels(run_dir):
    path = os.path.join(run_dir, "claim_labels.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run label_claims.py first")
    lab = {}
    for line in open(path):
        line = line.strip()
        if line:
            r = json.loads(line)
            lab[r["rollout"]] = r.get("label")
    return lab


def early_submit_turns(records):
    """(early_turn, submit_turn) per spec, or None to skip.
       early = first SET turn + 1 ; submit = last SUBMIT turn."""
    set_turns = [r["turn"] for r in records if r.get("action") == "set"]
    sub_turns = [r["turn"] for r in records if r.get("action") == "submit"]
    if not set_turns or not sub_turns:
        return None
    early, submit = set_turns[0] + 1, sub_turns[-1]
    if early == submit:        # submitted on the turn right after first SET -> no early/late split
        return None
    return early, submit


def turn_actfile(records, turn):
    for r in records:
        if r["turn"] == turn:
            return (r.get("meta") or {}).get("activations")
    return None


def build(run_dir, layer_stride=2, k=5):
    labels = load_claim_labels(run_dir)
    rollouts = sorted(glob.glob(os.path.join(run_dir, "rollout_*")))
    data, layer_idx = [], None
    skip = dict(no_files=0, no_label=0, label_other=0, forced=0,
                no_split=0, no_act=0)

    for rd in rollouts:
        rid = os.path.basename(rd)
        cf, sf, tf = (os.path.join(rd, x) for x in
                      ("case.json", "submission.json", "transcript.jsonl"))
        if not all(os.path.exists(p) for p in (cf, sf, tf)):
            skip["no_files"] += 1; continue
        lab = labels.get(rid)
        if lab is None:
            skip["no_label"] += 1; continue
        if lab not in ("OPTIMAL", "DOUBT"):
            skip["label_other"] += 1; continue
        if json.load(open(sf)).get("forced"):
            skip["forced"] += 1; continue

        records = [json.loads(l) for l in open(tf) if l.strip()]
        es = early_submit_turns(records)
        if es is None:
            skip["no_split"] += 1; continue
        et, st = es
        ea, sa = turn_actfile(records, et), turn_actfile(records, st)
        if not ea or not sa:
            skip["no_act"] += 1; continue
        ep = os.path.join(rd, "activations", ea)
        sp = os.path.join(rd, "activations", sa)
        if not (os.path.exists(ep) and os.path.exists(sp)):
            skip["no_act"] += 1; continue

        eacts, sacts = load_npz_acts(ep), load_npz_acts(sp)
        if layer_idx is None:
            layer_idx = list(range(0, eacts.shape[0], layer_stride))
        elr, slr = reduce_loci(eacts, k=k), reduce_loci(sacts, k=k)
        data.append(dict(
            rid=rid, label=lab,
            early={loc: elr[loc][layer_idx] for loc in LOCI},
            submit={loc: slr[loc][layer_idx] for loc in LOCI}))

    if not data:
        raise SystemExit("no qualifying rollouts -- check claim_labels.jsonl and captures")
    return data, np.asarray(layer_idx), skip


# --------------------------------------------------------------------------- #
def steering_vecs(E, S, n_comp):
    """E,S: (N,d) early/submit activations for one cell. Returns R: (N,nc) reduced
    per-rollout steering vectors (uncentered, origin preserved)."""
    P = np.vstack([E, S])
    sd = P.std(0) + 1e-6                      # pooled scale (pooled mean cancels in S-E)
    D = (S - E) / sd                          # (N,d) steering vector in d_model
    nc = min(n_comp, D.shape[0], D.shape[1])
    _, _, Vt = np.linalg.svd(D, full_matrices=False)   # uncentered PCA
    return D @ Vt[:nc].T                       # project, keep origin


def class_cosine(R, y):
    mo, md = R[y == 1].mean(0), R[y == 0].mean(0)
    return float(mo @ md / ((np.linalg.norm(mo) * np.linalg.norm(md)) + 1e-12))


def perm_test(R, y, n_perm, seed=0):
    rng = np.random.default_rng(seed)
    obs = class_cosine(R, y)
    null = np.array([class_cosine(R, rng.permutation(y)) for _ in range(n_perm)])
    p_low = (1 + np.sum(null <= obs)) / (1 + n_perm)   # separation => obs below null
    return obs, null, p_low


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--layer-stride", type=int, default=2)
    ap.add_argument("--k", type=int, default=5, help="window for first5/last5 means")
    ap.add_argument("--pca", type=int, default=50)
    ap.add_argument("--perm", type=int, default=5000)
    args = ap.parse_args()

    data, layers, skip = build(args.run, layer_stride=args.layer_stride, k=args.k)
    y = np.array([1 if d["label"] == "OPTIMAL" else 0 for d in data])
    n_opt, n_dbt = int((y == 1).sum()), int((y == 0).sum())

    print(f"\nqualifying rollouts: {len(data)}   OPTIMAL: {n_opt}   DOUBT: {n_dbt}")
    print("skipped: " + "  ".join(f"{kk}={vv}" for kk, vv in skip.items()))
    if n_opt < 5 or n_dbt < 5:
        print("  !! a class is tiny -- cosine/permutation are unreliable; gather more labelled submits.")

    # cosine grid over (locus, layer)
    grid = {loc: np.full(len(layers), np.nan) for loc in LOCI}
    Rcache = {}
    for loc in LOCI:
        for j in range(len(layers)):
            E = np.stack([d["early"][loc][j] for d in data]).astype(np.float64)
            S = np.stack([d["submit"][loc][j] for d in data]).astype(np.float64)
            R = steering_vecs(E, S, args.pca)
            Rcache[(loc, j)] = R
            grid[loc][j] = class_cosine(R, y)

    print(f"\ncos(mean OPTIMAL, mean DOUBT) per cell  (LOW = directions diverge):")
    print(f"{'locus':12s} " + " ".join(f"L{int(l):02d}" for l in layers))
    for loc in LOCI:
        print(f"{loc:12s} " + " ".join(f"{c:+.2f}" if not np.isnan(c) else "  -  "
                                       for c in grid[loc]))

    # argmin cell = most divergent
    flat = [(loc, j, grid[loc][j]) for loc in LOCI for j in range(len(layers))
            if not np.isnan(grid[loc][j])]
    loc_b, j_b, cos_b = min(flat, key=lambda t: t[2])
    obs, null, p = perm_test(Rcache[(loc_b, j_b)], y, args.perm)
    print(f"\nmost-divergent cell: '{loc_b}' @ L{int(layers[j_b])}   cosine {cos_b:+.3f}")
    print(f"  permutation null: mean {null.mean():+.3f}  sd {null.std():.3f}   "
          f"p(obs ≤ null) = {p:.4f}   ({args.perm} shuffles)")
    print("  -> p small AND cosine below null  => OPTIMAL and DOUBT steering vectors")
    print("     point in genuinely different directions here (the doubt axis is real).")
    print("     p ~ 0.5 / cosine ~ null  => no separable divergence at this cell.")

    # a few next-lowest cells for context
    print("\n  next most-divergent cells:")
    for loc, j, c in sorted(flat, key=lambda t: t[2])[1:5]:
        print(f"    {loc:12s} L{int(layers[j]):02d}   cosine {c:+.3f}")

    out = os.path.join(args.run, "steer_divergence.npz")
    R_b = Rcache[(loc_b, j_b)]
    np.savez(out, cosine_grid=np.vstack([grid[loc] for loc in LOCI]),
             loci=np.array(LOCI), layers=layers,
             best_locus=loc_b, best_layer=int(layers[j_b]),
             best_cosine=float(cos_b), perm_p=float(p),
             mean_optimal=R_b[y == 1].mean(0).astype(np.float32),
             mean_doubt=R_b[y == 0].mean(0).astype(np.float32),
             n_opt=n_opt, n_dbt=n_dbt)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()