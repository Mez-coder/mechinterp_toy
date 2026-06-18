"""Offline pilot: WHICH token locus (and layer) best carries the model's latent
read on plan optimality, *beyond what the visible margin already tells us*.

No GPU, no model. Reads what a capture run already wrote under
runs/<run_name>/rollout_XXXX/:
    case.json          m0/G/C/beta/priority           -> recompute true margins
    submission.json    optimum_margin_priority        -> the hidden optimum
    transcript.jsonl   per-turn weight_vec, resp_len, activations filename
    activations/turn_XX.npz   acts (L+1, n_pos, d)     -> the residual stream

Run the capture first with cfg.capture_tokens = 'assistant' (every generated
token) so this script has the full token axis to slice. Then:

    python probe_pilot.py --run runs/csat
    python probe_pilot.py --run runs/csat --layer-stride 1 --pca 64

WHAT IT MEASURES
----------------
Label for turn t = the gap of the plan the model was LOOKING AT that turn:
        gap_t = optimum_margin_priority  -  priority_margin(w_observed_{t})
where w_observed_t is the weight vector that produced the table shown at turn t
(i.e. the PREVIOUS turn's SET result; zeros at turn 1). This is the model's read
on a state it has actually seen -- not on the weights it is about to propose.

The number that matters is NOT the probe's R2. It is

        dR2 = R2(activation probe)  -  R2(observable baseline)

The observable baseline sees [visible priority margin, turn, resp_len] -- i.e.
everything the model can read straight off the prompt. A probe that only decodes
the visible margin scores ~0 here. Positive dR2 on HELD-OUT problems is the part
of the gap the model must be estimating internally: the hidden optimum.

WHY GROUPED-CV BY ROLLOUT IS LOAD-BEARING
-----------------------------------------
Within one problem the optimum is constant, so gap = const - visible_margin and
any probe "wins" by decoding a number that's already in the prompt. We therefore
hold out WHOLE rollouts (each rollout = one jittered problem); on unseen problems
the constant differs, so a real win requires representing the achievable optimum.
=> The pilot is only meaningful if optimum_margin_priority actually VARIES across
   rollouts. The script prints that spread; if it's ~flat, raise cfg.case_jitter
   (or vary n_obj) and recapture before trusting any dR2.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np

try:                                   # registers the bfloat16 dtype for np.load
    import ml_dtypes  # noqa: F401
except Exception:
    ml_dtypes = None

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score


# --------------------------------------------------------------------------- #
# ground-truth recompute (mirror of CouplingEnv.margins; pure numpy)          #
# --------------------------------------------------------------------------- #
def priority_margin(w, case):
    w = np.clip(np.asarray(w, float), 0.0, 1.0)
    m0 = np.asarray(case["m0"], float)
    G = np.asarray(case["G"], float)
    C = np.asarray(case["C"], float)
    beta = float(case["beta"])
    gain = 1.0 - np.exp(-beta * w)
    harm = w * w
    m = m0 + G * gain - C.dot(harm)
    return float(m[int(case["priority"])])


# --------------------------------------------------------------------------- #
# token-locus reduction: acts (L+1, n_pos, d) -> {locus: (L+1, d)}            #
# --------------------------------------------------------------------------- #
def reduce_loci(acts, k=5):
    """One vector per (locus, layer). Token axis is collapsed here so we never
    hold the full (L+1, n_pos, d) tensor for every turn in RAM."""
    n_pos = acts.shape[1]
    kk = max(1, min(k, n_pos))
    lo, hi = n_pos // 3, max(n_pos // 3 + 1, (2 * n_pos) // 3)   # middle third
    return {
        "first":      acts[:, 0, :],
        "first5_mean": acts[:, :kk, :].mean(axis=1),
        "mid_mean":   acts[:, lo:hi, :].mean(axis=1),
        "last5_mean": acts[:, -kk:, :].mean(axis=1),
        "last":       acts[:, -1, :],            # commit/newline token (recorder's decision idx)
        "all_mean":   acts.mean(axis=1),
    }


LOCI = ["first", "first5_mean", "mid_mean", "last5_mean", "last", "all_mean"]


def load_npz_acts(path):
    d = np.load(path)
    acts = d["acts"]
    if acts.dtype != np.float32:
        if ml_dtypes is None:
            raise RuntimeError(
                f"{path} is stored as {acts.dtype} but ml_dtypes is not installed; "
                "`pip install ml_dtypes` to read bfloat16 captures.")
        acts = acts.astype(np.float32)
    return acts                              # (L+1, n_pos, d)


# --------------------------------------------------------------------------- #
# dataset assembly                                                            #
# --------------------------------------------------------------------------- #
def build_dataset(run_dir, layer_stride=2, k=5, max_rollouts=None):
    rollouts = sorted(glob.glob(os.path.join(run_dir, "rollout_*")))
    if max_rollouts:
        rollouts = rollouts[:max_rollouts]
    if not rollouts:
        raise SystemExit(f"no rollout_* dirs under {run_dir}")

    X = {loc: [] for loc in LOCI}            # each -> list of (n_layers, d)
    y, groups, base = [], [], []             # gap, rollout idx, [margin, turn, resp_len]
    opt_per_problem = []
    layer_idx = None
    n_skip_infeasible = 0

    for gi, rd in enumerate(rollouts):
        cf = os.path.join(rd, "case.json")
        sf = os.path.join(rd, "submission.json")
        tf = os.path.join(rd, "transcript.jsonl")
        if not (os.path.exists(cf) and os.path.exists(sf) and os.path.exists(tf)):
            continue
        case = json.load(open(cf))
        sub = json.load(open(sf))
        if not sub.get("optimum_feasible") or sub.get("optimum_margin_priority") is None:
            n_skip_infeasible += 1
            continue
        opt_mp = float(sub["optimum_margin_priority"])
        opt_per_problem.append(opt_mp)

        records = [json.loads(l) for l in open(tf) if l.strip()]
        prev_w = [0.0] * int(case["n_obj"])          # state shown at turn 1 = zeros
        for rec in records:
            actfile = (rec.get("meta") or {}).get("activations")
            if not actfile:                          # forced_submit etc. -> no capture
                prev_w = rec.get("weight_vec", prev_w)
                continue
            path = os.path.join(rd, "activations", actfile)
            if not os.path.exists(path):
                prev_w = rec.get("weight_vec", prev_w)
                continue

            observed_w = prev_w                      # the table the model READ this turn
            gap = opt_mp - priority_margin(observed_w, case)
            vis_margin = priority_margin(observed_w, case)
            resp_len = float((rec.get("meta") or {}).get("resp_len", 0))

            acts = load_npz_acts(path)               # (L+1, n_pos, d)
            if layer_idx is None:
                layer_idx = list(range(0, acts.shape[0], layer_stride))
            loci = reduce_loci(acts, k=k)
            for loc in LOCI:
                X[loc].append(loci[loc][layer_idx])  # (n_layers, d)

            y.append(gap)
            groups.append(gi)
            base.append([vis_margin, float(rec["turn"]), resp_len])
            prev_w = rec.get("weight_vec", prev_w)   # advance observed state

    if not y:
        raise SystemExit("no captured turns found -- did you run with "
                         "cfg.capture=True and cfg.capture_tokens='assistant'?")

    X = {loc: np.stack(v).astype(np.float32) for loc, v in X.items()}   # (N, n_layers, d)
    return (X, np.asarray(y), np.asarray(groups), np.asarray(base),
            np.asarray(layer_idx), np.asarray(opt_per_problem), n_skip_infeasible)


# --------------------------------------------------------------------------- #
# evaluation                                                                  #
# --------------------------------------------------------------------------- #
def cv_r2(Xmat, y, groups, n_splits, pca, alphas):
    steps = [StandardScaler()]
    if pca:
        steps.append(PCA(n_components=min(pca, Xmat.shape[0] - n_splits, Xmat.shape[1])))
    steps.append(RidgeCV(alphas=alphas))
    pipe = make_pipeline(*steps)
    cv = GroupKFold(n_splits=n_splits)
    pred = cross_val_predict(pipe, Xmat, y, groups=groups, cv=cv)
    return r2_score(y, pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir, e.g. runs/csat")
    ap.add_argument("--layer-stride", type=int, default=2)
    ap.add_argument("--k", type=int, default=5, help="window for first5/last5 means")
    ap.add_argument("--pca", type=int, default=50, help="PCA comps before ridge (0=off)")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--max-rollouts", type=int, default=None)
    args = ap.parse_args()

    X, y, groups, base, layers, opt, n_inf = build_dataset(
        args.run, layer_stride=args.layer_stride, k=args.k, max_rollouts=args.max_rollouts)

    n_groups = len(np.unique(groups))
    n_splits = max(2, min(args.splits, n_groups))
    alphas = np.logspace(-1, 5, 13)

    print(f"\nsamples (model turns): {len(y)}   rollouts/problems: {n_groups}"
          f"   layers swept: {len(layers)}   d_model: {X['first'].shape[-1]}")
    if n_inf:
        print(f"skipped {n_inf} rollouts with infeasible optimum")
    print(f"gap (label)  range [{y.min():+.3f}, {y.max():+.3f}]  mean {y.mean():+.3f}  std {y.std():.3f}")
    print(f"optimum_margin_priority across problems  std {opt.std():.4f}  "
          f"range [{opt.min():.3f}, {opt.max():.3f}]")
    if opt.std() < 1e-3:
        print("  !! optimum barely varies across problems -- dR2 is NOT trustworthy.\n"
              "     raise cfg.case_jitter (or vary n_obj) and recapture.")

    # ---- observable baseline: everything the model can read off the prompt ----
    base_r2 = cv_r2(base, y, groups, n_splits, pca=0, alphas=alphas)
    print(f"\nobservable baseline R2 (margin + turn + resp_len): {base_r2:+.3f}")
    print("  -> probe must beat THIS. dR2 = probe R2 - baseline R2\n")

    # ---- sweep locus x layer ----
    print(f"{'locus':12s} " + " ".join(f"L{int(l):02d}" for l in layers))
    best = None
    summary = {}
    for loc in LOCI:
        cells, best_layer, best_d = [], None, -1e9
        for j, l in enumerate(layers):
            r2 = cv_r2(X[loc][:, j, :], y, groups, n_splits, pca=args.pca, alphas=alphas)
            d = r2 - base_r2
            cells.append(d)
            if d > best_d:
                best_d, best_layer = d, int(l)
            if best is None or d > best[2]:
                best = (loc, int(l), d, r2)
        summary[loc] = (best_layer, best_d)
        row = " ".join(f"{c:+.2f}" for c in cells)
        print(f"{loc:12s} {row}")

    print("\nbest dR2 per locus (layer):")
    for loc in LOCI:
        bl, bd = summary[loc]
        print(f"  {loc:12s} dR2 {bd:+.3f}  @ layer {bl}")
    print(f"\nWINNER: locus '{best[0]}' @ layer {best[1]}  "
          f"dR2 {best[2]:+.3f}  (probe R2 {best[3]:+.3f}, baseline {base_r2:+.3f})")
    print("\nFreeze this (locus, layer) and recapture the full run at that single "
          "position; everything downstream -- and the steering vector -- reads here.")


if __name__ == "__main__":
    main()