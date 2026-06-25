"""direction_extract.py -- extract SET ("keep searching") vs SUBMIT ("stop here")
activation directions from captured rollouts and find the layer where they are
maximally separable.

Run as a package module, next to rollout.py / recorder.py:

    python -m csat.direction_extract --run-dir runs/csat --win 4

What it does
------------
* Walks every rollout_* dir under --run-dir.
* Per rollout, reads transcript.jsonl to learn which turn was SET / SUBMIT, loads
  the matching activations/turn_XX.npz, selects the post-MLP residual at the
  requested layers (default: even hidden_state indices >= 2), pools the tokens
  around the action verb (+/- --win) with a mean, then means the SET turns
  together -> one SET vector per rollout; the single SUBMIT turn -> one SUBMIT
  vector per rollout.
* Uses ALL rollouts (no held-out split: the steering run is a fresh run). For
  every layer it computes the separability metric

        cos(SET_all, SUBMIT_all)
      ------------------------------------------------------------
      ( SUM_ij cos(SET_i,SET_j) ) * ( SUM_ij cos(SUBMIT_i,SUBMIT_j) )

  and reports the ARGMIN layer: SET vs SUBMIT least aligned (small numerator)
  while each class is internally coherent across rollouts (large denominator).
  Identity used: SUM_ij cos(X_i,X_j) = || sum_i unit(X_i) ||^2.
* Saves the pooled SET_all / SUBMIT_all vectors for every selected layer to
  <run-dir>/directions.npz (consumed by steering.py), plus a JSON summary.

`acts[j]` (== hidden-state index j) == output of decoder block j-1 == the
post-MLP residual stream; acts[0] is the embedding output (excluded by default).

Capture-window caveat
---------------------
With the current config (capture_tokens='lastk', capture_last_k=5) only the last
5 generated tokens per turn are stored -- the run-up to and including the action
line. --win pools across whatever of that window is available and centres on the
verb when it is inside the captured tokens. For a genuine "k before AND k after",
widen the capture (raise capture_last_k, or capture_tokens='assistant').
"""
from __future__ import annotations
import os, json, glob, argparse
import numpy as np

# bfloat16 captures need ml_dtypes registered before np.load can read them.
try:
    import ml_dtypes  # noqa: F401  (registers the bf16 numpy dtype as a side effect)
    _HAVE_BF16 = True
except Exception:
    _HAVE_BF16 = False

from .config import Config

SET_VERB, SUBMIT_VERB = "SET", "SUBMIT"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _unit_rows(M, eps=1e-12):
    """Row-wise unit-normalise an (N, d) matrix (zero rows -> stay zero)."""
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(n, eps)


def _coherence(M):
    """SUM_ij cos(M_i, M_j) == || sum_i unit(M_i) ||^2  for an (N, d) stack."""
    u = _unit_rows(M)
    return float(np.dot(u.sum(0), u.sum(0)))


def load_acts(path):
    """Return (acts_f32 (L+1,n_pos,d), token_ids, positions, decision_index)."""
    try:
        with np.load(path, allow_pickle=False) as z:
            acts = np.asarray(z["acts"])
            token_ids = np.asarray(z["token_ids"])
            positions = np.asarray(z["positions"])
            decision_index = int(z["decision_index"])
    except ValueError as e:                       # usually an unknown bf16 dtype
        if not _HAVE_BF16:
            raise RuntimeError(
                f"failed to read {path}; activations look like bfloat16 but "
                "ml_dtypes is not installed. Run `uv add ml_dtypes` (or "
                "`pip install ml_dtypes`) and retry.") from e
        raise
    return acts.astype(np.float32), token_ids, positions, decision_index


def turn_actions(transcript_path):
    """turn (int) -> action kind ('set'|'submit'|'parse_error'|'forced_submit')."""
    out = {}
    if not os.path.exists(transcript_path):
        return out
    with open(transcript_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "turn" in rec and "action" in rec:
                out[int(rec["turn"])] = rec["action"]
    return out


def locate_verb(token_ids, tokenizer, verb):
    """Index (into the captured token window) of the token that starts the LAST
    occurrence of `verb`. None if not found / no tokenizer. SUBMIT does not
    contain SET, so the two verbs never collide."""
    if tokenizer is None:
        return None
    text, spans = "", []
    for t in token_ids:
        p = tokenizer.decode([int(t)])
        spans.append((len(text), len(text) + len(p)))
        text += p
    pos = text.upper().rfind(verb.upper())
    if pos == -1:
        return None
    for i, (a, b) in enumerate(spans):
        if a <= pos < b:
            return i
    return None


def window_indices(n_pos, center, win):
    """Positions to pool over. center=None -> all captured positions."""
    if center is None:
        return list(range(n_pos))
    lo, hi = max(0, center - win), min(n_pos - 1, center + win)
    return list(range(lo, hi + 1))


def _pool_indices(n_pos, center, pool, win):
    """Which captured token positions to mean-pool. 'before' = all tokens strictly
    BEFORE the verb (excludes the literal SET/SUBMIT token AND the weight digits
    that follow it). 'around' = +/-win about the verb. center=None -> all tokens."""
    if center is None:
        return list(range(n_pos))
    if pool == "before":
        idx = list(range(0, center))
        return idx or list(range(n_pos))          # verb at pos 0 (rare at lastk=30/20)
    if pool == "around":
        return list(range(max(0, center - win), min(n_pos - 1, center + win) + 1))
    return list(range(n_pos))                     # 'all'


def turn_vector(npz_path, layer_sel, verb, win, tokenizer, pool="before"):
    """Per-turn vector: mean over the pooled tokens (pool='before' by default) of
    the post-MLP residual at the selected layers.
    Returns ((n_sel,d) float32, verb_was_found, n_tokens_pooled) or None."""
    if not os.path.exists(npz_path):
        return None
    acts, token_ids, _positions, _dec = load_acts(npz_path)   # (L+1, n_pos, d)
    center = locate_verb(token_ids, tokenizer, verb)          # may be None
    idx = _pool_indices(acts.shape[1], center, pool, win)
    sel = acts[np.ix_(layer_sel, idx)]                        # (n_sel, len(idx), d)
    return sel.mean(axis=1), (center is not None), len(idx)   # (n_sel, d)


def rollout_vectors(rdir, layer_sel, win, tokenizer, pool="before"):
    """SET vector = the FINAL SET turn before SUBMIT (crisp last-STEP->SUBMIT
    contrast, no dilution by early turns). SUBMIT = the single submit turn."""
    acts_dir = os.path.join(rdir, "activations")
    kinds = turn_actions(os.path.join(rdir, "transcript.jsonl"))
    set_turns = sorted(t for t, k in kinds.items() if k == "set")
    submit_turn = next((t for t, k in kinds.items() if k == "submit"), None)

    found = {"set_centered": 0, "set_total": 0, "submit_centered": 0, "set_ntok": 0}
    set_vec = None
    if set_turns:
        t = set_turns[-1]                                     # FINAL step before submit
        res = turn_vector(os.path.join(acts_dir, f"turn_{t:02d}.npz"),
                          layer_sel, SET_VERB, win, tokenizer, pool)
        if res is not None:
            set_vec, centered, ntok = res
            found.update(set_total=1, set_centered=int(centered), set_ntok=ntok)
    submit_vec = None
    if submit_turn is not None:
        res = turn_vector(os.path.join(acts_dir, f"turn_{submit_turn:02d}.npz"),
                          layer_sel, SUBMIT_VERB, win, tokenizer, pool)
        if res is not None:
            submit_vec, centered, _ = res
            found["submit_centered"] = int(centered)
    return set_vec, submit_vec, found



def select_layers(n_total, mode, include_embed):
    """n_total = L+1 hidden-state indices (0..L). Returns the indices to keep."""
    idx = list(range(n_total)) if mode == "all" else list(range(0, n_total, 2))
    if not include_embed and 0 in idx:
        idx = [j for j in idx if j != 0]          # drop embeddings (not post-MLP)
    return idx


def _turn_proj(npz, layer, verb, tok, pool, win, mid, dirv, denom):
    """Per-turn decision coordinate at `layer`: maps SET_all->-1, SUBMIT_all->+1."""
    acts, tids, _, _ = load_acts(npz)
    if layer >= acts.shape[0]:
        return None
    center = locate_verb(tids, tok, verb) if verb else None
    idx = _pool_indices(acts.shape[1], center, pool, win)
    v = acts[layer][idx].mean(0)
    return float(2.0 * np.dot(v - mid, dirv) / denom)



# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    cfg = Config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=os.path.join(cfg.out_dir, cfg.run_name),
                    help="directory of rollout_* dirs (default: runs/<run_name>)")
    ap.add_argument("--model", default=cfg.model_name,
                    help="HF id used only to load a tokenizer for verb centering")
    ap.add_argument("--win", type=int, default=4,
                    help="tokens each side of the verb when --pool around")
    ap.add_argument("--pool", choices=["before", "around", "all"], default="before",
                    help="which captured tokens to mean-pool (default: before the verb)")
    ap.add_argument("--held-out", type=int, default=5,
                    help="N rollouts excluded from building the vectors, used for the "
                         "per-turn projection trajectory plot")
    ap.add_argument("--held-out-seed", type=int, default=0)
    ap.add_argument("--proj-layer", type=int, default=None,
                    help="layer for the trajectory plot (default: selected best layer)")
    ap.add_argument("--layers", choices=["even", "all"], default="even")
    ap.add_argument("--include-embed", action="store_true",
                    help="keep hidden-state index 0 (embeddings) in the selection")
    
    ap.add_argument("--select", choices=["separation", "ratio", "abs"],
                    default="separation",
                    help="how to pick the best layer: 'separation' (default; "
                         "maximise (1-cos)·cohSET·cohSUB), 'ratio' (your literal "
                         "cos/(coh·coh), argmin), or 'abs' (orthogonal ideal)")
    ap.add_argument("--out", default=None,
                    help="output npz (default: <run-dir>/directions.npz)")
    args = ap.parse_args()

    run_dir = args.run_dir
    out_npz = args.out or os.path.join(run_dir, "directions.npz")

    # tokenizer (optional: only used to centre the window on the verb) ----------
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        print(f"[tok] loaded tokenizer for {args.model}")
    except Exception as e:
        print(f"[tok] no tokenizer ({e}); pooling ALL captured tokens per turn "
              "instead of centering on the verb.")

    rollout_dirs = sorted(glob.glob(os.path.join(run_dir, "rollout_*")))
    rollout_dirs = [d for d in rollout_dirs if os.path.isdir(d)]
    if not rollout_dirs:
        raise SystemExit(f"no rollout_* dirs under {run_dir}")

    # determine layer selection from the first readable npz ---------------------
    n_total = None
    for rd in rollout_dirs:
        g = sorted(glob.glob(os.path.join(rd, "activations", "turn_*.npz")))
        if g:
            acts, *_ = load_acts(g[0])
            n_total, d_model = acts.shape[0], acts.shape[2]
            break
    if n_total is None:
        raise SystemExit("found rollouts but no activation npz files.")
    layer_sel = select_layers(n_total, args.layers, args.include_embed)
    print(f"[layers] hidden-states 0..{n_total-1} (d_model={d_model}); "
          f"using {len(layer_sel)} layers: {layer_sel}")

    # held-out split: build the direction on TRAIN, validate trajectories on HELD
    rngho = np.random.default_rng(args.held_out_seed)
    perm = rngho.permutation(len(rollout_dirs))
    K = min(args.held_out, max(0, len(rollout_dirs) - 2))
    held_set = {rollout_dirs[i] for i in perm[:K]}
    train_dirs = [d for d in rollout_dirs if d not in held_set]
    held_dirs = [rollout_dirs[i] for i in perm[:K]]
    print(f"[split] {len(train_dirs)} train / {len(held_dirs)} held-out rollouts")

    # per-rollout vectors (TRAIN rollouts) --------------------------------------
    set_list, submit_list = [], []
    diag = {"set_total": 0, "set_centered": 0, "submit_centered": 0}
    for rd in train_dirs:
        sv, bv, found = rollout_vectors(rd, layer_sel, args.win, tokenizer, args.pool)
        for k in diag:
            diag[k] += found[k]
        if sv is not None:
            set_list.append(sv)
        if bv is not None:
            submit_list.append(bv)

    P, Q = len(set_list), len(submit_list)
    if P < 2 or Q < 2:
        raise SystemExit(f"need >=2 rollouts per class for a coherence metric; "
                         f"got SET={P}, SUBMIT={Q}.")
    SET = np.stack(set_list)        # (P, n_sel, d)
    SUB = np.stack(submit_list)     # (Q, n_sel, d)
    print(f"\n[pool] using all rollouts -> SET from {P}, SUBMIT from {Q}")
    print(f"[diag] SET turns pooled: {diag['set_total']} "
          f"(verb-centered {diag['set_centered']}); "
          f"SUBMIT verb-centered {diag['submit_centered']}/{Q}")

    # per-layer separability metrics -------------------------------------------
    # All three share the same ingredients per layer:
    #   cross   = cos(SET_all, SUBMIT_all)                signed alignment of the
    #                                                      two pooled directions
    #   coh_X   = SUM_ij cos(X_i, X_j) = ||sum unit(X)||^2 within-class coherence
    #   meanpair_X = coh_X / N_X^2   in (0,1], 1 iff a class shares one direction
    #
    # Selectors (best layer):
    #   separation (default): MAXIMISE (1 - cross) * meanpair_SET * meanpair_SUB
    #       -> SET & SUBMIT as different as possible (anti-aligned best) AND each
    #          class coherent. Behaves correctly for cross<0 (unlike the ratio).
    #   ratio: your literal formula, MINIMISE cross / (coh_SET * coh_SUB).
    #          Faithful, but mis-ranks when cross<0 (warns below).
    #   abs:   MINIMISE |cross| / (coh_SET * coh_SUB) -> orthogonal ideal.
    set_all = SET.mean(axis=0)      # (n_sel, d)  pooled directions
    sub_all = SUB.mean(axis=0)
    rows = []
    for li, j in enumerate(layer_sel):
        S, B = SET[:, li, :], SUB[:, li, :]                  # (P,d), (Q,d)
        cross = _cos(set_all[li], sub_all[li])               # signed
        coh_set, coh_sub = _coherence(S), _coherence(B)
        mp_set, mp_sub = coh_set / (P * P), coh_sub / (Q * Q)
        denom = coh_set * coh_sub
        rows.append(dict(
            layer=int(j), cos_cross=cross, coh_set=coh_set, coh_sub=coh_sub,
            set_mean_pair=mp_set, sub_mean_pair=mp_sub,
            separation=(1.0 - cross) * mp_set * mp_sub,                  # maximise
            ratio=(cross / denom) if denom > 0 else float("nan"),       # minimise
            abs_ratio=(abs(cross) / denom) if denom > 0 else float("nan"),  # minimise
        ))

    layers = np.array([r["layer"] for r in rows])
    sep = np.array([r["separation"] for r in rows], float)
    rat = np.array([r["ratio"] for r in rows], float)
    absr = np.array([r["abs_ratio"] for r in rows], float)

    sel = args.select
    if sel == "separation":
        best = int(np.nanargmax(sep)); score_name, score = "separation (max)", sep
    elif sel == "ratio":
        best = int(np.nanargmin(rat)); score_name, score = "ratio (min)", rat
    else:
        best = int(np.nanargmin(absr)); score_name, score = "abs-ratio (min)", absr
    rb = rows[best]

    sep_best = int(np.nanargmax(sep))
    rat_best = int(np.nanargmin(rat))
    print(f"\n[result] selector={sel!r} -> best hidden-state layer {rb['layer']} "
          f"({score_name}={score[best]:.4g})")
    print(f"         cos(SET_all,SUBMIT_all)={rb['cos_cross']:+.3f}  "
          f"SET meanpair={rb['set_mean_pair']:.3f}  SUBMIT meanpair={rb['sub_mean_pair']:.3f}")
    print(f"         argmax(separation)=L{rows[sep_best]['layer']}   "
          f"argmin(ratio)=L{rows[rat_best]['layer']}")
    if rows[best]["cos_cross"] < 0 and sel == "ratio":
        print("  [warn] picked layer has cos(SET,SUBMIT)<0: the literal ratio "
              "rewards LOW coherence in this regime. Prefer --select separation.")
    if sep_best != rat_best:
        print("  [note] separation and ratio disagree (expected wherever "
              "cos(SET,SUBMIT)<0). See the per-layer table / plot.")

    # plot ----------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True)
        ax0.plot(layers, [r["cos_cross"] for r in rows], "o-",
                 color="tab:purple", label="cos(SET_all, SUBMIT_all)  [want low/neg]")
        ax0.plot(layers, [r["set_mean_pair"] for r in rows], "s-",
                 color="tab:blue", label="SET coherence (mean pairwise)  [want high]")
        ax0.plot(layers, [r["sub_mean_pair"] for r in rows], "^-",
                 color="tab:green", label="SUBMIT coherence (mean pairwise)  [want high]")
        ax0.axhline(0, color="k", lw=0.6)
        ax0.set_ylabel("cosine"); ax0.legend(fontsize=8); ax0.grid(alpha=0.3)

        ax1.plot(layers, sep, "o-", color="tab:red",
                 label="separation = (1-cos)·mpSET·mpSUB  [max]")
        ax1.axvline(rows[sep_best]["layer"], color="tab:red", ls=":", lw=1,
                    label=f"argmax sep L{rows[sep_best]['layer']}")
        ax1.set_ylabel("separation (max = best)"); ax1.set_ylim(bottom=0)
        axb = ax1.twinx()
        axb.plot(layers, rat, "x--", color="tab:gray", alpha=0.8,
                 label="literal ratio = cos/(coh·coh)  [min]")
        axb.axvline(rows[rat_best]["layer"], color="tab:gray", ls=":", lw=1)
        axb.set_ylabel("literal ratio (min = best)")
        ax1.set_xlabel("hidden-state layer (post-MLP residual; even indices)")
        h0, l0 = ax1.get_legend_handles_labels()
        h1, l1 = axb.get_legend_handles_labels()
        ax1.legend(h0 + h1, l0 + l1, fontsize=8, loc="upper right")
        ax1.grid(alpha=0.3)
        fig.suptitle("SET vs SUBMIT separability by layer")
        fig.tight_layout()
        png = os.path.join(run_dir, "direction_separability_by_layer.png")
        fig.savefig(png, dpi=130)
        print(f"[plot] wrote {png}")
    except Exception as e:
        print(f"[plot] skipped ({e})")

    # save ----------------------------------------------------------------------
    np.savez_compressed(
        out_npz,
        layers=layers.astype(np.int32),
        set_all=set_all.astype(np.float32),
        submit_all=sub_all.astype(np.float32),
        win=np.int32(args.win),
        d_model=np.int32(d_model),
        n_set=np.int32(P),
        n_submit=np.int32(Q),
        best_layer=np.int32(rb["layer"]),         # per the chosen --select
        best_layer_separation=np.int32(rows[sep_best]["layer"]),
        best_layer_ratio=np.int32(rows[rat_best]["layer"]),
        select=np.array(sel),
        model_name=np.array(args.model),
    )
    with open(os.path.join(run_dir, "directions_summary.json"), "w") as f:
        json.dump(dict(run_dir=run_dir, win=args.win, n_set=P, n_submit=Q,
                       select=sel, best_layer=rb["layer"],
                       best_layer_separation=rows[sep_best]["layer"],
                       best_layer_ratio=rows[rat_best]["layer"],
                       layers=[int(x) for x in layers], per_layer=rows), f, indent=2)
    print(f"[save] {out_npz}  (+ directions_summary.json)")

    # held-out per-turn projection trajectories ---------------------------------
    proj_layer = args.proj_layer if args.proj_layer is not None else rb["layer"]
    if proj_layer not in layer_sel:
        print(f"[traj] proj-layer {proj_layer} not in selection -> using {rb['layer']}")
        proj_layer = rb["layer"]
    pli = layer_sel.index(proj_layer)
    a_v, b_v = set_all[pli], sub_all[pli]
    mid, dirv = (a_v + b_v) / 2.0, (b_v - a_v)
    denom = float(dirv @ dirv) + 1e-12
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        n_plotted = 0
        for rd in held_dirs:
            kinds = turn_actions(os.path.join(rd, "transcript.jsonl"))
            xs, ys, sub_t = [], [], None
            for t in sorted(kinds):
                verb = {"set": "SET", "submit": "SUBMIT"}.get(kinds[t])
                npz = os.path.join(rd, "activations", f"turn_{t:02d}.npz")
                if not os.path.exists(npz):
                    continue
                p = _turn_proj(npz, proj_layer, verb, tokenizer, args.pool, args.win,
                               mid, dirv, denom)
                if p is None:
                    continue
                xs.append(t); ys.append(p)
                if kinds[t] == "submit":
                    sub_t = t
            if not xs:
                continue
            rid = os.path.basename(rd).split("_")[-1]
            line, = ax.plot(xs, ys, "-o", ms=3, alpha=0.85, label=f"r{rid}")
            if sub_t is not None:
                ax.plot([sub_t], [ys[xs.index(sub_t)]], "*", ms=15,
                        color=line.get_color())
            n_plotted += 1
        ax.axhline(1, color="g", ls="--", lw=1)
        ax.axhline(-1, color="b", ls="--", lw=1)
        ax.text(0.01, 0.98, "SUBMIT_all = +1", color="g", transform=ax.transAxes,
                va="top", fontsize=8)
        ax.text(0.01, 0.02, "SET_all = -1", color="b", transform=ax.transAxes,
                va="bottom", fontsize=8)
        ax.set_xlabel("turn"); ax.set_ylabel("projection onto (SUBMIT-SET)  [SET=-1, SUBMIT=+1]")
        ax.set_title(f"Held-out per-turn projection @ layer {proj_layer}  (* = submit turn)")
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3); fig.tight_layout()
        png = os.path.join(run_dir, "heldout_turn_projection.png")
        fig.savefig(png, dpi=130)
        print(f"[traj] wrote {png}  ({n_plotted} held-out rollouts, layer {proj_layer})")
    except Exception as e:
        print(f"[traj] skipped ({e})")


if __name__ == "__main__":
    main()