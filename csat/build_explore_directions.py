"""build_explore_directions.py -- assemble the NEW steering direction
(SUBMIT_all - EXPLORE_all) and write it in the SAME schema as directions.npz, so
steering.py / transfer_studies.py consume it unchanged.

Positive pole (EXPLORE): per-rollout mean vectors written by label_exploration.py
  (<source>/exploration/rollout_XXXX.npz, key mean_vec, shape (L+1, d)).
Negative pole (SUBMIT): identical to direction_extract.py -- the before-verb pool
  of the submit turn from the captured turn_XX.npz. We literally call
  direction_extract.rollout_vectors and keep only its submit_vec, so this pole is
  byte-for-byte the one you already use.

Layer selection / separability metric / output keys all mirror direction_extract,
so the only thing that changed between directions.npz and directions_explore.npz
is what the positive pole represents.

    # 1) extract submit pole + label/extract explore pole first:
    python -m csat.direction_extract  --run-dir runs/csat          # (for token-norm etc; optional)
    python -m csat.label_exploration  --source-run-dir runs/csat
    # 2) build the new direction:
    python -m csat.build_explore_directions --source-run-dir runs/csat
    # -> runs/csat/directions_explore.npz  (best_layer chosen by separation)

Then steer with it (see the steering.py / transfer_studies.py --vector-type flag):
    python -m csat.steering --vector-type explore --env sine --alphas -1 0 1
"""
from __future__ import annotations
import os, json, glob, argparse
import numpy as np

try:
    import ml_dtypes  # noqa: F401  (lets np.load read bf16 captures for the submit pole)
except Exception:
    pass

from .config import Config
from . import direction_extract as de


def _rid_num(name):
    import re
    m = re.search(r"rollout_(\d+)", name)
    return int(m.group(1)) if m else None


def load_explore_pole(src):
    """Return {rollout_idx: mean_vec (L+1, d)} from exploration/rollout_*.npz."""
    out = {}
    for p in sorted(glob.glob(os.path.join(src, "exploration", "rollout_*.npz"))):
        n = _rid_num(os.path.basename(p))
        if n is None:
            continue
        with np.load(p) as z:
            out[n] = np.asarray(z["mean_vec"]).astype(np.float32)   # (L+1, d)
    return out


def main():
    cfg = Config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-run-dir", default=os.path.join(cfg.out_dir, cfg.run_name))
    ap.add_argument("--model", default=cfg.model_name,
                    help="HF id (tokenizer only, for verb-centred SUBMIT pooling)")
    ap.add_argument("--layers", choices=["even", "all"], default="even")
    ap.add_argument("--include-embed", action="store_true")
    ap.add_argument("--pool", choices=["before", "around", "all"], default="before",
                    help="token pool for the SUBMIT pole (match your extraction)")
    ap.add_argument("--win", type=int, default=4)
    ap.add_argument("--select", choices=["separation", "ratio", "abs"],
                    default="separation")
    ap.add_argument("--out", default=None,
                    help="output npz (default <source>/directions_explore.npz)")
    args = ap.parse_args()

    src = args.source_run_dir
    out_npz = args.out or os.path.join(src, "directions_explore.npz")

    explore = load_explore_pole(src)
    if len(explore) < 2:
        raise SystemExit(f"need >=2 rollouts with an explore vector; found "
                         f"{len(explore)}. Run label_exploration.py (without --no-acts).")
    n_total, d_model = next(iter(explore.values())).shape
    layer_sel = de.select_layers(n_total, args.layers, args.include_embed)
    print(f"[layers] hidden-states 0..{n_total-1} (d_model={d_model}); "
          f"{len(layer_sel)} layers: {layer_sel}")

    # tokenizer for the SUBMIT pole's verb-centred pooling (optional) -----------
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        print(f"[tok] loaded tokenizer for {args.model}")
    except Exception as e:
        print(f"[tok] no tokenizer ({e}); SUBMIT pole pools all captured tokens.")

    # pair each explore rollout with its SUBMIT-pole vector (same rollout dir) ---
    expl_list, sub_list = [], []
    n_sub_centered = 0
    for ridx, evec in sorted(explore.items()):
        rdir = os.path.join(src, f"rollout_{ridx:04d}")
        if not os.path.isdir(rdir):
            continue
        _setv, subv, found = de.rollout_vectors(rdir, layer_sel, args.win,
                                                tokenizer, args.pool)
        if subv is None:
            continue                                 # no submit turn captured -> drop pair
        expl_list.append(evec[layer_sel])            # (n_sel, d)
        sub_list.append(subv)                        # (n_sel, d)
        n_sub_centered += int(found.get("submit_centered", 0))

    P = Q = len(expl_list)
    if P < 2:
        raise SystemExit(f"only {P} rollouts had BOTH an explore vector and a "
                         "captured submit turn; need >=2.")
    EXPL = np.stack(expl_list)        # (P, n_sel, d)
    SUB = np.stack(sub_list)          # (P, n_sel, d)
    print(f"[pool] paired rollouts: {P}  (SUBMIT verb-centered {n_sub_centered}/{Q})")

    expl_all = EXPL.mean(axis=0)      # (n_sel, d)
    sub_all = SUB.mean(axis=0)
    rows = []
    for li, j in enumerate(layer_sel):
        S, B = EXPL[:, li, :], SUB[:, li, :]
        cross = de._cos(expl_all[li], sub_all[li])
        coh_s, coh_b = de._coherence(S), de._coherence(B)
        mp_s, mp_b = coh_s / (P * P), coh_b / (Q * Q)
        denom = coh_s * coh_b
        rows.append(dict(layer=int(j), cos_cross=cross,
                         set_mean_pair=mp_s, sub_mean_pair=mp_b,
                         separation=(1.0 - cross) * mp_s * mp_b,
                         ratio=(cross / denom) if denom > 0 else float("nan"),
                         abs_ratio=(abs(cross) / denom) if denom > 0 else float("nan")))

    layers = np.array([r["layer"] for r in rows])
    sep = np.array([r["separation"] for r in rows], float)
    rat = np.array([r["ratio"] for r in rows], float)
    absr = np.array([r["abs_ratio"] for r in rows], float)
    if args.select == "separation":
        best = int(np.nanargmax(sep)); sname, score = "separation (max)", sep
    elif args.select == "ratio":
        best = int(np.nanargmin(rat)); sname, score = "ratio (min)", rat
    else:
        best = int(np.nanargmin(absr)); sname, score = "abs-ratio (min)", absr
    rb = rows[best]
    sep_best = int(np.nanargmax(sep)); rat_best = int(np.nanargmin(rat))
    print(f"\n[result] selector={args.select!r} -> best hidden-state layer "
          f"{rb['layer']} ({sname}={score[best]:.4g})")
    print(f"         cos(EXPLORE_all,SUBMIT_all)={rb['cos_cross']:+.3f}  "
          f"EXPLORE meanpair={rb['set_mean_pair']:.3f}  "
          f"SUBMIT meanpair={rb['sub_mean_pair']:.3f}")
    print(f"         argmax(separation)=L{rows[sep_best]['layer']}  "
          f"argmin(ratio)=L{rows[rat_best]['layer']}")

    # NOTE: keys mirror directions.npz exactly. The positive pole 'set_all' now
    # holds the EXPLORE vectors, so steering.py (which computes submit_all-set_all)
    # works unchanged: alpha>0 -> SUBMIT/stop, alpha<0 -> EXPLORE/keep going.
    np.savez_compressed(
        out_npz,
        layers=layers.astype(np.int32),
        set_all=expl_all.astype(np.float32),         # EXPLORE pole (drop-in name)
        submit_all=sub_all.astype(np.float32),
        win=np.int32(args.win),
        d_model=np.int32(d_model),
        n_set=np.int32(P), n_submit=np.int32(Q),
        best_layer=np.int32(rb["layer"]),
        best_layer_separation=np.int32(rows[sep_best]["layer"]),
        best_layer_ratio=np.int32(rows[rat_best]["layer"]),
        select=np.array(args.select),
        model_name=np.array(args.model),
        pole=np.array("explore"),                    # marker so you can tell them apart
    )
    with open(os.path.join(src, "directions_explore_summary.json"), "w") as f:
        json.dump(dict(source_run_dir=src, pole="explore", win=args.win,
                       n_pairs=P, select=args.select, best_layer=rb["layer"],
                       best_layer_separation=rows[sep_best]["layer"],
                       best_layer_ratio=rows[rat_best]["layer"],
                       layers=[int(x) for x in layers], per_layer=rows), f, indent=2)
    print(f"[save] {out_npz}  (+ directions_explore_summary.json)")


if __name__ == "__main__":
    main()