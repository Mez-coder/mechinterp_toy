"""steering.py -- build a SUBMIT-minus-SET steering vector at a chosen layer and
causally apply it to the post-MLP residual during the parabola pipeline, sweeping
the steering strength alpha and saving transcripts.

Run as a package module, next to rollout.py:

    # 1) extract directions first (writes runs/csat/directions.npz):
    python -m csat.direction_extract --run-dir runs/csat

    # 2) sweep steering at the chosen layer:
    python -m csat.steering --layer 18 --frac 0.4 \
        --alphas -1 -0.5 0 0.5 1 --n-rollouts 20 --repeats 1

Layer convention (matches direction_extract.py / recorder.py)
-------------------------------------------------------------
`--layer j` is a *hidden-state index*: hidden_states[j] == output of decoder
block j-1 == the post-MLP residual stream. So to steer that exact point we add to
the OUTPUT of decoder layer (j-1). j must be >= 1.

What `alpha` scales
-------------------
The steering vector is (SUBMIT_all - SET_all), unit-normalised, then scaled to
`frac` x (mean per-token residual norm at this layer)  -- so at |alpha|=1 you add
~frac of one token's worth of norm. `alpha` then scales (and signs) that:
  alpha > 0  -> toward SUBMIT  (expect: stop earlier / less over-optimisation)
  alpha < 0  -> toward SET     (expect: keep searching / more over-optimisation)
Transcripts for each alpha land in a distinct run_name so nothing collides.
"""
from __future__ import annotations
import os, json, argparse, contextlib
import numpy as np

try:
    import ml_dtypes  # noqa: F401  (lets np.load read bf16 activation captures)
except Exception:
    pass

from .config import Config
from .agents import ModelAgent
from .rollout import build_env, run_rollout
from . import io_utils as io
from . import direction_extract as de


# --------------------------------------------------------------------------- #
# directions + token-norm
# --------------------------------------------------------------------------- #
def load_direction(directions_path, layer):
    """Return (set_vec, submit_vec) at hidden-state index `layer`, shape (d,).
    Reads the pooled-over-all-rollouts keys set_all/submit_all written by the
    current direction_extract.py (falls back to the older set_train/submit_train)."""
    z = np.load(directions_path, allow_pickle=True)
    layers = list(z["layers"].astype(int))
    if layer not in layers:
        raise SystemExit(f"layer {layer} not in saved directions {layers}; "
                         "re-run direction_extract.py with --layers all, or pick "
                         "one of the saved layers.")
    li = layers.index(layer)
    set_key = "set_all" if "set_all" in z.files else "set_train"
    sub_key = "submit_all" if "submit_all" in z.files else "submit_train"
    return (z[set_key][li].astype(np.float32),
            z[sub_key][li].astype(np.float32))


def mean_token_norm_at_layer(run_dir, layer, n_sample=300, seed=0):
    """Mean L2 norm of a single token's post-MLP residual at hidden-state index
    `layer`, sampled from captured turns. acts[layer] is exactly that residual."""
    import glob
    files = sorted(glob.glob(os.path.join(run_dir, "rollout_*",
                                          "activations", "turn_*.npz")))
    if not files:
        raise SystemExit(f"no activation npz under {run_dir} to estimate token norm.")
    rng = np.random.default_rng(seed)
    if len(files) > n_sample:
        files = [files[i] for i in rng.choice(len(files), n_sample, replace=False)]
    norms = []
    for p in files:
        try:
            with np.load(p, allow_pickle=False) as zz:
                acts = np.asarray(zz["acts"]).astype(np.float32)   # (L+1, n_pos, d)
        except Exception:
            continue
        if layer >= acts.shape[0]:
            continue
        norms.append(np.linalg.norm(acts[layer], axis=-1).reshape(-1))   # per token
    if not norms:
        raise SystemExit(f"could not estimate token norm at layer {layer}.")
    return float(np.concatenate(norms).mean())


def build_steering_vector(layer, directions_path, run_dir, frac=0.4):
    """SUBMIT_all - SET_all at `layer`, unit-normalised then scaled to
    frac * mean_token_norm(layer). Returns (vec (d,), info dict)."""
    set_t, sub_t = load_direction(directions_path, layer)
    raw = sub_t - set_t
    nrm = np.linalg.norm(raw)
    if nrm == 0:
        raise SystemExit("SUBMIT_train == SET_train at this layer; nothing to steer.")
    unit = raw / nrm
    tok_norm = mean_token_norm_at_layer(run_dir, layer)
    vec = (unit * (frac * tok_norm)).astype(np.float32)
    info = dict(layer=int(layer), frac=float(frac), raw_norm=float(nrm),
                mean_token_norm=float(tok_norm),
                steer_norm=float(np.linalg.norm(vec)))
    return vec, info


# --------------------------------------------------------------------------- #
# locating the decoder stack + the steering hook
# --------------------------------------------------------------------------- #
def find_decoder_layers(model, override=None):
    """Return the ModuleList of decoder blocks. `override` is a dotted attr path
    (e.g. 'model.language_model.layers') if auto-detection picks the wrong one."""
    import torch.nn as nn
    if override:
        obj = model
        for part in override.split("."):
            obj = getattr(obj, part)
        return obj
    cands = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            child = mod[0]
            if (hasattr(child, "mlp") or hasattr(child, "self_attn")
                    or "DecoderLayer" in type(child).__name__):
                cands.append((len(mod), name, mod))
    if not cands:
        raise RuntimeError("could not locate a decoder-layer ModuleList; "
                           "pass --layers-attr to point at it explicitly.")
    cands.sort(key=lambda t: t[0])                 # longest stack = main decoder
    n, name, mod = cands[-1]
    print(f"[hook] decoder stack: '{name}' with {n} layers")
    return mod


@contextlib.contextmanager
def steering_active(model, block_idx, steer_vec, alpha, layers_attr=None):
    """Add alpha * steer_vec to the OUTPUT of decoder block `block_idx` for the
    duration of the context. block_idx = (hidden-state layer) - 1."""
    import torch
    layers = find_decoder_layers(model, layers_attr)
    if not (0 <= block_idx < len(layers)):
        raise SystemExit(f"block_idx {block_idx} out of range (0..{len(layers)-1}); "
                         "check --layer.")
    p = next(model.parameters())
    steer = torch.tensor(steer_vec, dtype=p.dtype, device=p.device)
    add = (alpha * steer)

    def hook(_module, _inputs, output):
        if alpha == 0.0:
            return output
        if isinstance(output, tuple):
            hs = output[0]
            hs = hs + add.to(hs.dtype).to(hs.device)
            return (hs,) + tuple(output[1:])
        return output + add.to(output.dtype).to(output.device)

    handle = layers[block_idx].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# --------------------------------------------------------------------------- #
# parabola pipeline, steered, for one alpha
# --------------------------------------------------------------------------- #
def run_steered(cfg, agent, steer_vec, layer, alpha, n_rollouts, repeats,
                base_run_name, env_kind="parabola", layers_attr=None):
    """Run the single-margin pipeline under steering=alpha in `env_kind`; save
    transcripts under a per-alpha run_name (no case_spread, no capture)."""
    block_idx = layer - 1                          # hidden_states[layer] = layers[layer-1] out
    cfg.env_kind = env_kind
    if env_kind == "sine":
        cfg.n_obj = 1                              # 1D action; the stopper reads cfg.n_obj
    cfg.capture = False                            # don't re-forward / contaminate under steering
    cfg.run_name = f"{base_run_name}_{env_kind}_L{layer}_a{alpha:+.2f}".replace("+", "p").replace("-", "m")

    run_dir = cfg.run_dir()
    with open(os.path.join(run_dir, "steer_meta.json"), "w") as f:
        json.dump(dict(layer=int(layer), block_idx=int(block_idx), alpha=float(alpha),
                       env_kind=cfg.env_kind, n_rollouts=int(n_rollouts),
                       repeats=int(repeats)), f, indent=2)

    seeds = list(range(cfg.seed_start, cfg.seed_start + n_rollouts))
    start = io.next_rollout_idx(run_dir)
    print(f"\n=== alpha={alpha:+.2f}  layer={layer}  -> {run_dir} "
          f"(resume from idx {start}) ===")

    with steering_active(agent.model, block_idx, steer_vec, alpha, layers_attr):
        for case_id, seed in enumerate(seeds):
            env = build_env(cfg); env.reset(seed=seed, wide=getattr(cfg, "wide_cases", True))
            opt = env.optimum()                    # parabola optimum is exact & cheap
            for rep in range(repeats):
                idx = case_id * repeats + rep
                if idx < start:
                    continue
                r = run_rollout(cfg, idx, agent, seed=seed,
                                case_id=case_id, rep=rep, opt=opt)
                print(f"  a={alpha:+.2f} rollout {idx:04d} "
                      f"(case {case_id:03d} rep {rep}): submitted={r['submitted']} "
                      f"forced={r['forced']} first_pass_turn={r['first_pass_turn']}")


def run_unsteered_capture(cfg, agent, env_kind, n_rollouts, repeats, base_run_name):
    """Run rollouts in `env_kind` with NO steering and capture ON, so each turn's
    last-k activations are saved for projection. Returns the run_dir."""
    cfg.env_kind = env_kind
    if env_kind == "sine":
        cfg.n_obj = 1
    cfg.capture = True
    cfg.run_name = f"{base_run_name}_{env_kind}_nosteer"
    run_dir = cfg.run_dir()
    seeds = list(range(cfg.seed_start, cfg.seed_start + n_rollouts))
    start = io.next_rollout_idx(run_dir)
    print(f"\n=== no-steer capture in {env_kind} -> {run_dir} (resume {start}) ===")
    for case_id, seed in enumerate(seeds):
        env = build_env(cfg); env.reset(seed=seed, wide=getattr(cfg, "wide_cases", True))
        opt = env.optimum()
        for rep in range(repeats):
            idx = case_id * repeats + rep
            if idx < start:
                continue
            r = run_rollout(cfg, idx, agent, seed=seed, case_id=case_id, rep=rep, opt=opt)
            print(f"  nosteer {idx:04d}: submitted={r['submitted']} forced={r['forced']} "
                  f"first_pass_turn={r['first_pass_turn']}")
    return run_dir


def plot_projection_trajectories(run_dir, directions, layer, tok, pool, win,
                                 out_png, title, max_lines=12):
    """Per-turn projection of (last-k-before-verb) activations onto the loaded
    SUBMIT-SET axis (SET_all=-1, SUBMIT_all=+1) for rollouts in run_dir. Reuses
    the held-out-trajectory logic so steered-env and source-env plots match."""
    import glob, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    set_v, sub_v = load_direction(directions, layer)
    mid, dirv = (set_v + sub_v) / 2.0, (sub_v - set_v)
    denom = float(dirv @ dirv) + 1e-12
    dirs = sorted(glob.glob(os.path.join(run_dir, "rollout_*")))[:max_lines]
    fig, ax = plt.subplots(figsize=(8, 5)); n = 0
    for rd in dirs:
        kinds = de.turn_actions(os.path.join(rd, "transcript.jsonl"))
        xs, ys, sub_t = [], [], None
        for t in sorted(kinds):
            verb = {"set": "SET", "submit": "SUBMIT"}.get(kinds[t])
            npz = os.path.join(rd, "activations", f"turn_{t:02d}.npz")
            if not os.path.exists(npz):
                continue
            p = de._turn_proj(npz, layer, verb, tok, pool, win, mid, dirv, denom)
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
            ax.plot([sub_t], [ys[xs.index(sub_t)]], "*", ms=15, color=line.get_color())
        n += 1
    ax.axhline(1, color="g", ls="--", lw=1); ax.axhline(-1, color="b", ls="--", lw=1)
    ax.text(0.01, 0.98, "SUBMIT_all = +1", color="g", transform=ax.transAxes, va="top", fontsize=8)
    ax.text(0.01, 0.02, "SET_all = -1", color="b", transform=ax.transAxes, va="bottom", fontsize=8)
    ax.set_xlabel("turn"); ax.set_ylabel("projection onto (SUBMIT-SET)  [source-env axis]")
    ax.set_title(title); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"[proj] wrote {out_png}  ({n} rollouts, layer {layer})")



def main():
    cfg = Config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, default=None,
                    help="hidden-state index to steer (>=1). Defaults to the "
                         "best_layer saved in directions.npz by direction_extract.py")
    ap.add_argument("--directions", default=None,
                    help="directions.npz (default: <source run-dir>/directions.npz)")
    ap.add_argument("--source-run-dir", default=os.path.join(cfg.out_dir, cfg.run_name),
                    help="run dir the directions + token-norm come from (default runs/<run_name>)")
    ap.add_argument("--frac", type=float, default=0.4,
                    help="steering magnitude as a fraction of mean token norm (default 0.4)")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[-1.0, -0.5, 0.0, 0.5, 1.0],
                    help="steering strengths to sweep")
    ap.add_argument("--n-rollouts", type=int, default=20, help="cases per alpha")
    ap.add_argument("--repeats", type=int, default=1, help="rollouts per case")
    ap.add_argument("--model", default=None, help="HF id override")
    ap.add_argument("--run-name", default="csat_steer",
                    help="base run_name; each alpha gets <name>_L<layer>_a<alpha>")
    ap.add_argument("--layers-attr", default=None,
                    help="dotted path to the decoder ModuleList if auto-detect is wrong")
    ap.add_argument("--env", choices=["parabola", "sine", "coupling"], default="parabola",
                    help="environment to run the test in (default: parabola)")
    ap.add_argument("--mode", choices=["steer", "project"], default="steer",
                    help="'steer': alpha sweep (Test 1). 'project': no steering, "
                         "capture + plot per-turn projection onto the source axis (Test 2)")
    ap.add_argument("--pool", choices=["before", "around", "all"], default="before",
                    help="token pool for projection (match your extraction)")
    ap.add_argument("--win", type=int, default=4, help="win for --pool around")
    args = ap.parse_args()

    if args.model:
        cfg.model_name = args.model

    directions = args.directions or os.path.join(args.source_run_dir, "directions.npz")
    if not os.path.exists(directions):
        raise SystemExit(f"{directions} not found; run direction_extract.py first.")

    layer = args.layer
    if layer is None:                               # default to the saved best layer
        z = np.load(directions, allow_pickle=True)
        if "best_layer" not in z.files:
            raise SystemExit("no best_layer in directions.npz; pass --layer.")
        layer = int(z["best_layer"])
        print(f"[steer] --layer not given; using saved best_layer = {layer}")
    if layer < 1:
        raise SystemExit("--layer must be >= 1 (layer 0 is embeddings, not post-MLP).")

    # tokenizer (for verb-centered pooling in projection / shared with extraction)
    tok = None
    try:
        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained(cfg.model_name)
        except Exception:
            tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    except Exception as e:
        print(f"[proj] no tokenizer ({e}); pooling all captured tokens.")

    agent = ModelAgent(cfg)                         # loads the model once

    if args.mode == "project":
        # Test 2: NO steering. Run env rollouts with capture, project each turn's
        # last-k-before-verb activations onto the SOURCE (e.g. parabola) axis.
        run_dir = run_unsteered_capture(cfg, agent, args.env,
                                        n_rollouts=args.n_rollouts,
                                        repeats=args.repeats, base_run_name=args.run_name)
        out_png = os.path.join(run_dir, f"projection_{args.env}_L{layer}.png")
        plot_projection_trajectories(run_dir, directions, layer, tok,
                                     args.pool, args.win, out_png,
                                     title=f"{args.env} (no steering) projected on "
                                           f"{os.path.basename(args.source_run_dir)} axis @ L{layer}")
        print("\n[done] projection transfer test complete.")
        return

    # Test 1: steering sweep in args.env, using the SOURCE-derived steer vector.
    steer_vec, info = build_steering_vector(layer, directions,
                                            args.source_run_dir, frac=args.frac)
    print(f"[steer] layer {info['layer']}  |steer|={info['steer_norm']:.2f}  "
          f"= {info['frac']:.0%} of mean token norm {info['mean_token_norm']:.2f}  "
          f"(raw |SUBMIT-SET|={info['raw_norm']:.3f})")
    for alpha in args.alphas:
        run_steered(cfg, agent, steer_vec, layer, float(alpha),
                    n_rollouts=args.n_rollouts, repeats=args.repeats,
                    base_run_name=args.run_name, env_kind=args.env,
                    layers_attr=args.layers_attr)
    print("\n[done] alpha sweep complete.")


if __name__ == "__main__":
    main()