"""steering.py -- load the SOURCE (e.g. parabola) SUBMIT-minus-SET direction and
causally apply it to the post-MLP residual. This is the SAME interface used by the
1D/story transfer studies; the proton env REUSES the already-extracted vector
(directions.npz) rather than building its own, so this is a genuine cross-
environment steering test.

Layer convention (matches direction_extract.py / recorder.py): `layer j` is a
hidden-state index, hidden_states[j] == output of decoder block j-1 == post-MLP
residual. To steer that point we add to the OUTPUT of decoder block (j-1).

build_steering_vector: (SUBMIT_all - SET_all) at `layer`, unit-normalised, scaled
to frac x mean-per-token residual norm AT THE SOURCE. alpha then scales/signs it:
  alpha > 0 -> toward SUBMIT  (stop earlier / less over-optimisation)
  alpha < 0 -> toward SET     (keep searching / more over-optimisation)
"""
from __future__ import annotations
import os, glob, contextlib
import numpy as np

try:
    import ml_dtypes  # noqa: F401  (lets np.load read bf16 activation captures)
except Exception:
    pass


# --------------------------------------------------------------------------- #
# directions + token-norm  (all SOURCE-derived)
# --------------------------------------------------------------------------- #
def load_direction(directions_path, layer):
    """Return (set_vec, submit_vec) at hidden-state index `layer`, shape (d,).
    Reads pooled-over-all-rollouts keys set_all/submit_all (falls back to the
    older set_train/submit_train)."""
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
    `layer`, sampled from captured turns in `run_dir` (the SOURCE env)."""
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
    frac * mean_token_norm(layer) AT THE SOURCE. Returns (vec (d,), info)."""
    set_t, sub_t = load_direction(directions_path, layer)
    raw = sub_t - set_t
    nrm = np.linalg.norm(raw)
    if nrm == 0:
        raise SystemExit("SUBMIT_all == SET_all at this layer; nothing to steer.")
    unit = raw / nrm
    tok_norm = mean_token_norm_at_layer(run_dir, layer)
    vec = (unit * (frac * tok_norm)).astype(np.float32)
    info = dict(layer=int(layer), frac=float(frac), raw_norm=float(nrm),
                mean_token_norm=float(tok_norm),
                steer_norm=float(np.linalg.norm(vec)))
    return vec, info


# --------------------------------------------------------------------------- #
# locating the decoder stack + the steering hook  (architecture-robust)
# --------------------------------------------------------------------------- #
def find_decoder_layers(model, override=None):
    """Return the ModuleList of decoder blocks. Works for plain decoder-only,
    MoE, and VLM models (where the LLM decoder is nested under a vision-language
    wrapper) by picking the LONGEST ModuleList whose blocks look like decoder
    layers. `override` is a dotted attr path (e.g. 'model.language_model.layers')
    if auto-detection picks the wrong stack."""
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
                           "pass layers_attr to point at it explicitly.")
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
                         "check the layer.")
    p = next(model.parameters())
    steer = torch.tensor(steer_vec, dtype=p.dtype, device=p.device)
    add = (alpha * steer)

    def hook(_module, _inputs, output):
        if alpha == 0.0:
            return output
        if isinstance(output, tuple):
            hs = output[0] + add.to(output[0].dtype).to(output[0].device)
            return (hs,) + tuple(output[1:])
        return output + add.to(output.dtype).to(output.device)

    handle = layers[block_idx].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()