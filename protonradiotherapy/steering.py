"""Steering vectors for the satisficing study.

The vector is the SAME construction validated on the 1D-sine and story tasks:

    v_steer[layer] = mean residual at the DECISION token over SUBMIT turns
                   - mean residual at the DECISION token over SET turns

captured across all layers from ~100 rollouts (the decision token is the last
generated token, which the recorder always stores last; with capture_tokens=
'lastk' we average the last-k run-up to it). `build_steering_vector` writes a
directions.npz with one vector per hidden-state layer; `steering_active` adds
alpha*v to a decoder block's residual output via a forward hook for
counterfactual replay; `load_direction` fetches a single layer's vector.

Layer convention (shared with the old transfer studies): layer index j is a
hidden_states index, and the hook is placed on decoder block j-1 whose output ==
hidden_states[j].
"""
from __future__ import annotations
import os, json, glob, contextlib
import numpy as np


def _load_npz_acts(path):
    """Load a per-turn activation file as float32 (handles bf16 storage)."""
    try:
        import ml_dtypes  # noqa: registers bfloat16 for numpy load
    except Exception:
        pass
    z = np.load(path, allow_pickle=False)
    acts = z["acts"]
    if acts.dtype != np.float32:
        acts = acts.astype(np.float32)
    return acts, int(z["decision_index"])


def build_steering_vector(run_dir, out_path=None, mean_last_k=True):
    """Contrast SUBMIT vs SET decision-token residuals across all rollouts.

    Returns vecs of shape (n_layers, d_model) and writes directions.npz with
    keys: v (the contrast), mu_submit, mu_set, n_submit, n_set.
    """
    submit_acc, set_acc = None, None
    n_sub, n_set = 0, 0
    for rd in sorted(glob.glob(os.path.join(run_dir, "rollout_*"))):
        tpath = os.path.join(rd, "transcript.jsonl")
        if not os.path.exists(tpath):
            continue
        kind_by_turn = {}
        with open(tpath) as f:
            for line in f:
                r = json.loads(line)
                if "turn" in r and r.get("action") in ("set", "submit"):
                    kind_by_turn[int(r["turn"])] = r["action"]
        adir = os.path.join(rd, "activations")
        for turn, kind in kind_by_turn.items():
            apath = os.path.join(adir, f"turn_{turn:02d}.npz")
            if not os.path.exists(apath):
                continue
            acts, didx = _load_npz_acts(apath)            # (L+1, n_pos, d)
            vec = acts.mean(axis=1) if mean_last_k else acts[:, didx, :]
            if kind == "submit":
                submit_acc = vec if submit_acc is None else submit_acc + vec
                n_sub += 1
            else:
                set_acc = vec if set_acc is None else set_acc + vec
                n_set += 1
    if n_sub == 0 or n_set == 0:
        raise RuntimeError(f"need both classes; got SUBMIT={n_sub} SET={n_set}")
    mu_sub = submit_acc / n_sub
    mu_set = set_acc / n_set
    v = mu_sub - mu_set
    out_path = out_path or os.path.join(run_dir, "directions.npz")
    np.savez_compressed(out_path, v=v, mu_submit=mu_sub, mu_set=mu_set,
                        n_submit=n_sub, n_set=n_set)
    return v


def load_direction(path, layer):
    z = np.load(path)
    return z["v"][layer]


@contextlib.contextmanager
def steering_active(model, block_idx, vec, alpha):
    """Add alpha*vec to the residual output of decoder block `block_idx`.

    `vec` is a (d_model,) numpy array for the corresponding hidden_states layer.
    Hooking block j-1 steers hidden_states[j] (the transfer-study convention).
    """
    import torch
    layers = _decoder_layers(model)
    blk = layers[block_idx]
    v = torch.tensor(np.asarray(vec), dtype=next(model.parameters()).dtype,
                     device=next(model.parameters()).device)

    def hook(_module, _inp, out):
        if isinstance(out, tuple):
            h = out[0]
            h = h + alpha * v
            return (h,) + tuple(out[1:])
        return out + alpha * v

    handle = blk.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _decoder_layers(model):
    """Locate the list of decoder blocks across common HF architectures."""
    for attr in ("model", "language_model"):
        m = getattr(model, attr, None)
        if m is not None:
            inner = getattr(m, "model", m)
            if hasattr(inner, "layers"):
                return inner.layers
            if hasattr(m, "layers"):
                return m.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError("could not locate decoder layers on this model")
