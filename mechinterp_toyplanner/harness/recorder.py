"""Activation capture: after the model generates an action, re-forward the full
sequence with output_hidden_states and save the residual stream at selected
token positions across ALL layers. The 'decision' token (last generated) is the
position where SET-vs-SUBMIT is committed -- it is always the LAST captured
position, flagged via decision_index.

Which positions are kept (cfg.capture_tokens):
  'decision'  -> just the commit token (cheapest)
  'lastk'     -> the last `last_k` generated tokens, i.e. the run-up to and
                 including the commit (default; captures the decision locus
                 without storing the whole scratchpad)
  'assistant' -> every generated token (most expensive; token axis grows with
                 the scratchpad)

Saved npz per turn:
  acts   : (n_layers+1, n_positions, d_model)  [capture_dtype]
  positions     : absolute token indices captured
  decision_index: index into `positions` of the commit token (always last)
  token_ids     : the captured tokens

Storage dtype note
------------------
The model runs in bfloat16, whose exponent range matches float32 (~3.4e38).
Residual streams routinely contain 'massive activations' far above the float16
range (~6.5e4), so casting them to float16 overflows to +/-inf and silently
corrupts exactly the outlier features interp work cares about. We therefore:
  - 'bfloat16' : 2 bytes, LOSSLESS for a bf16 model (recommended). Needs the
                 `ml_dtypes` package (uv add ml_dtypes); also import it when
                 loading the npz back. Falls back to float32 if unavailable.
  - 'float32'  : 4 bytes, always safe.
  - 'float16'  : 2 bytes, used ONLY if the activations actually fit; otherwise
                 we warn and store float32 so nothing is lost.

This module imports torch lazily so the --human path needs no GPU/torch.
"""
from __future__ import annotations
import os
import warnings
import numpy as np

_FP16_MAX = 65504.0


def _to_storage(arr_f32, dtype):
    """Cast float32 activations to the requested storage dtype without silently
    losing out-of-range values."""
    if dtype == "float32":
        return arr_f32
    if dtype == "bfloat16":
        try:
            import ml_dtypes
            return arr_f32.astype(ml_dtypes.bfloat16)
        except Exception:
            warnings.warn(
                "capture_dtype='bfloat16' but ml_dtypes is not installed; "
                "storing activations as float32 (run `uv add ml_dtypes` to halve "
                "the file size losslessly).")
            return arr_f32
    if dtype == "float16":
        peak = float(np.nanmax(np.abs(arr_f32))) if arr_f32.size else 0.0
        if peak > _FP16_MAX:
            warnings.warn(
                f"activations peak at {peak:.0f} (> float16 max {_FP16_MAX:.0f}); "
                "storing as float32 to avoid overflow to inf. Use "
                "capture_dtype='bfloat16' for 2-byte lossless storage.")
            return arr_f32
        return arr_f32.astype(np.float16)
    raise ValueError(f"unknown capture dtype {dtype!r}")


def _positions(tokens, prompt_len, seq, last_k):
    """Absolute token indices to capture. The commit token (seq-1) is always the
    last entry, so decision_index = len(positions) - 1."""
    if tokens == "decision":
        pos = [seq - 1]
    elif tokens == "assistant":
        pos = list(range(prompt_len, seq))
    else:  # 'lastk' (default)
        k = max(int(last_k), 1)
        pos = list(range(max(prompt_len, seq - k), seq))
    return pos or [seq - 1]


def capture_and_save(model, full_ids, prompt_len, path, tokens="lastk",
                     dtype="float16", last_k=5):
    import torch
    with torch.no_grad():
        out = model(full_ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states                      # tuple (L+1) of (1, seq, d_model)
    seq = full_ids.shape[1]
    positions = _positions(tokens, prompt_len, seq, last_k)
    idx = torch.tensor(positions, device=hs[0].device)
    stack = torch.stack([h[0].index_select(0, idx) for h in hs])   # (L+1, n_pos, d)
    acts = stack.to(torch.float32).cpu().numpy()                   # bf16 -> f32 (lossless)
    np.savez_compressed(
        path,
        acts=_to_storage(acts, dtype),
        positions=np.array(positions),
        decision_index=len(positions) - 1,
        token_ids=full_ids[0, positions].cpu().numpy(),
    )