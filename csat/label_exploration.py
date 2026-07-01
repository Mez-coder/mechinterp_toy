"""label_exploration.py -- have the SAME model read each SET turn's reasoning in
isolation and flag the tokens that show *active exploration* ("try a region not
tried before / change tack / reconsider the current setting"), then read those
tokens' IN-TASK activations to build the positive pole of a steering direction.

Why this exists
---------------
The current direction is (SUBMIT_all - final_SET_all). The final-SET pole is
contaminated by the *act of setting* (the commit to emit a SET line), so negative
steering can just re-SET the same weights instead of exploring. Here we relabel:
keep SUBMIT as the "stop now" pole, but replace the positive pole with the
activations at the tokens the model itself marks as exploration.

Two distinct passes per SET turn (this is the important part):
  1) LABELING pass  -- a fresh, isolated conversation: the model reads the turn's
     reasoning text and returns the exploration spans VERBATIM. Pure annotation.
  2) ACTIVATION pass -- we re-forward the turn AS IT HAPPENED IN THE TASK
     (deterministic env replay rebuilds the exact context, mirroring rollout.py),
     and pool the residual stream at the flagged token positions. So the vector
     encodes *exploring while solving*, not *reading about exploring*.

Capture-window note: your turn_XX.npz only stores the last capture_last_k tokens,
which usually excludes the exploration reasoning. That is why we re-forward
(reusing recorder._forward_hidden_states) instead of reading the saved npz.

Run as a package module, next to rollout.py / direction_extract.py:

    python -m csat.label_exploration --source-run-dir runs/csat
    # smoke test on a few rollouts, see what the model picks, skip activations:
    python -m csat.label_exploration --source-run-dir runs/csat \
        --max-rollouts 3 --no-acts

Outputs under <source-run-dir>:
    exploration/rollout_XXXX.npz   per-rollout explore pole: turn_vecs (T,L+1,d),
                                   mean_vec (L+1,d), turns, n_tokens
    exploration/labels.jsonl       one record per labelled turn (selection, spans,
                                   token positions, matched flag)
    exploration_snippets.md        human-readable: each trace with the model's
                                   selected spans wrapped in <<< >>>
"""
from __future__ import annotations
import os, re, json, glob, argparse
import numpy as np

from .config import Config
from .agents import ModelAgent
from .rollout import build_env
from .dsl import split_thinking
from .prompts import system_prompt_for, render_case_for, render_feedback_for
from .recorder import _forward_hidden_states


# --------------------------------------------------------------------------- #
# labeling prompt + parsing
# --------------------------------------------------------------------------- #
LABEL_INSTRUCTION = (
    "Below is a reasoning trace produced by an agent solving a trial-and-error "
    "optimisation task. The agent repeatedly adjusts numeric weights, sees a "
    "margin/feedback table, and decides whether to keep adjusting or stop.\n\n"
    "Identify the spans of text where the agent is ACTIVELY EXPLORING: deciding "
    "to try a setting or region it has not tried, changing tack, or reconsidering "
    "its current weights in favour of a different option. Do NOT select text that "
    "merely restates the current plan, reports/repeats the feedback, confirms it "
    "is satisfied, or commits to stopping. Do NOT select the final action command "
    "line itself (the 'SET ...' or 'SUBMIT ...' line).\n\n"
    "Copy each selected span VERBATIM from the trace. Respond with ONLY a JSON "
    "array of strings, each an exact substring of the trace. If there is no such "
    "text, respond with []." 
)


def label_messages(response_text):
    body = f"{LABEL_INSTRUCTION}\n\nReasoning trace:\n<<<\n{response_text}\n>>>"
    return [{"role": "user", "content": body}]


def _label_inputs(agent, messages):
    out = agent.processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", enable_thinking=False)
    ids = out["input_ids"] if hasattr(out, "keys") else out
    return ids.to(agent.model.device)


def generate_label(agent, response_text, max_new=256):
    """Greedy (deterministic) annotation of one trace. Returns the raw model text."""
    import torch
    ids = _label_inputs(agent, label_messages(response_text))
    with torch.no_grad():
        out = agent.model.generate(
            ids, max_new_tokens=max_new, do_sample=False,
            pad_token_id=agent.processor.tokenizer.eos_token_id,
            attention_mask=torch.ones_like(ids))
    return agent.processor.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


_JSON_STR = re.compile(r'"(?:[^"\\]|\\.)*"', re.DOTALL)


def _salvage_strings(s):
    """Recover every COMPLETE JSON string literal from `s`, even if the enclosing
    array was never closed (truncated generation). The final partial element (no
    closing quote) is naturally excluded."""
    out = []
    for m in _JSON_STR.finditer(s):
        try:
            v = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(v, str) and v.strip():
            out.append(v)
    return out


def parse_selections(text):
    """Pull the JSON array of strings out of the model's answer; tolerant of
    surrounding prose / code fences AND of truncation (a budget-clipped array is
    salvaged element-by-element). Returns list[str] (possibly empty)."""
    if not text:
        return []
    text = text.replace("```json", "```").strip()
    i = text.find("[")
    if i == -1:
        return []
    # strict parse if the array closes cleanly
    depth, j = 0, None
    for k in range(i, len(text)):
        if text[k] == "[":
            depth += 1
        elif text[k] == "]":
            depth -= 1
            if depth == 0:
                j = k + 1
                break
    if j is not None:
        try:
            arr = json.loads(text[i:j])
            return [s for s in arr if isinstance(s, str) and s.strip()]
        except json.JSONDecodeError:
            pass
    # truncated / malformed -> salvage complete string elements
    return _salvage_strings(text[i:])


# --------------------------------------------------------------------------- #
# span -> token index mapping
# --------------------------------------------------------------------------- #
def token_pieces(tokenizer, ids):
    """Per-token decoded strings for a 1D id list (so concatenation ~= the text)."""
    return [tokenizer.decode([int(t)]) for t in ids]


def _norm_with_map(s):
    """Collapse runs of whitespace to one space; return (norm, map) where
    map[k] = index in the ORIGINAL string of norm[k]."""
    out, mp, prev_space = [], [], False
    for i, ch in enumerate(s):
        if ch.isspace():
            if prev_space:
                continue
            out.append(" "); mp.append(i); prev_space = True
        else:
            out.append(ch); mp.append(i); prev_space = False
    return "".join(out), mp


def match_span(pieces, target):
    """Return (token_indices, (char_lo, char_hi)) for `target` within the text
    reconstructed from `pieces`. Tries exact then whitespace-normalised. Empty
    list if not found."""
    spans, pos = [], 0
    for p in pieces:
        spans.append((pos, pos + len(p))); pos += len(p)
    full = "".join(pieces)
    t = target.strip()
    if not t:
        return [], None

    lo = full.find(t)
    if lo != -1:
        hi = lo + len(t)
    else:                                            # normalised fallback
        fn, fmap = _norm_with_map(full)
        tn, _ = _norm_with_map(t)
        p = fn.find(tn)
        if p == -1:
            return [], None
        lo, hi = fmap[p], fmap[p + len(tn) - 1] + 1

    idx = [k for k, (a, b) in enumerate(spans) if a < hi and b > lo]
    return idx, (lo, hi)


# --------------------------------------------------------------------------- #
# deterministic in-task replay (mirrors rollout.py message construction)
# --------------------------------------------------------------------------- #
def load_records(transcript_path):
    recs = []
    with open(transcript_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    recs.sort(key=lambda r: r.get("turn", 0))
    return recs


def iter_turn_contexts(cfg, seed, records):
    """Yield (turn, kind, response, context_messages) for each turn, where
    context_messages is exactly what the model saw to generate that turn.
    Advances env+messages between turns the same way rollout.py does."""
    env = build_env(cfg)
    env.reset(seed=seed, wide=getattr(cfg, "wide_cases", True))
    messages = [{"role": "system", "content": system_prompt_for(cfg)},
                {"role": "user", "content": render_case_for(env, cfg)}]
    for rec in records:
        turn = int(rec.get("turn", 0))
        kind = rec.get("action", "")
        resp = rec.get("response", "") or ""
        yield turn, kind, resp, list(messages)

        # advance to set up the next turn (mirror rollout.py exactly)
        answer, _ = split_thinking(resp, cfg.enable_thinking)
        messages.append({"role": "assistant", "content": answer})
        weights = rec.get("weights") or {}
        if kind in ("submit", "forced_submit"):
            for i, w in weights.items():
                ii = int(i)
                if 0 <= ii < env.n_obj:
                    env.set_weight(ii, w)
            break
        if kind == "set":
            for i, w in weights.items():
                ii = int(i)
                if 0 <= ii < env.n_obj:
                    env.set_weight(ii, w)
            messages.append({"role": "user",
                             "content": render_feedback_for(env, cfg, env.feedback(),
                                                            turn=turn, max_turns=cfg.max_turns)})
        else:                                        # parse_error
            messages.append({"role": "user",
                             "content": f"Could not parse an action ({rec.get('error','')}).\n"
                             + render_feedback_for(env, cfg, turn=turn, max_turns=cfg.max_turns)})


# --------------------------------------------------------------------------- #
# in-task activation pooling at flagged tokens
# --------------------------------------------------------------------------- #
def intask_pooled_acts(agent, context_messages, response_text, flagged_resp_idx):
    """Re-forward [generation prompt for this turn] ++ [response tokens] and mean-
    pool the residual stream over the flagged response-token positions.
    Returns (L+1, d) float32, or None. Also returns the response token ids/pieces
    via the caller path (we recompute pieces here for indexing)."""
    import torch
    tok = agent.processor.tokenizer
    prompt_ids = agent._build_inputs(context_messages)               # (1, P)
    resp_ids = tok(response_text, add_special_tokens=False,
                   return_tensors="pt").input_ids.to(prompt_ids.device)   # (1, R)
    R = resp_ids.shape[1]
    flagged = [i for i in flagged_resp_idx if 0 <= i < R]
    if not flagged:
        return None, R
    full = torch.cat([prompt_ids, resp_ids], dim=1)
    P = prompt_ids.shape[1]
    abs_idx = torch.tensor([P + i for i in flagged], device=full.device)
    hs = _forward_hidden_states(agent.model, full)                   # tuple (L+1) of (1,seq,d)
    stack = torch.stack([h[0].index_select(0, abs_idx) for h in hs]) # (L+1, n_flag, d)
    pooled = stack.to(torch.float32).mean(axis=1).cpu().numpy()      # (L+1, d)
    del hs, stack, full
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pooled, R


# --------------------------------------------------------------------------- #
# per-rollout driver
# --------------------------------------------------------------------------- #
def cfg_for_rollout(base_cfg, rdir):
    """Clone base cfg but force env_kind / n_obj to match this rollout's case.json
    so the deterministic replay reproduces the original landscape."""
    import copy
    cfg = copy.copy(base_cfg)
    case_path = os.path.join(rdir, "case.json")
    if os.path.exists(case_path):
        with open(case_path) as f:
            case = json.load(f)
        cfg.env_kind = case.get("env_kind", cfg.env_kind)
        cfg.n_obj = int(case.get("n_obj", cfg.n_obj))
    return cfg


def process_rollout(agent, base_cfg, rdir, do_acts, label_max_new,
                    md_lines, label_records, max_turns_per_rollout=None):
    """Returns (turn_vecs (list of (L+1,d)), turns (list of int)) for SET turns
    with at least one matched exploration span."""
    case_path = os.path.join(rdir, "case.json")
    tr_path = os.path.join(rdir, "transcript.jsonl")
    if not (os.path.exists(case_path) and os.path.exists(tr_path)):
        return [], []
    with open(case_path) as f:
        seed = int(json.load(f).get("seed", 0))
    cfg = cfg_for_rollout(base_cfg, rdir)
    records = load_records(tr_path)
    rid = os.path.basename(rdir)
    tok = agent.processor.tokenizer

    turn_vecs, turns = [], []
    n_set = 0
    for turn, kind, resp, ctx in iter_turn_contexts(cfg, seed, records):
        if kind != "set" or not resp.strip():
            continue
        if max_turns_per_rollout is not None and n_set >= max_turns_per_rollout:
            break
        n_set += 1

        raw = generate_label(agent, resp, max_new=label_max_new)
        sels = parse_selections(raw)

        # map each selection onto the response token sequence (for acts + display)
        resp_ids = tok(resp, add_special_tokens=False)["input_ids"]
        pieces = token_pieces(tok, resp_ids)
        flagged_idx, char_ranges = [], []
        for s in sels:
            idx, cr = match_span(pieces, s)
            if idx:
                flagged_idx.extend(idx)
                if cr:
                    char_ranges.append(cr)
        flagged_idx = sorted(set(flagged_idx))
        matched = len(flagged_idx) > 0

        rec = dict(rollout=rid, turn=turn, n_selections=len(sels),
                   matched=matched, n_tokens=len(flagged_idx),
                   selections=sels, raw_answer=raw,
                   token_positions=flagged_idx)
        label_records.append(rec)

        # human-readable dump (wrap matched spans in <<< >>>)
        md_lines.append(f"### {rid}  turn {turn}  "
                        f"({'matched' if matched else 'NO MATCH'}, "
                        f"{len(sels)} selection(s))")
        md_lines.append("```")
        md_lines.append(_wrap_spans(resp, char_ranges))
        md_lines.append("```")
        md_lines.append(f"model JSON: `{raw[:500]}`\n")

        if matched and do_acts:
            pooled, _R = intask_pooled_acts(agent, ctx, resp, flagged_idx)
            if pooled is not None:
                turn_vecs.append(pooled)
                turns.append(turn)
    return turn_vecs, turns


def _wrap_spans(text, char_ranges):
    """Insert <<< >>> around each (lo,hi) char range (merged, sorted)."""
    if not char_ranges:
        return text
    rs = sorted(set(char_ranges))
    merged = [list(rs[0])]
    for lo, hi in rs[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    out, prev = [], 0
    for lo, hi in merged:
        out.append(text[prev:lo]); out.append("<<<"); out.append(text[lo:hi])
        out.append(">>>"); prev = hi
    out.append(text[prev:])
    return "".join(out)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    cfg = Config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-run-dir", default=os.path.join(cfg.out_dir, cfg.run_name),
                    help="run dir of rollout_* dirs to label (default runs/<run_name>)")
    ap.add_argument("--model", default=None, help="HF id override")
    ap.add_argument("--max-rollouts", type=int, default=None,
                    help="only process the first N rollouts (smoke test)")
    ap.add_argument("--max-turns-per-rollout", type=int, default=None,
                    help="cap SET turns labelled per rollout (smoke test)")
    ap.add_argument("--label-max-new", type=int, default=1024,
                    help="max new tokens for the annotation generation")
    ap.add_argument("--no-acts", action="store_true",
                    help="label + dump snippets only; skip the activation re-forward")
    args = ap.parse_args()

    if args.model:
        cfg.model_name = args.model
    src = args.source_run_dir
    out_dir = os.path.join(src, "exploration")
    os.makedirs(out_dir, exist_ok=True)

    rollout_dirs = sorted(d for d in glob.glob(os.path.join(src, "rollout_*"))
                          if os.path.isdir(d))
    if args.max_rollouts is not None:
        rollout_dirs = rollout_dirs[:args.max_rollouts]
    if not rollout_dirs:
        raise SystemExit(f"no rollout_* dirs under {src}")
    print(f"[label] {len(rollout_dirs)} rollouts under {src}; "
          f"acts={'off' if args.no_acts else 'on'}")

    agent = ModelAgent(cfg)                          # your loader

    md_lines = ["# Exploration-span selections\n"]
    label_records = []
    n_rollout_vecs = 0
    for rdir in rollout_dirs:
        rid = os.path.basename(rdir)
        tvecs, turns = process_rollout(
            agent, cfg, rdir, do_acts=not args.no_acts,
            label_max_new=args.label_max_new, md_lines=md_lines,
            label_records=label_records,
            max_turns_per_rollout=args.max_turns_per_rollout)
        n_matched = sum(1 for r in label_records if r["rollout"] == rid and r["matched"])
        n_total = sum(1 for r in label_records if r["rollout"] == rid)
        print(f"  {rid}: {n_matched}/{n_total} SET turns matched"
              + (f", {len(tvecs)} turn-vectors" if not args.no_acts else ""))
        if tvecs:
            arr = np.stack(tvecs).astype(np.float32)          # (T, L+1, d)
            np.savez_compressed(os.path.join(out_dir, f"{rid}.npz"),
                                turn_vecs=arr, mean_vec=arr.mean(0),
                                turns=np.array(turns, dtype=np.int32))
            n_rollout_vecs += 1

    with open(os.path.join(out_dir, "labels.jsonl"), "w") as f:
        for r in label_records:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(src, "exploration_snippets.md"), "w") as f:
        f.write("\n".join(md_lines))

    n_match = sum(r["matched"] for r in label_records)
    print(f"\n[label] labelled {len(label_records)} SET turns; "
          f"{n_match} had a matched span; "
          f"{n_rollout_vecs} rollouts produced an explore vector.")
    print(f"[label] wrote {out_dir}/labels.jsonl, "
          f"{os.path.join(src,'exploration_snippets.md')}"
          + ("" if args.no_acts else f", {out_dir}/rollout_*.npz"))
    if args.no_acts:
        print("[label] re-run without --no-acts to compute the explore vectors, "
              "then build the direction with build_explore_directions.py")


if __name__ == "__main__":
    main()