"""trigger_spans.py -- turn a closed-loop trigger run into READABLE text spans.

For every trigger event in a `--study trigger` run it pulls, from the saved data
(no model re-run needed):

  * window_text   -- the k tokens whose pooled projection crossed the threshold
                     (i.e. the run-up that "looked like SUBMIT" and fired the steer)
  * steered_text  -- the tokens generated WHILE the steer vector was being injected
  * after_text    -- the next few tokens after steering stopped (how the cadence
                     resumed)

Inputs are the per-rollout `trigger_trace.json` (token-level [tok_i, proj, steering]
written by run_trigger) and `transcript.jsonl` (the response text per turn). Token
indices are aligned to the response by re-tokenising the saved response text with
the same tokenizer (the approach label_exploration.py already relies on); this is
exact enough to read, and a per-turn alignment note is recorded.

Run as a package module, next to transfer_studies.py:

    python -m csat.trigger_spans --run-dir runs/csat_transfer_coupling_L22_trigger

Outputs under <run-dir>:
    trigger_spans/rollout_XXXX.json   list of events (texts + token/char ranges)
    trigger_spans.md                  every fired turn with the window in <<>> and
                                      the steered span in [[ ]], read in context
"""
from __future__ import annotations
import os, re, json, glob, argparse


# --------------------------------------------------------------------------- #
# pure helpers (unit-testable without a model/tokenizer)
# --------------------------------------------------------------------------- #
def steering_runs(trace):
    """From a trace of (tok_i, proj, steering) rows, return contiguous runs where
    steering is True as (start_tok, end_tok_exclusive, proj_at_start)."""
    runs, start = [], None
    last = None
    for row in trace:
        ti, proj, steer = int(row[0]), row[1], bool(row[2])
        if steer and start is None:
            start, start_proj = ti, proj
        if (not steer) and start is not None:
            runs.append((start, last + 1, start_proj)); start = None
        last = ti
    if start is not None:
        runs.append((start, last + 1, start_proj))
    return runs


def char_spans(pieces):
    """Cumulative (lo, hi) char offsets for per-token decoded `pieces`."""
    spans, pos = [], 0
    for p in pieces:
        spans.append((pos, pos + len(p))); pos += len(p)
    return spans


def _slice_text(pieces, lo, hi):
    lo = max(0, lo); hi = min(len(pieces), hi)
    return "".join(pieces[lo:hi]), (lo, hi)


def build_events(pieces, trace, k, steer_k, after_n):
    """One record per steering run: the firing window (k tokens up to & incl. the
    trigger), the steered span, and a few tokens after. Token indices are clipped
    to the available `pieces`."""
    n = len(pieces)
    events = []
    for (s, e, proj) in steering_runs(trace):
        win_txt, win_rng = _slice_text(pieces, s - k + 1, s + 1)
        steer_txt, steer_rng = _slice_text(pieces, s, e)
        after_txt, after_rng = _slice_text(pieces, e, e + after_n)
        events.append(dict(
            trigger_tok=s, proj_at_trigger=(None if proj is None else float(proj)),
            n_steered=(min(e, n) - s), steer_run=[s, e],
            window_tok=list(win_rng), steered_tok=list(steer_rng),
            after_tok=list(after_rng),
            window_text=win_txt, steered_text=steer_txt, after_text=after_txt))
    return events


def mark_response(pieces, events):
    """Render the full response with each firing window wrapped <<...>> and each
    steered span wrapped [[...]] (token-index based; overlaps render in order)."""
    spans = char_spans(pieces)
    full = "".join(pieces)
    # collect (char_pos, marker) insertions
    ins = []
    for ev in events:
        wl, wh = ev["window_tok"]
        sl, sh = ev["steered_tok"]
        if wh > wl:
            ins.append((spans[wl][0], "<<")); ins.append((spans[wh - 1][1], ">>"))
        if sh > sl:
            ins.append((spans[sl][0], "[[")); ins.append((spans[sh - 1][1], "]]"))
    # apply right-to-left so offsets stay valid
    out = full
    for pos, mark in sorted(ins, key=lambda t: -t[0]):
        out = out[:pos] + mark + out[pos:]
    return out


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def load_transcript_text(path):
    """turn -> response text from a transcript.jsonl."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "turn" in rec:
                out[int(rec["turn"])] = rec.get("response", "") or ""
    return out


def token_pieces(tok, text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return [tok.decode([int(i)]) for i in ids]


def load_tokenizer(model_name):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except Exception:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    from .config import Config
    cfg = Config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="a --study trigger run dir (has rollout_*/trigger_trace.json)")
    ap.add_argument("--model", default=cfg.model_name,
                    help="HF id for the tokenizer (must match the run's model)")
    ap.add_argument("--after", type=int, default=24,
                    help="tokens to show AFTER each steered span (cadence resume)")
    ap.add_argument("--k", type=int, default=None,
                    help="firing-window size (default: read from composite_meta.json)")
    ap.add_argument("--steer-k", type=int, default=None,
                    help="injection length (default: read from composite_meta.json)")
    args = ap.parse_args()

    run_dir = args.run_dir
    meta = {}
    mp = os.path.join(run_dir, "composite_meta.json")
    if os.path.exists(mp):
        with open(mp) as f:
            meta = json.load(f)
    k = args.k if args.k is not None else int(meta.get("k", 20))
    steer_k = args.steer_k if args.steer_k is not None else int(meta.get("steer_k", 20))

    rollouts = sorted(glob.glob(os.path.join(run_dir, "rollout_*")))
    rollouts = [d for d in rollouts if os.path.isdir(d)]
    if not rollouts:
        raise SystemExit(f"no rollout_* dirs under {run_dir}")

    tok = load_tokenizer(args.model)
    out_dir = os.path.join(run_dir, "trigger_spans")
    os.makedirs(out_dir, exist_ok=True)
    md = [f"# Trigger spans  ({os.path.basename(run_dir)})",
          f"window `<<...>>` = the {k} tokens that projected toward SUBMIT and "
          f"fired the steer;  `[[...]]` = tokens generated under injection "
          f"(steer_k={steer_k}).\n"]

    n_events = 0
    for rd in rollouts:
        rid = os.path.basename(rd)
        tp = os.path.join(rd, "trigger_trace.json")
        if not os.path.exists(tp):
            continue
        with open(tp) as f:
            traces = json.load(f)
        texts = load_transcript_text(os.path.join(rd, "transcript.jsonl"))

        roll_events = []
        for tk in sorted(traces, key=lambda s: int(s)):
            trace = traces[tk] or []
            turn = int(tk)
            resp = texts.get(turn, "")
            if not trace or not resp:
                continue
            pieces = token_pieces(tok, resp)
            note = None
            if len(pieces) != len(trace):
                note = (f"token count differs: retokenised={len(pieces)} "
                        f"trace={len(trace)} (indices clipped)")
            events = build_events(pieces, trace, k, steer_k, args.after)
            if not events:
                continue
            for ev in events:
                ev.update(rollout=rid, turn=turn, align_note=note)
            roll_events.extend(events)
            n_events += len(events)

            md.append(f"### {rid} · turn {turn}  "
                      f"({len(events)} trigger{'s' if len(events) != 1 else ''}"
                      + (f"; {note}" if note else "") + ")")
            md.append("```")
            md.append(mark_response(pieces, events))
            md.append("```")
            for j, ev in enumerate(events):
                pj = ("n/a" if ev["proj_at_trigger"] is None
                      else f"{ev['proj_at_trigger']:+.3f}")
                md.append(f"- trigger {j} @ tok {ev['trigger_tok']} "
                          f"(proj {pj}, steered {ev['n_steered']} tok)")
                md.append(f"    window : …{ev['window_text']!r}")
                md.append(f"    steered: {ev['steered_text']!r}")
                md.append(f"    after  : {ev['after_text']!r}…")
            md.append("")

        if roll_events:
            with open(os.path.join(out_dir, f"{rid}.json"), "w") as f:
                json.dump(roll_events, f, indent=2)

    with open(os.path.join(run_dir, "trigger_spans.md"), "w") as f:
        f.write("\n".join(md))
    print(f"[spans] {n_events} trigger events across {len(rollouts)} rollouts "
          f"(k={k}, steer_k={steer_k})")
    print(f"[spans] wrote {out_dir}/rollout_*.json and "
          f"{os.path.join(run_dir, 'trigger_spans.md')}")


if __name__ == "__main__":
    main()