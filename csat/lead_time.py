"""lead_time.py -- how far AHEAD of the verbalised action does the detector fire?

The anticipation study showed the SUBMIT-SET projection carries no TURN-level
foresight: on SET turns it is at chance for "submits within m turns". The claim
that remains testable is TOKEN-level foresight WITHIN the submit turn: does the
pooled projection cross the threshold while the model is still mid-deliberation,
before the SUBMIT verb (or the committed action line) is emitted?

For every turn of a `--study trigger` run this script measures, offline:

  verb_tok     -- token index where the action verb (SUBMIT / SET) starts in the
                  re-tokenised response
  first_cross  -- first token whose recorded pooled projection crosses the
                  threshold (the trace is causal: the projection at token i uses
                  only tokens <= i, so a crossing before verb_tok is genuinely
                  pre-verbalisation IN TOKENS)
  lead         -- verb_tok - first_cross   (>0 = detector fired early)
  window_text  -- the k tokens up to the crossing (what the detector "saw")
  lead_span    -- the text BETWEEN the crossing and the verb (what the model
                  said after latent commitment but before verbalising it)

plus, on SET turns, the false-fire rate: crossings on turns that did NOT submit.
A detector is only a useful early-warning monitor if it fires early on submit
turns AND rarely on set turns; both numbers are reported, together with a
turn-level AUROC of the within-turn peak projection (submit vs set turns).

Honest caveats, printed with the results:
  * lead is a LOWER bound on latent lead: the pooled window is k tokens, so the
    detector cannot react faster than the pooling smooths (and cannot fire at
    all inside the first k-1 tokens of a turn).
  * lexical vs latent is decided by READING window_text / lead_span: if the
    window already says "I'll finalise", the crossing is verbalised commitment,
    not latent state. The .md output exists to make that reading easy.
  * run this on a MONITOR-ONLY pass (--alpha 0): in a steered run every token
    after the first injection is contaminated. The script warns if it detects
    steered tokens before the first crossing of a later turn.

Usage (after a monitor-only trigger run):

    python -m csat.transfer_studies --study trigger --alpha 0 \
        --env coupling --steer-proj 0.0 --k 20 --n-rollouts 12 \
        --run-name csat_monitor
    python -m csat.lead_time --run-dir runs/csat_monitor_coupling_L22_trigger

Outputs under <run-dir>:
    lead_times.json   per-turn records + summary
    lead_times.png    lead histogram; crossing rates; peak-proj separation
    lead_times.md     each submit turn with {{CROSS}} and <<VERB>> marked inline
"""
from __future__ import annotations
import os, json, glob, argparse
import numpy as np

from .trigger_spans import token_pieces, load_tokenizer, char_spans


# --------------------------------------------------------------------------- #
# pure helpers (unit-testable with any tokenizer-like object)
# --------------------------------------------------------------------------- #
def load_transcript(path):
    """turn -> dict(action, response) from a trigger run's transcript.jsonl."""
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
                out[int(rec["turn"])] = dict(action=rec.get("action"),
                                             response=rec.get("response", "") or "")
    return out


# Heuristic stop-language lexicon: phrases that VERBALISE the decision to stop.
# Deliberately broad; the test asks whether the detector fires before ANY of
# these appear, so false positives in the lexicon only make the test harder to
# pass (conservative in the right direction). Override with --markers.
STOP_MARKERS = [
    r"\bsubmit\w*", r"\bfinali[sz]\w*", r"\bgood enough\b", r"\boptimal\b",
    r"\boptimum\b", r"\bmaximum\b", r"\bmaximi[sz]ed\b", r"\bsettle\b",
    r"\bconclu\w*", r"\bdone\b", r"\bstop\b", r"\bno further\b",
    r"\bbest so far\b", r"\bhappy with\b", r"\bsatisf\w*", r"\bplateau\w*",
    r"\bexhausted\b", r"\bcannot improve\b", r"\bno (?:more|room)\b",
]


def _marker_re(markers):
    import re as _re
    return _re.compile("|".join(markers), _re.IGNORECASE)


def first_marker_tok(pieces, mre):
    """(token_index, matched_text) of the FIRST stop-language marker in the
    response, or (None, None)."""
    spans = char_spans(pieces)
    text = "".join(pieces)
    m = mre.search(text)
    if not m:
        return None, None
    for i, (a, b) in enumerate(spans):
        if a <= m.start() < b:
            return i, m.group(0)
    return None, None


def window_at_frac(pieces, frac, k):
    """The k-token window ENDING at fractional position `frac` of the turn."""
    n = len(pieces)
    if n == 0:
        return ""
    end = max(1, min(n, int(round(frac * (n - 1))) + 1))
    return "".join(pieces[max(0, end - k):end])


def locate_verb_pieces(pieces, verb):
    """Token index (into `pieces`) where the LAST occurrence of `verb` starts,
    or None. Mirrors direction_extract.locate_verb but works on decoded pieces."""
    spans = char_spans(pieces)
    text = "".join(pieces)
    pos = text.upper().rfind(verb.upper())
    if pos == -1:
        return None
    for i, (a, b) in enumerate(spans):
        if a <= pos < b:
            return i
    return None


def first_crossing(trace, thr, mode="above", before_tok=None):
    """First (tok_i, proj) whose pooled projection crosses thr. Rows with proj
    None (buffer warm-up) are skipped. before_tok limits the search (exclusive)."""
    for row in trace:
        ti, proj = int(row[0]), row[1]
        if proj is None:
            continue
        if before_tok is not None and ti >= before_tok:
            break
        if (proj > thr) if mode == "above" else (proj < thr):
            return ti, float(proj)
    return None


def peak_proj(trace):
    vals = [r[1] for r in trace if r[1] is not None]
    return float(max(vals)) if vals else None


def analyze_turn(pieces, trace, action, thr, mode, k, max_span_chars=400,
                 mre=None):
    """One record per turn. For submit turns, lead is measured to the verb; for
    set turns any crossing at all is a (potential) false fire. If mre (compiled
    stop-marker regex) is given, submit turns also get the LEXICAL timing:
    marker_tok (first stop-language token), marker_lead = marker_tok - cross_tok
    (positive = detector fired BEFORE any stop-language existed in the text)."""
    n_tok = len(trace)
    verb = "SUBMIT" if action == "submit" else ("SET" if action == "set" else None)
    verb_tok = locate_verb_pieces(pieces, verb) if verb else None
    # crossing anywhere in the turn (false-fire stat), and strictly pre-verb
    cross_any = first_crossing(trace, thr, mode)
    cross_pre = (first_crossing(trace, thr, mode, before_tok=verb_tok)
                 if verb_tok is not None else cross_any)
    rec = dict(action=action, n_tokens=n_tok, n_pieces=len(pieces),
               verb_tok=verb_tok, peak_proj=peak_proj(trace),
               crossed=bool(cross_any),
               cross_tok=(cross_any[0] if cross_any else None),
               cross_proj=(cross_any[1] if cross_any else None),
               detectable=bool(n_tok >= k))       # turns shorter than k can't fire
    if action == "set" and cross_any:              # false fire: keep its window
        ct = cross_any[0]
        rec["window_text"] = "".join(pieces[max(0, ct - k + 1):ct + 1])
        rec["after_text"] = "".join(pieces[ct + 1:ct + 1 + k])[:max_span_chars]
    if action == "submit" and verb_tok is not None:
        rec["crossed_before_verb"] = bool(cross_pre)
        if mre is not None:
            mt, mtxt = first_marker_tok(pieces, mre)
            rec["marker_tok"], rec["marker_text"] = mt, mtxt
            if cross_pre and mt is not None:
                rec["marker_lead"] = int(mt - cross_pre[0])
                rec["crossed_before_marker"] = bool(cross_pre[0] < mt)
            elif cross_pre and mt is None:
                rec["crossed_before_marker"] = True   # no stop-language at all
        if cross_pre:
            ct = cross_pre[0]
            rec.update(lead_tokens=int(verb_tok - ct),
                       lead_frac=float((verb_tok - ct) / max(verb_tok, 1)),
                       cross_frac=float(ct / max(n_tok - 1, 1)),
                       window_text="".join(pieces[max(0, ct - k + 1):ct + 1]),
                       precross_text="".join(
                           pieces[max(0, ct - 2 * k + 1):max(0, ct - k + 1)]),
                       lead_span="".join(pieces[ct + 1:verb_tok])[:max_span_chars])
        elif cross_any:                            # fired only at/after the verb
            rec.update(lead_tokens=int(verb_tok - cross_any[0]))
    return rec


def summarize(records):
    subs = [r for r in records if r["action"] == "submit" and r["verb_tok"] is not None
            and r["detectable"]]
    sets_ = [r for r in records if r["action"] == "set" and r["detectable"]]
    out = dict(n_submit_turns=len(subs), n_set_turns=len(sets_),
               n_undetectable=sum(1 for r in records if not r["detectable"]))
    if subs:
        pre = [r for r in subs if r.get("crossed_before_verb")]
        out["frac_crossed_before_verb"] = len(pre) / len(subs)
        leads = np.array([r["lead_tokens"] for r in pre], float)
        if len(leads):
            out.update(lead_median=float(np.median(leads)),
                       lead_iqr=[float(np.percentile(leads, 25)),
                                 float(np.percentile(leads, 75))],
                       lead_max=int(leads.max()))
    if sets_:
        out["set_false_fire_rate"] = float(np.mean([r["crossed"] for r in sets_]))
        # clustering: are false fires spread across rollouts, or concentrated in a
        # few (e.g. a deadlocked rollout repeatedly contemplating giving up)?
        by_roll = {}
        for r in sets_:
            by_roll.setdefault(r["rollout"], []).append(r["crossed"])
        rates = {g: float(np.mean(v)) for g, v in by_roll.items()}
        out["set_false_fire_by_rollout"] = rates
        out["n_rollouts_with_false_fires"] = sum(1 for v in rates.values() if v > 0)
    # turn-level separation of the within-turn PEAK projection
    ys = [1] * len(subs) + [0] * len(sets_)
    ps = [r["peak_proj"] for r in subs] + [r["peak_proj"] for r in sets_]
    ok = [(p, y) for p, y in zip(ps, ys) if p is not None]
    if ok and 0 < sum(y for _, y in ok) < len(ok):
        p = np.array([a for a, _ in ok]); y = np.array([b for _, b in ok], bool)
        order = np.argsort(p, kind="mergesort")
        ranks = np.empty(len(p)); sx = p[order]; i = 0; r0 = 1
        while i < len(p):
            j = i
            while j + 1 < len(p) and sx[j + 1] == sx[i]:
                j += 1
            ranks[order[i:j + 1]] = 0.5 * (r0 + r0 + (j - i)); r0 += j - i + 1; i = j + 1
        n1, n0 = int(y.sum()), int((~y).sum())
        out["peak_proj_auroc_submit_vs_set"] = \
            float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return out


def lexical_summary(records, set_bank, mre):
    """Formalises two claims about the crossings on submit turns:

    (1) ORDERING: the detector fires before the text contains ANY stop-language
        (STOP_MARKERS). frac_crossed_before_marker; marker_lead = tokens from
        crossing to the first marker (positive = detector earlier).
    (2) UNREMARKABLE WINDOWS: the k-token window the detector was reading at the
        crossing contains stop-language no more often than (a) the window one k
        earlier in the SAME turn and (b) position-matched windows from SET turns
        (set_bank: {decile: [window texts]}). If rate(cross) >> rate(matched),
        the detector is at least partly keying on surface stop-vocabulary; if
        the rates are similar while the ordering test passes, the crossing is
        not explained by shallow lexical content.

    Caveat (report it): the lexicon is heuristic, and 'lexically unremarkable'
    does not rule out the detector reading SEMANTIC content ('O1 has no room
    left') -- it rules out shallow stop-vocabulary matching."""
    subs = [r for r in records if r["action"] == "submit"
            and r.get("crossed_before_verb")]
    out = {}
    with_marker_info = [r for r in subs if "crossed_before_marker" in r]
    if with_marker_info:
        out["frac_crossed_before_marker"] = float(
            np.mean([r["crossed_before_marker"] for r in with_marker_info]))
        leads = [r["marker_lead"] for r in with_marker_info
                 if r.get("marker_lead") is not None]
        if leads:
            leads = np.asarray(leads, float)
            out["marker_lead_median"] = float(np.median(leads))
            out["marker_lead_iqr"] = [float(np.percentile(leads, 25)),
                                      float(np.percentile(leads, 75))]
        out["n_no_marker_at_all"] = int(sum(1 for r in with_marker_info
                                            if r.get("marker_tok") is None))

    def _rate(texts):
        texts = [t for t in texts if t]
        if not texts:
            return None
        return float(np.mean([bool(mre.search(t)) for t in texts]))

    out["window_marker_rate_cross"] = _rate([r.get("window_text") for r in subs])
    out["window_marker_rate_precross"] = _rate([r.get("precross_text")
                                                for r in subs])
    # position-matched SET windows: for each crossing, take set-turn windows at
    # the nearest decile to the crossing's fractional position
    matched = []
    for r in subs:
        f = r.get("cross_frac")
        if f is None or not set_bank:
            continue
        dec = min(set_bank, key=lambda d: abs(d - f))
        matched.extend(set_bank[dec])
    out["window_marker_rate_set_matched"] = _rate(matched)
    out["n_set_matched_windows"] = len(matched)
    return out


def plot_lexical(records, lex, out_png, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    leads = [r.get("marker_lead") for r in records
             if r["action"] == "submit" and r.get("marker_lead") is not None]
    if leads:
        ax1.hist(leads, bins=max(5, min(20, len(leads))), color="tab:blue",
                 alpha=0.8)
    ax1.axvline(0, color="tab:red", ls="--", lw=1)
    ax1.set_xlabel("first stop-language token - crossing token")
    ax1.set_ylabel("submit turns")
    ax1.set_title("Crossing vs first stop-language\n(>0 = detector fired first)",
                  fontsize=9)
    ax1.grid(alpha=0.3)
    labels = ["crossing\nwindow", "same turn,\nk earlier", "SET turns,\nmatched pos."]
    vals = [lex.get("window_marker_rate_cross"),
            lex.get("window_marker_rate_precross"),
            lex.get("window_marker_rate_set_matched")]
    xs = [i for i, v in enumerate(vals) if v is not None]
    ax2.bar([labels[i] for i in xs], [vals[i] for i in xs],
            color=["tab:blue", "tab:gray", "tab:green"], alpha=0.8)
    for i, x in enumerate(xs):
        ax2.text(i, vals[x] + 0.02, f"{vals[x]:.0%}", ha="center", fontsize=9)
    ax2.set_ylim(0, 1.1)
    ax2.set_title(f"Stop-language rate in {k}-token windows", fontsize=9)
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"[lead] wrote {out_png}")


def mark_turn(pieces, rec):
    """Response text with {{CROSS}} at the crossing token and <<VERB>> markers."""
    spans = char_spans(pieces)
    full = "".join(pieces)
    ins = []
    if rec.get("cross_tok") is not None and rec["cross_tok"] < len(spans):
        ins.append((spans[rec["cross_tok"]][1], "{{CROSS}}"))
    if rec.get("verb_tok") is not None and rec["verb_tok"] < len(spans):
        ins.append((spans[rec["verb_tok"]][0], "<<VERB>>"))
    out = full
    for pos, mark in sorted(ins, key=lambda t: -t[0]):
        out = out[:pos] + mark + out[pos:]
    return out


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def plot_leads(records, summ, out_png, k, thr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.0))

    leads = [r["lead_tokens"] for r in records
             if r["action"] == "submit" and r.get("crossed_before_verb")]
    if leads:
        ax1.hist(leads, bins=max(5, min(25, len(leads))), color="tab:blue", alpha=0.8)
    ax1.axvline(k, color="tab:red", ls="--", lw=1,
                label=f"pooling window k={k}\n(leads < k partly smoothing)")
    ax1.set_xlabel("lead (tokens before the verb)"); ax1.set_ylabel("submit turns")
    ax1.set_title("Detector lead on submit turns"); ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    labels = ["submit:\npre-verb", "submit:\nat/after verb\nor never", "set:\nfalse fire"]
    n_sub = max(summ.get("n_submit_turns", 0), 1)
    pre = summ.get("frac_crossed_before_verb", 0.0)
    vals = [pre, 1.0 - pre, summ.get("set_false_fire_rate", 0.0)]
    ax2.bar(labels, vals, color=["tab:green", "tab:gray", "tab:red"], alpha=0.75)
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
    ax2.set_ylim(0, 1.12); ax2.set_title(f"Crossing rates (thr={thr:g})")
    ax2.grid(alpha=0.3, axis="y")

    ps = [r["peak_proj"] for r in records if r["action"] == "set"
          and r["peak_proj"] is not None]
    pb = [r["peak_proj"] for r in records if r["action"] == "submit"
          and r["peak_proj"] is not None]
    ax3.hist(ps, bins=20, alpha=0.6, label="set turns", color="tab:blue",
             density=True)
    ax3.hist(pb, bins=20, alpha=0.6, label="submit turns", color="tab:orange",
             density=True)
    ax3.axvline(thr, color="tab:red", ls="--", lw=1, label="threshold")
    auc = summ.get("peak_proj_auroc_submit_vs_set")
    ax3.set_title("Within-turn PEAK projection"
                  + (f"  (AUROC {auc:.3f})" if auc is not None else ""))
    ax3.set_xlabel("peak pooled projection"); ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"[lead] wrote {out_png}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    from .config import Config
    cfg = Config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="a --study trigger run dir (rollout_*/trigger_trace.json)")
    ap.add_argument("--model", default=cfg.model_name,
                    help="HF id for the tokenizer (must match the run's model)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="crossing threshold (default: steer_proj from "
                         "composite_meta.json)")
    ap.add_argument("--mode", choices=["above", "below"], default=None,
                    help="crossing direction (default: from composite_meta.json)")
    ap.add_argument("--k", type=int, default=None,
                    help="pooling window (default: from composite_meta.json)")
    ap.add_argument("--md-max", type=int, default=40,
                    help="max submit turns to render in lead_times.md")
    ap.add_argument("--markers", nargs="+", default=None,
                    help="override the stop-language regex lexicon "
                         "(default: built-in STOP_MARKERS)")
    args = ap.parse_args()

    run_dir = args.run_dir
    meta = {}
    mp = os.path.join(run_dir, "composite_meta.json")
    if os.path.exists(mp):
        with open(mp) as f:
            meta = json.load(f)
    thr = args.threshold if args.threshold is not None else float(meta.get("steer_proj", 0.0))
    mode = args.mode or meta.get("trigger", "above")
    k = args.k if args.k is not None else int(meta.get("k", 20))
    alpha = meta.get("alpha")
    if alpha not in (None, 0, 0.0):
        print(f"[lead] WARNING: this run injected with alpha={alpha}; every token "
              "after the first injection is contaminated by the steer. For a clean "
              "lead-time measurement re-run the trigger study with --alpha 0.")

    tok = load_tokenizer(args.model)
    mre = _marker_re(args.markers or STOP_MARKERS)
    records, md_blocks, ff_blocks = [], [], []
    set_bank = {round(f, 1): [] for f in np.arange(0.1, 1.0, 0.1)}
    rollouts = sorted(d for d in glob.glob(os.path.join(run_dir, "rollout_*"))
                      if os.path.isdir(d))
    if not rollouts:
        raise SystemExit(f"no rollout_* dirs under {run_dir}")

    for rd in rollouts:
        rid = os.path.basename(rd)
        tp = os.path.join(rd, "trigger_trace.json")
        if not os.path.exists(tp):
            continue
        with open(tp) as f:
            traces = json.load(f)
        turns = load_transcript(os.path.join(rd, "transcript.jsonl"))
        for tk in sorted(traces, key=lambda s: int(s)):
            turn = int(tk)
            trace = traces[tk] or []
            info = turns.get(turn, {})
            action, resp = info.get("action"), info.get("response", "")
            if not trace or not resp or action not in ("set", "submit"):
                continue
            pieces = token_pieces(tok, resp)
            if action == "set" and len(pieces) >= k:
                for f in set_bank:                 # position bank for matching
                    set_bank[f].append(window_at_frac(pieces, f, k))
            rec = analyze_turn(pieces, trace, action, thr, mode, k, mre=mre)
            rec.update(rollout=rid, turn=turn,
                       align_note=(None if len(pieces) == len(trace) else
                                   f"retok={len(pieces)} trace={len(trace)}"))
            records.append(rec)
            if action == "submit" and len(md_blocks) < args.md_max:
                lead = rec.get("lead_tokens")
                mk = rec.get("marker_lead")
                mtxt = rec.get("marker_text")
                minfo = (f", first stop-language {mk:+d} tok "
                         f"({mtxt!r})" if mk is not None else
                         (", no stop-language found"
                          if rec.get("marker_tok") is None
                          and "crossed_before_marker" in rec else ""))
                pk = rec.get("peak_proj")
                pks = f"{pk:+.3f}" if pk is not None else "n/a (turn < k)"
                md_blocks.append(
                    f"### {rid} · turn {turn}  "
                    f"(lead={lead if lead is not None else 'n/a'} tok"
                    f"{minfo}, peak={pks})"
                    + (f"; {rec['align_note']}" if rec["align_note"] else "")
                    + "\n```\n" + mark_turn(pieces, rec) + "\n```\n")
            elif action == "set" and rec["crossed"] and len(ff_blocks) < args.md_max:
                ff_blocks.append(
                    f"### {rid} · turn {turn}  (SET turn false fire, "
                    f"proj={rec['cross_proj']:+.3f}, peak={rec['peak_proj']:+.3f})\n"
                    f"- window : ...{rec.get('window_text', '')!r}\n"
                    f"- after  : {rec.get('after_text', '')!r}...\n")

    if not records:
        raise SystemExit("no usable turns (traces + transcripts) found.")
    summ = summarize(records)
    lex = lexical_summary(records, set_bank, mre)
    summ["lexical"] = lex

    print(f"\n[lead] threshold={thr:g} ({mode}), k={k}; "
          f"{summ['n_submit_turns']} submit turns, {summ['n_set_turns']} set turns "
          f"({summ['n_undetectable']} turns shorter than k skipped)")
    if "frac_crossed_before_verb" in summ:
        print(f"   submit turns crossing BEFORE the verb : "
              f"{summ['frac_crossed_before_verb']:.0%}")
    if "lead_median" in summ:
        lo, hi = summ["lead_iqr"]
        print(f"   lead (tokens): median {summ['lead_median']:.0f}  "
              f"IQR [{lo:.0f}, {hi:.0f}]  max {summ['lead_max']}")
        print(f"   note: leads under ~k={k} are partly pooling smoothing; leads "
              "well beyond k are the interesting ones.")
    if "set_false_fire_rate" in summ:
        print(f"   set-turn false-fire rate               : "
              f"{summ['set_false_fire_rate']:.0%}  "
              f"(in {summ['n_rollouts_with_false_fires']} rollout(s); per-rollout "
              "rates in lead_times.json)")
    if "peak_proj_auroc_submit_vs_set" in summ:
        print(f"   peak-projection AUROC (submit vs set)  : "
              f"{summ['peak_proj_auroc_submit_vs_set']:.3f}")
    if "frac_crossed_before_marker" in lex:
        print(f"   [lexical] crossings BEFORE any stop-language : "
              f"{lex['frac_crossed_before_marker']:.0%}"
              + (f"  ({lex['n_no_marker_at_all']} turn(s) had no stop-language "
                 "at all)" if lex.get("n_no_marker_at_all") else ""))
        if "marker_lead_median" in lex:
            lo, hi = lex["marker_lead_iqr"]
            print(f"   [lexical] crossing-to-first-marker (tokens): median "
                  f"{lex['marker_lead_median']:+.0f}  IQR [{lo:+.0f}, {hi:+.0f}]  "
                  "(positive = detector first)")
        rc = lex.get("window_marker_rate_cross")
        rp = lex.get("window_marker_rate_precross")
        rs = lex.get("window_marker_rate_set_matched")
        parts = [f"crossing {rc:.0%}" if rc is not None else None,
                 f"k-earlier {rp:.0%}" if rp is not None else None,
                 f"SET-matched {rs:.0%} (n={lex.get('n_set_matched_windows', 0)})"
                 if rs is not None else None]
        print("   [lexical] stop-language rate in windows       : "
              + "  vs  ".join(p for p in parts if p))
        print("   [lexical] caveat: lexicon is heuristic; similar rates rule out "
              "shallow stop-vocab matching,\n"
              "             not the detector reading semantic content of the "
              "deliberation.")
    print("   -> now READ lead_times.md: if the {{CROSS}} window already "
          "verbalises stopping, the lead is lexical, not latent; if SET-turn "
          "false fires show the model CONSIDERING stopping, they are "
          "near-decisions, not noise.")

    with open(os.path.join(run_dir, "lead_times.json"), "w") as f:
        json.dump(dict(run_dir=run_dir, threshold=thr, mode=mode, k=k,
                       summary=summ, records=records), f, indent=2)
    with open(os.path.join(run_dir, "lead_times.md"), "w") as f:
        f.write(f"# Lead times ({os.path.basename(run_dir)})\n"
                f"`{{{{CROSS}}}}` = first threshold crossing (thr={thr:g}, k={k}); "
                f"`<<VERB>>` = action verb.\nText between the two markers is the "
                f"lead span -- what the model said after latent commitment.\n\n"
                + "\n".join(md_blocks)
                + ("\n\n## SET-turn false fires\n"
                   "Windows that crossed the threshold on turns that did NOT "
                   "submit. If these read as the model CONSIDERING stopping "
                   "(then continuing), the detector tracks stop-contemplation "
                   "and a higher threshold separates contemplation from "
                   "commitment.\n\n" + "\n".join(ff_blocks) if ff_blocks else ""))
    print(f"[lead] wrote lead_times.json and lead_times.md under {run_dir}")
    plot_leads(records, summ, os.path.join(run_dir, "lead_times.png"), k, thr)
    plot_lexical(records, lex, os.path.join(run_dir, "lead_lexical.png"), k)


if __name__ == "__main__":
    main()