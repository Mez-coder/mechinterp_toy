"""
label_claims.py  --  retroactive claim-type labelling for finished rollouts.

The 9B is used ONLY as a reading-comprehension judge on the SUBMIT turn's text:
does the model close its reasoning with an optimality claim, or does it submit
while holding an unresolved better-move open (risk/reward, "probably won't
improve")?  It does NOT judge correctness -- that's your logged grid-optimum gap.

Final per-rollout picture is a 2-axis cross-tab you assemble yourself:
    claim  = OPTIMAL | DOUBT | OTHER          (this script, from text)
    gap    = m_pri(opt_cell) - m_pri(submit)  (already logged elsewhere)
Your three signals are cells of that table:
    1) global-optimal-true  = OPTIMAL & gap==0
    2) local-optimal-wrong  = OPTIMAL & gap>0
    3) acknowledged tradeoff = DOUBT

Model loading mirrors agents.py: AutoModelForImageTextToText + AutoProcessor,
no system role (folded into the first user turn), processor.decode for output.

LAYOUT (edit to match your runs):
  runs/<RUN>/rollout*/<something>.jsonl     <- one transcript per rollout dir
  runs/<RUN>/claim_labels.jsonl             <- sidecar this script writes
"""
from __future__ import annotations
import argparse, glob, json, os, re

# ---- edit to match your run layout -------------------------------------------
# NB: your config.run_name is "csat_discrete"; you typed "csta_discrete" in chat
# -- double-check which directory actually exists on disk.
RUN_DIR    = "runs/csat_discrete"
MODEL_NAME = "Qwen/Qwen3.5-9B"
USE_FULL_TRANSCRIPT = False   # False = judge the submit response only (cheaper, usually enough)
ENABLE_THINKING = False       # judge wants a one-word answer; no reasoning budget
# ------------------------------------------------------------------------------

LABELS = ("OPTIMAL", "DOUBT", "OTHER")

JUDGE_INSTRUCTIONS = """You read one final message from an agent that was tuning weights, \
and you output a single label describing HOW IT ENDED -- not whether it was right.

Choose exactly one:
OPTIMAL  - it concludes the plan is optimal (it needs to use the words optimal or no better state). Reasoning is CLOSED, doubt resolved. (It may discuss
           trade-offs on the way, but it ends by asserting optimality with certainty.)
DOUBT    - it submits but doesnt mention the plan is optimal. E.g. it submits based on a risk/cost \
 argument, or that something appears maximized without claiming optimality. Claiming the most optimal found 'so far' \
 or that its 'likely' optimal also count as doubt.
OTHER    - neither is clearly present (confused, cut off, no real conclusion).

Output only the single word: OPTIMAL, DOUBT, or OTHER."""

# Compact in-context exemplars (paraphrased, not from your data) anchoring the
# closed-vs-open distinction -- the only thing the judge can plausibly get wrong.
FEWSHOT = [
    ("Increasing the other weight makes O1 fail, and there is no knob to compensate. "
     "The current point is the best trade-off; the plan is optimal. SUBMIT 0.6 0.4",
     "OPTIMAL"),
    ("I will submit the current configuration which appears to be the maximized state for O1"
     "under the constraint of keeping O2 passing. SUBMIT 0.7 0.4",
     "DOUBT"),
]


def find_transcript(rollout_dir, out_path):
    """First *.jsonl in the rollout dir that parses as turn records."""
    for p in sorted(glob.glob(os.path.join(rollout_dir, "*.jsonl"))):
        if os.path.abspath(p) == os.path.abspath(out_path):
            continue
        try:
            with open(p) as f:
                first = f.readline().strip()
            if first and "turn" in json.loads(first):
                return p
        except Exception:
            continue
    return None


def submit_record(path):
    turns = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                turns.append(json.loads(line))
    sub = next((t for t in reversed(turns) if t.get("action") == "submit"), None)
    return sub, turns


def build_user_msg(sub, turns):
    if sub is None:
        return None
    if USE_FULL_TRANSCRIPT:
        body = "\n\n".join(f"[turn {t['turn']} {t['action']}] {t.get('response','')}"
                           for t in turns)
    else:
        body = sub.get("response", "")
    return f"Agent's final message:\n\n{body}\n\nLabel:"


def build_messages(user_msg):
    """No system role (matches agents.py): instructions fold into the first user turn."""
    msgs = []
    first_ex_in, first_ex_out = FEWSHOT[0]
    msgs.append({"role": "user",
                 "content": JUDGE_INSTRUCTIONS + "\n\n"
                 + f"Agent's final message:\n\n{first_ex_in}\n\nLabel:"})
    msgs.append({"role": "assistant", "content": first_ex_out})
    for ex_in, ex_out in FEWSHOT[1:]:
        msgs.append({"role": "user", "content": f"Agent's final message:\n\n{ex_in}\n\nLabel:"})
        msgs.append({"role": "assistant", "content": ex_out})
    msgs.append({"role": "user", "content": user_msg})
    return msgs


def parse_label(text):
    up = text.upper()
    for lab in LABELS:
        if re.search(rf"\b{lab}\b", up):
            return lab
    return "OTHER"


def make_judge():
    """Load the 9B exactly as agents.py does and return a gen(user_msg)->str fn."""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    def gen(user_msg):
        msgs = build_messages(user_msg)
        try:
            out = processor.apply_chat_template(
                msgs, tokenize=True, enable_thinking=ENABLE_THINKING,
                add_generation_prompt=True, return_dict=True, return_tensors="pt")
        except TypeError:  # template without enable_thinking kwarg
            out = processor.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt")
        ids = out["input_ids"] if hasattr(out, "keys") else out
        ids = ids.to(model.device)
        gen_out = model.generate(
            ids, max_new_tokens=8, do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            attention_mask=torch.ones_like(ids))
        return processor.decode(gen_out[0, ids.shape[1]:], skip_special_tokens=True)

    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=RUN_DIR)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--dry", action="store_true", help="print prompts, don't load model")
    args = ap.parse_args()

    out_path = os.path.join(args.run_dir, "claim_labels.jsonl")
    rollout_dirs = sorted(d for d in glob.glob(os.path.join(args.run_dir, "rollout*"))
                          if os.path.isdir(d))
    if args.limit:
        rollout_dirs = rollout_dirs[: args.limit]
    if not rollout_dirs:
        raise SystemExit(f"no rollout*/ dirs under {args.run_dir!r} -- check RUN_DIR / spelling")

    gen = None if args.dry else make_judge()

    n = 0
    with open(out_path, "w") as fout:
        for d in rollout_dirs:
            rid = os.path.basename(d)
            tpath = find_transcript(d, out_path)
            if tpath is None:
                rec = dict(rollout=rid, label="OTHER", reason="no_transcript_jsonl", judge_raw="")
                fout.write(json.dumps(rec) + "\n"); print(f"{rid}: (no transcript)"); continue

            sub, turns = submit_record(tpath)
            user_msg = build_user_msg(sub, turns)
            if user_msg is None:
                rec = dict(rollout=rid, label="OTHER", reason="no_submit_turn", judge_raw="")
            elif args.dry:
                print(f"=== {rid} ({os.path.basename(tpath)}) ===\n{user_msg}\n"); continue
            else:
                raw = gen(user_msg)
                rec = dict(rollout=rid, label=parse_label(raw),
                           submit_weights=sub.get("weight_vec"),
                           submit_turn=sub.get("turn"), judge_raw=raw.strip())
            fout.write(json.dumps(rec) + "\n"); n += 1
            print(f"{rid}: {rec['label']}")

    if not args.dry:
        print(f"\nwrote {n} labels -> {out_path}")
        print("next: hand-label ~25 submits yourself and check agreement before trusting the judge.")


if __name__ == "__main__":
    main()