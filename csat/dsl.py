"""Forgiving parser for the 2-action DSL + feedback rendering.

The model may reason freely, but its reply must contain ONE action:
    SET w1 w2 w3      positional weights for O1..On (each 0..1), then re-evaluate
    SUBMIT            finalise the current plan

Parsing rules (lenient on surrounding prose, strict on the action itself):
  - the numbers must come IMMEDIATELY after SET (only spaces/commas/colons/'='
    between), so the verb "set" inside reasoning is NOT mistaken for an action.
  - SET requires exactly n_obj numbers, positional for O1..On; wrong count or any
    negative weight is a parse_error (the loop re-asks with the reason).
  - SET takes precedence over SUBMIT.
Weights are clipped to [0,1] by the env.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import re

_SET = re.compile(r'\bSET\b', re.IGNORECASE)
_SUBMIT = re.compile(r'\bSUBMIT\b', re.IGNORECASE)
# a contiguous run of numbers that begins IMMEDIATELY after SET (only spaces,
# commas, colons or '=' may sit between SET and the first number). This rejects
# prose like "...set weights for O1, O2, O3..." because 'weights' is not a number.
_SET_RUN = re.compile(r'[\s:=,]*(-?\d+(?:\.\d+)?(?:[ \t,]+-?\d+(?:\.\d+)?)*)')
_NUM = re.compile(r'-?\d+(?:\.\d+)?')


@dataclass
class Action:
    kind: str                                    # 'set' | 'submit' | 'parse_error'
    weights: dict = field(default_factory=dict)  # {obj_index (0-based): value}
    error: str = ""
    raw: str = ""


def parse_action(text, n_obj):
    raw = text or ""
    example = "SET " + " ".join("0.3" for _ in range(n_obj))
    for m in _SET.finditer(raw):                 # try each 'SET'; skip prose ones
        run = _SET_RUN.match(raw[m.end():])
        if not run:
            continue                             # 'SET' not immediately followed by a number
        nums = _NUM.findall(run.group(1))
        if len(nums) != n_obj:
            return Action('parse_error', raw=raw,
                          error=f"SET must be immediately followed by exactly {n_obj} "
                                f"numbers, e.g. '{example}'; got {len(nums)}")
        vals = [float(x) for x in nums]
        bad = {i + 1: v for i, v in enumerate(vals) if v < 0}
        if bad:
            return Action('parse_error', raw=raw,
                          error=f"negative weights not allowed: {bad}")
        return Action('set', weights={i: vals[i] for i in range(n_obj)}, raw=raw)
    # SUBMIT counts only when it stands alone on a line or is the final token,
    # so prose like "I should not submit yet" does NOT end the episode.
    for ln in raw.splitlines():
        if re.fullmatch(r'[\W_]*submit[\W_]*', ln.strip(), re.IGNORECASE):
            return Action('submit', raw=raw)
    toks = raw.split()
    if toks and re.fullmatch(r'[\W_]*submit[\W_]*', toks[-1], re.IGNORECASE):
        return Action('submit', raw=raw)
    return Action('parse_error', raw=raw,
                  error=f"no action found; write '{example}' or 'SUBMIT'")


def render_feedback(rows, turn=None, max_turns=None, note=None, priority=None):
    """Signed-margin table the model sees after each evaluation."""
    n = len(rows)
    lines = []
    if turn is not None:
        lines.append(f"[turn {turn}/{max_turns}]")
    lines.append(f"{'obj':5s} {'weight':>7s} {'margin':>8s}  status")
    n_pass = 0
    for r in rows:
        n_pass += int(r['ok'])
        status = 'PASS' if r['ok'] else 'FAIL'
        star = ' *' if (priority is not None and r['obj'] == priority) else ''
        lines.append(f"O{r['obj'] + 1:<4d} {r['weight']:7.3f} {r['margin']:+8.3f}  {status}{star}")
    lines.append(f"objectives passing: {n_pass}/{n}  (margin>0 = under limit)")
    if priority is not None:
        lines.append(f"PRIORITY = O{priority + 1} (*): keep EVERY objective passing, "
                     f"then make O{priority + 1}'s margin as large as you can.")
    if note:
        lines.append(note)
    example = "SET " + " ".join("0.3" for _ in range(n))
    lines.append(f"Reply with one weight (0-1) per objective in order, e.g. '{example}', "
                 "or 'SUBMIT' to finalise.")
    return "\n".join(lines)