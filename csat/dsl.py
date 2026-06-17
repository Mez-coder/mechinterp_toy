"""Forgiving parser for the 2-action DSL + feedback rendering.

The model emits free text containing ONE action:
    SET O1=0.5, O2=0.3      set objective weights in [0,1], then re-evaluate
    SUBMIT                  finalise the current plan

Lenient on purpose (a 9B is messy): scan for Oi=value assignments and a SUBMIT
token. Assignments take precedence, so to submit the model must send SUBMIT with
no assignments. Negative weights are a parse error; values are clipped to [0,1]
by the env.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import re

_ASSIGN = re.compile(r'\bO(\d+)\s*[=:]\s*(-?\d+(?:\.\d+)?)', re.IGNORECASE)
_SUBMIT = re.compile(r'\bSUBMIT\b', re.IGNORECASE)


@dataclass
class Action:
    kind: str                                    # 'set' | 'submit' | 'parse_error'
    weights: dict = field(default_factory=dict)  # {obj_index (0-based): value}
    error: str = ""
    raw: str = ""


def parse_action(text):
    raw = text or ""
    assigns = {int(m.group(1)) - 1: float(m.group(2)) for m in _ASSIGN.finditer(raw)}
    if assigns:
        bad = {k + 1: v for k, v in assigns.items() if v < 0}
        if bad:
            return Action('parse_error', raw=raw,
                          error=f"negative weights not allowed: {bad}")
        return Action('set', weights=assigns, raw=raw)
    if _SUBMIT.search(raw):
        return Action('submit', raw=raw)
    return Action('parse_error', raw=raw,
                  error="no action found; use 'SET Oi=value, ...' or 'SUBMIT'")


def render_feedback(rows, turn=None, max_turns=None, note=None):
    """Signed-margin table the model sees after each evaluation."""
    lines = []
    if turn is not None:
        lines.append(f"[turn {turn}/{max_turns}]")
    lines.append(f"{'obj':5s} {'weight':>7s} {'margin':>8s}  status")
    n_pass = 0
    for r in rows:
        n_pass += int(r['ok'])
        status = 'PASS' if r['ok'] else 'FAIL'
        lines.append(f"O{r['obj'] + 1:<4d} {r['weight']:7.3f} {r['margin']:+8.3f}  {status}")
    lines.append(f"objectives passing: {n_pass}/{len(rows)}  "
                 "(margin>0 = under limit; more margin is better but costs the others)")
    if note:
        lines.append(note)
    lines.append("Action -> SET Oi=value[, Oj=value]  (weights 0..1)  |  SUBMIT")
    return "\n".join(lines)
