"""Forgiving parser for the 2-tool action DSL, and feedback rendering.

The model emits free text that should contain ONE action:
    SET OAR1=5.0, OAR2=3.0      (tune OAR weights, then re-optimise)
    SUBMIT                       (finalise the current plan)

Parsing is deliberately lenient (a 4B will be messy): we scan for OARk=value
assignments and for a SUBMIT token. Assignments take precedence over SUBMIT, so
to submit the model must send SUBMIT with no assignments.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import re

_ASSIGN = re.compile(r'\b(OAR\d+|CTV98|CTV2|CTV)\s*[=:]\s*(-?\d+(?:\.\d+)?)', re.IGNORECASE)
_SUBMIT = re.compile(r'\bSUBMIT\b', re.IGNORECASE)


def handle_for(structure, metric):
    if structure == 'CTV':
        return 'CTV2' if metric == 'D2%' else 'CTV98'
    return structure


def resolve_handle(handle):
    h = handle.upper()
    if h == 'CTV2':  return 'CTV', 'D2%'
    if h == 'CTV98': return 'CTV', 'D98%'
    if h == 'CTV':   return 'CTV', None
    return h, None


@dataclass
class Action:
    kind: str                       # 'set' | 'submit' | 'parse_error'
    weights: dict = field(default_factory=dict)
    error: str = ""
    raw: str = ""


def parse_action(text: str) -> Action:
    raw = text or ""
    assigns = {m.group(1).upper(): float(m.group(2)) for m in _ASSIGN.finditer(raw)}
    if assigns:
        bad = {k: v for k, v in assigns.items() if v < 0}
        if bad:
            return Action('parse_error', raw=raw,
                          error=f"negative weights not allowed: {bad}")
        return Action('set', weights=assigns, raw=raw)
    if _SUBMIT.search(raw):
        return Action('submit', raw=raw)
    return Action('parse_error', raw=raw,
                  error="no action found; use 'SET OARi=value, ...' or 'SUBMIT'")


def render_feedback(rows, turn=None, max_turns=None, note=None) -> str:
    """Plain-text DVH feedback table the model sees after each optimise."""
    lines = []
    if turn is not None:
        lines.append(f"[turn {turn}/{max_turns}]")
    lines.append(f"{'handle':8s} {'metric':6s} {'value':>7s} {'limit':>7s} "
                 f"{'weight':>7s}  status")
    n_oar = n_met = 0
    for r in rows:
        is_oar = r['structure'].startswith('OAR')
        if is_oar:
            n_oar += 1; n_met += int(r['ok'])
        status = 'OK' if r['ok'] else ('--' if is_oar else 'cover!')
        wt = f"{r['weight']:.1f}"
        h = handle_for(r['structure'], r['metric'])
        lines.append(f"{h:8s} {r['metric']:6s} {r['value']:7.1f} "
                     f"{r['limit_gy']:7.1f} {wt:>7s}  {status}")
    lines.append(f"OAR limits met: {n_met}/{n_oar}")
    if note:
        lines.append(note)
    lines.append("Action -> SET CTV98=value / CTV2=value / OARi=value[, ...]  |  SUBMIT")
    return "\n".join(lines)
