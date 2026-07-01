"""Forgiving parser for the 2-tool angle/weight DSL, and feedback rendering.

The model emits free reasoning text that should end with ONE action:
    [SET 90=1.0, 270=0.8, 45=0.5]    -> beams at those gantry angles (deg) with
                                         those global weights; trace + optimise +
                                         re-evaluate (this REPLACES the whole plan)
    [SUBMIT]                          -> finalise the current plan

Parsing is lenient (a 9B is messy): we scan for `angle=weight` pairs after a SET,
and for a SUBMIT token. The LAST action by position wins (the model finishes with
its decision). A bare `[SET 90, 270]` is accepted with unit weights.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import re

_SET_VERB = re.compile(r'\bSET\b', re.IGNORECASE)
_PAIR = re.compile(r'(-?\d+(?:\.\d+)?)\s*(?:[=:]\s*(-?\d+(?:\.\d+)?))?')
_SUBMIT_BRACKET = re.compile(r'\[\s*SUBMIT\s*\]', re.IGNORECASE)
_SUBMIT_BARE = re.compile(r'^[\s\[\(>*\-]*SUBMIT[\s\]\)\.\!]*$', re.IGNORECASE | re.MULTILINE)


@dataclass
class Action:
    kind: str                       # 'set' | 'submit' | 'parse_error'
    angles: list = field(default_factory=list)
    weights: list = field(default_factory=list)
    error: str = ""
    raw: str = ""


def parse_action(text: str) -> Action:
    raw = text or ""
    cands = []
    for m in _SET_VERB.finditer(raw):
        seg = raw[m.end():]
        cut = len(seg)
        for ch in ('\n', ']'):
            i = seg.find(ch)
            if i != -1:
                cut = min(cut, i)
        seg = seg[:cut]
        angles, weights = [], []
        for pm in _PAIR.finditer(seg):
            a = pm.group(1)
            if a is None:
                continue
            w = pm.group(2)
            angles.append(float(a))
            weights.append(float(w) if w is not None else 1.0)
        if angles:
            cands.append((m.start(), 'set', (angles, weights)))
    for m in _SUBMIT_BRACKET.finditer(raw):
        cands.append((m.start(), 'submit', None))
    for m in _SUBMIT_BARE.finditer(raw):
        cands.append((m.start(), 'submit', None))

    if not cands:
        return Action('parse_error', raw=raw,
                      error="no action found; use '[SET angle=weight, ...]' or '[SUBMIT]'")
    cands.sort(key=lambda c: c[0])
    _, kind, payload = cands[-1]
    if kind == 'set':
        angles, weights = payload
        if any(w < 0 for w in weights):
            return Action('parse_error', raw=raw, error="negative weights not allowed")
        if any(not (0 <= a < 360.0001) for a in angles):
            return Action('parse_error', raw=raw,
                          error="angles must be gantry degrees in [0,360)")
        return Action('set', angles=angles, weights=weights, raw=raw)
    return Action('submit', raw=raw)


def render_feedback(rows, angles=None, weights=None, turn=None, max_turns=None,
                    note=None, passes=None, scale=100.0) -> str:
    """Plain-text OAR table the model sees after each SET. Doses are shown as
    % of Rx (scale=100). Coverage is assumed and not shown -- the only objective
    is to push every OAR hotspot as far below its limit as possible."""
    lines = []
    if turn is not None:
        lines.append(f"[turn {turn}/{max_turns}]")
    if angles is not None:
        beams = ", ".join(f"{a:.0f}deg(w={w:.2f})" for a, w in zip(angles, weights))
        lines.append(f"beams: {beams if beams else '(none)'}")
    lines.append("OAR hotspot doses (% of Rx) -- push these as LOW as you can:")
    lines.append(f"{'OAR':16s} {'hotspot':>8s} {'limit':>8s}  status")
    for r in rows:
        label = f"{r['structure']} ({r.get('color','?')})"
        status = 'OK' if r['ok'] else 'OVER LIMIT'
        lines.append(f"{label:16s} {r['value']*scale:8.2f} "
                     f"{r['limit']*scale:8.2f}  {status}")
    if passes is not None:
        lines.append(f"PLAN STATUS: {'all OARs within limits' if passes else 'an OAR is OVER its limit'}")
    if note:
        lines.append(note)
    lines.append("Action -> [SET angle=weight, ...]  |  [SUBMIT]")
    return "\n".join(lines)