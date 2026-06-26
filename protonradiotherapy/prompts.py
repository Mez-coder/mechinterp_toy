"""System prompt and case presentation for the 2D proton sandbox.

The model is a VISION model: every turn it is shown an image and must reason
over it. At case start it sees the clean phantom (blue target, red OARs); after
each SET it sees the resulting dose wash (turbo colormap) with the target drawn
as a solid white contour and OARs dashed. Beam paths are implicit in the dose,
so no arrows are drawn.
"""
from __future__ import annotations
from .dsl import render_feedback

SYSTEM_PROMPT = """\
You are planning a 2D proton radiotherapy treatment. A circular water phantom \
contains one tumour (CTV, drawn in blue) that must be covered by the prescription \
dose, and one or more organs-at-risk (OARs, drawn in red) that each have a maximum \
allowed dose.

You control the BEAMS directly. Each beam has:
  - an ANGLE: the gantry direction in degrees, 0-359 (0 = from the +x side, 90 = \
from the +y side, increasing counter-clockwise); the beam enters the phantom from \
that direction and travels toward the centre, stopping inside the tumour (Bragg peak).
  - a WEIGHT (>= 0): how much this beam contributes relative to the others.
You may use up to {max_beams} beams. An internal optimiser then makes each beam \
deliver uniform dose to the tumour, and your weights set how the beams are mixed. \
Because protons stop AT the tumour, choosing angles that approach the tumour \
WITHOUT crossing an OAR is how you spare that organ -- there is no separate OAR \
knob, only geometry (angles) and mixing (weights).

Each turn you are shown an IMAGE of the current dose and a table of doses (value \
vs limit, OK / over / cold). You then take exactly ONE action:
  [SET a1=w1, a2=w2, ...]   -> place beams at those angles with those weights and \
re-optimise; this REPLACES the whole plan (any beam not listed is removed). You \
then see the new dose image and table.
  [SUBMIT]                  -> finalise the current plan.

Goal: keep the tumour covered (D98% above its floor, D2% below its ceiling) while \
pushing every OAR dose as far below its limit as you can by steering beams around \
the organs. Submit once you are satisfied.\
"""


def system_prompt_for(cfg):
    return SYSTEM_PROMPT.format(max_beams=cfg.max_beams)


def render_case(env, max_turns) -> str:
    """Text shown alongside the clean phantom image at case start."""
    oar_lines = []
    for n, m in env.oar_metric.items():
        oar_lines.append(f"  {n}: limit {m} <= {env.oar_limit[n]:.3f} (dose/Rx)")
    s = ["New proton plan. The image shows the phantom: tumour (blue), OARs (red).",
         "Structures and OAR limits:"]
    s.extend(oar_lines)
    s.append(f"Tumour coverage target: D98% >= {env.d98_acc:.3f}, D2% <= {env.d2_acc:.3f} (dose/Rx).")
    s.append("No beams are placed yet. Choose your beams.")
    s.append("")
    s.append(render_feedback(env.get_feedback(), angles=[], weights=[],
                             turn=0, max_turns=max_turns,
                             passes=env.plan_passes()))
    return "\n".join(s)


def render_feedback_for(env, feedback, angles, weights, turn, max_turns, note=None):
    return render_feedback(feedback, angles=angles, weights=weights,
                           turn=turn, max_turns=max_turns, note=note,
                           passes=env.plan_passes())
