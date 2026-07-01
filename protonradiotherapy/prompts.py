"""System prompt and case presentation for the 2D proton sandbox.

The model is a VISION model. At case start it sees the clean phantom (blue
target; each OAR in its own colour with a legend) -- this is where the OAR
colour key is established. After each SET it sees the dose wash (% of Rx) with
the target as a white contour and each OAR as a contour in its own colour.

Objective: target coverage is handled automatically (assumed), so the ONLY job
is to keep every OAR's hotspot under its limit and push it as LOW as possible by
choosing beam angles that avoid the organs and weighting the beams.
"""
from __future__ import annotations
from .dsl import render_feedback
import numpy as np

SYSTEM_PROMPT = """\
You are planning a 2D proton radiotherapy treatment. A circular water phantom \
contains one tumour (CTV, shown in blue) and one or more organs-at-risk (OARs), \
each drawn in its OWN colour with a label and legend in the first image.

Target coverage is handled for you automatically: whatever beams you pick, an \
internal optimiser makes the tumour receive the full uniform prescription dose. \
So you do NOT need to worry about covering the tumour. Your ONLY objective is the \
ORGANS-AT-RISK: each OAR has a maximum allowed hotspot dose, and you want every \
OAR hotspot as far BELOW its limit as you can get it. Doses are reported as % of \
the prescription (so 100 = full dose, and an OAR reading 4.0 means 4% of Rx).

You control the BEAMS. Each beam has:
  - an ANGLE: gantry direction in degrees, 0-359 (0 = from the +x side, 90 = from \
the +y side, increasing counter-clockwise); the beam enters from that direction, \
travels toward the centre, and stops inside the tumour (Bragg peak).
  - a WEIGHT (>= 0): how much this beam contributes relative to the others.
You may use up to {max_beams} beams. Because protons stop AT the tumour and \
deposit dose all along the way in, an OAR gets hit when a beam's path crosses it. \
So you lower an OAR's hotspot by steering beam angles AROUND that organ (and by \
down-weighting beams that must pass near it). There is no separate OAR knob -- \
only geometry (angles) and mixing (weights).

Each turn you are shown an IMAGE of the current dose and a table of OAR hotspot \
doses vs their limits. You then take exactly ONE action:
  [SET a1=w1, a2=w2, ...]   -> place beams at those angles/weights and re-optimise; \
this REPLACES the whole plan (any beam not listed is removed). You then see the \
new dose image and table.
  [SUBMIT]                  -> finalise the current plan.

For example, if you wanted to set angles at 0, 45 and 270 degrees with the same weight, you would set a plan like so: [SET 0=1, 45=1, 270=1]\

Even once every OAR is within its limit, keep going if you can push the hotspots \
lower -- submit only when you judge the plan can't reasonably be improved.\
"""


def system_prompt_for(cfg):
    return SYSTEM_PROMPT.format(max_beams=cfg.max_beams)




def _bearing_dist(env, name):
    iy, ix = np.nonzero(env.structures[name])
    wx, wy = env.den.world_of_voxel(ix.mean(), iy.mean())
    bearing = float(np.degrees(np.arctan2(wy, wx)) % 360.0)
    dist = float(np.hypot(wx, wy))
    rx = env.case_meta.get(name, {}).get('rx_mm')
    ry = env.case_meta.get(name, {}).get('ry_mm')
    size = (0.5 * (rx + ry)) if (rx and ry) else float(np.sqrt(len(ix)) * env.voxel_mm / 2)
    return bearing, dist, size
 
 
def render_case(env, max_turns, scale=100.0) -> str:
    """Text shown at case start: full OAR geometry + limits (no image)."""
    s = ["New proton plan. The tumour is at the centre. Organs-at-risk (avoid "
         "entering beams from their bearing):"]
    for n in env.oar_metric:
        b, d, sz = _bearing_dist(env, n)
        s.append(f"  {n} ({env.oar_color[n]}): bearing {b:.0f} deg, "
                 f"{d:.0f} mm from centre, ~{2*sz:.0f} mm across, "
                 f"hotspot limit {env.oar_limit[n]*scale:.2f} %Rx")
    s.append("Coverage of the tumour is automatic; your only objective is to keep "
             "every OAR hotspot below its limit and push it as low as possible.")
    s.append("No beams are placed yet. Choose your beams.")
    s.append("")
    s.append(render_feedback(env.get_feedback(), angles=[], weights=[],
                             turn=0, max_turns=max_turns,
                             passes=env.plan_passes(), scale=scale))
    return "\n".join(s)
 
 
def render_feedback_for(env, feedback, angles, weights, turn, max_turns,
                        note=None, scale=100.0):
    return render_feedback(feedback, angles=angles, weights=weights,
                           turn=turn, max_turns=max_turns, note=note,
                           passes=env.plan_passes(), scale=scale)
 