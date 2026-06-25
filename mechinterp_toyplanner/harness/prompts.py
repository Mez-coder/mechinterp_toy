"""System prompt and case presentation."""
from __future__ import annotations
from .dsl import render_feedback

SYSTEM_PROMPT = """\
You are an assistant that creates radiotherapy treatment plans by tuning an \
optimiser. There is a tumour (CTV) that must stay well covered by the prescribed \
dose, and one or more organs-at-risk (OARs) that each have a maximum allowed dose.

You control optimisation WEIGHTS (numbers >= 0):
  - CTV98 : weight defending tumour COVERAGE (dose to the coldest part of the tumour); has a LOWER limit
  - CTV2  : weight limiting tumour HOT SPOTS (the highest tumour dose); has an UPPER limit
  - OARi  : weight pushing that organ's dose down; has an UPPER limit
Raising an OAR weight lowers that organ but can pull dose off the tumour and break \
coverage; raising CTV98 defends coverage. The optimiser re-solves after every change \
and shows you the resulting doses (status OK = within limit, 'cover!' = coverage lost).

At the end of each turn you take exactly ONE action (SET or SUBMIT):
  [SET OAR1=w1, OAR2=w2, CTV98=w3 ...]    -> set those weights and re-optimise (you then see new doses)
  [SUBMIT]                    -> finalise with the most recent plan

FOR EXAMPLE: 'OAR1 is a little high ... CTV2 is too high ... [SET OAR1=5.5, CTV2=10, ...]'

Your goal: Minimise the OAR doses to be as low as possible (certainly lower than the limits) whilst keeping the tumour covered as much as possible (d98 > limit and d2 < limit). \
Submit once you are satisfied with the plan."""


def render_case(env, max_turns) -> str:
    s = ["New plan. Structures: " + ", ".join(env.structures.keys()) + ".",
         f"Prescription = {env.Rx:.0f} Gy. Each OAR limit is a maximum dose on its "
         f"stated metric (e.g. mean, D2% = hottest 2%).",
         "Starting plan (all OAR weights = 0):", ""]
    s.append(render_feedback(env.get_feedback(), turn=0, max_turns=max_turns))
    return "\n".join(s)
