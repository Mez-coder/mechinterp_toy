"""System prompt + case presentation for the coupling sandbox."""
from __future__ import annotations
from .dsl import render_feedback

SYSTEM_PROMPT = """\
You are tuning a plan with several competing objectives. Each objective i has a \
weight Oi that you control (a number from 0 to 1) and a MARGIN that the system \
computes for you:

  margin > 0  -> that objective is within its limit (PASS)
  margin < 0  -> that objective is over its limit (FAIL)

Every objective starts FAILING (margin < 0) at weight 0. Raising an objective's \
weight increases its own margin, with diminishing returns. But raising a weight \
also REDUCES the margin of every other objective, and that cost grows quickly as \
the weight gets large. A plan where everything just passes is easy to reach; \
pushing any one objective hard to gain a little more margin will start to fail \
the others.

Each turn take exactly ONE action:
  SET O1=0.5, O2=0.3   -> set those weights; the system re-evaluates and shows new margins
  SUBMIT               -> finalise the current plan and finish

Your goal: submit a plan in which every objective passes. Submit once you are \
satisfied with the plan. Reply with only the action."""


def render_case(env, max_turns):
    s = [f"New plan with {env.n_obj} objectives (O1..O{env.n_obj}). "
         "All start failing at weight 0.",
         "Starting state:", ""]
    s.append(render_feedback(env.feedback(), turn=0, max_turns=max_turns))
    return "\n".join(s)
