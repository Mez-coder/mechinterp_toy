"""System prompt + case presentation for the coupling sandbox."""
from __future__ import annotations
from .dsl import render_feedback

SYSTEM_PROMPT = """\
You are tuning a plan with several competing objectives (O1, O2, ...). Each \
objective has a weight you control (a number from 0 to 1 (the sum of weights does not have to equal 1) and a MARGIN that the \
system computes for you:

  margin > 0  -> that objective is within its limit (PASS)
  margin < 0  -> that objective is over its limit (FAIL)

Every objective starts FAILING at weight 0. Raising an objective's weight \
increases its own margin, with diminishing returns. But raising a weight also \
REDUCES the margin of every other objective, and that cost grows quickly as the \
weight gets large.

For each plan, one objective is the PRIORITY (marked * and named below). Your \
goal has two parts, in order:
  (1) get EVERY objective passing (margin >= 0), then
  (2) make the PRIORITY objective's margin as large as you can WITHOUT failing
      any other objective.
There is a best achievable priority margin for each plan; aim for it. Submit once \
you believe the priority is as high as it can go with everything still passing.

You may think briefly first, but your reply must contain exactly ONE action:
  SET w1 w2 w3   -> set the weights of O1, O2, O3 in that order, each from 0 to 1
                    (e.g. SET 0.4 0.3 0.5); the system re-evaluates and shows new margins
  SUBMIT         -> finalise the current plan and finish

Give one number per objective, in order. As this is your first pass I suggest you immediately respond with 'SET 0.1 0.1 0.1'."""


def render_case(env, max_turns):
    s = [f"New plan with {env.n_obj} objectives (O1..O{env.n_obj}). "
         "All start failing at weight 0.",
         f"PRIORITY objective for this plan: O{env.priority + 1}.",
         "Starting state:", ""]
    s.append(render_feedback(env.feedback(), turn=0, max_turns=max_turns,
                             priority=env.priority))
    return "\n".join(s)