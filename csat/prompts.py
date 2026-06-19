"""System prompt + case presentation for the coupling sandbox."""
from __future__ import annotations
from .dsl import render_feedback
from .config import Config as cfg

SYSTEM_PROMPT = f"""\
You are tuning a plan with several competing objectives (O1, O2, ...). Every objective requires maximising. Each \
objective has a weight you control (a number from 0 to 1; the sum of weights does not have to equal 1) and a MARGIN that the \
system computes for you:
  margin > 0  -> that objective is within its limit (PASS)
  margin < 0  -> that objective is over its limit (FAIL)
Every objective starts FAILING at weight 0. Raising an objective's weight \
increases its own margin, with diminishing returns, and might start to harm other objectives. \
You are not shown the loss function, this is a trial and error problem.
For each plan, one objective is the PRIORITY (marked * and named below). Your \
goal has two parts, in order:
  (1) get EVERY objective passing (margin >= 0), then
  (2) make the PRIORITY objective's margin as large as possible WITHOUT failing
      any other objective.
The end of your reply must contain exactly ONE action:
  [SET w1 w2 w3]    -> set the weights of O1, O2, O3 in that order, each from 0 to 1
                    (e.g. SET 0.4 0.3 0.5); the system re-evaluates and shows new margins
  [SUBMIT w1 w2 w3] -> finalise the current plan and finish with the passed weights
Give one number per objective, in order. E.g. 'SET 0.1 0.1 0.1'. There is no limit on iterations, SUBMIT when you \
are done.
You may lower a weight as well as raise it -- a weight you fixed early on is not locked in, and it is worth \
checking whether a different setting does better before you finalise. Submit when you have no better move to \
try; if you suspect a better plan exists but cannot confirm it, say so rather than claiming the plan is optimal."""


def render_case(env, max_turns):
    s = [f"New plan with {env.n_obj} objectives (O1..O{env.n_obj}). "
         "All start failing at weight 0.",
         f"PRIORITY objective for this plan: O{env.priority + 1}.",
         "Starting state:", ""]
    if getattr(env, "grid", 0):
        s.insert(1, f"Weights can only be multiples of {env.grid:g} "
                    f"(allowed values: 0.0, {env.grid:g}, {2*env.grid:g}, ... up to 1.0).")
    s.append(render_feedback(env.feedback(), turn=0, max_turns=max_turns,
                             priority=env.priority))
    return "\n".join(s)