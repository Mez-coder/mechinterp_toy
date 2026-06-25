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
Raising an objective's weight \
increases its own margin, with diminishing returns, and will start to harm other objectives. \
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
checking whether a different setting does better before you finalise. Submit when you are happy."""


PARABOLA_SYSTEM_PROMPT = """\
You are tuning a plan described by several coordinates (w1, w2, ...), each a number from 0 to 1 that you control, and the weights need not be equal. \
The system computes a single MARGIN for the whole plan:
  margin > 0  -> the plan is within its limit (PASS)
  margin < 0  -> the plan is over its limit (FAIL)
A higher margin is a better plan. There is one best setting of the coordinates that maximises the margin; you are \
not shown where it is or how the margin is computed -- this is a trial-and-error search.
Your goal has two parts, in order:
  (1) get the plan PASSING (margin >= 0), then
  (2) make the margin as LARGE as you can by adjusting the coordinates.
The end of your reply must contain exactly ONE action:
  [SET w1 w2 ...]    -> set every coordinate (each 0 to 1, in order); the system re-evaluates and shows the new margin
  [SUBMIT w1 w2 ...] -> finalise the plan with those coordinates
Submit when you have an optimal plan."""


SINE_SYSTEM_PROMPT = """\
You are tuning a plan described by coordinate x1 which takes a continuous value from 0 to 1 that you control. \
The system computes a single MARGIN for the whole plan:
  margin > 0  -> the plan is within its limit (PASS)
  margin < 0  -> the plan is over its limit (FAIL)
A higher margin is a better plan. There is one best setting of the coordinate that maximises the margin; you are \
not shown where it is or how the margin is computed -- this is a trial-and-error search for the global optimum.
Your goal has two parts, in order:
  (1) get the plan PASSING (margin >= 0), then
  (2) make the margin as LARGE as you can by adjusting the coordinate.
The end of your reply must contain exactly ONE action:
  [SET x1]    -> set the coordinate (between 0 to 1); the system re-evaluates and shows the new margin
  [SUBMIT x1] -> finalise the plan with that coordinate
Submit when you have an optimal plan."""


def render_case(env, max_turns):
    s = [f"New plan with {env.n_obj} objectives (O1..O{env.n_obj}). "
         "Every objective starts failing at weight 0; you begin from a "
         "mid-range plan (all weights 0.5), shown below.",
         f"PRIORITY objective for this plan: O{env.priority + 1}.",
         "Starting state:", ""]
    if getattr(env, "grid", 0):
        s.insert(1, f"Weights can only be multiples of {env.grid:g} "
                    f"(allowed values: 0.0, {env.grid:g}, {2*env.grid:g}, ... up to 1.0).")
    s.append(render_feedback(env.feedback(), turn=0, max_turns=max_turns,
                             priority=env.priority))
    return "\n".join(s)

def render_case_parabola(env, max_turns):
    s = [f"New plan with {env.n_obj} coordinates (w1..w{env.n_obj}). "
         "You begin from a mid-range plan (all coordinates 0.5), shown below.",
         "There is a single hidden margin to maximise; find the coordinates that make it as large as possible.",
         "Starting state:", ""]
    s.append(render_feedback(env.feedback(), turn=0, max_turns=max_turns, priority=None))
    return "\n".join(s)

def system_prompt_for(cfg):
    kind = getattr(cfg, "env_kind", "coupling")

    if kind == "parabola":
      return PARABOLA_SYSTEM_PROMPT 
    elif kind == "sine":
      return SINE_SYSTEM_PROMPT
    else:
      return SYSTEM_PROMPT


def render_case_for(env, cfg):
    kind = getattr(cfg, "env_kind", "coupling")
    if kind in ("parabola", "sine"):
        return render_case_parabola(env, cfg.max_turns)
    return render_case(env, cfg.max_turns)


def render_feedback_for(env, cfg, fb=None, *, turn=None, max_turns=None, note=None):
    """Env-aware per-turn feedback render.

    Suppresses the PRIORITY star/line for single-margin envs (parabola, sine)
    so the in-loop feedback matches their framing instead of silently falling
    back to coupling language. Mirrors the dispatch used by system_prompt_for /
    render_case_for so the rollout loop never reads env.priority directly.
    """
    fb = env.feedback() if fb is None else fb
    kind = getattr(cfg, "env_kind", "coupling")
    priority = None if kind in ("parabola", "sine") else env.priority
    return render_feedback(fb, turn=turn, max_turns=max_turns, note=note,
                           priority=priority)