"""The rollout loop: one PlanningEnv per rollout (live in RAM = source of truth),
the 2-tool dispatch, and per-turn logging/activation capture.

Tools collapsed to two:
  set_weights(dict)  -> write weights, optimise once, return feedback
  submit             -> snapshot + end
"""
from __future__ import annotations
import os
from protoplan.env import PlanningEnv
from protoplan.physics import Geometry2D
from .dsl import parse_action, render_feedback, resolve_handle
from .prompts import SYSTEM_PROMPT, render_case
from . import io_utils as io


def build_env(cfg, geom):
    return PlanningEnv(Rx=cfg.Rx, geom=geom, cov_tol_pct=cfg.cov_tol_pct,
                       constraint_sigma_frac=cfg.constraint_sigma_frac,
                       constraint_tighten_frac=cfg.constraint_tighten_frac,
                       n_oar=cfg.n_oar, oar_overlap_bias=cfg.oar_overlap_bias)


def run_rollout(cfg, geom, idx, agent):
    seed = cfg.seed_start + idx
    env = build_env(cfg, geom)
    env.reset(seed=seed)
    d = io.rollout_dir(cfg.run_dir(), idx)
    io.save_case(d, env, seed)

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": render_case(env, cfg.max_turns)}]

    submitted = forced = False
    for turn in range(1, cfg.max_turns + 1):
        cap = os.path.join(d, "activations", f"turn_{turn:02d}.npz") if cfg.capture else None
        text, meta = agent.act(messages, capture_path=cap if agent.is_model else None)
        action = parse_action(text)
        messages.append({"role": "assistant", "content": text})

        rec = dict(turn=turn, action=action.kind, weights=action.weights,
                   error=action.error, response=text, meta=meta)

        if action.kind == "submit":
            snap = env.submit()['plan']
            io.save_submission(d, snap, dict(submit_turn=turn, forced=False,
                                             n_turns=turn))
            io.append_transcript(d, rec)
            submitted = True
            break

        if action.kind == "set":
            for handle, w in action.weights.items():
                struct, metric = resolve_handle(handle)
                if struct in env.struct_idx:
                    env.set_weight(struct, w, metric=metric)
            fb = env.optimise()
            io.write_objectives(d, env)
            io.save_dose(d, turn, env.dose)
            fb_text = render_feedback(fb, turn=turn, max_turns=cfg.max_turns)
            rec["coverage_ok"] = env._coverage_ok()
            messages.append({"role": "user", "content": fb_text})
        else:  # parse error -> tell the agent, let it retry (costs a turn)
            msg = f"Could not parse an action ({action.error}).\n" + \
                  render_feedback(env.get_feedback(), turn=turn, max_turns=cfg.max_turns)
            messages.append({"role": "user", "content": msg})

        io.append_transcript(d, rec)

    if not submitted:                              # budget exhausted -> forced submit
        snap = env.submit()['plan']
        io.save_submission(d, snap, dict(submit_turn=cfg.max_turns, forced=True,
                                         n_turns=cfg.max_turns))
        io.append_transcript(d, dict(turn=cfg.max_turns, action="forced_submit",
                                     weights={}, error="", response="", meta={}))
        forced = True

    return dict(idx=idx, seed=seed, submitted=submitted, forced=forced, dir=d)
