"""One rollout = one case played to SUBMIT (or the turn budget).

Each turn the model is shown an IMAGE (clean phantom at case start, dose wash
after each SET) plus the DVH table, and emits one action. Activations are
captured per turn (last-k decision run-up). The live env is the source of truth;
everything is also logged to disk for replay/analysis.
"""
from __future__ import annotations
import os
import numpy as np

from .env import PlanningEnv
from .dsl import parse_action
from .prompts import system_prompt_for, render_case, render_feedback_for
from . import io_utils as io


def build_env(cfg):
    return PlanningEnv(nx=cfg.nx, ny=cfg.ny, voxel_mm=cfg.voxel_mm,
                       radius_mm=cfg.radius_mm, Rx=cfg.Rx, opt_mode=cfg.opt_mode,
                       max_beams=cfg.max_beams, n_oar=(cfg.n_oar or 4),
                       march_mm=cfg.march_mm, spacing_mm=cfg.spacing_mm,
                       energy_step_mev=cfg.energy_step_mev,
                       lateral_sigma_mm=cfg.lateral_sigma_mm,
                       d98_floor_pct=cfg.d98_floor_pct, d2_ceil_pct=cfg.d2_ceil_pct,
                       constraint_tighten_frac=cfg.constraint_tighten_frac,
                       baseline_beams=cfg.baseline_beams, inner_iters=cfg.inner_iters)


def run_rollout(cfg, agent, idx, seed):
    d = io.rollout_dir(cfg.run_dir(), idx)
    env = build_env(cfg)
    env.reset(seed=seed)
    io.save_case(d, env, seed)

    imgdir = os.path.join(d, "images")
    case_img = os.path.join(imgdir, "phantom.png")        # written by save_case
    messages = [{"role": "system", "content": system_prompt_for(cfg)},
                {"role": "user", "content": render_case(env, cfg.max_turns),
                 "image_path": case_img}]

    submitted = False
    for turn in range(1, cfg.max_turns + 1):
        cap = os.path.join(d, "activations", f"turn_{turn:02d}.npz")
        text, meta = agent.act(messages, capture_path=cap if agent.is_model else None)
        action = parse_action(text)
        messages.append({"role": "assistant", "content": text})
        rec = dict(turn=turn, action=action.kind, angles=action.angles,
                   weights=action.weights, error=action.error,
                   response=text, **{k: v for k, v in meta.items() if k != "source"})

        if action.kind == "submit":
            env.submit()
            rec.update(passes=bool(env.plan_passes()))
            io.append_transcript(d, rec)
            io.save_submission(d, env.snapshot(), dict(turn=turn, seed=int(seed)))
            submitted = True
            break

        if action.kind == "set":
            fb, note = env.set_plan(action.angles, action.weights)
            io.save_dose(d, turn, env.dose)
            img = os.path.join(imgdir, f"dose_turn_{turn:02d}.png")
            env.render_dose(img, turn=turn)
            rec.update(passes=bool(env.plan_passes()), note=note)
            io.append_transcript(d, rec)
            messages.append({"role": "user",
                             "content": render_feedback_for(env, fb, env.angles,
                                                            env.global_w, turn,
                                                            cfg.max_turns, note=note),
                             "image_path": img})
        else:  # parse error -> nudge and let the model retry
            io.append_transcript(d, rec)
            messages.append({"role": "user",
                             "content": f"Could not parse an action ({action.error}). "
                                        "Reply with [SET a=w, ...] or [SUBMIT].",
                             "image_path": case_img})

    if not submitted:
        io.save_submission(d, env.snapshot(),
                           dict(turn=cfg.max_turns, seed=int(seed), forced=True))
    return env.snapshot()
