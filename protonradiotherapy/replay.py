"""Counterfactual replay -- the new-env analogue of the composite transfer test.

Take a completed rollout, roll the conversation back to the turn where the model
chose SUBMIT, DROP that SUBMIT, and re-generate from that point with the steering
vector applied (negative alpha pushes toward "keep optimising", positive toward
"stop"). Everything before the branch point is identical to the original run, so
any change in behaviour is attributable to the injected vector.

This reuses `steering_active(model, block_idx, vec, alpha)` exactly as the 1D and
story studies did; only the rollback/branch bookkeeping is env-specific.
"""
from __future__ import annotations
import os, json, copy
import numpy as np

from .dsl import parse_action
from .prompts import render_feedback_for
from .steering import load_direction, steering_active
from .rollout import build_env
from . import io_utils as io


def _load_transcript(rollout_dir):
    recs = []
    with open(os.path.join(rollout_dir, "transcript.jsonl")) as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def _rebuild_to(env, recs, branch_turn, cfg, imgdir):
    """Replay SET actions [1 .. branch_turn-1] to reconstruct env + messages
    exactly as they were just before `branch_turn`."""
    from .prompts import system_prompt_for, render_case
    case_img = os.path.join(imgdir, "phantom.png")
    messages = [{"role": "system", "content": system_prompt_for(cfg)},
                {"role": "user", "content": render_case(env, cfg.max_turns),
                 "image_path": case_img}]
    for r in recs:
        if r["turn"] >= branch_turn:
            break
        messages.append({"role": "assistant", "content": r["response"]})
        if r["action"] == "set":
            fb, note = env.set_plan(r["angles"], r["weights"])
            img = os.path.join(imgdir, f"dose_turn_{r['turn']:02d}.png")
            messages.append({"role": "user",
                             "content": render_feedback_for(env, fb, env.angles,
                                                            env.global_w, r["turn"],
                                                            cfg.max_turns, note=note),
                             "image_path": img})
    return messages


def replay_branch(cfg, agent, rollout_dir, layer, alpha, directions_path,
                  out_dir=None):
    """Branch at the SUBMIT turn under steering. Returns the new submission snapshot."""
    recs = _load_transcript(rollout_dir)
    submit_rec = next((r for r in recs if r["action"] == "submit"), None)
    if submit_rec is None:
        raise RuntimeError("no SUBMIT turn in this rollout; nothing to branch")
    branch_turn = submit_rec["turn"]

    seed = json.load(open(os.path.join(rollout_dir, "case.json")))["seed"]
    env = build_env(cfg); env.reset(seed=seed)
    imgdir = os.path.join(rollout_dir, "images")
    messages = _rebuild_to(env, recs, branch_turn, cfg, imgdir)

    vec = load_direction(directions_path, layer)
    block_idx = layer - 1                      # hidden_states[j] <- block j-1

    out_dir = out_dir or os.path.join(rollout_dir, f"branch_L{layer}_a{alpha}")
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    open(os.path.join(out_dir, "transcript.jsonl"), "w").close()

    submitted = False
    with steering_active(agent.model, block_idx, vec, alpha):
        for turn in range(branch_turn, cfg.max_turns + 1):
            text, meta = agent.act(messages, capture_path=None)
            action = parse_action(text)
            messages.append({"role": "assistant", "content": text})
            rec = dict(turn=turn, action=action.kind, angles=action.angles,
                       weights=action.weights, steered=dict(layer=layer, alpha=alpha))
            if action.kind == "submit":
                env.submit(); rec.update(passes=bool(env.plan_passes()))
                io.append_transcript(out_dir, rec); submitted = True; break
            if action.kind == "set":
                fb, note = env.set_plan(action.angles, action.weights)
                img = os.path.join(out_dir, "images", f"dose_turn_{turn:02d}.png")
                env.render_dose(img, turn=turn)
                rec.update(passes=bool(env.plan_passes()), note=note)
                io.append_transcript(out_dir, rec)
                messages.append({"role": "user",
                                 "content": render_feedback_for(env, fb, env.angles,
                                                                env.global_w, turn,
                                                                cfg.max_turns, note=note),
                                 "image_path": img})
            else:
                io.append_transcript(out_dir, rec)
                messages.append({"role": "user", "content":
                                 "Reply with [SET a=w, ...] or [SUBMIT].",
                                 "image_path": os.path.join(imgdir, "phantom.png")})
    snap = env.snapshot()
    io.save_submission(out_dir, snap, dict(branch_turn=int(branch_turn),
                                           layer=int(layer), alpha=float(alpha),
                                           forced=not submitted))
    return snap
