"""Counterfactual branch in the proton env using the SOURCE (parabola) vector.

This is the proton analogue of the composite transfer test: take a finished
rollout, roll the conversation back to the turn where the model chose SUBMIT,
DROP that SUBMIT, and re-generate from there under the SOURCE-derived steering
vector. alpha<0 pushes toward SET ("keep optimising"), alpha>0 toward SUBMIT
("stop"). The vector is NEVER rebuilt here -- it is loaded from the source
directions.npz exactly as the 1D/story studies do.
"""
from __future__ import annotations
import os, json
import numpy as np

from .dsl import parse_action
from .prompts import system_prompt_for, render_case, render_feedback_for
from .steering import build_steering_vector, steering_active
from .rollout import build_env
from . import io_utils as io


def _load_transcript(rollout_dir):
    recs = []
    with open(os.path.join(rollout_dir, "transcript.jsonl")) as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def _rebuild_to(env, recs, branch_turn, cfg, imgdir):
    """Replay SET actions [1..branch_turn-1] to reconstruct env + messages exactly
    as they were just before `branch_turn` (images reused from the source run)."""
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


def replay_branch(cfg, agent, rollout_dir, layer=None, alpha=-1.0,
                  directions_path=None, source_run_dir=None, frac=None,
                  layers_attr=None, out_dir=None):
    """Branch at the SUBMIT turn under the source vector. Returns the snapshot."""
    source_run_dir = source_run_dir or cfg.source_run_dir
    directions_path = directions_path or cfg.directions_path or \
        os.path.join(source_run_dir, "directions.npz")
    frac = cfg.steer_frac if frac is None else frac
    layers_attr = layers_attr or cfg.layers_attr
    if layer is None:
        layer = cfg.steer_layer
        if layer is None:
            z = np.load(directions_path, allow_pickle=True)
            layer = int(z["best_layer"])
    block_idx = layer - 1

    recs = _load_transcript(rollout_dir)
    submit_rec = next((r for r in recs if r["action"] == "submit"), None)
    if submit_rec is None:
        raise RuntimeError("no SUBMIT turn in this rollout; nothing to branch")
    branch_turn = submit_rec["turn"]

    seed = json.load(open(os.path.join(rollout_dir, "case.json")))["seed"]
    env = build_env(cfg); env.reset(seed=seed)
    imgdir = os.path.join(rollout_dir, "images")
    messages = _rebuild_to(env, recs, branch_turn, cfg, imgdir)

    steer_vec, info = build_steering_vector(layer, directions_path, source_run_dir, frac)
    print(f"[branch] layer {layer} |steer|={info['steer_norm']:.2f} "
          f"({info['frac']:.0%} of source token norm {info['mean_token_norm']:.2f}) alpha={alpha}")

    out_dir = out_dir or os.path.join(rollout_dir, f"branch_L{layer}_a{alpha:+.2f}")
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    open(os.path.join(out_dir, "transcript.jsonl"), "w").close()

    submitted = False
    with steering_active(agent.model, block_idx, steer_vec, alpha, layers_attr):
        for turn in range(branch_turn, cfg.max_turns + 1):
            text, _ = agent.act(messages, capture_path=None)   # no capture under steering
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
    io.save_submission(out_dir, snap, dict(branch_turn=int(branch_turn), layer=int(layer),
                                           alpha=float(alpha), forced=not submitted))
    return snap