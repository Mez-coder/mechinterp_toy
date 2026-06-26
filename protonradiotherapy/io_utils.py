"""Per-rollout persistence. Disk is the record/recovery path; the live
PlanningEnv in RAM is the source of truth during a rollout."""
from __future__ import annotations
import os, json
import numpy as np


def rollout_dir(run_dir, idx):
    d = os.path.join(run_dir, f"rollout_{idx:04d}")
    os.makedirs(os.path.join(d, "activations"), exist_ok=True)
    os.makedirs(os.path.join(d, "images"), exist_ok=True)
    return d


def save_case(d, env, seed):
    masks = {name: m for name, m in env.structures.items()}
    np.savez_compressed(os.path.join(d, "case.npz"), seed=seed,
                        nx=env.nx, ny=env.ny, voxel_mm=env.voxel_mm, **masks)
    with open(os.path.join(d, "case.json"), "w") as f:
        json.dump(dict(seed=int(seed),
                       oar_metric=env.oar_metric,
                       oar_limit={k: float(v) for k, v in env.oar_limit.items()},
                       d98_floor=float(env.d98_acc), d2_ceil=float(env.d2_acc),
                       opt_mode=env.opt_mode, max_beams=env.max_beams), f, indent=2)
    np.savez_compressed(os.path.join(d, "baseline.npz"), dose=env.baseline_dose)
    # clean phantom image shown to the model at case start
    env.render_phantom(os.path.join(d, "images", "phantom.png"))
    open(os.path.join(d, "transcript.jsonl"), "w").close()


def save_dose(d, turn, dose):
    np.savez_compressed(os.path.join(d, "activations", f"dose_turn_{turn:02d}.npz"),
                        dose=dose)


def append_transcript(d, record):
    with open(os.path.join(d, "transcript.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")


def save_submission(d, snapshot, meta):
    np.savez_compressed(os.path.join(d, "submission.npz"), dose=snapshot["dose"])
    out = dict(meta)
    out["feedback"] = snapshot["feedback"]
    out["oar_val"] = {k: float(v) for k, v in snapshot["oar_val"].items()}
    out["coverage_ok"] = bool(snapshot["coverage_ok"])
    out["passes"] = bool(snapshot["passes"])
    out["angles"] = snapshot["angles"]
    out["global_w"] = snapshot["global_w"]
    with open(os.path.join(d, "submission.json"), "w") as f:
        json.dump(out, f, indent=2)
