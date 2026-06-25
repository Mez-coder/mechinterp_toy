"""Per-rollout persistence. Disk is the record/recovery path, NOT the control
path -- the live PlanningEnv in RAM is the source of truth during a rollout."""
from __future__ import annotations
import os, json
import numpy as np


def rollout_dir(run_dir, idx):
    d = os.path.join(run_dir, f"rollout_{idx:04d}")
    os.makedirs(os.path.join(d, "activations"), exist_ok=True)
    return d


def save_case(d, env, seed):
    masks = {name: m for name, m in env.structures.items()}
    np.savez_compressed(os.path.join(d, "case.npz"), seed=seed,
                        nx=env.geom.nx, ny=env.geom.ny, voxel_mm=env.geom.voxel_mm,
                        **masks)
    with open(os.path.join(d, "case.json"), "w") as f:
        json.dump(dict(seed=int(seed), oar_metric=env.oar_metric,
                       ctv_floor={k: float(v) for k, v in env.ctv_floor.items()},
                       baseline_oar={k: float(v) for k, v in env.baseline['oar_val'].items()}),
                  f, indent=2)
    np.savez_compressed(os.path.join(d, "baseline.npz"),
                        dose=env.baseline['dose'], weights=env.baseline['weights'])
    write_objectives(d, env)
    open(os.path.join(d, "transcript.jsonl"), "w").close()   # fresh log per (re)run


def write_objectives(d, env):
    with open(os.path.join(d, "objectives.json"), "w") as f:
        json.dump(env.get_objectives(), f, indent=2)


def save_dose(d, turn, dose):
    np.savez_compressed(os.path.join(d, "activations", f"dose_turn_{turn:02d}.npz"), dose=dose)


def append_transcript(d, record):
    with open(os.path.join(d, "transcript.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")


def save_submission(d, snapshot, meta):
    np.savez_compressed(os.path.join(d, "submission.npz"), dose=snapshot['dose'])
    out = dict(meta)
    out['feedback'] = snapshot['feedback']
    out['oar_val'] = {k: float(v) for k, v in snapshot['oar_val'].items()}
    out['coverage_ok'] = bool(snapshot['coverage_ok'])
    with open(os.path.join(d, "submission.json"), "w") as f:
        json.dump(out, f, indent=2)
