"""Per-rollout persistence. Disk is the record path; the live CouplingEnv in RAM
is the source of truth during a rollout."""
from __future__ import annotations
import os, json


def rollout_dir(run_dir, idx):
    d = os.path.join(run_dir, f"rollout_{idx:04d}")
    os.makedirs(os.path.join(d, "activations"), exist_ok=True)
    return d


def save_case(d, env, seed):
    with open(os.path.join(d, "case.json"), "w") as f:
        json.dump(dict(seed=int(seed), n_obj=env.n_obj, beta=env.beta,
                       m0=env.m0_case.tolist(), G=env.G_case.tolist(),
                       C=env.C_case.tolist()), f, indent=2)
    open(os.path.join(d, "transcript.jsonl"), "w").close()   # fresh log per (re)run


def append_transcript(d, record):
    with open(os.path.join(d, "transcript.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")


def save_submission(d, snapshot, meta):
    out = dict(meta)
    out.update(weights=snapshot['weights'].tolist(),
               margins=snapshot['margins'].tolist(),
               all_pass=snapshot['all_pass'],
               total_margin=snapshot['total_margin'],
               total_weight=snapshot['total_weight'])
    with open(os.path.join(d, "submission.json"), "w") as f:
        json.dump(out, f, indent=2)
