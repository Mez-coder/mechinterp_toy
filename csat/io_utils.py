"""Per-rollout persistence. Disk is the record path; the live CouplingEnv in RAM
is the source of truth during a rollout."""
from __future__ import annotations
import os, json
import glob, re

def rollout_dir(run_dir, idx):
    d = os.path.join(run_dir, f"rollout_{idx:04d}")
    os.makedirs(os.path.join(d, "activations"), exist_ok=True)
    return d

def next_rollout_idx(run_dir):
    """One past the highest COMPLETED rollout (has submission.json), so reruns add
    new seeds instead of overwriting finished ones. A crashed dir with no
    submission.json is re-run rather than counted."""
    idxs = []
    for p in glob.glob(os.path.join(run_dir, "rollout_*")):
        m = re.search(r"rollout_(\d+)$", os.path.basename(p))
        if m and os.path.exists(os.path.join(p, "submission.json")):
            idxs.append(int(m.group(1)))
    return max(idxs) + 1 if idxs else 0

def save_case(d, env, seed):
    with open(os.path.join(d, "case.json"), "w") as f:
        json.dump(dict(seed=int(seed), n_obj=env.n_obj, beta=env.beta,
                       priority=int(env.priority),
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
               total_weight=snapshot['total_weight'],
               priority=snapshot['priority'],
               margin_priority=snapshot['margin_priority'])
    with open(os.path.join(d, "submission.json"), "w") as f:
        json.dump(out, f, indent=2)