"""Quick-look: optimality-gap (loss) curves, one figure per case.
Reconstructs per-turn priority-margin gap offline from logged weight_vecs,
since the gap is only stored at submit, not per turn."""
import os, json, glob
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# adjust this import to your package layout:
from csat.coupling_env import CouplingEnv


def _env_from_case(case):
    """Rebuild the exact landscape from a saved case.json."""
    env = CouplingEnv(n_obj=case["n_obj"], beta=case["beta"])
    env.m0_case = np.asarray(case["m0"], float)
    env.G_case  = np.asarray(case["G"],  float)
    env.C_case  = np.asarray(case["C"],  float)
    env.beta    = case["beta"]
    env.priority = case["priority"]
    return env


def _load_rollout(d):
    """Return (case_id, rollout_idx, turns, gaps) for one rollout dir, or None."""
    cpath, spath, tpath = (os.path.join(d, f) for f in
                           ("case.json", "submission.json", "transcript.jsonl"))
    if not (os.path.exists(cpath) and os.path.exists(spath) and os.path.exists(tpath)):
        return None
    with open(cpath) as f: case = json.load(f)
    with open(spath) as f: sub  = json.load(f)
    if not sub.get("optimum_feasible"):            # no feasible optimum -> no gap to plot
        return None
    opt_mp = sub["optimum_margin_priority"]
    case_id = sub.get("case_id")
    rollout_idx = int(os.path.basename(d).split("_")[-1])

    env = _env_from_case(case)
    k = env.priority
    turns, gaps, passing = [], [], []
    with open(tpath) as f:
        for line in f:
            rec = json.loads(line)
            wv = rec.get("weight_vec")
            if wv is None:
                continue
            m = env.margins(np.asarray(wv, float))         # all objectives
            turns.append(rec["turn"])
            gaps.append(opt_mp - float(m[k]))              # loss: optimum - current priority
            passing.append(bool(np.all(m >= 0)))          # feasible at this turn?
    if not turns:
        return None
    return case_id, rollout_idx, np.array(turns), np.array(gaps), np.array(passing)


def plot_loss_curves(run_dir, clip_neg=True):
    by_case = defaultdict(list)
    worst_neg = 0.0
    for d in sorted(glob.glob(os.path.join(run_dir, "rollout_*"))):
        if not os.path.isdir(d):
            continue
        r = _load_rollout(d)
        if r is None:
            continue
        case_id, idx, turns, gaps, passing = r          # <- now 5 values
        worst_neg = min(worst_neg, gaps.min())
        if clip_neg:
            gaps = np.clip(gaps, 0.0, None)
        by_case[case_id].append((idx, turns, gaps, passing))

    if worst_neg < -0.01:
        print(f"[warn] gap went to {worst_neg:.3f} below zero on some turn — "
              f"likely MC-optimum noise; treat near-zero gaps as 'at optimum'.")

    for case_id in sorted(by_case, key=lambda c: (c is None, c)):
        rollouts = sorted(by_case[case_id], key=lambda t: t[0])
        plt.figure(figsize=(7, 4.5))
        for idx, turns, gaps, passing in rollouts:
            line, = plt.plot(turns, gaps, lw=1.3, label=f"rollout_{idx:04d}")  # connecting line
            c = line.get_color()
            pm = passing                                  # boolean mask
            plt.scatter(turns[pm],  gaps[pm],  marker="o", s=22, color=c, zorder=3)
            plt.scatter(turns[~pm], gaps[~pm], marker="x", s=34, color=c, zorder=3, linewidths=1.5)
        plt.axhline(0, color="k", lw=0.6, ls="--", alpha=0.5)
        plt.xlabel("turn"); plt.ylabel("optimality gap  (optimum − priority margin)")
        plt.title(f"case {case_id}  —  {len(rollouts)} rollouts")
        plt.legend(fontsize=8, ncol=2)
        plt.gca().invert_yaxis()        # gap shrinks downward = 'descending to optimum'
        plt.tight_layout()
        plt.savefig(f"case number {case_id}")


if __name__ == "__main__":
    plot_loss_curves("runs/csat")