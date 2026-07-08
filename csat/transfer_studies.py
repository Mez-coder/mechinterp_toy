"""transfer_studies.py -- one entry point for the steering studies. The heavy
lifting (vector build, hook, steered/unsteered rollouts, projection, cosine
compare) lives in steering.py; this module imports it and adds the transfer tests.

  --study composite  baseline rollout; branch at the final-SET (or submit) turn
                     and re-generate under steering (does it land a better plan?).
  --study story      OOD: continue half a story under +/-alpha; length modulation.
  --study steer      alpha sweep in an env (porting steering.py's 'steer' mode).
  --study project    no steering; project per-turn acts onto the source axis
                     (porting steering.py's 'project' mode).
  --study compare    per-layer cosine of this vector vs another directions file
                     (no model loaded).

  python -m csat.transfer_studies --study composite --env sine \
      --source-run-dir runs/csat --alphas -0.5 -1.0 --n-rollouts 12
  python -m csat.transfer_studies --study steer --env parabola \
      --source-run-dir runs/csat --alphas -1 -0.5 0 0.5 1 --n-rollouts 20
  python -m csat.transfer_studies --study project --env sine --source-run-dir runs/csat
  python -m csat.transfer_studies --study compare --vector-type explore --source-run-dir runs/csat

Layer convention is shared with steering.py: --layer j is a hidden-state index;
the hook is placed on decoder block j-1 (whose output == hidden_states[j]).
"""
from __future__ import annotations
import os
# reduce CUDA fragmentation OOMs on long multi-turn branches. Must be set before
# torch initialises CUDA (i.e. before ModelAgent loads the model), so it lives at
# import time. Harmless if already exported in the environment.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json, argparse, copy, contextlib, gc
import numpy as np

from .config import Config
from .agents import ModelAgent
from .rollout import build_env
from .dsl import parse_action, split_thinking
from .prompts import system_prompt_for, render_case_for, render_feedback_for
from .steering import (build_steering_vector, steering_active, load_direction,
                       run_steered, run_unsteered_capture, compare_directions,
                       TriggerController, apply_control)
from . import io_utils as io


def _free():
    """Best-effort GPU/host memory release between rollouts (no-op without torch)."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ===========================================================================
# Composite study (Test 1)
# ===========================================================================
def _margin_now(env):
    s = env.snapshot()
    return float(s["margin_priority"]), bool(s["all_pass"])


def _advance(cfg, agent, env, messages, start_turn, snapshot_each=False,
             stop_after_turn=None, tag=None, controller=None):
    """Run the SET/SUBMIT loop from `start_turn` on (env, messages) IN PLACE.
    Returns records, per-turn margins, optional per-turn-start snapshots, and the
    submit info. No activation capture. Any steering hook is installed by the
    caller via a `with steering_active(...)` around this call.

    If `controller` (a TriggerController) is given, it is reset before each turn's
    generation and its per-turn record (projection trace, n_steered, ...) is
    attached to that turn's record under 'trigger'."""
    records, margins, snaps = [], [], {}
    submitted, submit_turn, snap = False, None, None
    last = cfg.max_turns if stop_after_turn is None else min(cfg.max_turns, stop_after_turn)
    for turn in range(start_turn, last + 1):
        if tag:
            print(f"      {tag} turn {turn}/{last}", end="\r", flush=True)
        if snapshot_each:
            snaps[turn] = (copy.deepcopy(env), list(messages))
        if controller is not None:
            controller.reset_turn()
        text, meta = agent.act(messages, capture_path=None)
        tinfo = controller.pop_turn() if controller is not None else None
        _free()                                       # release the just-generated KV/graph
        answer, _ = split_thinking(text, cfg.enable_thinking)
        action = parse_action(answer, env.n_obj)
        messages.append({"role": "assistant", "content": answer})
        rec = dict(turn=turn, action=action.kind, weights=action.weights,
                   error=action.error, response=text)
        if tinfo is not None:
            rec["trigger"] = tinfo

        if action.kind == "submit":
            for i, w in action.weights.items():
                if 0 <= i < env.n_obj:
                    env.set_weight(i, w)
            m, ap = _margin_now(env)
            rec.update(all_pass=ap, weight_vec=env.w.tolist(), margin=m)
            margins.append((turn, m)); records.append(rec)
            snap = env.submit()["plan"]; submitted = True; submit_turn = turn
            break

        if action.kind == "set":
            for i, w in action.weights.items():
                if 0 <= i < env.n_obj:
                    env.set_weight(i, w)
            m, ap = _margin_now(env)
            rec.update(all_pass=ap, weight_vec=env.w.tolist(), margin=m)
            margins.append((turn, m)); records.append(rec)
            messages.append({"role": "user",
                             "content": render_feedback_for(env, cfg, env.feedback(),
                                                            turn=turn, max_turns=cfg.max_turns)})
        else:                                     # parse error: costs a turn
            m, ap = _margin_now(env)
            rec.update(all_pass=ap, weight_vec=env.w.tolist(), margin=m)
            margins.append((turn, m)); records.append(rec)
            messages.append({"role": "user",
                             "content": f"Could not parse an action ({action.error}).\n"
                             + render_feedback_for(env, cfg, turn=turn, max_turns=cfg.max_turns)})

    if not submitted:
        snap = env.submit()["plan"]
    return dict(records=records, margins=margins, snaps=snaps, submitted=submitted,
                submit_turn=submit_turn, snapshot=snap,
                n_turns=(submit_turn if submit_turn else last))


def run_composite(cfg, agent, steer_vec, layer, seed, alphas, out_run_dir, idx,
                  steer_ctx=steering_active, branch_extra=10, inject_at="final_set",
                  opt=None, case_id=None, rep=None, branch_repeats=1,
                  null_branches=None, gen_only=True):
    """One case: baseline (alpha 0) run ONCE; then for each alpha, branch at the
    baseline BRANCH TURN and re-generate that turn onward under steering.

    branch_repeats (B): how many INDEPENDENT steered continuations to fan out from
      the SAME baseline branch point, per alpha. The prefix SETs are already done,
      so this is cheap -- only the post-branch turns are generated B times. It
      isolates the vector's effect from baseline-sampling noise and shows whether
      injecting converges to one exploration or several.

    null_branches (default = branch_repeats): how many UNSTEERED (alpha=0)
      re-branches to run from the SAME branch point. With temperature > 0 the
      steered branch differs from the baseline in TWO ways -- the vector AND
      resampling -- so the honest null is 'resample without the vector'. These
      land in the summary as keys '+0.00_b01', '+0.00_b02', ... ('+0.00_b00'
      stays the untouched baseline). Compare steered submit rates / margins
      against the null branches, not against the baseline alone.

    inject_at: 'final_set' (last SET before the baseline submit; default) or
      'submit' (the submit turn itself)."""
    block_idx = layer - 1
    env = build_env(cfg); env.reset(seed=seed, wide=getattr(cfg, "wide_cases", True))
    if opt is None:                                   # optimum is a property of the
        opt = env.optimum(samples=getattr(cfg, "optimum_samples", 50000))  # landscape
    opt_m = opt.get("margin_priority") if opt.get("feasible") else None
    messages = [{"role": "system", "content": system_prompt_for(cfg)},
                {"role": "user", "content": render_case_for(env, cfg)}]
    base = _advance(cfg, agent, env, messages, 1, snapshot_each=True, tag='base')

    rdir = io.rollout_dir(out_run_dir, idx)
    io.save_case(rdir, env, seed)

    # choose the branch turn from the baseline trajectory
    T_sub = base["submit_turn"]
    branch_turn, branch_kind = T_sub, "submit"
    if inject_at == "final_set" and T_sub is not None:
        sets = [r["turn"] for r in base["records"]
                if r["action"] == "set" and r["turn"] < T_sub]
        if sets:
            branch_turn, branch_kind = max(sets), "final_set"
        # else: model submitted with no prior SET -> fall back to the submit turn

    B = max(1, int(branch_repeats))
    NB = B if null_branches is None else max(0, int(null_branches))
    realizations = {(0.0, 0): base}                   # (alpha, branch_rep) -> result
    if base["submitted"] and branch_turn is not None and branch_turn in base["snaps"]:
        env_b, msgs_b = base["snaps"][branch_turn]    # state at the START of the branch turn
        stop_at = min(cfg.max_turns, branch_turn + branch_extra)   # bound branch compute
        # (alpha, branch_rep) pairs to run: null re-branches first (alpha 0, b>=1
        # so b=0 stays the untouched baseline), then the steered fan-out.
        todo = [(0.0, 1 + b) for b in range(NB)]
        todo += [(a, b) for a in alphas if a != 0.0 for b in range(B)]
        for a, b in todo:
            e2, m2 = copy.deepcopy(env_b), list(msgs_b)
            with steer_ctx(agent.model, block_idx, steer_vec, a, gen_only=gen_only):
                res = _advance(cfg, agent, e2, m2, branch_turn, snapshot_each=False,
                               stop_after_turn=stop_at, tag=f"a{a:+.1f}b{b}")
            realizations[(a, b)] = res
            _free()
            kind = "null " if a == 0.0 else ""
            print(f"    {kind}alpha {a:+.1f} branch {b}: ran turns {branch_turn}.."
                  f"{res['n_turns']} (submitted={res['submitted']})")
    else:
        print(f"  [composite] rollout {idx:04d} baseline forced / no branch turn; "
              f"only alpha 0 saved.")

    prefix_recs = [r for r in base["records"]
                   if branch_turn is None or r["turn"] < branch_turn]
    prefix_marg = [(t, m) for (t, m) in base["margins"]
                   if branch_turn is None or t < branch_turn]

    trajectories, summary, rows = {}, {}, []
    for (a, b), res in realizations.items():
        is_base = (a == 0.0 and b == 0)
        recs = res["records"] if is_base else prefix_recs + res["records"]
        traj = res["margins"] if is_base else prefix_marg + res["margins"]
        key = f"{a:+.2f}_b{b:02d}"
        with open(os.path.join(rdir, f"transcript_a{key}.jsonl"), "w") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
        final_m = traj[-1][1] if traj else None
        # best margin: separates persistence (did it keep going) from competence
        # (did going longer find a better point). post-branch = only the turns
        # this branch actually generated; whole = including the shared prefix.
        post = [m for (t, m) in res["margins"] if m is not None]
        whole = [m for (t, m) in traj if m is not None]
        best_post = (float(max(post)) if post else None)
        best_whole = (float(max(whole)) if whole else None)
        st, submitted, nt = res["submit_turn"], res["submitted"], res["n_turns"]
        for rec in recs:                              # tidy per-turn rows (for plotting)
            mg = rec.get("margin")
            rows.append(dict(
                idx=idx, seed=seed, alpha=a, branch_rep=b, turn=rec.get("turn"),
                action=rec.get("action"), margin=mg, all_pass=rec.get("all_pass"),
                submitted=submitted, submit_turn=st, n_turns=nt,
                is_submit=bool(submitted and st is not None
                               and rec.get("turn") == st and rec.get("action") == "submit"),
                is_branch=(not is_base),          # False = untouched baseline
                branch_turn=branch_turn, branch_kind=branch_kind, optimum_margin=opt_m,
                gap=((opt_m - mg) if (opt_m is not None and mg is not None) else None)))
        trajectories[(a, b)] = traj
        summary[key] = dict(alpha=a, branch_rep=b, submit_turn=st, n_turns=nt,
                            submitted=submitted, final_margin=final_m,
                            best_margin=best_whole, best_margin_post=best_post,
                            best_gap=((opt_m - best_whole)
                                      if (opt_m is not None and best_whole is not None)
                                      else None),
                            optimum_margin=opt_m,
                            gap=((opt_m - final_m)
                                 if (opt_m is not None and final_m is not None) else None),
                            all_pass=bool(res["snapshot"]["all_pass"]))
    with open(os.path.join(rdir, "composite_summary.json"), "w") as f:
        json.dump(dict(idx=idx, seed=seed, case_id=case_id, rep=rep,
                       branch_turn=branch_turn, branch_kind=branch_kind,
                       baseline_submit_turn=T_sub, optimum_margin=opt_m,
                       branch_repeats=B, summary=summary), f, indent=2)
    return dict(idx=idx, seed=seed, trajectories=trajectories, summary=summary,
                rows=rows, T=branch_turn, branch_kind=branch_kind)


def _iter_summary(r):
    """Yield (alpha, branch_rep, entry) from a composite_summary dict, robust to
    the key format 'sA.BC_bDD'. b=0 at alpha 0 is the untouched baseline; b>=1 at
    alpha 0 are the resampled null branches."""
    for key, entry in r["summary"].items():
        a = entry.get("alpha")
        b = entry.get("branch_rep", 0)
        if a is None:                                  # fall back to parsing the key
            try:
                astr, bstr = key.rsplit("_b", 1)
                a, b = float(astr), int(bstr)
            except (ValueError, AttributeError):
                continue
        yield float(a), int(b), entry


def _fnum(v):
    return float(v) if v is not None else float("nan")


def summarize_branch_null(results, seed=0, n_boot=2000):
    """Compare steered branches against the alpha=0 NULL branches (resampled from
    the same branch point), which is the honest counterfactual under sampling.

    Per alpha, over cases that have >=1 null branch and >=1 branch at that alpha:
      submit_rate        -- fraction of branches that submitted within the window
      mean_final_margin  -- over branches
      mean_delta_margin  -- per-case (mean steered margin - mean null margin),
                            averaged over cases, with a paired case-level
                            bootstrap 95% CI
      delta_submit_rate  -- same construction on the submitted indicator
    The pairing is by CASE (same branch point), so baseline-sampling noise
    cancels; the bootstrap resamples cases, respecting the clustering."""
    per_case = {}                                      # idx -> {alpha: [entry dicts]}
    for r in results:
        d = per_case.setdefault(r["idx"], {})
        for a, b, e in _iter_summary(r):
            if a == 0.0 and b == 0:
                continue                               # untouched baseline: not a branch
            d.setdefault(a, []).append(dict(
                final=_fnum(e.get("final_margin")),
                best=_fnum(e.get("best_margin")),      # nan for pre-best_margin runs
                submitted=bool(e.get("submitted")),
                bad_submit=bool(e.get("submitted")) and (e.get("all_pass") is False)))
    alphas = sorted({a for d in per_case.values() for a in d if a != 0.0})
    rng = np.random.default_rng(seed)
    out = {}

    def _case_means(a, key):
        """per-case (mean steered <key> - mean null <key>) over paired cases."""
        dv = []
        for d in per_case.values():
            nulls, steered = d.get(0.0), d.get(a)
            if not nulls or not steered:
                continue
            nv = [e[key] for e in nulls if not np.isnan(e[key])]
            sv = [e[key] for e in steered if not np.isnan(e[key])]
            if nv and sv:
                dv.append(float(np.mean(sv)) - float(np.mean(nv)))
        return np.asarray(dv, float)

    def _case_submit_delta(a):
        dv = []
        for d in per_case.values():
            nulls, steered = d.get(0.0), d.get(a)
            if not nulls or not steered:
                continue
            dv.append(np.mean([e["submitted"] for e in steered])
                      - np.mean([e["submitted"] for e in nulls]))
        return np.asarray(dv, float)

    def _boot_ci(x):
        if len(x) < 2:
            return (None, None)
        idx = rng.integers(0, len(x), size=(n_boot, len(x)))
        means = np.nanmean(x[idx], axis=1)
        return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))

    def _flat_stats(flat):
        row = dict(n_branches=len(flat))
        if flat:
            row.update(
                submit_rate=float(np.mean([e["submitted"] for e in flat])),
                mean_final_margin=float(np.nanmean([e["final"] for e in flat])),
                bad_submit_rate=float(np.mean([e["bad_submit"] for e in flat])))
            bests = [e["best"] for e in flat if not np.isnan(e["best"])]
            if bests:
                row["mean_best_margin"] = float(np.mean(bests))
        return row

    # null row
    null_flat = [e for d in per_case.values() for e in d.get(0.0, [])]
    if null_flat:
        out[0.0] = dict(role="null (resampled, unsteered)", **_flat_stats(null_flat))
    for a in alphas:
        flat = [e for d in per_case.values() for e in d.get(a, [])]
        row = dict(role="steered", **_flat_stats(flat))
        dm = _case_means(a, "final")
        db = _case_means(a, "best")
        ds = _case_submit_delta(a)
        row["n_paired_cases"] = int(len(dm))
        if len(dm):
            lo, hi = _boot_ci(dm)
            row.update(mean_delta_margin_vs_null=float(np.nanmean(dm)),
                       delta_margin_ci95=[lo, hi])
            lo, hi = _boot_ci(ds)
            row.update(delta_submit_rate_vs_null=float(np.nanmean(ds)),
                       delta_submit_ci95=[lo, hi])
        if len(db):
            lo, hi = _boot_ci(db)
            row.update(mean_delta_best_vs_null=float(np.nanmean(db)),
                       delta_best_ci95=[lo, hi])
        out[a] = row
    return out


def print_branch_null(summ):
    if 0.0 not in summ:
        print("[composite] no null branches found (run with --null-branches >= 1 "
              "to get the resampled alpha-0 counterfactual).")
        return
    s0 = summ[0.0]
    print(f"\n[composite] branch-vs-null (paired by case; null = resampled alpha 0):")
    print(f"   null      : n={s0['n_branches']}  submit_rate={s0['submit_rate']:.0%}  "
          f"final={s0['mean_final_margin']:+.4f}"
          + (f"  best={s0['mean_best_margin']:+.4f}" if "mean_best_margin" in s0 else "")
          + (f"  bad_submit={s0['bad_submit_rate']:.0%}"
             if s0.get("bad_submit_rate") else ""))
    for a, s in sorted(summ.items()):
        if a == 0.0:
            continue
        line = (f"   alpha {a:+.1f}: n={s['n_branches']}  "
                f"submit_rate={s['submit_rate']:.0%}  "
                f"final={s['mean_final_margin']:+.4f}")
        if "mean_best_margin" in s:
            line += f"  best={s['mean_best_margin']:+.4f}"
        if s.get("bad_submit_rate"):
            line += f"  bad_submit={s['bad_submit_rate']:.0%}"
        if s.get("delta_submit_rate_vs_null") is not None:
            line += f"\n              d_submit={s['delta_submit_rate_vs_null']:+.2f}"
            lo, hi = s.get("delta_submit_ci95", (None, None))
            if lo is not None:
                line += f" [{lo:+.2f},{hi:+.2f}]"
        if s.get("mean_delta_margin_vs_null") is not None:
            lo, hi = s["delta_margin_ci95"]
            line += f"  d_final={s['mean_delta_margin_vs_null']:+.4f}"
            if lo is not None:
                line += f" [{lo:+.4f},{hi:+.4f}]"
        if s.get("mean_delta_best_vs_null") is not None:
            lo, hi = s["delta_best_ci95"]
            line += f"  d_best={s['mean_delta_best_vs_null']:+.4f}"
            if lo is not None:
                line += f" [{lo:+.4f},{hi:+.4f}]"
        if s.get("n_paired_cases") is not None:
            line += f"  (paired cases={s['n_paired_cases']})"
        print(line)
    print("   reading: d_submit < 0 = the vector suppresses commitment "
          "(persistence); d_best > 0 = the extra turns actually found better "
          "points (competence conversion). bad_submit = finalised while an "
          "objective was FAILING -- watch for steering eroding constraints.")


def summarize_composite(results, alphas):
    """Aggregate across rollouts: mean final priority margin, mean gap to the case
    optimum, mean improvement over baseline, and fraction of rollouts where the
    steered run ended with a STRICTLY larger margin than its own baseline.
    (Aggregates over branch reps; deltas are vs the UNTOUCHED baseline -- for the
    resampling-honest comparison use summarize_branch_null.)"""
    def _mean_over_reps(r, a, key, base_only=False):
        vals = [_fnum(e.get(key)) for aa, b, e in _iter_summary(r)
                if aa == a and (b == 0 if base_only else not (a == 0.0 and b == 0))]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    base_fm = np.array([_mean_over_reps(r, 0.0, "final_margin", base_only=True)
                        for r in results])
    out = {}
    for a in [0.0] + [x for x in alphas if x != 0.0]:
        fm = np.array([_mean_over_reps(r, a, "final_margin", base_only=(a == 0.0))
                       for r in results])
        gap = np.array([_mean_over_reps(r, a, "gap", base_only=(a == 0.0))
                        for r in results])
        row = dict(mean_final_margin=(float(np.nanmean(fm))
                                      if not np.all(np.isnan(fm)) else None),
                   mean_gap=(float(np.nanmean(gap)) if not np.all(np.isnan(gap)) else None),
                   n=int(np.sum(~np.isnan(fm))))
        if a == 0.0:
            row.update(mean_delta_vs_base=0.0, frac_improved=None)
        else:
            delta = fm - base_fm
            valid = ~np.isnan(delta)
            row.update(
                mean_delta_vs_base=(float(np.nanmean(delta)) if valid.any() else None),
                frac_improved=(float(np.sum(delta[valid] > 1e-9)) / int(valid.sum())
                               if valid.any() else None))
        out[a] = row
    return out


def plot_composite(results, alphas, out_png, ncol=3):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = len(results); ncol = min(ncol, max(1, n)); nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow), squeeze=False)
    colors = {0.0: "black"}
    palette = ["tab:orange", "tab:red", "tab:purple", "tab:brown"]
    for k, a in enumerate(alphas):
        colors[a] = palette[k % len(palette)]
    for ax in axes.flat:
        ax.axis("off")
    for i, res in enumerate(results):
        ax = axes[i // ncol][i % ncol]; ax.axis("on")
        for a in [0.0] + list(alphas):
            traj = res["trajectories"].get(a)
            if not traj:
                continue
            xs, ys = zip(*traj)
            lab = "baseline (0)" if a == 0.0 else f"alpha {a:+.1f}"
            ax.plot(xs, ys, "-o", ms=3, color=colors[a], label=lab,
                    lw=(2 if a == 0.0 else 1.5), alpha=0.9)
            st = res["summary"][a]["submit_turn"]
            if st is not None:
                yv = dict(traj).get(st)
                if yv is not None:
                    ax.plot([st], [yv], "*", ms=13, color=colors[a])
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set_title(f"rollout {res['idx']:04d} (seed {res['seed']})", fontsize=9)
        ax.set_xlabel("turn"); ax.set_ylabel("margin"); ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Composite: margin vs turn (* = submit). Negative alpha branches at "
                 "the baseline submit turn.", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"[composite] wrote {out_png}  ({len(results)} rollouts)")



# ===========================================================================
# Trigger study (closed-loop) -- monitor a projection, inject when it crosses
# ===========================================================================
def run_trigger(cfg, agent, controller, env_kind, n_rollouts, repeats, out_run_dir):
    """Run unsteered task rollouts under the closed-loop TriggerController; build
    tidy kind='trigger' rows (margin, all_pass, detector projection, n_steered per
    turn) plus per-rollout transcripts and a per-token projection trace sidecar."""
    cfg.env_kind = env_kind
    if env_kind == "sine":
        cfg.n_obj = 1
    cfg.capture = False
    os.makedirs(out_run_dir, exist_ok=True)
    seeds = list(range(cfg.seed_start, cfg.seed_start + n_rollouts))
    rows = []
    with controller:                                   # install the hook for the run
        for case_id, seed in enumerate(seeds):
            opt = build_env(cfg)
            opt.reset(seed=seed, wide=getattr(cfg, "wide_cases", True))
            optd = opt.optimum(samples=getattr(cfg, "optimum_samples", 50000))
            opt_m = optd.get("margin_priority") if optd.get("feasible") else None
            for rep in range(repeats):
                idx = case_id * repeats + rep
                env = build_env(cfg); env.reset(seed=seed, wide=getattr(cfg, "wide_cases", True))
                messages = [{"role": "system", "content": system_prompt_for(cfg)},
                            {"role": "user", "content": render_case_for(env, cfg)}]
                res = _advance(cfg, agent, env, messages, 1, controller=controller,
                               tag=f"trig{idx}")
                st, submitted, nt = res["submit_turn"], res["submitted"], res["n_turns"]
                rdir = io.rollout_dir(out_run_dir, idx); io.save_case(rdir, env, seed)
                traces = {}
                with open(os.path.join(rdir, "transcript.jsonl"), "w") as f:
                    for rec in res["records"]:
                        ti = rec.get("trigger") or {}
                        traces[rec["turn"]] = ti.get("trace")
                        slim = {k: v for k, v in rec.items() if k != "trigger"}
                        slim["trigger"] = {kk: ti.get(kk) for kk in
                                           ("proj_end", "proj_max", "proj_min", "n_steered",
                                            "n_triggers", "fired", "n_tokens")}
                        f.write(json.dumps(slim) + "\n")
                with open(os.path.join(rdir, "trigger_trace.json"), "w") as f:
                    json.dump(traces, f)
                for rec in res["records"]:
                    t = rec["turn"]; mg = rec.get("margin"); ti = rec.get("trigger") or {}
                    rows.append(dict(
                        idx=idx, seed=seed, case_id=case_id, rep=rep, branch_rep=0,
                        alpha=None, turn=t, action=rec.get("action"), margin=mg,
                        all_pass=rec.get("all_pass"), proj=ti.get("proj_end"),
                        proj_max=ti.get("proj_max"),
                        n_steered=ti.get("n_steered"), submitted=submitted,
                        submit_turn=st, n_turns=nt,
                        is_submit=bool(submitted and st is not None
                                       and t == st and rec.get("action") == "submit"),
                        branch_turn=None, branch_kind=None, optimum_margin=opt_m,
                        gap=((opt_m - mg) if (opt_m is not None and mg is not None) else None)))
                fired_turns = sum(1 for r in res["records"]
                                  if (r.get("trigger") or {}).get("n_steered"))
                steered_tok = sum((r.get("trigger") or {}).get("n_steered", 0) or 0
                                  for r in res["records"])
                print(f"  trig {idx:04d} (seed {seed}): submitted={submitted} "
                      f"fired_turns={fired_turns} steered_tokens={steered_tok}")
                _free()
    return rows


# ===========================================================================
# Story study (Test 2) -- out-of-distribution length modulation
# ===========================================================================
DEFAULT_STORY_PROMPTS = [
    "Write a short anecdote about a lighthouse keeper who finds a message in a bottle.",
    "Write a short anecdote about a child who befriends a robot in an abandoned factory.",
    "Write a short anecdote about two strangers who share a long train journey.",
    "Write a short anecdote about a baker whose bread grants vivid dreams.",
    "Write a short anecdote about an astronaut who discovers a garden on a dead planet.",
]


def _story_prompt_ids(agent, prompt):
    import torch
    msgs = [{"role": "user", "content": prompt}]
    kw = dict(tokenize=True, add_generation_prompt=True,
              return_dict=True, return_tensors="pt")
    try:
        # CRITICAL: without this, Qwen3 templates default to THINKING mode and
        # every story opens a <think> block that rambles to max_new.
        out = agent.processor.apply_chat_template(
            msgs, enable_thinking=agent.cfg.enable_thinking, **kw)
    except TypeError:                                # template without the kwarg
        out = agent.processor.apply_chat_template(msgs, **kw)
    ids = out["input_ids"] if hasattr(out, "keys") else out
    return ids.to(agent.model.device)


def _eos_ids(agent):
    """ALL ids that legitimately end a chat turn: the tokenizer eos, everything
    in the model's generation_config.eos_token_id (int or list), and <|im_end|>
    if it exists. On Qwen these are not all the same id."""
    ids = set()
    tk = agent.processor.tokenizer
    if tk.eos_token_id is not None:
        ids.add(int(tk.eos_token_id))
    g = getattr(agent.model.generation_config, "eos_token_id", None)
    for v in (g if isinstance(g, (list, tuple)) else [g]):
        if v is not None:
            ids.add(int(v))
    try:
        im = tk.convert_tokens_to_ids("<|im_end|>")
        if im is not None and im >= 0 and im != getattr(tk, "unk_token_id", -1):
            ids.add(int(im))
    except Exception:
        pass
    return ids


def _gen_continuation(agent, input_ids, max_new):
    """Free generation (no action stopper); returns the continuation token ids."""
    import torch
    out = agent.model.generate(
        input_ids, max_new_tokens=max_new,
        do_sample=agent.cfg.temperature > 0, temperature=agent.cfg.temperature,
        eos_token_id=sorted(_eos_ids(agent)),
        pad_token_id=agent.processor.tokenizer.eos_token_id,
        attention_mask=torch.ones_like(input_ids))
    return out[0, input_ids.shape[1]:]


def _concat_prefix(pids, half_list):
    import torch
    return torch.cat([pids, torch.tensor([half_list], device=pids.device)], dim=1)


def _trim_eos(ids, eos):
    """Trim trailing end-of-turn ids; eos may be an int or a set of ints."""
    eos = {eos} if isinstance(eos, int) else set(eos)
    ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    while ids and ids[-1] in eos:
        ids.pop()
    return ids


def run_story_study(agent, steer_vec, layer, prompts, alphas, n_repeats, max_new,
                    steer_ctx=steering_active, encode_fn=_story_prompt_ids,
                    gen_fn=_gen_continuation, concat_fn=_concat_prefix,
                    gen_only=True, ctl_vec=None):
    """For each repeat: generate a baseline story, cut at the midpoint, and continue
    the first half unsteered (alpha 0) and under each alpha (steering every NEW
    token; the prefix encode is unsteered when gen_only). If ctl_vec is given
    (matched-norm random vector), each alpha is ALSO run with the control from
    the SAME prefix, so real and control land in one row/plot. Returns rows of
    continuation lengths + decoded texts (with a 'capped' flag when a continuation
    hit max_new -- those lengths are censored, so report frac_capped alongside
    means). Hypotheses:
        alpha > 0 (toward SUBMIT/stop)     -> continuation SHORTER than alpha 0
        alpha < 0 (toward SET/keep going)  -> continuation LONGER  than alpha 0"""
    block_idx = layer - 1
    eos = _eos_ids(agent)
    tk = agent.processor.tokenizer
    rows = []
    for r in range(n_repeats):
        prompt = prompts[r % len(prompts)]
        pids = encode_fn(agent, prompt)
        base_cont = _trim_eos(gen_fn(agent, pids, max_new), eos)
        base_len = len(base_cont)
        half = max(1, base_len // 2)
        prefix = concat_fn(pids, base_cont[:half])
        cont, capped, texts = {}, {}, {}
        cont_ctl, capped_ctl, texts_ctl = {}, {}, {}
        for a in [0.0] + list(alphas):
            if a == 0.0:
                c = _trim_eos(gen_fn(agent, prefix, max_new), eos)
            else:
                with steer_ctx(agent.model, block_idx, steer_vec, a,
                               gen_only=gen_only):
                    c = _trim_eos(gen_fn(agent, prefix, max_new), eos)
            cont[a] = len(c)
            capped[a] = bool(len(c) >= max_new)   # hit the ceiling: length censored
            texts[a] = tk.decode(c)
            if ctl_vec is not None and a != 0.0:  # matched-norm control, same prefix
                with steer_ctx(agent.model, block_idx, ctl_vec, a,
                               gen_only=gen_only):
                    c2 = _trim_eos(gen_fn(agent, prefix, max_new), eos)
                cont_ctl[a] = len(c2)
                capped_ctl[a] = bool(len(c2) >= max_new)
                texts_ctl[a] = tk.decode(c2)
        row = dict(repeat=r, prompt=prompt, base_len=base_len, half=half,
                   base_capped=bool(base_len >= max_new),
                   base_text=tk.decode(base_cont),
                   cont=cont, capped=capped, texts=texts)
        if ctl_vec is not None:
            row.update(cont_ctl=cont_ctl, capped_ctl=capped_ctl,
                       texts_ctl=texts_ctl)
        rows.append(row)
        deltas = " ".join(f"a{a:+.1f}:{cont[a]}({cont[a]-cont[0.0]:+d})"
                          + ("^" if capped[a] else "")
                          + (f"/ctl:{cont_ctl[a]}" + ("^" if capped_ctl[a] else "")
                             if a in cont_ctl else "")
                          for a in alphas)
        print(f"  story r{r}: base_len={base_len}{'^' if base_len >= max_new else ''} "
              f"half={half} cont0={cont[0.0]}"
              f"{'^' if capped[0.0] else ''}  {deltas}   (^ = hit max_new)")
    return rows

def summarize_story(rows, alphas, ctl=False):
    """Mean continuation length per alpha, its spread (std) and standard error
    (sem), and mean signed delta vs alpha 0. frac_capped = fraction of
    continuations that hit max_new (length-censored; when high, BOTH mean_len
    and its std/sem are biased -- the mean pulled down, the variance deflated --
    so treat those bars as unreliable). ctl=True summarises the control arm
    (cont_ctl/capped_ctl) against the SAME alpha-0 continuations."""
    ck, pk = ("cont_ctl", "capped_ctl") if ctl else ("cont", "capped")
    def _get(r, d, a):
        v = r.get(d, {})
        return v.get(a, v.get(str(a)))                # json round-trip safety
    def _cap(a):
        vals = [_get(r, pk, a) for r in rows]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None
    def _spread(arr):                                 # (std, sem)
        if len(arr) > 1:
            sd = float(arr.std(ddof=1))
            return sd, sd / np.sqrt(len(arr))
        return 0.0, 0.0
    out = {}
    c0 = np.array([_get(r, "cont", 0.0) for r in rows], float)
    if not ctl:
        sd0, se0 = _spread(c0)
        out[0.0] = dict(mean_len=float(c0.mean()), std_len=sd0, sem_len=se0,
                        mean_delta=0.0, n=len(rows), frac_capped=_cap(0.0))
    for a in alphas:
        ca = np.array([v for v in (_get(r, ck, a) for r in rows)
                       if v is not None], float)
        if not len(ca):
            continue
        c0a = np.array([_get(r, "cont", 0.0) for r in rows
                        if _get(r, ck, a) is not None], float)
        sda, sea = _spread(ca)
        out[a] = dict(mean_len=float(ca.mean()), std_len=sda, sem_len=sea,
                      mean_delta=float((ca - c0a).mean()), n=int(len(ca)),
                      frac_capped=_cap(a))
    return out


def plot_story_bars(summ, summ_ctl, alphas, out_png, max_new,
                    errbar="sem", cap_hatch=0.99):
    """Grouped bars: mean continuation length per alpha, real vector vs
    matched-norm control, unsteered as the reference bar. Error bars show
    errbar='sem' (uncertainty of the mean, default) or 'std' (spread of
    continuations); errbar=None disables them. Bars with frac_capped >= cap_hatch
    are hatched (length-censored -> mean and error bar both untrustworthy)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    err_key = {"sem": "sem_len", "std": "std_len", None: None}[errbar]
    order = [0.0] + sorted(a for a in alphas if a != 0.0)
    x = np.arange(len(order))
    w = 0.36 if summ_ctl else 0.6
    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(order), 4.2))
    def _bars(summary, offset, color):
        for xi, a in zip(x, order):
            s = summary.get(a) or summary.get(str(a))
            if not s:
                continue
            cap = s.get("frac_capped") or 0.0
            err = (s.get(err_key) or 0.0) if err_key else 0.0
            ax.bar(xi + offset, s["mean_len"], w, color=color,
                   alpha=0.85,
                   edgecolor="black", linewidth=0.6,
                   yerr=err, capsize=3, error_kw=dict(ecolor="black", lw=0.8))
            ax.text(xi + offset, s["mean_len"] + err + max_new * 0.01,
                    f"{s['mean_len']:.0f}" + (f"\n{cap:.0%}^" if cap > 0 else ""),
                    ha="center", fontsize=7)
    _bars(summ, -w / 2 if summ_ctl else 0.0, "tab:blue")
    if summ_ctl:
        _bars(summ_ctl, +w / 2, "red")
    legend_handles = [Patch(facecolor="tab:blue", alpha=0.85, label="steer vector")]
    if summ_ctl:
        legend_handles.append(
            Patch(facecolor="red", alpha=0.85,
                  label="random control (matched norm)"))
    #legend_handles.append(
    #    ax.axhline(max_new, color="tab:red", ls=":", lw=1,
    #               label=f"max_new={max_new}"))
    ax.set_xticks(x)
    ax.set_xticklabels(["unsteered" if a == 0.0 else f"\u03b1 {a:+.1f}"
                        for a in order])
    ax.set_ylabel("mean continuation length (tokens)")
    ax.set_title("Story continuation length: steer vs matched-norm control\n")
    ax.legend(handles=legend_handles, fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"[story] wrote {out_png}")

import os, json

def load_or_run_story_study(json_path, *run_args, max_new, **run_kwargs):
    """Guarded runner: if json_path already holds results for THIS max_new,
    load and return them instead of re-running. If the file is missing -- or was
    made with a different max_new -- run the study and save. Keying on max_new is
    the point: bumping it changes the generations, so a plain 'file exists?'
    check would silently reuse stale, censored data."""
    if os.path.exists(json_path):
        with open(json_path) as f:
            blob = json.load(f)
        if isinstance(blob, dict) and "rows" in blob:      # new format w/ meta
            saved_max_new, rows = blob.get("max_new"), blob["rows"]
        else:                                              # old bare-list format
            saved_max_new, rows = None, blob
        if saved_max_new == max_new:
            print(f"[story] loaded {len(rows)} rows from {json_path} "
                  f"(max_new={max_new}, skipping simulation)")
            return rows
        why = ("no max_new recorded" if saved_max_new is None
               else f"saved max_new={saved_max_new}")
        print(f"[story] {json_path} is stale ({why} != {max_new}) -> re-running")
    rows = run_story_study(*run_args, max_new=max_new, **run_kwargs)
    with open(json_path, "w") as f:
        json.dump({"max_new": max_new, "rows": rows}, f)
    print(f"[story] wrote {len(rows)} rows to {json_path} (max_new={max_new})")
    return rows


def plot_story(rows, alphas, out_png):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cats = [0.0] + list(alphas)
    means = [np.mean([r["cont"][a] for r in rows]) for a in cats]
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = np.arange(len(cats))
    ax.bar(xs, means, color=["black"] + ["tab:red" if a < 0 else "tab:green" for a in alphas],
           alpha=0.55, width=0.6)
    for j, a in enumerate(cats):                   # scatter individual repeats
        ys = [r["cont"][a] for r in rows]
        ax.scatter(np.full(len(ys), j) + np.random.uniform(-0.08, 0.08, len(ys)),
                   ys, color="k", s=18, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(["unsteered (0)"] + [f"alpha {a:+.1f}" for a in alphas])
    ax.set_ylabel("continuation length (tokens, from the half-story)")
    ax.set_title("Story continuation length vs steering\n"
                 "(toward SUBMIT should shorten, toward SET should lengthen)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out_png, dpi=130)
    print(f"[story] wrote {out_png}")


# ===========================================================================
# CLI
# ===========================================================================
def main():
    cfg = Config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", choices=["composite", "story", "steer", "project",
                                        "compare", "trigger"], required=True,
                    help="composite/story (transfer tests); steer (alpha sweep in an "
                         "env); project (no steering, project per-turn acts onto the "
                         "axis); compare (per-layer cosine vs another directions file); "
                         "trigger (closed-loop: monitor a projection, inject when it "
                         "crosses a threshold).")
    ap.add_argument("--source-run-dir", default=os.path.join(cfg.out_dir, cfg.run_name))
    ap.add_argument("--directions", default=None)
    ap.add_argument("--vector-type", choices=["set", "explore"], default="set",
                    help="'set': SUBMIT-finalSET (directions.npz). "
                         "'explore': SUBMIT-EXPLORE (directions_explore.npz).")
    ap.add_argument("--inject-at", choices=["final_set", "submit"], default="final_set",
                    help="composite branch point: 'final_set' (last SET before the "
                         "baseline submit; default) or 'submit' (the submit turn).")
    ap.add_argument("--layer", type=int, default=None,
                    help="hidden-state index (default: best_layer in directions.npz)")
    ap.add_argument("--frac", type=float, default=0.4)
    ap.add_argument("--alphas", type=float, nargs="+", default=None,
                    help="composite: negatives (default -0.5 -1.0); story: signed "
                         "(default -0.5 0.5); steer: default -1 -0.5 0 0.5 1.")
    ap.add_argument("--model", default=None)
    ap.add_argument("--run-name", default="csat_transfer")
    ap.add_argument("--layers-attr", default=None,
                    help="dotted path to the decoder ModuleList if auto-detect is wrong")
    # project / compare
    ap.add_argument("--pool", choices=["before", "around", "all"], default="before",
                    help="project: token pool for the projection (match extraction)")
    ap.add_argument("--win", type=int, default=4, help="project: window for --pool around")
    ap.add_argument("--compare-to", default=None,
                    help="compare: directions file to compare against "
                         "(default <source-run-dir>/directions.npz)")
    # trigger (closed-loop)
    ap.add_argument("--detect-vec", choices=["submit", "explore"], default="submit",
                    help="trigger: axis to MONITOR (submit=directions.npz, "
                         "explore=directions_explore.npz)")
    ap.add_argument("--steer-vec", choices=["submit", "explore"], default="explore",
                    help="trigger: axis to INJECT when triggered")
    ap.add_argument("--steer-proj", type=float, default=0.0,
                    help="trigger: affine-projection threshold (SET=-1, SUBMIT=+1)")
    ap.add_argument("--trigger", choices=["above", "below"], default="above",
                    help="trigger: fire when projection is above/below --steer-proj")
    ap.add_argument("--k", type=int, default=10,
                    help="trigger: monitor window (mean-pooled last-k tokens)")
    ap.add_argument("--steer-k", type=int, default=20,
                    help="trigger: tokens to inject once fired (then re-arm)")
    ap.add_argument("--alpha", type=float, default=-1.0,
                    help="trigger: scalar on the steer vector (e.g. -1 toward EXPLORE)")
    # composite
    ap.add_argument("--env", choices=["parabola", "sine", "coupling"], default="sine")
    ap.add_argument("--n-rollouts", type=int, default=12)
    ap.add_argument("--repeats", type=int, default=1,
                    help="composite: repeats per case (same landscape, fresh "
                         "sampling). Overlaid on one subplot per case. Alias: --n-repeats.")
    ap.add_argument("--branch-repeats", type=int, default=1,
                    help="composite: fan out N steered continuations from the SAME "
                         "baseline branch point per alpha (cheap; reuses the prefix). "
                         "Use with --n-rollouts 1 to study one baseline's divergence.")
    ap.add_argument("--null-branches", type=int, default=None,
                    help="composite: N UNSTEERED (alpha 0) re-branches from the same "
                         "branch point -- the resampling-honest null the steered "
                         "branches are compared against (default: = --branch-repeats; "
                         "3+ recommended when temperature > 0).")
    ap.add_argument("--control", choices=["none", "random", "random-orth"],
                    default="none",
                    help="replace the steering vector with a matched-norm random "
                         "vector ('random'), or one also orthogonalised to the real "
                         "axis ('random-orth'). Applies to steer/composite/story and "
                         "the trigger's INJECTED vector. Run names get a _ctl tag.")
    ap.add_argument("--control-seed", type=int, default=0,
                    help="seed for the random control vector")
    ap.add_argument("--steer-prefill", action="store_true",
                    help="ALSO steer prefill forwards (old behaviour). Default is "
                         "generation-only, so the model's reading of the prompt/"
                         "feedback numbers is never perturbed.")
    ap.add_argument("--with-control", action="store_true",
                    help="story: additionally run a matched-norm RANDOM control "
                         "at every alpha from the same prefixes, so real and "
                         "control bars share one plot (story_bars.png). Uses "
                         "--control-seed. Unlike --control, this does NOT replace "
                         "the real vector.")
    ap.add_argument("--gen-max-new", type=int, default=None,
                    help="override cfg.max_new_tokens for the per-turn optimisation "
                         "generation (composite). Lower (e.g. 1024-1536) to cut OOM.")
    ap.add_argument("--plot-view", choices=["auto", "repeats", "fanout"], default="auto",
                    help="auto: 'fanout' if --branch-repeats>1 else 'repeats'.")
    ap.add_argument("--plot-passing", choices=["segment", "marker", "none"],
                    default="segment",
                    help="encode all_pass: 'segment' dims/thins failing spans + "
                         "hollow markers; 'marker' only marker fill; 'none'.")
    ap.add_argument("--branch-extra", type=int, default=10,
                    help="max turns a steered branch may run past the baseline submit "
                         "(bounds compute on coupling/parabola)")
    # story
    ap.add_argument("--n-repeats", type=int, default=None,
                    help="repeats: story default 5; composite uses --repeats unless "
                         "this is given (then it overrides --repeats).")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--prompt", default=None, help="fix a single story prompt")
    args = ap.parse_args()

    if args.model:
        cfg.model_name = args.model
    _dfile = "directions_explore.npz" if args.vector_type == "explore" else "directions.npz"
    directions = args.directions or os.path.join(args.source_run_dir, _dfile)
    if not os.path.exists(directions):
        raise SystemExit(f"{directions} not found; run direction_extract.py "
                         "(or build_explore_directions.py for --vector-type explore) first.")

    repeats = args.n_repeats if args.n_repeats is not None else args.repeats

    # --- compare: pure numpy, no model load ---
    if args.study == "compare":
        compare_to = args.compare_to or os.path.join(args.source_run_dir, "directions.npz")
        if not os.path.exists(compare_to):
            raise SystemExit(f"{compare_to} not found; pass --compare-to.")
        pairs = compare_directions(directions, compare_to)
        print(f"[compare] per-layer cosine of the (SUBMIT-SET) steering axis\n"
              f"          NEW = {directions}\n          OLD = {compare_to}")
        for L, c in pairs:
            print(f"   L{L:>2d}: cos = {c:+.4f}")
        if pairs:
            cs = np.array([c for _, c in pairs], float)
            print(f"[compare] over {len(pairs)} shared layers: mean {np.nanmean(cs):+.4f}  "
                  f"min {np.nanmin(cs):+.4f}  max {np.nanmax(cs):+.4f}")
        return

    layer = args.layer
    if layer is None:
        z = np.load(directions, allow_pickle=True)
        if "best_layer" not in z.files:
            raise SystemExit("no best_layer in directions; pass --layer.")
        layer = int(z["best_layer"])
        print(f"[transfer] using saved best_layer = {layer}")
    if layer < 1:
        raise SystemExit("--layer must be >= 1.")

    if args.gen_max_new:
        cfg.max_new_tokens = int(args.gen_max_new)   # OOM lever for generated turns
    agent = ModelAgent(cfg)

    # --- steer: alpha sweep in args.env using the SOURCE-derived steer vector ---
    if args.study == "steer":
        from . import composite_plot as cp
        alphas = args.alphas if args.alphas is not None else [-1.0, -0.5, 0.0, 0.5, 1.0]
        steer_vec, info = build_steering_vector(layer, directions,
                                                args.source_run_dir, frac=args.frac)
        print(f"[steer] layer {info['layer']}  |steer|={info['steer_norm']:.2f}  "
              f"= {info['frac']:.0%} of mean token norm {info['mean_token_norm']:.2f}")
        steer_vec, ctl_tag = apply_control(steer_vec, args.control, args.control_seed)
        alpha_dirs = []
        for alpha in alphas:
            rd = run_steered(cfg, agent, steer_vec, layer, float(alpha),
                             n_rollouts=args.n_rollouts, repeats=repeats,
                             base_run_name=args.run_name, env_kind=args.env,
                             layers_attr=args.layers_attr,
                             gen_only=(not args.steer_prefill), tag=ctl_tag)
            alpha_dirs.append(rd)
        out_run_dir = os.path.join(cfg.out_dir,
                                   f"{args.run_name}_{args.env}_L{layer}{ctl_tag}_steer")
        cp.rebuild_steer(alpha_dirs, out_run_dir, write=True)   # tidy CSV (kind=steer)
        cp.plot(out_run_dir, passing=args.plot_passing)         # auto-detects kind
        print(f"[done] steer sweep complete -> {out_run_dir}")
        return

    # --- project: NO steering; project per-turn acts onto the source axis ---
    if args.study == "project":
        from . import composite_plot as cp
        run_dir = run_unsteered_capture(cfg, agent, args.env, n_rollouts=args.n_rollouts,
                                        repeats=repeats, base_run_name=args.run_name)
        cp.rebuild_project(run_dir, directions, layer, args.pool, args.win,
                           cfg.model_name, write=True)          # tidy CSV (kind=project)
        cp.plot(run_dir)                                        # auto-detects kind
        print(f"[done] projection complete -> {run_dir}")
        return

    # --- trigger: closed-loop monitor + inject ---
    if args.study == "trigger":
        from . import composite_plot as cp

        def _dir_for(which):
            f = "directions_explore.npz" if which == "explore" else "directions.npz"
            p = os.path.join(args.source_run_dir, f)
            if not os.path.exists(p):
                raise SystemExit(f"{p} not found (needed for --{which}).")
            return p

        detect_path = _dir_for(args.detect_vec)
        steer_path = _dir_for(args.steer_vec)
        set_v, sub_v = load_direction(detect_path, layer)            # detector axis
        steer_vec, info = build_steering_vector(layer, steer_path,
                                                args.source_run_dir, frac=args.frac)
        # control applies to the INJECTED vector only; the detector axis stays real
        # (a random injection under a real detector tests whether closed-loop gains
        # come from the direction or from perturbation per se).
        steer_vec, ctl_tag = apply_control(steer_vec, args.control, args.control_seed)
        print(f"[trigger] detect={args.detect_vec} steer={args.steer_vec}{ctl_tag} "
              f"alpha={args.alpha:+.2f} L{layer} steer_proj={args.steer_proj} "
              f"k={args.k} steer_k={args.steer_k} ({args.trigger}); "
              f"|steer|={info['steer_norm']:.2f}")
        ctrl = TriggerController(agent.model, layer - 1, set_v, sub_v, steer_vec,
                                 args.alpha, args.steer_proj, args.k, args.steer_k,
                                 trigger=args.trigger, layers_attr=args.layers_attr)
        out_run_dir = os.path.join(cfg.out_dir,
                                   f"{args.run_name}_{args.env}_L{layer}{ctl_tag}_trigger")
        rows = run_trigger(cfg, agent, ctrl, args.env, args.n_rollouts, repeats, out_run_dir)
        cp.write_data(out_run_dir, rows, meta=dict(
            kind="trigger", env=args.env, layer=layer, detect_vec=args.detect_vec,
            steer_vec=args.steer_vec, alpha=args.alpha, steer_proj=args.steer_proj,
            k=args.k, steer_k=args.steer_k, trigger=args.trigger))
        cp.plot(out_run_dir, passing=args.plot_passing)
        print(f"[done] trigger study complete -> {out_run_dir}")
        return

    # composite / story both need the steer vector
    steer_vec, info = build_steering_vector(layer, directions, args.source_run_dir, frac=args.frac)
    print(f"[transfer] layer {layer}  |steer|={info['steer_norm']:.2f} "
          f"({info['frac']:.0%} of token norm {info['mean_token_norm']:.2f})")
    steer_vec, ctl_tag = apply_control(steer_vec, args.control, args.control_seed)
    gen_only = not args.steer_prefill

    if args.study == "composite":
        alphas = args.alphas if args.alphas is not None else [-0.5, -1.0]
        cfg.env_kind = args.env
        if args.env == "sine":
            cfg.n_obj = 1
        cfg.capture = False
        cfg.run_name = (f"{args.run_name}_composite_{args.env}"
                        f"_{args.vector_type}_{args.inject_at}_L{layer}{ctl_tag}")
        out_run_dir = cfg.run_dir()
        seeds = list(range(cfg.seed_start, cfg.seed_start + args.n_rollouts))
        from . import composite_plot as cp
        null_b = (args.null_branches if args.null_branches is not None
                  else args.branch_repeats)
        if null_b < 2 and cfg.temperature > 0:
            print("[composite] note: temperature > 0 with only "
                  f"{null_b} null branch(es); the resampled-null comparison will "
                  "be noisy. Consider --null-branches 3 (or more).")

        # write run meta up front so a later crash still leaves a titled rebuild
        cp.write_data(out_run_dir, [], meta=dict(
            study="composite", env=args.env, vector_type=args.vector_type,
            inject_at=args.inject_at, layer=layer, frac=args.frac,
            alphas=list(alphas), n_rollouts=args.n_rollouts, repeats=repeats,
            branch_repeats=args.branch_repeats, null_branches=null_b,
            control=args.control, control_seed=args.control_seed,
            gen_only=gen_only,
            model_name=cfg.model_name, run_name=cfg.run_name,
            source_run_dir=args.source_run_dir)) if not os.path.exists(
            os.path.join(out_run_dir, "composite_meta.json")) else None

        n_done = n_run = 0
        for case_id, seed in enumerate(seeds):
            # optimum is a property of the landscape -> compute ONCE per case and
            # reuse across repeats (mirrors run.py; avoids MC noise / recompute)
            env0 = build_env(cfg); env0.reset(seed=seed, wide=getattr(cfg, "wide_cases", True))
            opt = env0.optimum(samples=getattr(cfg, "optimum_samples", 50000))
            for rep in range(repeats):
                idx = case_id * repeats + rep
                done_marker = os.path.join(out_run_dir, f"rollout_{idx:04d}",
                                           "composite_summary.json")
                if os.path.exists(done_marker):        # resume: skip finished rollouts
                    n_done += 1
                    continue
                print(f"[composite] case {case_id} rep {rep} (seed {seed})")
                run_composite(cfg, agent, steer_vec, layer, seed, alphas,
                              out_run_dir, idx, branch_extra=args.branch_extra,
                              inject_at=args.inject_at, opt=opt,
                              case_id=case_id, rep=rep,
                              branch_repeats=args.branch_repeats,
                              null_branches=null_b, gen_only=gen_only)
                n_run += 1
                _free()                                # release between rollouts
        if n_done:
            print(f"[composite] resumed: skipped {n_done} finished rollouts, "
                  f"ran {n_run} new.")

        # ---- rebuild the tidy table from ALL completed rollout dirs (durable;
        #      includes resumed rollouts) and draw from it ----
        all_rows, _meta = cp.rebuild_from_dirs(out_run_dir, write=True)
        cp.plot(out_run_dir, view=args.plot_view, passing=args.plot_passing,
                stars="steered")
        summ, agg_alphas = cp.aggregate_from_rows(all_rows)
        print(f"\n[composite] vector={args.vector_type}  inject_at={args.inject_at}  "
              f"env={args.env}  (final priority margin; higher better; "
              f"gap = optimum - final):")
        for a in [0.0] + list(agg_alphas):
            s = summ.get(a, {})
            if s.get("mean_final_margin") is None:
                continue
            line = f"   alpha {a:+.1f}: mean_final_margin={s['mean_final_margin']:+.4f}"
            if s.get("mean_gap") is not None:
                line += f"  mean_gap={s['mean_gap']:+.4f}"
            if a != 0.0 and s.get("mean_delta_vs_base") is not None:
                line += f"  delta_vs_base={s['mean_delta_vs_base']:+.4f}"
            if a != 0.0 and s.get("frac_improved") is not None:
                line += f"  improved={s['frac_improved']:.0%}"
            print(line)
        with open(os.path.join(out_run_dir, "composite_aggregate.json"), "w") as f:
            json.dump({str(k): v for k, v in summ.items()}, f, indent=2)

        # ---- resampled-null comparison (loads composite_summary.json from disk,
        #      so it also covers rollouts finished in earlier resumed runs) ----
        import glob as _glob
        results = []
        for p in sorted(_glob.glob(os.path.join(out_run_dir, "rollout_*",
                                                "composite_summary.json"))):
            try:
                with open(p) as f:
                    results.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
        nsumm = summarize_branch_null(results)
        print_branch_null(nsumm)
        with open(os.path.join(out_run_dir, "composite_null_summary.json"), "w") as f:
            json.dump({str(k): v for k, v in nsumm.items()}, f, indent=2)
        print("[done] composite study complete.")
        return

    # story
    alphas = args.alphas if args.alphas is not None else [-0.5, 0.5]
    prompts = [args.prompt] if args.prompt else DEFAULT_STORY_PROMPTS
    cfg.run_name = f"{args.run_name}_story_L{layer}{ctl_tag}"
    out_run_dir = cfg.run_dir()
    ctl_vec = None
    if args.with_control:
        if args.control != "none":
            print("[story] note: --with-control runs the REAL vector plus a "
                  "matched-norm random control in one pass; ignoring --control "
                  "(which would have REPLACED the vector).")
        from .steering import control_vector
        ctl_vec = control_vector(steer_vec, seed=args.control_seed)
    rows = run_story_study(agent, steer_vec, layer, prompts, alphas,
                           n_repeats=(args.n_repeats if args.n_repeats is not None else 5),
                           max_new=args.max_new, gen_only=gen_only,
                           ctl_vec=ctl_vec)
    with open(os.path.join(out_run_dir, "story_rows.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(out_run_dir, "story_texts.json"), "w") as f:
        json.dump([{k: r[k] for k in
                    ("repeat", "prompt", "base_text", "texts", "texts_ctl")
                    if k in r} for r in rows], f, indent=2)
    summ = summarize_story(rows, alphas)
    summ_ctl = summarize_story(rows, alphas, ctl=True) if ctl_vec is not None else None
    print("\n[story] mean continuation length per alpha (delta vs unsteered):")
    for a in [0.0] + list(alphas):
        s = summ.get(a)
        if not s:
            continue
        cap = (f"  capped={s['frac_capped']:.0%}"
               if s.get("frac_capped") is not None else "")
        line = (f"   alpha {a:+.1f}: mean_len={s['mean_len']:.1f}  "
                f"mean_delta={s['mean_delta']:+.1f}{cap}")
        if summ_ctl and a in summ_ctl:
            sc = summ_ctl[a]
            capc = (f" capped={sc['frac_capped']:.0%}"
                    if sc.get("frac_capped") is not None else "")
            line += (f"   | control: mean_len={sc['mean_len']:.1f} "
                     f"delta={sc['mean_delta']:+.1f}{capc}")
        print(line)
    plot_story_bars(summ, summ_ctl, alphas,
                    os.path.join(out_run_dir, "story_bars.png"), args.max_new)
    try:
        plot_story(rows, alphas, os.path.join(out_run_dir, "story_lengths.png"))
    except Exception as e:
        print(f"[story] per-repeat plot skipped ({e})")
    print("[done] story study complete. Read story_texts.json for coherence: "
          "shorter-but-well-ended vs truncated-mid-sentence are different claims.")


if __name__ == "__main__":
    main()