"""transfer_studies.py -- two stronger tests of what the SUBMIT-SET vector encodes.

Both reuse the SOURCE (e.g. parabola) steering vector from directions.npz; neither
re-extracts. Run as a package module next to steering.py:

  # Composite (Test 1): baseline rollout untouched; at the SUBMIT turn, branch and
  # re-generate that turn (and onward) under negative alpha (toward "keep going").
  python -m csat.transfer_studies --study composite --env sine \
      --source-run-dir runs/csat --alphas -0.5 -1.0 --n-rollouts 12

  # Story (Test 2): out-of-distribution. Generate a story, cut it in half, continue
  # the half under +alpha (stop) and -alpha (continue), steering EVERY new token;
  # does +alpha shorten and -alpha lengthen the continuation vs unsteered?
  python -m csat.transfer_studies --study story \
      --source-run-dir runs/csat --alphas -0.5 0.5 --n-repeats 5

Layer convention is shared with steering.py: --layer j is a hidden-state index;
the hook is placed on decoder block j-1 (whose output == hidden_states[j]).
"""
from __future__ import annotations
import os, json, argparse, copy, contextlib
import numpy as np

from .config import Config
from .agents import ModelAgent
from .rollout import build_env
from .dsl import parse_action, split_thinking
from .prompts import system_prompt_for, render_case_for, render_feedback_for
from .steering import build_steering_vector, steering_active, load_direction
from . import io_utils as io


# ===========================================================================
# Composite study (Test 1)
# ===========================================================================
def _margin_now(env):
    s = env.snapshot()
    return float(s["margin_priority"]), bool(s["all_pass"])


def _advance(cfg, agent, env, messages, start_turn, snapshot_each=False,
             stop_after_turn=None, tag=None):
    """Run the SET/SUBMIT loop from `start_turn` on (env, messages) IN PLACE.
    Returns records, per-turn margins, optional per-turn-start snapshots, and the
    submit info. No activation capture. Any steering hook is installed by the
    caller via a `with steering_active(...)` around this call."""
    records, margins, snaps = [], [], {}
    submitted, submit_turn, snap = False, None, None
    last = cfg.max_turns if stop_after_turn is None else min(cfg.max_turns, stop_after_turn)
    for turn in range(start_turn, last + 1):
        if tag:
            print(f"      {tag} turn {turn}/{last}", end="\r", flush=True)
        if snapshot_each:
            snaps[turn] = (copy.deepcopy(env), list(messages))
        text, meta = agent.act(messages, capture_path=None)
        answer, _ = split_thinking(text, cfg.enable_thinking)
        action = parse_action(answer, env.n_obj)
        messages.append({"role": "assistant", "content": answer})
        rec = dict(turn=turn, action=action.kind, weights=action.weights,
                   error=action.error, response=text)

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
                  steer_ctx=steering_active, branch_extra=10):
    """One case: baseline (alpha 0) untouched; then for each alpha, branch at the
    baseline SUBMIT turn and re-run that turn onward under steering. Saves one
    transcript per alpha and returns the per-alpha margin trajectories."""
    block_idx = layer - 1
    env = build_env(cfg); env.reset(seed=seed, wide=getattr(cfg, "wide_cases", True))
    messages = [{"role": "system", "content": system_prompt_for(cfg)},
                {"role": "user", "content": render_case_for(env, cfg)}]
    base = _advance(cfg, agent, env, messages, 1, snapshot_each=True, tag='base')

    rdir = io.rollout_dir(out_run_dir, idx)
    io.save_case(rdir, env, seed)

    realizations = {0.0: base}
    T = base["submit_turn"]
    if base["submitted"] and T is not None:
        env_b, msgs_b = base["snaps"][T]          # state at the START of the submit turn
        stop_at = min(cfg.max_turns, T + branch_extra)   # bound branch compute
        for a in alphas:
            e2, m2 = copy.deepcopy(env_b), list(msgs_b)
            with steer_ctx(agent.model, block_idx, steer_vec, a):
                realizations[a] = _advance(cfg, agent, e2, m2, T, snapshot_each=False,
                                           stop_after_turn=stop_at, tag=f"a{a:+.1f}")
            print(f"    alpha {a:+.1f}: ran turns {T}..{realizations[a]['n_turns']} "
                  f"(submitted={realizations[a]['submitted']})")
    else:
        print(f"  [composite] rollout {idx:04d} baseline was forced (no submit "
              f"decision to flip); only alpha 0 saved.")

    prefix_recs = [r for r in base["records"] if T is None or r["turn"] < T]
    prefix_marg = [(t, m) for (t, m) in base["margins"] if T is None or t < T]

    trajectories, summary = {}, {}
    for a, res in realizations.items():
        recs = res["records"] if a == 0.0 else prefix_recs + res["records"]
        traj = res["margins"] if a == 0.0 else prefix_marg + res["margins"]
        with open(os.path.join(rdir, f"transcript_a{a:+.2f}.jsonl"), "w") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
        trajectories[a] = traj
        summary[a] = dict(submit_turn=res["submit_turn"], n_turns=res["n_turns"],
                          submitted=res["submitted"],
                          final_margin=(traj[-1][1] if traj else None),
                          all_pass=bool(res["snapshot"]["all_pass"]))
    with open(os.path.join(rdir, "composite_summary.json"), "w") as f:
        json.dump(dict(idx=idx, seed=seed, baseline_submit_turn=T, summary=summary), f, indent=2)
    return dict(idx=idx, seed=seed, trajectories=trajectories, summary=summary, T=T)


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
# Story study (Test 2) -- out-of-distribution length modulation
# ===========================================================================
DEFAULT_STORY_PROMPTS = [
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Write a short story about a child who befriends a robot in an abandoned factory.",
    "Write a short story about two strangers who share a long train journey.",
    "Write a short story about a baker whose bread grants vivid dreams.",
    "Write a short story about an astronaut who discovers a garden on a dead planet.",
]


def _story_prompt_ids(agent, prompt):
    import torch
    msgs = [{"role": "user", "content": prompt}]
    out = agent.processor.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt")
    ids = out["input_ids"] if hasattr(out, "keys") else out
    return ids.to(agent.model.device)


def _gen_continuation(agent, input_ids, max_new):
    """Free generation (no action stopper); returns the continuation token ids."""
    import torch
    out = agent.model.generate(
        input_ids, max_new_tokens=max_new,
        do_sample=agent.cfg.temperature > 0, temperature=agent.cfg.temperature,
        pad_token_id=agent.processor.tokenizer.eos_token_id,
        attention_mask=torch.ones_like(input_ids))
    return out[0, input_ids.shape[1]:]


def _concat_prefix(pids, half_list):
    import torch
    return torch.cat([pids, torch.tensor([half_list], device=pids.device)], dim=1)


def _trim_eos(ids, eos):
    ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    while ids and ids[-1] == eos:
        ids.pop()
    return ids


def run_story_study(agent, steer_vec, layer, prompts, alphas, n_repeats, max_new,
                    steer_ctx=steering_active, encode_fn=_story_prompt_ids,
                    gen_fn=_gen_continuation, concat_fn=_concat_prefix):
    """For each repeat: generate a baseline story, cut at the midpoint, and continue
    the first half unsteered (alpha 0) and under each alpha (steering EVERY new
    token). Returns rows of continuation lengths. Hypotheses:
        alpha > 0 (toward SUBMIT/stop)     -> continuation SHORTER than alpha 0
        alpha < 0 (toward SET/keep going)  -> continuation LONGER  than alpha 0"""
    block_idx = layer - 1
    eos = agent.processor.tokenizer.eos_token_id
    rows = []
    for r in range(n_repeats):
        prompt = prompts[r % len(prompts)]
        pids = encode_fn(agent, prompt)
        base_cont = _trim_eos(gen_fn(agent, pids, max_new), eos)
        base_len = len(base_cont)
        half = max(1, base_len // 2)
        prefix = concat_fn(pids, base_cont[:half])
        cont = {}
        for a in [0.0] + list(alphas):
            if a == 0.0:
                c = _trim_eos(gen_fn(agent, prefix, max_new), eos)
            else:
                with steer_ctx(agent.model, block_idx, steer_vec, a):
                    c = _trim_eos(gen_fn(agent, prefix, max_new), eos)
            cont[a] = len(c)
        rows.append(dict(repeat=r, prompt=prompt, base_len=base_len, half=half, cont=cont))
        deltas = " ".join(f"a{a:+.1f}:{cont[a]}({cont[a]-cont[0.0]:+d})"
                          for a in alphas)
        print(f"  story r{r}: base_len={base_len} half={half} cont0={cont[0.0]}  {deltas}")
    return rows


def summarize_story(rows, alphas):
    """Mean continuation length per alpha and mean signed delta vs alpha 0."""
    out = {}
    c0 = np.array([r["cont"][0.0] for r in rows], float)
    out[0.0] = dict(mean_len=float(c0.mean()), mean_delta=0.0, n=len(rows))
    for a in alphas:
        ca = np.array([r["cont"][a] for r in rows], float)
        out[a] = dict(mean_len=float(ca.mean()),
                      mean_delta=float((ca - c0).mean()), n=len(rows))
    return out


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
    ap.add_argument("--study", choices=["composite", "story"], required=True)
    ap.add_argument("--source-run-dir", default=os.path.join(cfg.out_dir, cfg.run_name))
    ap.add_argument("--directions", default=None)
    ap.add_argument("--layer", type=int, default=None,
                    help="hidden-state index (default: best_layer in directions.npz)")
    ap.add_argument("--frac", type=float, default=0.4)
    ap.add_argument("--alphas", type=float, nargs="+", default=None,
                    help="composite: negatives (default -0.5 -1.0); story: signed (default -0.5 0.5)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--run-name", default="csat_transfer")
    ap.add_argument("--layers-attr", default=None)
    # composite
    ap.add_argument("--env", choices=["parabola", "sine", "coupling"], default="sine")
    ap.add_argument("--n-rollouts", type=int, default=12)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--branch-extra", type=int, default=10,
                    help="max turns a steered branch may run past the baseline submit "
                         "(bounds compute on coupling/parabola)")
    # story
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--prompt", default=None, help="fix a single story prompt")
    args = ap.parse_args()

    if args.model:
        cfg.model_name = args.model
    directions = args.directions or os.path.join(args.source_run_dir, "directions.npz")
    if not os.path.exists(directions):
        raise SystemExit(f"{directions} not found; run direction_extract.py first.")
    layer = args.layer
    if layer is None:
        z = np.load(directions, allow_pickle=True)
        layer = int(z["best_layer"])
        print(f"[transfer] using saved best_layer = {layer}")
    if layer < 1:
        raise SystemExit("--layer must be >= 1.")

    steer_vec, info = build_steering_vector(layer, directions, args.source_run_dir, frac=args.frac)
    print(f"[transfer] layer {layer}  |steer|={info['steer_norm']:.2f} "
          f"({info['frac']:.0%} of token norm {info['mean_token_norm']:.2f})")

    agent = ModelAgent(cfg)

    if args.study == "composite":
        alphas = args.alphas if args.alphas is not None else [-0.5, -1.0]
        cfg.env_kind = args.env
        if args.env == "sine":
            cfg.n_obj = 1
        cfg.capture = False
        cfg.run_name = f"{args.run_name}_composite_{args.env}_L{layer}"
        out_run_dir = cfg.run_dir()
        seeds = list(range(cfg.seed_start, cfg.seed_start + args.n_rollouts))
        results = []
        for case_id, seed in enumerate(seeds):
            for rep in range(args.repeats):
                idx = case_id * args.repeats + rep
                print(f"[composite] case {case_id} rep {rep} (seed {seed})")
                results.append(run_composite(cfg, agent, steer_vec, layer, seed,
                                             alphas, out_run_dir, idx,
                                             branch_extra=args.branch_extra))
        plot_composite(results, alphas, os.path.join(out_run_dir, "composite_margins.png"))
        print("[done] composite study complete.")
        return

    # story
    alphas = args.alphas if args.alphas is not None else [-0.5, 0.5]
    prompts = [args.prompt] if args.prompt else DEFAULT_STORY_PROMPTS
    cfg.run_name = f"{args.run_name}_story_L{layer}"
    out_run_dir = cfg.run_dir()
    rows = run_story_study(agent, steer_vec, layer, prompts, alphas,
                           n_repeats=args.n_repeats, max_new=args.max_new)
    with open(os.path.join(out_run_dir, "story_rows.json"), "w") as f:
        json.dump(rows, f, indent=2)
    summ = summarize_story(rows, alphas)
    print("\n[story] mean continuation length per alpha (delta vs unsteered):")
    for a in [0.0] + list(alphas):
        s = summ[a]
        print(f"   alpha {a:+.1f}: mean_len={s['mean_len']:.1f}  mean_delta={s['mean_delta']:+.1f}")
    plot_story(rows, alphas, os.path.join(out_run_dir, "story_lengths.png"))
    print("[done] story study complete.")


if __name__ == "__main__":
    main()