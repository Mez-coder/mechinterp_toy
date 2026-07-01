"""Cross-environment steering verification in the proton env, reusing the SOURCE
(parabola) SUBMIT-minus-SET vector from directions.npz -- never rebuilt here.

Two tests (same idea as the 1D/story transfer_studies):

  composite (Test 1): for each seed, run an unsteered baseline to SUBMIT; then
    branch at that SUBMIT turn and re-run it onward under each alpha. alpha<0
    -> toward SET (keep optimising / push OAR doses lower), alpha>0 -> toward
    SUBMIT (stop earlier). We track a per-turn satisficing margin and plot how
    the branch diverges from baseline.

  project (Test 2): run unsteered proton rollouts WITH capture, then project each
    turn's last-k decision activations onto the SOURCE SUBMIT-SET axis
    (SET=-1, SUBMIT=+1). If the axis transfers, the projection should climb as
    the model approaches its own SUBMIT -- in a different environment.

  python -m protontherapy.transfer --study composite --source-run-dir runs/csat \
      --layer 22 --alphas -1.0 -0.5 0.0 0.5 1.0 --n-rollouts 12
  python -m protontherapy.transfer --study project --source-run-dir runs/csat \
      --layer 22 --n-rollouts 12
"""
from __future__ import annotations
import os, json, copy, glob, argparse
import numpy as np

from .config import RunConfig
from .dsl import parse_action
from .prompts import system_prompt_for, render_case, render_feedback_for
from .steering import build_steering_vector, steering_active, load_direction
from .rollout import build_env, run_rollout
from . import io_utils as io


# --------------------------------------------------------------------------- #
# satisficing margin: min signed headroom across all constraints (Rx units)
# --------------------------------------------------------------------------- #
def _margin(env, scale=100.0):
    """Satisficing margin in % of Rx: how far the WORST OAR sits below its limit
    (positive = all OARs within limits, with headroom). Coverage is assumed."""
    head = [(r["limit"] - r["value"]) for r in env.get_feedback()]
    worst = float(min(head)) * scale if head else 0.0
    return worst, worst


def _advance(cfg, agent, env, messages, start_turn, snapshot_each=False,
             stop_after_turn=None, tag=None, imgdir=None):
    """Run the SET/SUBMIT loop from `start_turn` on (env, messages) IN PLACE.
    No capture (steering is applied by the caller). If `imgdir` is given, each
    SET turn renders the dose wash there and feeds it to the model (the VLM needs
    the image to reason); otherwise feedback is text-only."""
    records, margins, snaps = [], [], {}
    submitted, submit_turn = False, None
    last = cfg.max_turns if stop_after_turn is None else min(cfg.max_turns, stop_after_turn)
    for turn in range(start_turn, last + 1):
        if tag:
            print(f"      {tag} turn {turn}/{last}", end="\r", flush=True)
        if snapshot_each:
            snaps[turn] = (copy.deepcopy(env), list(messages))
        text, _ = agent.act(messages, capture_path=None)
        action = parse_action(text)
        messages.append({"role": "assistant", "content": text})
        rec = dict(turn=turn, action=action.kind, angles=action.angles,
                   weights=action.weights, response=text)
        if action.kind == "submit":
            env.submit(); mar, oar = _margin(env)
            rec.update(passes=bool(env.plan_passes()), margin=mar, oar_headroom=oar)
            margins.append((turn, mar)); records.append(rec)
            submitted, submit_turn = True, turn
            break
        if action.kind == "set":
            env.set_plan(action.angles, action.weights)
            mar, oar = _margin(env)
            rec.update(passes=bool(env.plan_passes()), margin=mar, oar_headroom=oar)
            margins.append((turn, mar)); records.append(rec)
            img = None
            if imgdir:
                os.makedirs(imgdir, exist_ok=True)
                img = os.path.join(imgdir, f"dose_turn_{turn:02d}.png")
                env.render_dose(img, turn=turn)
            messages.append({"role": "user",
                             "content": render_feedback_for(env, env.get_feedback(),
                                                            env.angles, env.global_w,
                                                            turn, cfg.max_turns),
                             "image_path": img})
        else:
            records.append(rec)
            messages.append({"role": "user",
                             "content": "Reply with [SET a=w, ...] or [SUBMIT].",
                             "image_path": None})
    return dict(records=records, margins=margins, snaps=snaps,
                submitted=submitted, submit_turn=submit_turn,
                n_turns=(submit_turn or last), snapshot=env.snapshot())


# --------------------------------------------------------------------------- #
# Test 1: composite branch at the baseline SUBMIT turn under an alpha sweep
# --------------------------------------------------------------------------- #
def run_composite(cfg, agent, steer_vec, layer, seed, alphas, out_run_dir, idx,
                  layers_attr=None, branch_extra=10):
    block_idx = layer - 1
    env = build_env(cfg); env.reset(seed=seed)
    rdir = io.rollout_dir(out_run_dir, idx)
    io.save_case(rdir, env, seed)                       # writes phantom.png too
    imgdir = os.path.join(rdir, "images")
    messages = [{"role": "system", "content": system_prompt_for(cfg)},
                {"role": "user", "content": render_case(env, cfg.max_turns),
                 "image_path": os.path.join(imgdir, "phantom.png")}]
    base = _advance(cfg, agent, env, messages, 1, snapshot_each=True, tag="base",
                    imgdir=imgdir)

    realizations = {0.0: base}
    T = base["submit_turn"]
    if base["submitted"] and T is not None:
        env_b, msgs_b = base["snaps"][T]
        stop_at = min(cfg.max_turns, T + branch_extra)
        for a in alphas:
            if a == 0.0:
                continue
            e2, m2 = copy.deepcopy(env_b), list(msgs_b)
            with steering_active(agent.model, block_idx, steer_vec, a, layers_attr):
                realizations[a] = _advance(cfg, agent, e2, m2, T,
                                           stop_after_turn=stop_at, tag=f"a{a:+.1f}",
                                           imgdir=os.path.join(rdir, f"branch_a{a:+.2f}"))
            print(f"    alpha {a:+.1f}: turns {T}..{realizations[a]['n_turns']} "
                  f"submitted={realizations[a]['submitted']}")
    else:
        print(f"  [composite] rollout {idx:04d} baseline forced; only alpha 0 saved.")

    prefix = [(t, m) for (t, m) in base["margins"] if T is None or t < T]
    trajectories, summary = {}, {}
    for a, res in realizations.items():
        traj = res["margins"] if a == 0.0 else prefix + res["margins"]
        with open(os.path.join(rdir, f"transcript_a{a:+.2f}.jsonl"), "w") as f:
            for rec in (res["records"] if a == 0.0 else res["records"]):
                f.write(json.dumps(rec) + "\n")
        trajectories[a] = traj
        summary[a] = dict(submit_turn=res["submit_turn"], n_turns=res["n_turns"],
                          submitted=res["submitted"],
                          final_margin=(traj[-1][1] if traj else None),
                          passes=bool(res["snapshot"]["passes"]))
    with open(os.path.join(rdir, "composite_summary.json"), "w") as f:
        json.dump(dict(idx=idx, seed=seed, baseline_submit_turn=T, summary=summary),
                  f, indent=2)
    return dict(idx=idx, seed=seed, trajectories=trajectories, summary=summary, T=T)


def plot_composite(results, alphas, out_png, ncol=3):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = len(results); ncol = min(ncol, max(1, n)); nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow), squeeze=False)
    colors = {0.0: "black"}
    palette = ["tab:orange", "tab:red", "tab:purple", "tab:brown", "tab:green"]
    for k, a in enumerate([a for a in alphas if a != 0.0]):
        colors[a] = palette[k % len(palette)]
    for ax in axes.flat:
        ax.axis("off")
    for i, res in enumerate(results):
        ax = axes[i // ncol][i % ncol]; ax.axis("on")
        for a in [0.0] + [x for x in alphas if x != 0.0]:
            traj = res["trajectories"].get(a)
            if not traj:
                continue
            xs = [t for t, _ in traj]; ys = [m for _, m in traj]
            ax.plot(xs, ys, "-o", ms=3, color=colors.get(a, "gray"),
                    label=f"a={a:+.1f}", alpha=0.9 if a == 0.0 else 0.8)
        if res["T"]:
            ax.axvline(res["T"], color="gray", ls=":", lw=1)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"seed {res['seed']}"); ax.grid(alpha=0.3)
        ax.set_xlabel("turn"); ax.set_ylabel("worst-OAR margin (%Rx)")
        if i == 0:
            ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_png, dpi=130)
    print(f"[composite] wrote {out_png}")


# --------------------------------------------------------------------------- #
# Test 2: project the proton env's per-turn activations onto the SOURCE axis
# --------------------------------------------------------------------------- #
def _turn_projection(npz_path, layer, set_v, sub_v):
    with np.load(npz_path, allow_pickle=False) as z:
        acts = np.asarray(z["acts"]).astype(np.float32)        # (L+1, n_pos, d)
    if layer >= acts.shape[0]:
        return None
    v = acts[layer].mean(axis=0)                               # pool last-k tokens
    mid = (set_v + sub_v) / 2.0
    dirv = sub_v - set_v
    denom = float(dirv @ dirv) / 2.0 + 1e-12                   # SET=-1, SUBMIT=+1
    return float(((v - mid) @ dirv) / denom)


def project_runs(run_dir, directions_path, layer, out_png, max_lines=12):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    set_v, sub_v = load_direction(directions_path, layer)
    dirs = sorted(glob.glob(os.path.join(run_dir, "rollout_*")))[:max_lines]
    fig, ax = plt.subplots(figsize=(8, 5)); n = 0
    for rd in dirs:
        tpath = os.path.join(rd, "transcript.jsonl")
        if not os.path.exists(tpath):
            continue
        kinds = {}
        for line in open(tpath):
            r = json.loads(line)
            if r.get("action") in ("set", "submit"):
                kinds[int(r["turn"])] = r["action"]
        xs, ys, sub_t = [], [], None
        for t in sorted(kinds):
            npz = os.path.join(rd, "activations", f"turn_{t:02d}.npz")
            if not os.path.exists(npz):
                continue
            p = _turn_projection(npz, layer, set_v, sub_v)
            if p is None:
                continue
            xs.append(t); ys.append(p)
            if kinds[t] == "submit":
                sub_t = t
        if not xs:
            continue
        line, = ax.plot(xs, ys, "-o", ms=3, alpha=0.85,
                        label=os.path.basename(rd).split("_")[-1])
        if sub_t is not None:
            ax.plot([sub_t], [ys[xs.index(sub_t)]], "*", ms=15, color=line.get_color())
        n += 1
    ax.axhline(1, color="g", ls="--", lw=1); ax.axhline(-1, color="b", ls="--", lw=1)
    ax.text(0.01, 0.98, "SUBMIT_all = +1", color="g", transform=ax.transAxes, va="top", fontsize=8)
    ax.text(0.01, 0.02, "SET_all = -1", color="b", transform=ax.transAxes, va="bottom", fontsize=8)
    ax.set_xlabel("turn"); ax.set_ylabel("projection onto source (SUBMIT-SET) axis")
    ax.set_title(f"proton env projected on source axis @ L{layer}")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"[project] wrote {out_png}  ({n} rollouts, layer {layer})")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", choices=["composite", "project"], default="composite")
    ap.add_argument("--run-name", default="proton_transfer")
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--source-run-dir", default="runs/csat")
    ap.add_argument("--directions", default=None)
    ap.add_argument("--layer", type=int, default=None,
                    help="hidden-state index (default: steer_layer/best_layer)")
    ap.add_argument("--frac", type=float, default=0.4)
    ap.add_argument("--alphas", type=float, nargs="+", default=[-1.0, -0.5, 0.0, 0.5, 1.0])
    ap.add_argument("--n-rollouts", type=int, default=12)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--layers-attr", default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    cfg = RunConfig(run_name=args.run_name, out_root=args.out_root,
                    n_rollouts=args.n_rollouts, seed_start=args.seed_start,
                    source_run_dir=args.source_run_dir, steer_frac=args.frac,
                    layers_attr=args.layers_attr)
    if args.model:
        cfg.model_name = args.model
    directions = args.directions or os.path.join(args.source_run_dir, "directions.npz")
    if not os.path.exists(directions):
        raise SystemExit(f"{directions} not found; extract the source vector first.")
    layer = args.layer if args.layer is not None else cfg.steer_layer
    if layer is None:
        layer = int(np.load(directions, allow_pickle=True)["best_layer"])
    os.makedirs(cfg.run_dir(), exist_ok=True)

    from .agents import ModelAgent
    agent = ModelAgent(cfg)

    if args.study == "project":
        # unsteered rollouts WITH capture, then project onto the source axis
        for i in range(cfg.n_rollouts):
            run_rollout(cfg, agent, i, cfg.seed_start + i)
        out_png = os.path.join(cfg.run_dir(), f"projection_L{layer}.png")
        project_runs(cfg.run_dir(), directions, layer, out_png)
        return

    steer_vec, info = build_steering_vector(layer, directions, args.source_run_dir, args.frac)
    print(f"[transfer] layer {layer} |steer|={info['steer_norm']:.2f} "
          f"({info['frac']:.0%} of source token norm {info['mean_token_norm']:.2f})")
    results = []
    for i in range(cfg.n_rollouts):
        results.append(run_composite(cfg, agent, steer_vec, layer,
                                     cfg.seed_start + i, args.alphas,
                                     cfg.run_dir(), i, layers_attr=args.layers_attr))
    plot_composite(results, args.alphas, os.path.join(cfg.run_dir(), f"composite_L{layer}.png"))
    print("[done] composite transfer complete.")


if __name__ == "__main__":
    main()