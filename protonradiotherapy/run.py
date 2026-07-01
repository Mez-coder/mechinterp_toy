"""Entry point.

  # play N rollouts with the vision model, capturing activations
  python -m protontherapy.run --run-name r0 --n-rollouts 100

  # debug the loop with no model (canned actions)
  python -m protontherapy.run --scripted "[SET 30=1,120=1,210=0.7,300=1]" "[SUBMIT]"

  # roleplay it yourself
  python -m protontherapy.run --human --n-rollouts 1

  # counterfactual branch at the SUBMIT turn using the SOURCE (parabola) vector
  python -m protontherapy.run --replay runs/r0/rollout_0000 \
      --source-run-dir runs/csat --layer 22 --alpha -1.0

The cross-environment steering STUDY (alpha sweep + projection) lives in
protontherapy.transfer; the source vector is NEVER rebuilt here -- it is loaded
from <source-run-dir>/directions.npz exactly as the 1D/story studies do.
"""
from __future__ import annotations
import argparse, os

from .config import RunConfig
from .rollout import run_rollout
from .agents import HumanAgent, ScriptedAgent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="run0")
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--n-rollouts", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--opt-mode", choices=["sfo", "mfo"], default="sfo")
    ap.add_argument("--model", default=None)
    ap.add_argument("--human", action="store_true")
    ap.add_argument("--scripted", nargs="*", default=None)
    # counterfactual branch with the source vector
    ap.add_argument("--replay", default=None, help="rollout dir to branch at SUBMIT")
    ap.add_argument("--source-run-dir", default="runs/csat",
                    help="SOURCE run dir holding directions.npz + token-norm")
    ap.add_argument("--directions", default=None)
    ap.add_argument("--layer", type=int, default=None, help="hidden-state index of the vector")
    ap.add_argument("--alpha", type=float, default=-1.0)
    ap.add_argument("--frac", type=float, default=None)
    ap.add_argument("--layers-attr", default=None,
                    help="dotted path to decoder ModuleList if auto-detect picks wrong")
    args = ap.parse_args()

    cfg = RunConfig(run_name=args.run_name, out_root=args.out_root,
                    n_rollouts=args.n_rollouts, seed_start=args.seed_start,
                    max_turns=args.max_turns, opt_mode=args.opt_mode,
                    source_run_dir=args.source_run_dir)
    if args.model:
        cfg.model_name = args.model
    if args.directions:
        cfg.directions_path = args.directions
    if args.layer is not None:
        cfg.steer_layer = args.layer
    if args.layers_attr:
        cfg.layers_attr = args.layers_attr
    os.makedirs(cfg.run_dir(), exist_ok=True)
    cfg.save(os.path.join(cfg.run_dir(), "config.json"))

    if args.replay:
        from .agents import ModelAgent
        from .replay import replay_branch
        agent = ModelAgent(cfg)
        snap = replay_branch(cfg, agent, args.replay, layer=args.layer,
                             alpha=args.alpha, frac=args.frac,
                             layers_attr=args.layers_attr)
        print("branch submission:", snap["angles"], snap["global_w"],
              "passes=", snap["passes"])
        return

    if args.human:
        agent = HumanAgent()
    elif args.scripted is not None:
        agent = ScriptedAgent(args.scripted)
    else:
        from .agents import ModelAgent
        agent = ModelAgent(cfg)

    for i in range(cfg.n_rollouts):
        seed = cfg.seed_start + i
        snap = run_rollout(cfg, agent, i, seed)
        print(f"[rollout {i}] seed={seed} angles={snap['angles']} "
              f"passes={snap['passes']} coverage_ok={snap['coverage_ok']}")


if __name__ == "__main__":
    main()