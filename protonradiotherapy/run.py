"""Entry point.

  # play N rollouts with the vision model, capturing activations
  python -m protontherapy.run --run-name r0 --n-rollouts 100

  # debug the loop with no model (canned actions)
  python -m protontherapy.run --scripted "[SET 30=1,120=1,210=0.7,300=1]" "[SUBMIT]"

  # roleplay it yourself
  python -m protontherapy.run --human --n-rollouts 1

  # build the SUBMIT-vs-SET steering vector from a finished run
  python -m protontherapy.run --build-vector --run-name r0

  # counterfactual branch at the SUBMIT turn under steering
  python -m protontherapy.run --replay runs/r0/rollout_0000 --layer 18 --alpha -1.0 --run-name r0
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
    ap.add_argument("--human", action="store_true")
    ap.add_argument("--scripted", nargs="*", default=None)
    ap.add_argument("--build-vector", action="store_true")
    ap.add_argument("--replay", default=None, help="rollout dir to branch")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--alpha", type=float, default=-1.0)
    ap.add_argument("--directions", default=None)
    args = ap.parse_args()

    cfg = RunConfig(run_name=args.run_name, out_root=args.out_root,
                    n_rollouts=args.n_rollouts, seed_start=args.seed_start,
                    max_turns=args.max_turns, opt_mode=args.opt_mode)
    os.makedirs(cfg.run_dir(), exist_ok=True)
    cfg.save(os.path.join(cfg.run_dir(), "config.json"))

    if args.build_vector:
        from .steering import build_steering_vector
        v = build_steering_vector(cfg.run_dir())
        print(f"built steering vector {v.shape} -> {cfg.run_dir()}/directions.npz")
        return

    if args.replay:
        from .agents import ModelAgent
        from .replay import replay_branch
        agent = ModelAgent(cfg)
        directions = args.directions or os.path.join(cfg.run_dir(), "directions.npz")
        snap = replay_branch(cfg, agent, args.replay, args.layer, args.alpha, directions)
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
