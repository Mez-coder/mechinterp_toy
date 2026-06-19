"""Entry point.

    python -m csat.run --human            # roleplay the model yourself (no GPU)
    python -m csat.run --n 200            # run 200 model rollouts
    python -m csat.run --n 50 --model google/gemma-2-9b-it

Logs transcripts + (for the model) decision-token activations under
runs/<run_name>/rollout_XXXX/.
"""
from __future__ import annotations
import argparse
from .config import Config
from .agents import HumanAgent, ModelAgent
from .rollout import run_rollout
from . import io_utils as io
from .case_spread import pick_spread_seeds
from .rollout import build_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", action="store_true", help="roleplay the model yourself")
    ap.add_argument("--n", type=int, default=None, help="number of rollouts")
    ap.add_argument("--model", type=str, default=None, help="HF model id")
    args = ap.parse_args()

    cfg = Config()
    if args.n is not None:
        cfg.n_rollouts = args.n
    if args.model:
        cfg.model_name = args.model
    if args.human:
        cfg.capture = False

    agent = HumanAgent() if args.human else ModelAgent(cfg)
    start = io.next_rollout_idx(cfg.run_dir())
    if start:
        print(f"resuming: {start} completed rollouts found; continuing from idx {start}")
    seeds = pick_spread_seeds(cfg, n_want=cfg.n_rollouts, build_env=build_env,
                          seed_start=cfg.seed_start)
    for idx, seed in enumerate(seeds):
        r = run_rollout(cfg, idx, agent, seed=seed)          # <-- capture into r
        print(f"rollout {idx:04d}: submitted={r['submitted']} forced={r['forced']} "
            f"first_pass_turn={r['first_pass_turn']}")


if __name__ == "__main__":
    main()
