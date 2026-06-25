"""Entry point. Build the (cached) geometry once, then run N rollouts.

  uv run --with numpy --with scipy python run.py --human
  uv run --with numpy --with scipy --with torch --with transformers python run.py
  python run.py --human --rollouts 2 --max-turns 8 --run-name try1
"""
from __future__ import annotations
import argparse, os, time
from config import RunConfig
from protoplan.physics import Geometry2D
from harness.rollout import run_rollout
from harness.agents import HumanAgent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", action="store_true", help="roleplay the model yourself")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--rollouts", type=int, default=None)
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--seed-start", type=int, default=None)
    ap.add_argument("--no-capture", action="store_true")
    ap.add_argument("--config", default=None, help="load a saved RunConfig json")
    args = ap.parse_args()

    cfg = RunConfig.load(args.config) if args.config else RunConfig()
    if args.run_name: cfg.run_name = args.run_name
    if args.rollouts is not None: cfg.n_rollouts = args.rollouts
    if args.max_turns is not None: cfg.max_turns = args.max_turns
    if args.seed_start is not None: cfg.seed_start = args.seed_start
    if args.no_capture or args.human: cfg.capture = False

    os.makedirs(cfg.run_dir(), exist_ok=True)
    cfg.save(os.path.join(cfg.run_dir(), "config.json"))

    print(f"building/loading geometry ({cfg.nx}x{cfg.ny}@{cfg.voxel_mm}mm) ...")
    t0 = time.time()
    geom = Geometry2D(nx=cfg.nx, ny=cfg.ny, voxel_mm=cfg.voxel_mm,
                      n_energy_layers=cfg.n_energy_layers,
                      spot_half_extent_mm=cfg.spot_half_extent_mm,
                      cache_dir=cfg.cache_dir)
    print(f"  ready ({geom.n_spots} spots, {time.time()-t0:.1f}s)")

    if args.human:
        agent = HumanAgent()
    else:
        from harness.agents import ModelAgent
        print(f"loading model {cfg.model_name} ...")
        agent = ModelAgent(cfg)

    for idx in range(cfg.n_rollouts):
        print(f"\n########## rollout {idx} (seed {cfg.seed_start+idx}) ##########")
        res = run_rollout(cfg, geom, idx, agent)
        tag = "FORCED submit (budget)" if res["forced"] else \
              ("submitted" if res["submitted"] else "?")
        print(f"  -> {tag}; logs in {res['dir']}")

    print(f"\ndone. run dir: {cfg.run_dir()}")


if __name__ == "__main__":
    main()
