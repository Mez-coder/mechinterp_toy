"""Run configuration for the 2D proton angle/weight environment. Edit here."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json, os
from typing import Optional


@dataclass
class RunConfig:
    # run
    run_name: str = "run0"
    out_root: str = "runs"
    n_rollouts: int = 1
    seed_start: int = 0
    max_turns: int = 30                 # hard budget; runaway -> forced submit

    # geometry / phantom
    nx: int = 150
    ny: int = 150
    voxel_mm: float = 2.0
    radius_mm: float = 150.0
    Rx: float = 1.0                     # prescription (relative; doses are /Rx)

    # beams / action space
    max_beams: int = 4

    # physics / tracer
    opt_mode: str = "sfo"              # 'sfo' (default) | 'mfo'
    march_mm: float = 2.0
    spacing_mm: float = 2.0
    energy_step_mev: float = 3.0
    lateral_sigma_mm: float = 15.0      # 0 -> exact thin-ray + boundary-split
    inner_iters: int = 400

    # difficulty / scoring
    n_oar: Optional[int] = 6       # fixed OAR count; None -> random 2-4
    d98_floor_pct: float = 92.0
    d2_ceil_pct: float = 108.0
    constraint_tighten_frac: float = 0.0   # shift OAR limits BELOW baseline (harder)
    baseline_beams: int = 12

    # model (ignored in --human / --scripted)
    model_name: str = "Qwen/Qwen3.5-9B"
    device: str = "auto"
    temperature: float = 0.7
    max_new_tokens: int = 4096
    enable_thinking: bool = False

    # activation capture
    capture: bool = True
    capture_tokens: str = "lastk"      # 'decision' | 'lastk' | 'assistant'
    capture_last_k: int = 30           # k=30 decision run-up (changed from 50)
    capture_dtype: str = "bfloat16"    # 'bfloat16' (lossless) | 'float32' | 'float16'

    # cross-environment steering / transfer (REUSE the source vector; never rebuild)
    source_run_dir: str = "runs/csat_v0"  # SOURCE (e.g. parabola) run: directions + token-norm
    directions_path: Optional[str] = None   # default <source_run_dir>/directions.npz
    steer_layer: Optional[int] = 22    # hidden-state index of the extracted vector
    steer_frac: float = 0.4            # magnitude = frac * source mean-token-norm
    layers_attr: Optional[str] = None  # dotted path to decoder ModuleList if auto-detect is wrong

    def run_dir(self):
        return os.path.join(self.out_root, self.run_name)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))