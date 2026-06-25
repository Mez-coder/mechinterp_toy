"""Run configuration. Edit here, not in code."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from typing import Optional


@dataclass
class RunConfig:
    # run
    run_name: str = "run0"
    out_root: str = "runs"
    n_rollouts: int = 5
    seed_start: int = 0
    max_turns: int = 30            # hard budget; runaway -> forced submit

    # environment / geometry
    Rx: float = 50.4
    nx: int = 150
    ny: int = 150
    voxel_mm: float = 2.0
    n_energy_layers: int = 24
    spot_half_extent_mm: float = 60.0
    cache_dir: str = "cache"

    # ---- difficulty knobs -------------------------------------------------
    n_oar: Optional[int] = 4        # fixed OAR count; None = random 2-4
    oar_overlap_bias: float = 0.0      # 0 = spread; ->1 pulls OARs onto the CTV
    cov_tol_pct: float = 1.0           # coverage you may trade for sparing (smaller = harder)
    constraint_sigma_frac: float = 0.1 # spread of sampled OAR limit (fraction of baseline)
    constraint_tighten_frac: float = 0.1  # mean shift of limit BELOW baseline (the difficulty dial)

    # model (ignored in --human mode)
    model_name: str = "Qwen/Qwen3.5-9B"
    device: str = "auto"
    temperature: float = 0.7
    max_new_tokens: int = 4096

    # activation capture
    capture: bool = True
    capture_tokens: str = "lastk"       # 'decision' | 'lastk' | 'assistant'
    capture_last_k: int = 50             # tokens kept when capture_tokens=='lastk'
    capture_dtype: str = "float16"      # 'bfloat16' (lossless, needs ml_dtypes) | 'float32' | 'float16'

    def run_dir(self):
        import os
        return os.path.join(self.out_root, self.run_name)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))