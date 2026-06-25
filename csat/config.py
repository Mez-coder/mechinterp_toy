"""Run / model / env configuration."""
from __future__ import annotations
from dataclasses import dataclass
import os


@dataclass
class Config:
    # --- model ---
    model_name: str = "Qwen/Qwen3.5-9B"
    device: str = "cuda"
    temperature: float = 0.7
    max_new_tokens: int = 4096
    enable_thinking: bool = False
    # --- activation capture (see recorder.py) ---
    capture: bool = True
    capture_tokens: str = "lastk"        # 'decision' | 'lastk' | 'assistant'
    capture_last_k: int = 30
    capture_dtype: str = "bfloat16"      # lossless for a bf16 model (needs ml_dtypes)
    # --- rollout ---
    out_dir: str = "runs"
    run_name: str = "csat_v1"
    seed_start: int = 0
    n_rollouts: int = 200         # number of DISTINCT cases (landscapes)
    repeats_per_case: int = 1     # R: independent rollouts per case (same landscape, fresh sampling)
    max_turns: int = 30
    # --- env ---
    # --- env selection ---
    env_kind: str = "coupling"        # "coupling" | "parabola" | "sine"
    # --- parabola env (used when env_kind == "parabola") ---
    par_a: float = 0.9                 # floor tilt: 0 = flat floor, >0 = gentle central gradient
    par_b: float = 1.2                  # wall steepness (quartic)
    par_z_pass_frac: float = 0.8        # pass threshold as fraction of z at the 0.5 start

    ## coupling --
    n_obj: int = 3
    beta: float = 4.0
    grid: float = 0.0                   # 0 = continuous; 0.1 = snap weights to a 0.1 grid
    case_jitter: float = 0.1             # per-case randomisation of m0/G/C (0 = fixed)
    optimum_samples: int = 500000         # MC samples for the per-case ground-truth optimum
    m0: object = None                    # optional explicit arrays; else env defaults
    G: object = None
    C: object = None

    def run_dir(self):
        d = os.path.join(self.out_dir, self.run_name)
        os.makedirs(d, exist_ok=True)
        return d