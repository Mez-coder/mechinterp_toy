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
    max_new_tokens: int = 528
    # --- activation capture (see recorder.py) ---
    capture: bool = True
    capture_tokens: str = "lastk"        # 'decision' | 'lastk' | 'assistant'
    capture_last_k: int = 5
    capture_dtype: str = "bfloat16"      # lossless for a bf16 model (needs ml_dtypes)
    # --- rollout ---
    out_dir: str = "runs"
    run_name: str = "csat"
    seed_start: int = 0
    n_rollouts: int = 200
    max_turns: int = 12
    # --- env ---
    n_obj: int = 3
    beta: float = 4.0
    case_jitter: float = 0.1             # per-case randomisation of m0/G/C (0 = fixed)
    optimum_samples: int = 50000         # MC samples for the per-case ground-truth optimum
    m0: object = None                    # optional explicit arrays; else env defaults
    G: object = None
    C: object = None

    def run_dir(self):
        d = os.path.join(self.out_dir, self.run_name)
        os.makedirs(d, exist_ok=True)
        return d