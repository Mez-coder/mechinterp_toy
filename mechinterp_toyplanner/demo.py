"""Demo: build/cache the environment, generate a case, output the baseline plan
and the OAR constraints the model will receive. No optimisation past baseline --
the model (in the agent harness) does everything from here.

Run:  uv run --with numpy --with scipy python demo.py
"""
import json, time
from protoplan.env import PlanningEnv

env = PlanningEnv(nx=150, ny=150, voxel_mm=2.0, n_energy_layers=24,
                  spot_half_extent_mm=60.0, cov_tol_pct=2.0, cache_dir="cache")

fb = env.reset(seed=3)

print(f"baseline CTV coverage: D98={env.ctv_floor['D98_base']/env.Rx*100:.1f}% "
      f"D2={env.ctv_floor['D2_base']/env.Rx*100:.1f}%  "
      f"(coverage floor D98>={env.ctv_floor['D98_gy']/env.Rx*100:.1f}%)\n")

print("OAR constraints the model must satisfy (sampled below baseline):")
for o in env.objectives:
      if o['structure'].startswith('OAR'):
            b = env.baseline['oar_val'][o['structure']]
            print(f"  {o['structure']:5s} {o['metric']:5s}  baseline={b:5.1f} Gy  "
              f"-> limit={o['limit_gy']:5.1f} Gy")
      from protoplan.plotting import plot_dose, plot_dvh
      env.reset(seed=3)
      plot_dose(env); plot_dvh(env)   # or pass ax= to put them side by side
      import matplotlib.pyplot as plt; plt.show()


# This is exactly what the model receives & edits (objectives.json).
print("\nobjectives.json handed to the model:")
print(json.dumps(env.get_objectives(), indent=2))
