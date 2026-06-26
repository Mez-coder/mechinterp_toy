# protontherapy — 2D proton RT planning environment (MI testbed)

A physically-grounded 2D proton treatment-planning sandbox where a vision LLM
controls **beam angles** and **per-beam weights**; the backend does real
ray-tracing → Bethe-Bloch dose deposition → Bragg-peak filtering → pencil-beam
optimisation → DVH / PASS-FAIL. Built as the testbed for the "stop vs
keep-optimising" steering-vector study (the `[SET …]` / `[SUBMIT]` DSL, last-k
activation capture, and counterfactual-replay harness carry over from the 1D /
story experiments).

## Action space
Each turn the model sees an **image** (clean phantom at case start; dose wash
thereafter) plus a DVH table, and emits one action:

```
[SET a1=w1, a2=w2, ...]   place up to max_beams beams (gantry degrees = weight)
                          and re-optimise; this REPLACES the whole plan
[SUBMIT]                  finalise the current plan
```

## Pipeline (per SET)
1. **Trace** each angle (`tracer.py`): parallel pencil array entering the
   phantom circle toward centre; geometry-derived energy band (Emin/Emax both
   from the target's near/far depth, ~3 MeV steps); vectorised Bethe-Bloch march
   (2 mm steps, density = mean of step endpoints); per-energy **range
   straggling**; **bilinear (area) splatting** of deposited energy (angle-robust
   — replaces nearest-voxel, which striped on diagonals); optional lateral
   penumbra. Keep a ray iff its **Bragg peak voxel ∈ target**. Cached per angle.
2. **Optimise** (`optimize.py`): OAR-blind inner projected-gradient ("SGD")
   minimising masked MSE to a uniform Rx.
   * **SFO (default):** each beam solved *independently* to uniform target dose,
     so global weights `g_i` are a clean 2nd DOF that trade only OAR dose
     (geometric) — coverage is preserved for any simplex `g`.
   * **MFO (`opt_mode='mfo'`):** joint solve; weights then perturb coverage.
3. **Combine** beams with normalised global weights; compute **true DVH**
   (`dvh.py`) and PASS/FAIL; render the dose wash (`render.py`).

## Physics honesty notes
* Stopping power is a real (simplified) **Bethe-Bloch** form; the constant is
  self-calibrated so R(150 MeV)=157 mm in water, and the range table is built by
  integrating that same curve, so marched Bragg peaks land where the geometry
  predicts. Density-scaled, so heterogeneous tissue drops in via `density.py`.
* Range straggling and lateral penumbra are physical broadenings (a real proton
  Bragg peak is not a delta); set `lateral_sigma_mm=0` for the exact spec
  thin-ray + equal-boundary-split behaviour.
* The simplified 2D thin-pencil model with MSE objective realistically caps
  target D98 around ~92–94 %; the default `d98_floor_pct=92` reflects that and
  is a config knob.

## MI harness
* `recorder.py` — capture residual stream at the last **k=30** decision tokens,
  all layers, lossless bf16 storage.
* `steering.py` — `build_steering_vector` = mean(SUBMIT decision residual) −
  mean(SET decision residual) per layer; `steering_active(model, block_idx, vec,
  alpha)` forward-hook (same signature as the 1D/story studies).
* `replay.py` — counterfactual branch: roll back to the SUBMIT turn, drop it,
  re-generate under −alpha·v ("keep optimising") / +alpha·v ("stop").

## Usage
```bash
# play rollouts with the vision model (captures activations)
python -m protontherapy.run --run-name r0 --n-rollouts 100

# debug the loop with no model
python -m protontherapy.run --scripted "[SET 30=1,120=1,210=0.7,300=1]" "[SUBMIT]"

# build the steering vector, then branch under it
python -m protontherapy.run --build-vector --run-name r0
python -m protontherapy.run --replay runs/r0/rollout_0000 --layer 18 --alpha -1.0 --run-name r0
```

## Files
`density.py` material grid · `geometry.py` case generator · `stopping.py`
Bethe-Bloch + range table · `tracer.py` ray-trace influence builder ·
`optimize.py` SFO/MFO solver · `dvh.py` metrics · `render.py` images ·
`env.py` environment · `dsl.py` action parser · `prompts.py` ·
`agents.py` (Human/Scripted/Model) · `recorder.py` · `steering.py` ·
`replay.py` · `rollout.py` · `run.py` · `config.py`.

Defaults: 150×150 grid @ 2 mm, phantom R=150 mm, ≤4 beams, SFO, k=30.
