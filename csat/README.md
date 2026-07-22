# csat — Constrained-Satisficing / stop-vs-continue interpretability sandbox

A small mechanistic-interpretability harness for studying **when and how a language model decides to *stop* optimising**. The model plays a trial-and-error tuning game: it repeatedly proposes numeric weights (`SET`) and, when it judges the plan good enough, finalises it (`SUBMIT`). The central question is whether the residual stream carries a **linear "keep searching vs. stop here" direction** — one that can be read off, causally steered, and shown to transfer to unrelated tasks — and whether that direction genuinely *anticipates* the stop decision rather than merely detecting the `SUBMIT` token about to be typed.

Everything is deliberately stripped down: the "optimisation" is a deterministic function evaluated by the harness, so the *only* cognition left to the model is the stop/continue judgement. The model runs fully on-policy (it emits every token itself); the harness only truncates at a completed action line and records what happened.

---

## The task

Each rollout hands the model a hidden landscape and a two-verb DSL:

- `SET w1 w2 …` — set a weight in `[0,1]` per objective/coordinate; the harness re-evaluates and returns a **signed margin** table (margin > 0 = passing/under limit, < 0 = failing).
- `SUBMIT w1 w2 …` — finalise with those weights.

The goal is two-part: first get **every** objective passing, then push the *priority* objective's margin as high as possible without breaking the others. Every objective starts *failing* (the model must climb off a constraint-violating baseline), and pushing a weight helps that objective with diminishing returns while harming the others with accelerating cost. That split — "a passing plan is cheap, over-optimising bites" — is the satisfice-vs-overoptimise regime the project isolates and steers.

Three interchangeable environments wear the same `SET/SUBMIT` interface:

| env (`env_kind`) | file | shape | purpose |
|---|---|---|---|
| `coupling` | `coupling_env.py` | N objectives, analytic self-gain minus cross-harm | the main multi-objective satisficing task; per-case difficulty is randomised |
| `parabola` | `parabola_env.py` | convex N-D bowl, one global optimum | a single-margin task with no luck/local minima; "effort" = how far it pushes toward a known optimum |
| `sine` | `sine_env.py` | 1-D sinusoid, one minimum per period | a genuinely *different* search wearing the same interface — used to test whether the steering direction is task-agnostic |

Ground-truth optima are computed per case (Monte-Carlo or exact grid for `coupling`, closed-form for `parabola`/`sine`) so a submitted plan can be scored against the best reachable one (its *optimality gap*).

---

## The pipeline in four stages

```
 (1) GENERATE            (2) EXTRACT              (3) STEER / TRANSFER        (4) VALIDATE
 run.py  ────────────►  direction_extract.py ──► steering.py            ──► anticipation.py
   rollouts +             SUBMIT − SET axis        transfer_studies.py        lead_time.py
   activation capture     + best layer             composite_plot.py          trigger_spans.py
                        label_exploration.py ─┐
                        build_explore_dirs.py ┴► SUBMIT − EXPLORE axis
```

### 1. Generate rollouts and capture activations — `run.py`
Runs the model (or you, via `--human`) through many rollouts, logging a full transcript per turn and saving the residual stream at the decision tokens.

```bash
python -m csat.run --human            # roleplay it yourself, no GPU
python -m csat.run --n 200            # 200 model rollouts
python -m csat.run --n 50 --model google/gemma-2-9b-it
```

Writes, under `runs/<run_name>/rollout_XXXX/`: `case.json` (the landscape), `transcript.jsonl` (per-turn action + full reasoning text + whether the plan is already passing), `submission.json` (final plan, forced/voluntary, optimality gap), and `activations/turn_XX.npz` (residual stream at all layers for the captured token positions). Runs are **resumable** — completed rollouts are skipped.

### 2. Extract the direction — `direction_extract.py`
Walks a captured run, builds one `SET` vector (the final SET turn) and one `SUBMIT` vector per rollout by mean-pooling the residual around the action verb, and searches every layer for the one where the two classes are **most separable** (`SET` vs `SUBMIT` maximally anti-aligned while each class stays internally coherent across rollouts).

```bash
python -m csat.direction_extract --run-dir runs/csat --win 4
```

Writes `directions.npz` (the pooled `set_all`/`submit_all` vectors per layer + the chosen `best_layer`), a JSON summary, a per-layer separability plot, and a held-out per-turn projection-trajectory plot.

**Alternative positive pole (`EXPLORE` instead of `final-SET`).** The final-SET pole is contaminated by "the act of emitting a SET line," so negative steering can just re-SET the same weights. Two scripts fix this:
- `label_exploration.py` — has the *same model* read each SET turn's reasoning in isolation and mark the spans that show *active exploration* ("try a region not tried before / change tack"), then re-forwards the turn **as it happened in the task** (deterministic env replay) and pools activations at exactly those flagged tokens. So the vector encodes *exploring while solving*, not *reading about exploring*.
- `build_explore_directions.py` — pairs those EXPLORE vectors with the same SUBMIT pole and writes `directions_explore.npz` in the **identical schema** to `directions.npz`, so every downstream steering tool consumes it unchanged.

```bash
python -m csat.label_exploration --source-run-dir runs/csat
python -m csat.build_explore_directions --source-run-dir runs/csat
```

### 3. Steer and test transfer — `steering.py` / `transfer_studies.py` / `composite_plot.py`
`steering.py` builds the `(SUBMIT − SET)` vector at a chosen layer, scales it to a fraction of a token's typical residual norm, and **causally adds `alpha × vector`** to the post-MLP residual during generation (generation-only by default, so the model's *reading* of the prompt/feedback numbers is never perturbed). `alpha > 0` pushes toward SUBMIT (stop earlier), `alpha < 0` toward SET (keep searching).

`transfer_studies.py` is the single entry point for the studies (it imports the machinery from `steering.py`):

| `--study` | what it does |
|---|---|
| `steer` | alpha sweep in an env; does steering shift stop timing / final margin? |
| `project` | no steering — project each turn's activations onto the axis (read-only transfer check) |
| `composite` | run a baseline, then **branch** at the final-SET turn and re-generate under steering, comparing against resampled unsteered "null" branches (the honest counterfactual under sampling) |
| `story` | out-of-distribution: continue half a story under ±alpha and measure length modulation — tests whether the "keep going / stop" axis is task-agnostic |
| `trigger` | **closed-loop**: monitor the projection every token and *inject* the vector only when it crosses a threshold, then re-arm |
| `compare` | per-layer cosine between two direction files (e.g. explore vs. set) — no model loaded |

```bash
python -m csat.transfer_studies --study steer     --env parabola --source-run-dir runs/csat --alphas -1 -0.5 0 0.5 1
python -m csat.transfer_studies --study composite --env sine     --source-run-dir runs/csat --alphas -0.5 -1.0 --null-branches 3
python -m csat.transfer_studies --study story     --source-run-dir runs/csat --with-control
python -m csat.transfer_studies --study trigger   --env coupling --steer-vec explore --alpha -1 --k 10 --steer-k 20
```

`composite_plot.py` is fully decoupled from the model: it reads the tidy per-turn CSV each study writes and draws the figures (margin/gap-vs-turn per case, steer sweeps, projection trajectories, trigger traces, and paired steered-vs-null histograms). It can **rebuild the CSV from the rollout dirs after a crash** with no re-simulation, and includes matched-norm random-vector controls for direction-specificity.

### 4. Validate the direction — `anticipation.py` / `lead_time.py` / `trigger_spans.py`
The key confound: the direction was extracted from tokens just before the verb, so it might only encode "the next word is SUBMIT" — a lexical feature, not an intention.

- `anticipation.py` — does the projection predict "submits within the next *m* turns" for m ≥ 2, or only m = 1? It reports AUROC of the projection, of the turn index (a nuisance predictor, since SUBMIT happens late), and of the projection *residualised on turn index* — the last is the number to quote — all with cluster bootstraps over rollouts, plus an event-aligned curve.
- `lead_time.py` — *token-level* foresight within the submit turn: does the projection cross threshold *before* the SUBMIT verb (and before any stop-language like "finalise", "good enough")? Reports lead in tokens, the false-fire rate on non-submit turns, and a lexical analysis separating "fired before the words exist" from "fired on surface stop-vocabulary."
- `trigger_spans.py` — turns a closed-loop `trigger` run into **readable text**: for each firing, the window that crossed the threshold, the tokens generated under injection, and the cadence afterward, marked inline in the transcript.

---

## Shared building blocks

| file | role |
|---|---|
| `config.py` | one `Config` dataclass: model, env selection + params, rollout counts, capture settings, output paths. |
| `agents.py` | `HumanAgent` (you type actions) and `ModelAgent` (loads the HF model, generates on-policy, stops at the first complete action line, optionally captures activations). torch/transformers imported lazily so `--human` needs neither. |
| `rollout.py` | the SET/SUBMIT turn loop for one rollout; `build_env` dispatches on `env_kind`; records per-turn state so over-optimisation (how far *past* the first passing plan the model keeps pushing) is reconstructable offline. |
| `dsl.py` | forgiving parser for the two-verb DSL (lenient on surrounding prose, strict on the action itself), `<think>`-block splitting, and the signed-margin feedback table the model sees. |
| `prompts.py` | system prompts + case/feedback rendering, dispatched per env (multi-objective priority language for `coupling`, single-margin for `parabola`/`sine`). |
| `recorder.py` | activation capture: re-forwards the sequence with `output_hidden_states`, saves the residual at selected positions across all layers. Stores **bfloat16 losslessly** (via `ml_dtypes`) because residual streams contain "massive activations" that overflow float16. |
| `io_utils.py` | per-rollout persistence (case / transcript / submission JSON) and resume bookkeeping. |
| `case_spread.py` | for `coupling`, picks rollout seeds whose ground-truth optima are *spread* across the weight space rather than clustered at the rail — so the dataset covers informative trap/off-rail optima, not just easy cases. |

---

## Output artifacts (per study)

Rollout dirs carry `case.json`, `transcript*.jsonl`, `submission.json`/`composite_summary.json`, and `activations/*.npz`. Analysis steps add: `directions*.npz` + summaries and separability/trajectory PNGs (extract); `composite_turns.csv` + `composite_meta.json` + per-case PNGs (studies, via `composite_plot`); `anticipation_L*.json/.png`, `lead_times.json/.png/.md`, `trigger_spans.md` and `exploration_snippets.md` (validation + labelling, human-readable).

## Requirements

Python with `numpy`, `matplotlib`, `torch`, `transformers`, and `ml_dtypes` (for lossless bf16 activation storage/loading — without it capture silently falls back to float32). A CUDA GPU is needed for the model paths; `--human` roleplay and all the offline analysis/plot scripts run CPU-only. Run everything as package modules from the parent of `csat/` (`python -m csat.<script>`).

## A typical end-to-end run

```bash
# 1. generate data (model rollouts + activation capture)
python -m csat.run --n 200

# 2. extract the SUBMIT-vs-SET direction and pick the best layer
python -m csat.direction_extract --run-dir runs/csat

#    (optional) build the cleaner SUBMIT-vs-EXPLORE direction
python -m csat.label_exploration      --source-run-dir runs/csat
python -m csat.build_explore_directions --source-run-dir runs/csat

# 3. steer and test transfer
python -m csat.transfer_studies --study steer  --env parabola --source-run-dir runs/csat
python -m csat.transfer_studies --study story  --source-run-dir runs/csat --with-control

# 4. validate: does the axis anticipate, or just detect the verb?
python -m csat.transfer_studies --study project --env sine --source-run-dir runs/csat
python -m csat.anticipation --run-dir runs/csat_transfer_sine_nosteer --directions runs/csat/directions.npz
```
