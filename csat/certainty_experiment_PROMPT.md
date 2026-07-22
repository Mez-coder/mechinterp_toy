# Task: write a "model certainty" concept-vector experiment for the `csat` codebase

You are extending an existing mechanistic-interpretability codebase called **`csat`**. Your job is to write **one new, runnable Python module** (package-style, e.g. `csat/certainty.py`, plus a small plotting helper if you want) that implements the experiment specified below. Follow the conventions, style, and utilities of the existing code — **reuse the existing modules wherever possible instead of re-implementing them.**

Read the `README.md` first for the full picture of the codebase, then use the attached scripts as your building blocks.

---

## Files attached alongside this prompt

**Essential (read and reuse directly):**
- `README.md` — overview of the whole project and the four-stage pipeline.
- `config.py` — the `Config` dataclass (model, env, capture, rollout settings). Extend it if you need new knobs.
- `agents.py` — `ModelAgent`: loads the HF model, generates on-policy, stops at the first complete action line, and captures activations. **This is how you generate a turn and get activations.**
- `recorder.py` — `capture_and_save`, `_forward_hidden_states`, `_positions`. **This is how activations are re-forwarded and saved; adapt it for "all generated tokens at layer 22".**
- `rollout.py` — the single-sample SET/SUBMIT turn loop. Your generation loop is a modified version of this (10 samples per turn).
- `parabola_env.py` — the environment you will run (`env_kind="parabola"`, `n_obj=2`, weights `w0, w1`).
- `prompts.py` — system prompt + case/feedback rendering for parabola.
- `dsl.py` — `parse_action` (returns `kind ∈ {set, submit, parse_error}` and the proposed weights), `split_thinking`, `render_feedback`. **Use this to parse each sample's action and weights.**
- `direction_extract.py` — `locate_verb`, the before-verb pooling logic, `_cos`, `_coherence`, and the per-turn **projection-trajectory plot**. Reuse `locate_verb` for the token exclusion, and the projection style for stage G.
- `steering.py` — `steering_active` (the causal hook), `build_steering_vector`, `load_direction`, `mean_token_norm_at_layer`, `run_steered`, `plot_projection_trajectories`. **This is how you do stage H (causal steering).**

**Useful (reuse if convenient):**
- `io_utils.py` — per-rollout persistence + resume bookkeeping patterns.
- `transfer_studies.py` — the `_advance` turn loop and the composite "branch at the submit turn and re-generate under steering" pattern (stage H test 1) and the alpha-sweep pattern (stage H test 2).
- `composite_plot.py` — plotting/CSV infrastructure and the `_draw_traj` helper if you want consistent figures.

---

## Fixed parameters (do not change unless exposed as a CLI flag)

- Environment: **parabola**, `n_obj = 2` (two weights `w0`, `w1`).
- Layer of interest: **hidden-state index 22** (i.e. the post-MLP residual = output of decoder block 21). Everything — capture, concept vectors, projection, steering — is at layer 22.
- Iterations per turn: **10** (call this `R`).
- Temperature: **> 0** (default `0.7`, from `Config`) — required, or there is no variance to measure.
- Number of bins: **4** (`c0` = least certain … `c3` = most certain).

Expose sensible CLI flags (`--n-rollouts`, `--iters`, `--layer`, `--exclude-window`, `--frac`, `--alphas`, `--run-name`, `--source-run-dir`, etc.) with the defaults above. Make the data-generation and analysis phases separately runnable (e.g. `--phase {generate,analyse,steer,all}`), and make generation **resumable** like the existing code.

---

## What the script must do

### Stage A — Generate rollouts with 10 samples per turn (parabola)

Run `N` parabola rollouts (default e.g. `--n-rollouts 50`). Mirror the turn loop in `rollout.py`/`transfer_studies._advance`, but at **every turn**:

1. **Freeze the current message context** and draw **`R = 10` independent samples** of the model's next action (temperature > 0). Each sample is one "iteration".
2. For **each iteration**, re-forward and **save the residual stream at layer 22 for _all generated (assistant) tokens_** of that sample (adapt `recorder.capture_and_save` / `_forward_hidden_states`; store only layer 22 to keep storage manageable, but keep it configurable to all layers). Save `token_ids` too (needed to locate the verb later).
3. Parse each iteration's action with `dsl.parse_action`: record `kind` (`set`/`submit`/`parse_error`) and the proposed `(w0, w1)`.
4. **Advance the canonical trajectory using iteration 0's action** (apply its weights / submit). Iterations 1–9 are measurement-only. Continue to the next turn until iteration 0 submits or `max_turns` is reached.

**Storage / tagging.** Save activations to disk with a clear **`rollout / turn / iteration`** tag. Suggested layout:
```
runs/<run_name>/rollout_XXXX/
    case.json                       # the landscape (io_utils.save_case)
    transcript.jsonl                # canonical (iteration-0) path, one row per turn
    measurements.jsonl              # one row per (turn, iteration): kind, w0, w1, n_tokens, npz path
    turns/turn_TT/iter_II.npz       # layer-22 acts (n_tokens, d_model) + token_ids + action kind
```
`transcript.jsonl` should be compatible with the existing per-turn record shape (turn, action, weights, response, all_pass, weight_vec) so the existing projection/plot code can read it.

> **Caveat to handle:** the parabola system prompt asks for short replies, so some samples may have very few reasoning tokens. Count and report iterations whose usable (pre-action) region is empty; consider a `--min-reasoning-tokens` guard. Do not crash on `parse_error` samples — exclude them from weight-variance and from concept-vector averaging.

### Stage B — Per-turn measurements

After the 10 iterations of a turn, over the iterations that produced a **valid parseable action** (SET or SUBMIT — both carry weights):
- `var_w0` = variance of `w0` across iterations.
- `var_w1` = variance of `w1` across iterations.
- `n_set`, `n_submit` (and `n_valid = n_set + n_submit`).
- `split_uncertainty` = `1 − |n_set − n_submit| / n_valid` → 1.0 at a 50/50 split (**most** uncertain), 0.0 when unanimous. (You may use binary entropy of `p = n_submit/n_valid` instead; both peak at 50/50 — state which you used.)

Skip turns with `n_valid < 2` (variance undefined); report how many were skipped.

### Stage C/D — Certainty score and binning

Across **all (rollout, turn) records pooled together**:
1. Z-score each signal over the population: `z_w0 = zscore(var_w0)`, `z_w1 = zscore(var_w1)`, `z_split = zscore(split_uncertainty)`.
2. **Uncertainty score** `U = z_w0 + z_w1 + z_split` (higher ⇒ less certain).
3. **Certainty score** `C = −U` (higher ⇒ more certain).
4. **Bin `C` into 4 equal-count quartiles**: `c0` = bottom quartile (least certain) … `c3` = top quartile (most certain). Every (rollout, turn) gets a bin label. Save the full table (rollout, turn, var_w0, var_w1, split_uncertainty, U, C, bin) to CSV/JSON.

### Stage E — Bin-averaged concept vectors (layer 22)

For each captured **iteration**, compute its **pre-action pooled vector** at layer 22:
- Locate the action verb token index `p` in the sample using `direction_extract.locate_verb` (decode `token_ids`, find the last `SET`/`SUBMIT`).
- **Mean-pool the residual over tokens `[0 : p − exclude_window)`** where `exclude_window = 20` — i.e. **drop the action verb, the ~20 tokens immediately before it, and everything after.** This isolates the reasoning that precedes the decision.
- If `p` is not found or the usable region is empty, exclude that iteration from averaging (count it).

Each iteration inherits **its turn's bin label**. Then:
- `c_b` = mean of the pre-action pooled vectors over **all iterations whose turn is in bin `b`** (b = 0..3). Four vectors, each `(d_model,)` at layer 22.

Build the **6 difference (concept) vectors**, each oriented **higher-minus-lower certainty**:
```
c3−c0, c3−c1, c3−c2, c2−c0, c2−c1, c1−c0
```
Save all 6 (plus `c0..c3`).

> **Reuse tip for stage H:** ALSO save a directions file in the **same schema as `directions.npz`** (see `direction_extract.py` / `steering.load_direction`), e.g. `directions_certainty.npz` with `layers=[22]`, `set_all=[c0]` (uncertain pole), `submit_all=[c3]` (certain pole), `best_layer=22`. Then `steering.build_steering_vector` and `plot_projection_trajectories` work **unchanged**, with `alpha>0` ⇒ toward certainty (`c3`) and `alpha<0` ⇒ toward uncertainty (`c0`), because `submit_all − set_all = c3 − c0`.

### Stage F — Cosine-similarity heatmap of the 6 concept vectors

Compute the 6×6 cosine-similarity matrix among the six difference vectors (`direction_extract._cos`) and draw it as an annotated heatmap. Label rows/cols (`c3−c0`, …). Interpretation to print/caption: **uniformly high similarity ⇒ "certainty" is a roughly linear direction; low/mixed ⇒ the concept is curved/non-linear in this space.**

### Stage G — Certainty projected as a function of turn

Exactly analogous to the SUBMIT−SET projection-vs-turn plot the codebase already makes (`direction_extract` held-out trajectory / `steering.plot_projection_trajectories`), but using the **certainty axis `c3 − c0`**:
- Affine projection with `mid = (c0 + c3)/2`, `dir = (c3 − c0)`, `proj = 2·(v − mid)·dir / (dir·dir)` so `c0 → −1` (uncertain) and `c3 → +1` (certain).
- For each rollout, plot the per-turn projection along the **canonical (iteration-0) trajectory** using each turn's pre-action pooled vector (use iteration 0's vector, or the mean of the 10 — state which; mean is steadier). Star the submit turn. Reuse the existing plotting style.

### Stage H — Causal steering (injection) tests at layer 22

Build the certainty steering vector from **`c3 − c0`** via `steering.build_steering_vector` (loading `directions_certainty.npz`; unit-normalise then scale to `frac × mean_token_norm_at_layer(22)`). Hook it on decoder block **21** with `steering.steering_active`. Run in the **parabola** env. Two tests:

1. **Inject toward UNCERTAINTY at the stop point.** Run a baseline rollout; branch at the baseline **submit turn** (reuse the composite branch-at-submit pattern in `transfer_studies.run_composite`) and re-generate under **negative alpha** (toward `c0`). Hypothesis: injected doubt makes the model **keep going / submit less**, i.e. more turns, larger post-branch exploration. Compare against unsteered (alpha 0) re-branches (the resampling-honest null).

2. **Steer toward CERTAINTY early.** From early turns, inject **positive alpha** (toward `c3`). Hypothesis: the model **submits prematurely** — earlier submit turn, larger optimality gap vs the known parabola optimum.

Sweep `--alphas` (default `-1 -0.5 0 0.5 1`), save transcripts per alpha, and draw margin/gap-vs-turn plots (reuse `composite_plot`/`steering` plotting). Report submit-rate, submit-turn, final margin, and optimality gap per alpha.

---

## Deliverable requirements

- One coherent, **runnable** module (plus optional plotting helper), package-style (`python -m csat.certainty …`), matching the existing code's argparse + docstring + logging conventions. Put a module docstring at the top explaining the pipeline and giving example commands, exactly like the other scripts.
- **Reuse** `ModelAgent`, `recorder`, `dsl`, `parabola_env`, `prompts`, `direction_extract` helpers, and `steering` — do not re-implement generation, parsing, capture, the steering hook, or the projection math.
- Save all intermediate artifacts (activations, measurements table, `c0..c3`, the 6 concept vectors, `directions_certainty.npz`) so analysis and plotting can be re-run **without** re-running the model.
- Import torch/transformers lazily where the existing code does, so the pure-analysis/plot phases run CPU-only.
- Be explicit in comments/prints about every assumption (split metric used, exclusion window, quartile binning, which vector feeds projection/steering).

If any part of this spec is ambiguous or seems inconsistent with the attached code, state your interpretation in a comment and proceed — do not silently drop a requirement.
