"""certainty.py -- "model certainty" concept-vector experiment (parabola env).

Idea: at every turn, freeze the context and draw R=10 independent samples of the
model's next action (temperature > 0). Behavioural DISAGREEMENT across those
samples (variance of the proposed weights + the SET/SUBMIT split) is an
observable proxy for the model's uncertainty at that decision point. We bin the
per-turn certainty score into quartiles, average the layer-22 pre-action pooled
residuals per bin (c0 = least certain .. c3 = most certain), and ask whether
"certainty" is a linear direction: cosine structure of the 6 bin-difference
vectors (stage F), read-out along the canonical trajectory (stage G), and causal
steering along c3-c0 (stage H).

Stages
------
  A  generate: N parabola rollouts; at every turn 10 iterations, each with the
     layer-22 residual saved for ALL generated tokens. The canonical trajectory
     is advanced with ITERATION 0's action; iterations 1..9 are measurement-only.
  B  per-(rollout,turn) measurements: var_w0, var_w1, n_set/n_submit,
     split_uncertainty = 1 - |n_set - n_submit| / n_valid   (NOT binary entropy;
     both peak at a 50/50 split -- we use the linear form and say so).
  C/D certainty score C = -(z(var_w0) + z(var_w1) + z(split_uncertainty)),
     binned into 4 equal-count quartiles over the pooled population.
  E  bin-averaged concept vectors c0..c3 at layer 22 from the PRE-ACTION pooled
     vector of each iteration (mean over tokens [0 : verb_pos - exclude_window),
     i.e. the verb, the ~20 tokens before it, and everything after are dropped),
     plus the 6 higher-minus-lower differences (c3-c0, c3-c1, c3-c2, c2-c0,
     c2-c1, c1-c0). Also writes directions_certainty.npz in the directions.npz
     schema (set_all=[c0] uncertain pole, submit_all=[c3] certain pole) so
     steering.build_steering_vector / plot_projection_trajectories work unchanged.
  F  6x6 cosine heatmap of the difference vectors. Uniformly high similarity =>
     certainty is roughly one linear direction; low/mixed => curved/non-linear.
  G  per-rollout projection-vs-turn along the CANONICAL trajectory on the
     certainty axis (affine: c0 -> -1, c3 -> +1), submit turn starred. The
     per-turn vector is the MEAN of that turn's valid iteration vectors (steadier
     than iteration 0 alone; stated here on purpose).
  H  causal steering at layer 22 with unit(c3-c0) * frac * mean_token_norm:
     test 1: branch at the baseline submit turn, re-generate under alpha < 0
             (toward c0 / doubt) vs resampled unsteered null branches
             (reuses transfer_studies.run_composite, inject_at='submit');
     test 2: whole-rollout alpha sweep from turn 1 (positive alpha = premature
             confidence -> earlier submit, larger optimality gap;
             reuses steering.run_steered + composite_plot).

Example commands
----------------
    # A: generate (GPU)
    python -m csat.certainty --phase generate --n-rollouts 50 --run-name csat_certainty

    # B-G: analyse + plots (CPU-only; torch never imported)
    python -m csat.certainty --phase analyse --run-dir runs/csat_certainty

    # H: steering tests (GPU)
    python -m csat.certainty --phase steer --run-dir runs/csat_certainty \
        --alphas -1 -0.5 0 0.5 1 --steer-n-rollouts 12 --null-branches 3

    # everything
    python -m csat.certainty --phase all --n-rollouts 50

Storage layout (stage A), per rollout under runs/<run_name>/rollout_XXXX/:
    case.json                     landscape (io_utils.save_case)
    transcript.jsonl              canonical (iteration-0) path, one row per turn,
                                  same record shape as rollout.py
    submission.json               canonical submission (or forced), like run.py
    measurements.jsonl            one row per (turn, iteration):
                                  kind, w0, w1, n_tokens, npz path, error
    turns/turn_TT/iter_II.npz     acts (n_saved_layers, n_tokens, d_model) at the
                                  captured layer(s) + token_ids + kind

Assumptions stated up front (also printed where they bite):
  * layer 22 is a hidden-state index: hidden_states[22] = output of decoder
    block 21 = the post-MLP residual (recorder.py convention); steering hooks
    block 21 (steering.py convention block_idx = layer - 1).
  * split metric: 1 - |n_set - n_submit| / n_valid (linear, not entropy).
  * weight variance: sample variance (ddof=1) over valid iterations; turns with
    n_valid < 2 are skipped (variance undefined) and counted.
  * z-scores are population z-scores over all pooled (rollout, turn) records;
    a zero-std signal contributes 0 to U (warned).
  * binning: equal-count quartiles by rank of C (ties broken by stable sort).
  * pre-action pooling: tokens [0 : p - exclude_window) with exclude_window=20;
    iterations whose verb is not found, or whose usable region is shorter than
    --min-reasoning-tokens, are excluded from vector averaging and counted.
    If no tokenizer is available the verb position is approximated as
    n_tokens - 8 (the action stopper halts right after the action line, so the
    verb sits ~a weight-string away from the end); this fallback is counted.
  * stage G projects the per-turn MEAN of valid iteration vectors.
  * stage H scales the vector by mean token norm computed from THIS run's
    layer-22 captures (steering.mean_token_norm_at_layer expects the recorder's
    activations/turn_XX.npz format, which this run does not produce -- the
    scaling formula is identical, only the file walk differs).
"""
from __future__ import annotations
import os
# reduce CUDA fragmentation OOMs on long multi-turn runs (must precede torch
# CUDA init; harmless if already set). Mirrors transfer_studies.py.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json, glob, csv, argparse, gc, warnings
import numpy as np

try:
    import ml_dtypes  # noqa: F401  (registers bf16 so np.load can read captures)
except Exception:
    pass

from .config import Config
from .dsl import parse_action, split_thinking
from .prompts import system_prompt_for, render_case_for, render_feedback_for
from .rollout import build_env
from . import io_utils as io
from . import direction_extract as de     # locate_verb, _cos (numpy-only)
from . import recorder                    # _forward_hidden_states, _positions,
                                          # _to_storage (torch imported lazily)

DIFF_PAIRS = [(3, 0), (3, 1), (3, 2), (2, 0), (2, 1), (1, 0)]   # higher - lower
DIFF_LABELS = [f"c{a}-c{b}" for a, b in DIFF_PAIRS]


def _free():
    """Best-effort GPU/host memory release (no-op without torch)."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_tokenizer(model_name):
    """Tokenizer for verb location (same pattern as direction_extract.main)."""
    try:
        from transformers import AutoTokenizer
        try:
            return AutoTokenizer.from_pretrained(model_name)
        except Exception:
            return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"[tok] no tokenizer ({e}); falling back to approximate verb "
              "position n_tokens - 8 (counted separately).")
        return None


# =========================================================================== #
# Stage A -- generation: 10 iterations per turn, layer-22 capture of ALL
# generated tokens per iteration, canonical trajectory = iteration 0
# =========================================================================== #
def _capture_layer_indices(n_hidden, layer, mode):
    """Hidden-state indices to store per iteration. 'single' keeps storage
    manageable (the spec's default); 'all' keeps every hidden state."""
    if mode == "all":
        return list(range(n_hidden))
    return [int(layer)]


def _sample_iteration(agent, messages, capture_path, layer, capture_layers_mode):
    """One frozen-context sample: generate (reusing ModelAgent._build_inputs /
    _generate / _action_stopper -- we bypass agent.act only because we need the
    full token ids to re-forward and keep JUST layer `layer` for ALL generated
    tokens, which recorder.capture_and_save cannot do without storing every
    layer), then re-forward and save.  Returns (text, answer, action, n_tokens)."""
    import torch
    cfg = agent.cfg
    ids = agent._build_inputs(messages)
    prompt_len = ids.shape[1]
    full = agent._generate(ids, cfg.max_new_tokens, agent._action_stopper(prompt_len))
    text = agent.processor.decode(full[0, prompt_len:], skip_special_tokens=True).strip()
    answer, _ = split_thinking(text, cfg.enable_thinking)
    action = parse_action(answer, cfg.n_obj)

    # re-forward for hidden states (recorder machinery), keep selected layer(s)
    hs = recorder._forward_hidden_states(agent.model, full)   # tuple (L+1) of (1,seq,d)
    positions = recorder._positions("assistant", prompt_len, full.shape[1], 0)
    lsel = _capture_layer_indices(len(hs), layer, capture_layers_mode)
    idx = torch.tensor(positions, device=hs[0].device)
    stack = torch.stack([hs[j][0].index_select(0, idx) for j in lsel])  # (n_sel,n_pos,d)
    acts = stack.to(torch.float32).cpu().numpy()
    token_ids = full[0, positions].cpu().numpy()
    del hs, stack, full, ids
    _free()

    np.savez_compressed(
        capture_path,
        acts=recorder._to_storage(acts, cfg.capture_dtype),
        layers=np.array(lsel, np.int32),
        token_ids=token_ids,
        positions=np.array(positions),
        kind=np.array(action.kind))
    return text, answer, action, len(positions)


def _phase_generate(cfg, args):
    from .agents import ModelAgent            # lazy: needs torch/transformers
    if cfg.temperature <= 0:
        raise SystemExit("temperature must be > 0 -- with greedy decoding the 10 "
                         "iterations are identical and there is no variance to measure.")
    cfg.env_kind = "parabola"
    cfg.capture = False                       # we do our own per-iteration capture
    run_dir = cfg.run_dir()
    R = args.iters
    with open(os.path.join(run_dir, "certainty_meta.json"), "w") as f:
        json.dump(dict(model=cfg.model_name, env="parabola", n_obj=cfg.n_obj,
                       iters=R, layer=args.layer, capture_layers=args.capture_layers,
                       temperature=cfg.temperature, max_turns=cfg.max_turns,
                       n_rollouts=args.n_rollouts, seed_start=cfg.seed_start),
                  f, indent=2)

    agent = ModelAgent(cfg)
    n_empty_reasoning = 0                     # iterations with < min-reasoning tokens
    for ridx in range(args.n_rollouts):
        d = os.path.join(run_dir, f"rollout_{ridx:04d}")
        if os.path.exists(os.path.join(d, "submission.json")):   # resumable
            print(f"[gen] rollout {ridx:04d} already complete; skipping.")
            continue
        seed = cfg.seed_start + ridx
        env = build_env(cfg)
        env.reset(seed=seed, wide=getattr(cfg, "wide_cases", True))
        d = io.rollout_dir(run_dir, ridx)
        io.save_case(d, env, seed)
        open(os.path.join(d, "measurements.jsonl"), "w").close()  # fresh per (re)run
        opt = env.optimum()                   # parabola optimum: exact & cheap
        opt_meta = dict(optimum_feasible=opt.get("feasible", False),
                        optimum_margin_priority=opt.get("margin_priority"),
                        optimum_weights=opt.get("weights"),
                        case_id=ridx, rep=0, case_seed=seed)

        def _gap(snap):
            return (opt["margin_priority"] - snap["margin_priority"]) \
                if opt.get("feasible") else None

        messages = [{"role": "system", "content": system_prompt_for(cfg)},
                    {"role": "user", "content": render_case_for(env, cfg)}]
        submitted = False
        first_pass_turn = None

        for turn in range(1, cfg.max_turns + 1):
            tdir = os.path.join(d, "turns", f"turn_{turn:02d}")
            os.makedirs(tdir, exist_ok=True)
            iter0 = None                       # (text, answer, action) of iteration 0
            n_set = n_sub = 0
            for it in range(R):                # R frozen-context samples
                npz = os.path.join(tdir, f"iter_{it:02d}.npz")
                text, answer, action, ntok = _sample_iteration(
                    agent, messages, npz, args.layer, args.capture_layers)
                if it == 0:
                    iter0 = (text, answer, action)
                n_set += (action.kind == "set"); n_sub += (action.kind == "submit")
                if ntok <= args.min_reasoning_tokens:
                    n_empty_reasoning += 1
                row = dict(rollout=ridx, turn=turn, iteration=it, kind=action.kind,
                           w0=action.weights.get(0), w1=action.weights.get(1),
                           weights={str(k): v for k, v in action.weights.items()},
                           n_tokens=ntok, error=action.error,
                           npz=os.path.relpath(npz, d))
                with open(os.path.join(d, "measurements.jsonl"), "a") as f:
                    f.write(json.dumps(row) + "\n")
            print(f"[gen] rollout {ridx:04d} turn {turn:02d}: "
                  f"{n_set} SET / {n_sub} SUBMIT / {R - n_set - n_sub} parse_error "
                  f"(canonical -> {iter0[2].kind})")

            # -- advance the CANONICAL trajectory with iteration 0 (mirrors
            #    rollout.run_rollout; iterations 1..R-1 were measurement-only) --
            text, answer, action = iter0
            messages.append({"role": "assistant", "content": answer})
            rec = dict(turn=turn, action=action.kind, weights=action.weights,
                       error=action.error, response=text,
                       all_pass=env.all_pass(), weight_vec=env.w.tolist(),
                       meta=dict(source="model", iterations=R,
                                 temperature=cfg.temperature))

            if action.kind == "submit":
                for i, w in action.weights.items():
                    if 0 <= i < env.n_obj:
                        env.set_weight(i, w)
                if first_pass_turn is None and env.all_pass():
                    first_pass_turn = turn
                rec["all_pass"] = env.all_pass()
                rec["weight_vec"] = env.w.tolist()
                snap = env.submit()["plan"]
                io.save_submission(d, snap, dict(submit_turn=turn, forced=False,
                                                 n_turns=turn,
                                                 first_pass_turn=first_pass_turn,
                                                 optimality_gap=_gap(snap), **opt_meta))
                io.append_transcript(d, rec)
                submitted = True
                break

            if action.kind == "set":
                for i, w in action.weights.items():
                    if 0 <= i < env.n_obj:
                        env.set_weight(i, w)
                if first_pass_turn is None and env.all_pass():
                    first_pass_turn = turn
                rec["all_pass"] = env.all_pass()
                rec["weight_vec"] = env.w.tolist()
                messages.append({"role": "user",
                                 "content": render_feedback_for(env, cfg, env.feedback(),
                                                                turn=turn,
                                                                max_turns=cfg.max_turns)})
            else:                              # parse error: costs a turn
                messages.append({"role": "user",
                                 "content": f"Could not parse an action ({action.error}).\n"
                                 + render_feedback_for(env, cfg, turn=turn,
                                                       max_turns=cfg.max_turns)})
            io.append_transcript(d, rec)

        if not submitted:                      # budget exhausted -> forced submit
            snap = env.submit()["plan"]
            io.save_submission(d, snap, dict(submit_turn=cfg.max_turns, forced=True,
                                             n_turns=cfg.max_turns,
                                             first_pass_turn=first_pass_turn,
                                             optimality_gap=_gap(snap), **opt_meta))
            io.append_transcript(d, dict(turn=cfg.max_turns, action="forced_submit",
                                         weights={}, error="", response="",
                                         all_pass=env.all_pass(),
                                         weight_vec=env.w.tolist(), meta={}))
        print(f"[gen] rollout {ridx:04d} done (submitted={submitted}).")
    print(f"[gen] iterations with <= {args.min_reasoning_tokens} generated tokens: "
          f"{n_empty_reasoning} (their pre-action region will be empty in stage E).")


# =========================================================================== #
# Stages B/C/D -- measurements, certainty score, quartile bins
# =========================================================================== #
def load_measurements(run_dir):
    """All measurement rows across rollout_* dirs (adds abs npz path)."""
    rows = []
    for d in sorted(glob.glob(os.path.join(run_dir, "rollout_*"))):
        p = os.path.join(d, "measurements.jsonl")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                r["npz_abs"] = os.path.join(d, r["npz"])
                rows.append(r)
    return rows


def _zscore(x, name):
    """Population z-score; zero-variance signals contribute 0 (warned)."""
    x = np.asarray(x, float)
    sd = x.std()
    if sd == 0 or not np.isfinite(sd):
        warnings.warn(f"signal {name!r} has zero variance across the population; "
                      "it contributes 0 to the uncertainty score U.")
        return np.zeros_like(x)
    return (x - x.mean()) / sd


def build_turn_table(meas_rows, n_bins=4, min_valid=2):
    """Stages B-D. Returns (records, n_skipped): one record per (rollout, turn)
    with var_w0, var_w1, n_set/n_submit, split_uncertainty, U, C and bin."""
    groups = {}
    for r in meas_rows:
        groups.setdefault((int(r["rollout"]), int(r["turn"])), []).append(r)

    records, n_skipped = [], 0
    for (ridx, turn) in sorted(groups):
        rows = groups[(ridx, turn)]
        valid = [r for r in rows if r["kind"] in ("set", "submit")
                 and r.get("w0") is not None and r.get("w1") is not None]
        n_set = sum(r["kind"] == "set" for r in valid)
        n_sub = sum(r["kind"] == "submit" for r in valid)
        n_valid = len(valid)
        if n_valid < min_valid:                # variance undefined
            n_skipped += 1
            continue
        w0 = np.array([r["w0"] for r in valid], float)
        w1 = np.array([r["w1"] for r in valid], float)
        # split metric: linear 1 - |n_set-n_submit|/n_valid (NOT entropy); both
        # peak at 50/50 -- we use the linear form throughout, as documented.
        split = 1.0 - abs(n_set - n_sub) / n_valid
        records.append(dict(rollout=ridx, turn=turn,
                            var_w0=float(w0.var(ddof=1)),     # sample variance
                            var_w1=float(w1.var(ddof=1)),
                            n_set=n_set, n_submit=n_sub, n_valid=n_valid,
                            n_parse_error=len(rows) - n_valid,
                            split_uncertainty=float(split)))
    if not records:
        raise SystemExit("no (rollout, turn) records with enough valid iterations; "
                         "did the generate phase run?")

    # stages C/D: pooled z-scores -> U -> C -> equal-count quartiles ------------
    zw0 = _zscore([r["var_w0"] for r in records], "var_w0")
    zw1 = _zscore([r["var_w1"] for r in records], "var_w1")
    zsp = _zscore([r["split_uncertainty"] for r in records], "split_uncertainty")
    U = zw0 + zw1 + zsp                       # higher => LESS certain
    C = -U                                    # higher => MORE certain
    N = len(records)
    order = np.argsort(C, kind="stable")      # equal-count bins by rank of C
    ranks = np.empty(N, int); ranks[order] = np.arange(N)
    bins = np.clip((ranks * n_bins) // N, 0, n_bins - 1)
    for r, u, c, b, z0, z1, zs in zip(records, U, C, bins, zw0, zw1, zsp):
        r.update(z_var_w0=float(z0), z_var_w1=float(z1), z_split=float(zs),
                 U=float(u), C=float(c), bin=int(b))
    print(f"[table] {N} (rollout, turn) records; {n_skipped} turns skipped "
          f"(n_valid < {min_valid}); bin sizes: "
          f"{[int((bins == b).sum()) for b in range(n_bins)]} "
          f"(c0 = least certain .. c{n_bins - 1} = most certain)")
    return records, n_skipped


def save_turn_table(run_dir, records, n_skipped):
    fields = ["rollout", "turn", "var_w0", "var_w1", "n_set", "n_submit",
              "n_valid", "n_parse_error", "split_uncertainty",
              "z_var_w0", "z_var_w1", "z_split", "U", "C", "bin"]
    path = os.path.join(run_dir, "certainty_table.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)
    with open(os.path.join(run_dir, "certainty_table.json"), "w") as f:
        json.dump(dict(n_records=len(records), n_skipped_turns=n_skipped,
                       split_metric="1 - |n_set - n_submit| / n_valid",
                       variance="sample (ddof=1) over valid iterations",
                       records=records), f, indent=2)
    print(f"[table] wrote {path} (+ .json)")


# =========================================================================== #
# Stage E -- per-iteration pre-action pooled vectors and bin means c0..c3
# =========================================================================== #
def _load_iter_acts(npz_path, layer):
    """(acts_at_layer (n_tokens, d) float32, token_ids) or None."""
    if not os.path.exists(npz_path):
        return None
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            layers = list(np.asarray(z["layers"]).astype(int))
            if layer not in layers:
                return None
            acts = np.asarray(z["acts"])[layers.index(layer)].astype(np.float32)
            token_ids = np.asarray(z["token_ids"])
    except ValueError as e:                        # usually an unknown bf16 dtype
        raise RuntimeError(
            f"failed to read {npz_path}; activations look like bfloat16 but "
            "ml_dtypes is not installed (`pip install ml_dtypes`).") from e
    return acts, token_ids


def iter_pooled_vector(npz_path, layer, tokenizer, kind, exclude_window,
                       min_reasoning_tokens):
    """Pre-action pooled vector for one iteration: mean of the layer-`layer`
    residual over tokens [0 : p - exclude_window), p = index of the LAST
    SET/SUBMIT verb token (direction_extract.locate_verb). Returns
    (vec | None, status) where status is 'ok' | 'ok_approx_verb' |
    'missing' | 'verb_not_found' | 'region_empty'."""
    loaded = _load_iter_acts(npz_path, layer)
    if loaded is None:
        return None, "missing"
    acts, token_ids = loaded
    verb = "SET" if kind == "set" else "SUBMIT"
    p = de.locate_verb(token_ids, tokenizer, verb)
    status = "ok"
    if p is None:
        if tokenizer is not None:
            return None, "verb_not_found"
        # no tokenizer: the action stopper halts right after the action line, so
        # the verb sits ~a weight-string from the end; approximate and count it.
        p = max(0, acts.shape[0] - 8)
        status = "ok_approx_verb"
    hi = p - exclude_window                     # drop verb + window before + after
    if hi < max(1, min_reasoning_tokens):
        return None, "region_empty"
    return acts[:hi].mean(axis=0).astype(np.float32), status


def build_pooled_vectors(run_dir, meas_rows, layer, tokenizer, exclude_window,
                         min_reasoning_tokens, cache_path=None, recompute=False):
    """Pooled pre-action vector for every VALID (set/submit) iteration, cached to
    pooled_vectors.npz so analysis/plots re-run without touching activations."""
    cache_path = cache_path or os.path.join(run_dir, "pooled_vectors.npz")
    if os.path.exists(cache_path) and not recompute:
        z = np.load(cache_path, allow_pickle=False)
        if (int(z["layer"]) == layer and int(z["exclude_window"]) == exclude_window):
            print(f"[pool] loaded {z['vecs'].shape[0]} cached vectors from "
                  f"{cache_path} (pass --recompute-pool to rebuild)")
            return dict(vecs=z["vecs"], rollout=z["rollout"], turn=z["turn"],
                        iteration=z["iteration"],
                        kinds=[k for k in z["kinds"]]), None
        print("[pool] cache exists but layer/exclude-window differ; rebuilding.")

    vecs, ro, tu, itn, kinds = [], [], [], [], []
    counts = dict(ok=0, ok_approx_verb=0, missing=0, verb_not_found=0,
                  region_empty=0, invalid_kind=0)
    for r in meas_rows:
        if r["kind"] not in ("set", "submit"):
            counts["invalid_kind"] += 1        # parse_error: excluded from averaging
            continue
        v, status = iter_pooled_vector(r["npz_abs"], layer, tokenizer, r["kind"],
                                       exclude_window, min_reasoning_tokens)
        counts[status] += 1
        if v is None:
            continue
        vecs.append(v); ro.append(r["rollout"]); tu.append(r["turn"])
        itn.append(r["iteration"]); kinds.append(r["kind"])
    if not vecs:
        raise SystemExit("no usable pre-action vectors (all excluded); check "
                         "--exclude-window vs typical reply length in the report.")
    out = dict(vecs=np.stack(vecs), rollout=np.array(ro, np.int32),
               turn=np.array(tu, np.int32), iteration=np.array(itn, np.int32),
               kinds=kinds)
    np.savez_compressed(cache_path, layer=np.int32(layer),
                        exclude_window=np.int32(exclude_window),
                        vecs=out["vecs"], rollout=out["rollout"], turn=out["turn"],
                        iteration=out["iteration"], kinds=np.array(kinds))
    print(f"[pool] {out['vecs'].shape[0]} usable iteration vectors "
          f"(d_model={out['vecs'].shape[1]}); exclusions: {counts}")
    return out, counts


def build_bin_vectors(run_dir, pooled, records, layer, exclude_window, n_bins,
                      model_name):
    """c_b = mean pooled vector over all iterations whose TURN is in bin b, plus
    the 6 higher-minus-lower differences. Also writes directions_certainty.npz
    in the directions.npz schema (set_all=[c0], submit_all=[c3], best_layer)."""
    bin_of = {(r["rollout"], r["turn"]): r["bin"] for r in records}
    labels = np.array([bin_of.get((int(a), int(b)), -1)
                       for a, b in zip(pooled["rollout"], pooled["turn"])])
    n_unbinned = int((labels < 0).sum())       # iterations on skipped turns
    cs, counts = [], []
    for b in range(n_bins):
        sel = pooled["vecs"][labels == b]
        if len(sel) == 0:
            raise SystemExit(f"bin c{b} has no usable iteration vectors; need more "
                             "rollouts or a smaller --exclude-window.")
        cs.append(sel.mean(axis=0).astype(np.float32)); counts.append(len(sel))
    diffs = np.stack([cs[a] - cs[b] for a, b in DIFF_PAIRS])
    print(f"[bins] iterations per bin: {counts} (+{n_unbinned} on skipped turns, "
          "unbinned)")

    np.savez_compressed(
        os.path.join(run_dir, "certainty_vectors.npz"),
        c=np.stack(cs), diffs=diffs, diff_labels=np.array(DIFF_LABELS),
        counts=np.array(counts, np.int32), layer=np.int32(layer),
        exclude_window=np.int32(exclude_window))
    # directions.npz-schema file: alpha>0 => toward c3 (certainty) because
    # submit_all - set_all = c3 - c0. Consumed unchanged by steering.py.
    np.savez_compressed(
        os.path.join(run_dir, "directions_certainty.npz"),
        layers=np.array([layer], np.int32),
        set_all=cs[0][None, :].astype(np.float32),        # uncertain pole
        submit_all=cs[n_bins - 1][None, :].astype(np.float32),  # certain pole
        win=np.int32(exclude_window),
        d_model=np.int32(cs[0].shape[0]),
        n_set=np.int32(counts[0]), n_submit=np.int32(counts[-1]),
        best_layer=np.int32(layer), select=np.array("certainty_quartiles"),
        model_name=np.array(model_name))
    print(f"[bins] wrote certainty_vectors.npz + directions_certainty.npz "
          f"(set_all=c0, submit_all=c{n_bins - 1}, layer {layer})")
    return cs, diffs


# =========================================================================== #
# Stage F -- cosine heatmap of the 6 difference vectors
# =========================================================================== #
def plot_cosine_heatmap(run_dir, diffs):
    n = len(diffs)
    M = np.array([[de._cos(diffs[i], diffs[j]) for j in range(n)] for i in range(n)])
    with open(os.path.join(run_dir, "certainty_cosine_matrix.json"), "w") as f:
        json.dump(dict(labels=DIFF_LABELS, cosine=M.tolist()), f, indent=2)
    off = M[~np.eye(n, dtype=bool)]
    print(f"[cos] off-diagonal cosine: mean {np.nanmean(off):+.3f}  "
          f"min {np.nanmin(off):+.3f}  max {np.nanmax(off):+.3f}")
    print("[cos] reading: uniformly HIGH similarity => certainty is roughly one "
          "linear direction at this layer; LOW/mixed => the concept is "
          "curved/non-linear in this space.")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(n)); ax.set_xticklabels(DIFF_LABELS, rotation=45, ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels(DIFF_LABELS)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                        fontsize=8, color=("white" if abs(M[i, j]) > 0.6 else "black"))
        fig.colorbar(im, ax=ax, label="cosine")
        ax.set_title("Certainty concept vectors: cosine similarity\n"
                     "(uniformly high = one linear 'certainty' direction)")
        fig.tight_layout()
        png = os.path.join(run_dir, "certainty_cosine_heatmap.png")
        fig.savefig(png, dpi=130)
        print(f"[cos] wrote {png}")
    except Exception as e:
        print(f"[cos] plot skipped ({e})")
    return M


# =========================================================================== #
# Stage G -- certainty projection vs turn along the canonical trajectory
# =========================================================================== #
def plot_certainty_trajectories(run_dir, pooled, cs, max_lines=12):
    """Affine projection (c0 -> -1, c3 -> +1) of the per-turn MEAN of valid
    iteration vectors, along the canonical (iteration-0) trajectory. Mirrors the
    projection style of direction_extract / steering.plot_projection_trajectories."""
    c0, c3 = cs[0], cs[-1]
    mid, dirv = (c0 + c3) / 2.0, (c3 - c0)
    denom = float(dirv @ dirv) + 1e-12

    # per-(rollout, turn) mean vector (mean of the <=R iteration vectors: steadier
    # than iteration 0 alone -- documented choice)
    per_turn = {}
    for v, ro, tu in zip(pooled["vecs"], pooled["rollout"], pooled["turn"]):
        per_turn.setdefault((int(ro), int(tu)), []).append(v)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[traj] plot skipped ({e})")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    n_plotted = 0
    for rd in sorted(glob.glob(os.path.join(run_dir, "rollout_*")))[:max_lines]:
        ridx = int(os.path.basename(rd).split("_")[-1])
        kinds = de.turn_actions(os.path.join(rd, "transcript.jsonl"))
        xs, ys, sub_t = [], [], None
        for t in sorted(kinds):
            vs = per_turn.get((ridx, t))
            if not vs:
                continue
            v = np.stack(vs).mean(axis=0)
            xs.append(t)
            ys.append(float(2.0 * np.dot(v - mid, dirv) / denom))
            if kinds[t] == "submit":
                sub_t = t
        if not xs:
            continue
        line, = ax.plot(xs, ys, "-o", ms=3, alpha=0.85, label=f"r{ridx:04d}")
        if sub_t is not None:
            ax.plot([sub_t], [ys[xs.index(sub_t)]], "*", ms=15, color=line.get_color())
        n_plotted += 1
    ax.axhline(1, color="g", ls="--", lw=1)
    ax.axhline(-1, color="b", ls="--", lw=1)
    ax.text(0.01, 0.98, "c3 (most certain) = +1", color="g",
            transform=ax.transAxes, va="top", fontsize=8)
    ax.text(0.01, 0.02, "c0 (least certain) = -1", color="b",
            transform=ax.transAxes, va="bottom", fontsize=8)
    ax.set_xlabel("turn")
    ax.set_ylabel("projection onto (c3 - c0)  [c0=-1, c3=+1]")
    ax.set_title("Certainty projection along the canonical trajectory "
                 "(* = submit turn; per-turn mean of the 10 iteration vectors)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3); fig.tight_layout()
    png = os.path.join(run_dir, "certainty_projection.png")
    fig.savefig(png, dpi=130)
    print(f"[traj] wrote {png}  ({n_plotted} rollouts)")


def _phase_analyse(args):
    run_dir = args.run_dir
    meas = load_measurements(run_dir)
    if not meas:
        raise SystemExit(f"no measurements.jsonl rows under {run_dir}; run "
                         "--phase generate first.")
    n_pe = sum(r["kind"] == "parse_error" for r in meas)
    print(f"[meas] {len(meas)} iteration rows ({n_pe} parse_error, excluded from "
          "weight-variance and vector averaging)")

    records, n_skipped = build_turn_table(meas, n_bins=args.n_bins,
                                          min_valid=args.min_valid)
    save_turn_table(run_dir, records, n_skipped)

    tokenizer = _load_tokenizer(args.model)
    pooled, counts = build_pooled_vectors(
        run_dir, meas, args.layer, tokenizer, args.exclude_window,
        args.min_reasoning_tokens, recompute=args.recompute_pool)
    cs, diffs = build_bin_vectors(run_dir, pooled, records, args.layer,
                                  args.exclude_window, args.n_bins, args.model)
    M = plot_cosine_heatmap(run_dir, diffs)
    plot_certainty_trajectories(run_dir, pooled, cs, max_lines=args.max_lines)

    with open(os.path.join(run_dir, "certainty_summary.json"), "w") as f:
        json.dump(dict(
            layer=args.layer, exclude_window=args.exclude_window,
            n_bins=args.n_bins, n_iteration_rows=len(meas),
            n_parse_error_iterations=n_pe, n_turn_records=len(records),
            n_skipped_turns=n_skipped, pooling_exclusions=counts,
            split_metric="1 - |n_set - n_submit| / n_valid",
            g_vector="per-turn mean of valid iteration vectors",
            offdiag_cos_mean=float(np.nanmean(M[~np.eye(len(M), dtype=bool)])),
        ), f, indent=2)
    print(f"[done] analysis complete -> {run_dir}")


# =========================================================================== #
# Stage H -- causal steering along (c3 - c0) at layer 22
# =========================================================================== #
def certainty_token_norm(run_dir, layer, n_sample=300, seed=0):
    """Mean per-token L2 norm of the layer-`layer` residual over this run's
    iteration captures. Same quantity as steering.mean_token_norm_at_layer, but
    that function walks the recorder's activations/turn_XX.npz layout (acts
    indexed by hidden-state number), which the certainty run does not produce."""
    files = sorted(glob.glob(os.path.join(run_dir, "rollout_*", "turns",
                                          "turn_*", "iter_*.npz")))
    if not files:
        raise SystemExit(f"no iteration npz under {run_dir} to estimate token norm.")
    rng = np.random.default_rng(seed)
    if len(files) > n_sample:
        files = [files[i] for i in rng.choice(len(files), n_sample, replace=False)]
    norms = []
    for p in files:
        loaded = _load_iter_acts(p, layer)
        if loaded is None:
            continue
        norms.append(np.linalg.norm(loaded[0], axis=-1).reshape(-1))
    if not norms:
        raise SystemExit(f"could not estimate token norm at layer {layer}.")
    return float(np.concatenate(norms).mean())


def build_certainty_steering_vector(run_dir, layer, frac):
    """unit(c3 - c0) * frac * mean_token_norm(layer). Reuses steering.load_direction
    on directions_certainty.npz (submit_all - set_all = c3 - c0), with the token
    norm from certainty_token_norm (see its docstring for why not
    steering.mean_token_norm_at_layer). alpha > 0 => toward certainty (c3)."""
    from .steering import load_direction
    directions = os.path.join(run_dir, "directions_certainty.npz")
    if not os.path.exists(directions):
        raise SystemExit(f"{directions} not found; run --phase analyse first.")
    c0, c3 = load_direction(directions, layer)
    raw = c3 - c0
    nrm = float(np.linalg.norm(raw))
    if nrm == 0:
        raise SystemExit("c3 == c0 at this layer; nothing to steer.")
    tok_norm = certainty_token_norm(run_dir, layer)
    vec = (raw / nrm * (frac * tok_norm)).astype(np.float32)
    info = dict(layer=int(layer), frac=float(frac), raw_norm=nrm,
                mean_token_norm=tok_norm, steer_norm=float(np.linalg.norm(vec)))
    return vec, info, directions


def _steer_sweep_report(alpha_dirs):
    """Per-alpha submit-rate, mean submit turn, mean final margin, mean optimality
    gap, read from the submission.json each run_steered rollout writes."""
    out = {}
    for ad in alpha_dirs:
        with open(os.path.join(ad, "steer_meta.json")) as f:
            alpha = float(json.load(f)["alpha"])
        subs = []
        for p in sorted(glob.glob(os.path.join(ad, "rollout_*", "submission.json"))):
            with open(p) as f:
                subs.append(json.load(f))
        if not subs:
            continue
        vol = [s for s in subs if not s.get("forced", False)]
        gaps = [s["optimality_gap"] for s in subs if s.get("optimality_gap") is not None]
        out[alpha] = dict(
            n=len(subs),
            submit_rate=len(vol) / len(subs),
            mean_submit_turn=(float(np.mean([s["submit_turn"] for s in vol]))
                              if vol else None),
            mean_final_margin=float(np.mean([s["margin_priority"] for s in subs])),
            mean_optimality_gap=(float(np.mean(gaps)) if gaps else None),
            run_dir=ad)
    print("\n[steer] alpha sweep (alpha > 0 = toward c3/certainty -- hypothesis: "
          "premature submit, larger gap; alpha < 0 = toward c0/doubt):")
    for a in sorted(out):
        s = out[a]
        st = (f"{s['mean_submit_turn']:.2f}" if s["mean_submit_turn"] is not None
              else "n/a")
        gp = (f"{s['mean_optimality_gap']:+.4f}"
              if s["mean_optimality_gap"] is not None else "n/a")
        print(f"   alpha {a:+.2f}: n={s['n']}  submit_rate={s['submit_rate']:.0%}  "
              f"mean_submit_turn={st}  final_margin={s['mean_final_margin']:+.4f}  "
              f"gap={gp}")
    return out


def _phase_steer(args):
    from .agents import ModelAgent
    from .steering import steering_active, run_steered
    from .transfer_studies import (run_composite, summarize_branch_null,
                                   print_branch_null, _free as ts_free)
    from . import composite_plot as cp

    run_dir = args.run_dir
    layer = args.layer
    cfg = Config()
    if args.model:
        cfg.model_name = args.model
    cfg.env_kind = "parabola"
    cfg.n_obj = args.n_obj
    cfg.capture = False
    cfg.out_dir = args.out_dir
    cfg.seed_start = args.seed_start
    cfg.max_turns = args.max_turns
    cfg.temperature = args.temperature

    vec, info, directions = build_certainty_steering_vector(run_dir, layer, args.frac)
    print(f"[steer] certainty axis (c3-c0) @ L{layer}: |steer|={info['steer_norm']:.2f} "
          f"= {info['frac']:.0%} of mean token norm {info['mean_token_norm']:.2f} "
          f"(raw |c3-c0|={info['raw_norm']:.3f})")

    agent = ModelAgent(cfg)
    neg_alphas = sorted(a for a in args.alphas if a < 0)
    report = {}

    # ---- Test 1: inject toward UNCERTAINTY (alpha < 0) at the baseline submit
    #      turn; compare against resampled unsteered null branches. Hypothesis:
    #      injected doubt suppresses submission / extends search. ----
    if neg_alphas and not args.skip_composite:
        cfg.run_name = f"{args.steer_run_name}_composite_submit_L{layer}"
        out_run_dir = cfg.run_dir()
        if not os.path.exists(os.path.join(out_run_dir, "composite_meta.json")):
            cp.write_data(out_run_dir, [], meta=dict(
                study="composite", env="parabola", vector_type="certainty",
                inject_at="submit", layer=layer, frac=args.frac,
                alphas=neg_alphas, n_rollouts=args.steer_n_rollouts,
                repeats=1, branch_repeats=args.branch_repeats,
                null_branches=args.null_branches, gen_only=True,
                model_name=cfg.model_name, run_name=cfg.run_name,
                source_run_dir=run_dir))
        seeds = list(range(cfg.seed_start, cfg.seed_start + args.steer_n_rollouts))
        for case_id, seed in enumerate(seeds):
            done = os.path.join(out_run_dir, f"rollout_{case_id:04d}",
                                "composite_summary.json")
            if os.path.exists(done):                     # resumable
                continue
            env0 = build_env(cfg)
            env0.reset(seed=seed, wide=getattr(cfg, "wide_cases", True))
            opt = env0.optimum()
            print(f"[composite] case {case_id} (seed {seed})")
            run_composite(cfg, agent, vec, layer, seed, neg_alphas, out_run_dir,
                          case_id, steer_ctx=steering_active,
                          branch_extra=args.branch_extra, inject_at="submit",
                          opt=opt, case_id=case_id, rep=0,
                          branch_repeats=args.branch_repeats,
                          null_branches=args.null_branches, gen_only=True)
            ts_free()
        cp.rebuild_from_dirs(out_run_dir, write=True)
        cp.plot(out_run_dir, stars="steered")
        results = []
        for p in sorted(glob.glob(os.path.join(out_run_dir, "rollout_*",
                                               "composite_summary.json"))):
            with open(p) as f:
                results.append(json.load(f))
        nsumm = summarize_branch_null(results)
        print_branch_null(nsumm)
        with open(os.path.join(out_run_dir, "composite_null_summary.json"), "w") as f:
            json.dump({str(k): v for k, v in nsumm.items()}, f, indent=2)
        report["composite_run_dir"] = out_run_dir
        print(f"[steer] test 1 (branch-at-submit, alpha<0) -> {out_run_dir}")

    # ---- Test 2: whole-rollout alpha sweep from turn 1 (positive alpha = steer
    #      toward CERTAINTY early). run_steered saves per-alpha transcripts;
    #      composite_plot rebuilds the margin/gap-vs-turn figures. ----
    if not args.skip_sweep:
        alpha_dirs = []
        for alpha in args.alphas:
            rd = run_steered(cfg, agent, vec, layer, float(alpha),
                             n_rollouts=args.steer_n_rollouts, repeats=1,
                             base_run_name=args.steer_run_name,
                             env_kind="parabola", gen_only=True)
            alpha_dirs.append(rd)
        sweep_dir = os.path.join(cfg.out_dir,
                                 f"{args.steer_run_name}_parabola_L{layer}_steer")
        cp.rebuild_steer(alpha_dirs, sweep_dir, write=True)
        cp.plot(sweep_dir)
        report["sweep"] = {str(k): v for k, v in _steer_sweep_report(alpha_dirs).items()}
        report["sweep_run_dir"] = sweep_dir

    report.update(steer_info=info, directions=directions, alphas=list(args.alphas))
    with open(os.path.join(run_dir, "certainty_steering_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] steering phase complete; report -> "
          f"{os.path.join(run_dir, 'certainty_steering_report.json')}")


# =========================================================================== #
# CLI
# =========================================================================== #
def main():
    cfg = Config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["generate", "analyse", "steer", "all"],
                    default="all",
                    help="generate (GPU, resumable) | analyse (CPU-only) | "
                         "steer (GPU) | all")
    ap.add_argument("--run-name", default="csat_certainty",
                    help="run name for the generate phase (runs/<run-name>)")
    ap.add_argument("--run-dir", "--source-run-dir", dest="run_dir", default=None,
                    help="rollout dir for analyse/steer (default runs/<run-name>)")
    ap.add_argument("--n-rollouts", type=int, default=50,
                    help="parabola rollouts to generate (stage A)")
    ap.add_argument("--iters", type=int, default=10,
                    help="R: frozen-context samples per turn (default 10)")
    ap.add_argument("--layer", type=int, default=22,
                    help="hidden-state index for capture/vectors/steering "
                         "(22 = output of decoder block 21)")
    ap.add_argument("--capture-layers", choices=["single", "all"], default="single",
                    help="store only --layer per iteration (default; keeps storage "
                         "manageable) or all hidden states")
    ap.add_argument("--exclude-window", type=int, default=20,
                    help="tokens dropped immediately BEFORE the action verb "
                         "(the verb itself and everything after are also dropped)")
    ap.add_argument("--min-reasoning-tokens", type=int, default=1,
                    help="minimum usable pre-action tokens for an iteration to "
                         "enter the vector averages")
    ap.add_argument("--n-bins", type=int, default=4,
                    help="certainty quantile bins (default 4: c0..c3)")
    ap.add_argument("--min-valid", type=int, default=2,
                    help="minimum valid iterations per turn (else the turn is "
                         "skipped; variance undefined below 2)")
    ap.add_argument("--recompute-pool", action="store_true",
                    help="ignore the pooled_vectors.npz cache")
    ap.add_argument("--max-lines", type=int, default=12,
                    help="rollouts drawn in the stage-G trajectory plot")
    ap.add_argument("--model", default=cfg.model_name, help="HF id override")
    ap.add_argument("--n-obj", type=int, default=2,
                    help="parabola dimensionality (spec: 2 -> weights w0, w1)")
    ap.add_argument("--max-turns", type=int, default=cfg.max_turns)
    ap.add_argument("--seed-start", type=int, default=cfg.seed_start)
    ap.add_argument("--temperature", type=float, default=cfg.temperature,
                    help="sampling temperature; must be > 0")
    ap.add_argument("--out-dir", default=cfg.out_dir)
    # stage H
    ap.add_argument("--frac", type=float, default=0.4,
                    help="steering magnitude as a fraction of mean token norm")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[-1.0, -0.5, 0.0, 0.5, 1.0],
                    help="steering strengths (negatives also feed the "
                         "branch-at-submit test)")
    ap.add_argument("--steer-n-rollouts", type=int, default=12,
                    help="cases per alpha in stage H")
    ap.add_argument("--steer-run-name", default="csat_certsteer",
                    help="base run name for the stage-H runs")
    ap.add_argument("--branch-repeats", type=int, default=1,
                    help="steered continuations per branch point per alpha")
    ap.add_argument("--null-branches", type=int, default=3,
                    help="unsteered (alpha 0) re-branches from the same branch "
                         "point -- the resampling-honest null")
    ap.add_argument("--branch-extra", type=int, default=10,
                    help="max turns a branch may run past the baseline submit")
    ap.add_argument("--skip-composite", action="store_true",
                    help="stage H: skip test 1 (branch-at-submit)")
    ap.add_argument("--skip-sweep", action="store_true",
                    help="stage H: skip test 2 (whole-rollout alpha sweep)")
    args = ap.parse_args()

    cfg.model_name = args.model
    cfg.run_name = args.run_name
    cfg.out_dir = args.out_dir
    cfg.n_obj = args.n_obj
    cfg.max_turns = args.max_turns
    cfg.seed_start = args.seed_start
    cfg.temperature = args.temperature
    if args.run_dir is None:
        args.run_dir = os.path.join(args.out_dir, args.run_name)

    if args.phase in ("generate", "all"):
        _phase_generate(cfg, args)
    if args.phase in ("analyse", "all"):
        _phase_analyse(args)
    if args.phase in ("steer", "all"):
        _phase_steer(args)


if __name__ == "__main__":
    main()
