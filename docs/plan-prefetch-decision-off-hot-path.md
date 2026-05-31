# Plan: Eliminating Per-Microstep Prefetch Decision Work – High Grad-Accum Sparse Manifest (Fixed-U)

**Branch**: `feature/upstream-sparse-runtime` (continuation)  
**Primary files**: `nanochat/dynamic_vocab.py`, `scripts/base_train.py`, `nanochat/dataloader.py`  
**Date**: 2026-05  
**Status**: New plan (post Phases 1-3 of original CPU-offload plan). Focus: the new dominant steady-state cost (~350-380ms `prefetch_launch` per microstep in 20-26× grad-accum regimes).  
**Context**: Profiling from d22 + 26× accum runs with `--sparse-debug-timing` shows that after successfully moving persistent slot *computation* off the hot path, the repeated *decision* logic for row prefetch (and associated persistent slot submission setup) now dominates main-thread CPU time inside long grad-accum windows.

---

## 1. Context & Motivation

After the successful completion of the original "Moving All CPU Work Off the Hot Path" plan (Phases 1-3), the persistent slot arriving/leaving decision logic and map mutations are now pure and safely runnable in the background via `_stage_prefetch_executor`. `prep_persistent_slot_ms` is near-zero in steady state, and `post_prep_python` overhead is negligible.

However, detailed timing in long grad-accum runs (20-26×) reveals a new dominant cost on the critical path:

- `sparse_next_fetch_ms` / the "fetch" bucket inside `fwdbwd` timing: consistently 350-390 ms per microstep.
- Breakdown: `loader: ~0.25 ms`, `h2d: ~5-6 ms`, **`prefetch_launch: 347-380 ms`**.

This `prefetch_launch` cost is spent on the **main thread** inside `DynamicVocabRuntime.prefetch_step()` (called for the *next* meta after every microstep in `base_train.py`).

The root cause is repeated execution of `_build_fixed_prefetch_requests()` (and related map queries before submitting the persistent slot task) on **every microstep**, even deep inside `preserve_resident_grads` windows where the inputs to the decision (current `grad_accum_ids_cpu` + persistent slot maps) are stable.

In a 26× grad-accum window this decision work runs 26 times per optimizer step instead of once. At d22 scale + large `grad_accum_u_max` (~27k) this costs hundreds of milliseconds of main-thread CPU per microstep — directly limiting steady-state throughput on long runs.

### Key Sources of the Current Cost (exhaustively from code + profiling)
- `_build_fixed_prefetch_requests` (dynamic_vocab.py:1306):
  - `index_select` on vocab-sized persistent inverse maps (`_fixed_input_global_to_slot_cpu`, etc.) using `grad_accum_ids_cpu`.
  - Boolean masking + `nonzero` to compute "arriving" ids for the next window (both input tables and lm_head).
  - Similar logic for lm_head reuse detection.
- Setup for the (now offloaded) persistent slot submission.
- All of the above executed unconditionally every microstep via the training loop after backward.

Existing infrastructure already provides the necessary signals:
- `grad_accum_steps`, `grad_accum_micro_step`, `is_grad_accum_boundary` in step_meta.
- `preserve_resident_grads` logic and `_same_grad_accum_window_ids`.
- Full window `grad_accum_active_ids` and union locals precomputed in the dataloader.
- `PersistentSlotTransition` + background executor pattern (proven by Phases 2-3).

---

## 2. Goals

1. **Primary**: Reduce the per-microstep `prefetch_launch` / `sparse_next_fetch_ms` cost by 5-10× (or more) inside long grad-accum windows by eliminating redundant decision work.
2. Preserve (or improve) correctness, debuggability, and the rich `--sparse-debug-timing` instrumentation.
3. Keep the design compatible with `torch.compile` + CUDA graphs (minimal new control flow or shape variation on the hot path).
4. Enable higher sustained tokens/sec on realistic long runs (hours) without changing training semantics or loss curves.

**Non-goals**: No requirement for intermediate states to be runnable in all combinations. Full manifest hoisting can be a later phase.

---

## 3. Constraints & Risks

- **Correctness on window boundaries** (highest risk): Prefetch decisions must be fresh and correct exactly when `not preserve_resident_grads` (new window starts). Any caching must be invalidated at the right moment.
- Grad-accum window semantics (`preserve_resident_grads`, `is_grad_accum_boundary`, union remapping via `_grad_accum_wte_local_to_slot_cpu`) must remain identical.
- The prefetch decisions feed both row staging *and* the persistent slot transition submission. Changes must not re-introduce the original race.
- `torch.compile` + CUDA graph sensitivity: New Python objects or variable control flow crossing the prepare/prefetch boundary can cause recompiles.
- Existing timing breakdown must continue to work (new timers for "prefetch decision" vs "launch" are encouraged).
- The dataloader already does substantial per-window precomputation; we should leverage it rather than duplicate.

---

## 4. Recommended Phased Approach (Prioritized by Expected Time Gains)

### Phase 0: Instrumentation & Baseline (Low risk, high diagnostic value)

**Goal**: Make the cost of the decision logic visible separately from launch/submission.

**Work**:
- Add fine-grained timers inside `prefetch_step` and `_build_fixed_prefetch_requests` (e.g., `prefetch_decision_cpu_ms`, `prefetch_submit_overhead_ms`).
- Extend the printed `fetch_ms` breakdown (or add under `--sparse-debug-timing`) to show decision vs. actual background submission time.
- Update the synthetic bench (`dev/sparse_cpu_offload_bench.py`) and any ad-hoc microbench to report these new counters.
- Capture a precise "before" profile on the current branch with the latest d22 + high-accum manifest (100+ steps, focus on steady-state windows).

**Deliverable**: Clear quantification that X% of the 350-380 ms `prefetch_launch` is decision work that is repeated inside resident windows.

**Rollback risk**: Very low.

---

### Phase 1: Window-Level Caching of Prefetch Decisions (Highest expected impact)

**Goal**: Compute the expensive prefetch decisions (row targets + persistent slot work) **once per grad-accum window** instead of every microstep. Cache the resulting "prefetch plan" for the duration of the resident window.

**Core insight** (validated by code + profiling):
- Inside a `preserve_resident_grads` window, both `grad_accum_ids_cpu` (fixed for the window) and the persistent slot maps are stable.
- Therefore `_build_fixed_prefetch_requests` returns (nearly) identical results for all microsteps except the last one before the next boundary.
- The heavy persistent slot transition for the *next* window can also be prepared once (we already submit it via the executor when we see the next meta).

**Implementation steps** (detailed):
1. Introduce a small cached state on `DynamicVocabRuntime`:
   - `self._cached_prefetch_plan: Optional[dict]` (or a lightweight dataclass `PrefetchPlan` containing the `requests` dict + any next-window persistent slot submission key).
   - `self._cached_prefetch_for_grad_accum_ids: Optional[torch.Tensor]`
   - `self._cached_prefetch_window_id: Optional[int]` (or use the existing `grad_accum_ids_cpu` identity + a generation counter).

2. Refactor `prefetch_step`:
   - Early return / fast path if we are inside a resident window and the incoming meta's `grad_accum_ids_cpu` matches the cached one.
   - On a detected boundary (or first step of a new window), compute the full plan (current `_build...` + submission of the persistent slot task for the *next* window) and store it.
   - For non-boundary steps inside the window, reuse the cached plan to set `_pending_*_future` (still submit the actual row prefetch work if needed, but skip the decision CPU work).

3. Integrate with existing signals:
   - Use `is_grad_accum_boundary`, `preserve_resident_grads`, and `_same_grad_accum_window_ids` (already computed in `_prepare_fixed_step`).
   - Invalidate the cache on any path that calls `_start_grad_accum_window`, `_invalidate_fixed_live_state`, or `disable_fixed_overlap_reuse` changes.

4. For the persistent slot side specifically:
   - The submission of `_compute_persistent_slot_state_for_meta` for the *next* window can be done once per window (when we first see the meta for the upcoming window) instead of every microstep. The result is consumed exactly once (in the next `_prepare_fixed_step` on the boundary).

5. Keep the "safe" (compute-every-time) path for the first few steps of a run and under a debug flag for validation.

**Expected time gain**: 5-10× reduction in `prefetch_launch` inside long windows (the dominant case for production runs). The remaining cost would be the actual submission + any boundary work.

**Risk**: Medium. Requires careful cache invalidation on boundaries. Rollback is easy (just disable the fast path).

**Deliverable**: Measurable drop in steady-state `prefetch_launch` / overall step time on high-accum manifests while preserving loss curves.

---

### Phase 2: Move Remaining Decision Work into the Background Executor (Complementary to Phase 1)

**Goal**: Even on window boundaries (where fresh decisions are required), perform as much of the CPU decision logic as possible in a background thread rather than on the main thread.

**Approach**:
- Extend the existing pattern used for `_compute_persistent_slot_state_for_meta`.
- Create a lightweight "PrefetchDecisionPlan" task that can be submitted to `_stage_prefetch_executor` (or a new dedicated low-priority executor).
- The task receives the relevant slice of step_meta + current persistent map state (or read-only views) and returns a ready-to-use `PrefetchPlan` object (the `requests` + any other small metadata).
- In `prefetch_step` (or a new `maybe_refresh_prefetch_plan` helper), do a non-blocking check or short wait for the plan from the previous submission, falling back to a fast synchronous path only when necessary.
- On window boundaries, submit the decision task for the *following* window as early as possible (ideally right after we apply the current transition).

**Synergy with Phase 1**: Phase 1 gives the biggest win by skipping work entirely inside windows. Phase 2 reduces the cost of the (fewer) times we *do* need fresh decisions.

**Additional idea – "instructions to GPU" variant**:
- The background task can produce not just id lists, but small pre-computed tensors or even launch some of the staging work itself.
- The main thread receives tiny "instructions" (e.g., a small tensor of ids or a pre-filled request object) that it can hand directly to the row prefetch machinery with almost zero Python overhead.

**Risk**: Medium. Requires careful lifetime management of the maps passed to the background task (they must not be mutated while the task is running). The existing debug guard + `PersistentSlotTransition` pattern gives us a template.

**Deliverable**: Further reduction in main-thread CPU time visible in `prefetch_launch` and `sparse_next_fetch_ms`, especially on boundaries.

---

### Phase 3: Hoist Prefetch Plans into the Manifest / Dataloader (Highest leverage, longer-term)

**Goal**: Move the *knowledge* of what will need to be prefetched out of the runtime entirely and into the data pipeline, where it can be computed once per manifest build (or per window) with full global visibility.

**Opportunities visible in current dataloader**:
- The manifest already contains the full sequence of windows and `grad_accum_active_ids`.
- The dataloader already walks ahead (`buffered_next_step_entry`, `preview_next_transition`, `apply_manifest_transition`).
- It already builds `grad_accum_global_to_slot` and the union local maps per window.

**Implementation ideas**:
- Extend the manifest schema (or a sidecar file) with per-window "prefetch plan" entries: the exact sets of ids that will need staging for the next window under the fixed-U + overlap-reuse policy.
- Or, have the dataloader compute and attach a compact "next_window_prefetch_ids" (or the equivalent of today's `requests` dict) into `step_meta` for the current window.
- In the runtime, `prefetch_step` (or the cached plan logic from Phase 1) can then be a near-no-op or a trivial copy when the plan is supplied by the dataloader.

This is the ultimate realization of the original plan's Phase 4 guidance ("Push more transition / slot decision work into the dataloader or manifest builder").

**Risk**: Lower for correctness if we keep a runtime fallback, but higher for manifest format compatibility and builder complexity. Should be done after Phases 1-2 have proven the value.

**Deliverable**: `prefetch_launch` cost drops to low single-digit milliseconds (or less) even on boundaries, because the expensive decisions are done offline or in the data loading pipeline.

---

### Phase 4: Measurement, Hardening, Documentation, and Cleanup

- Full before/after comparison on realistic long manifests (thousands of steps) using the updated timing harness.
- Ensure all existing tests pass, especially `tests/test_dynamic_vocab_grad_accum.py` (run with `PYTHONPATH=.`).
- Add or extend tests that specifically stress long grad-accum windows + prefetch plan caching (including boundary transitions and `disable_fixed_overlap_reuse` mode).
- Update code comments, the workspace memory entry, and user-facing docs.
- Consider a "prefetch decision level" or feature flag (0 = always compute, 1 = window caching, 2 = full background + dataloader plans) for future experiments and A/B testing.
- Clean up any duplicated decision logic introduced during the phases.

---

## 5. Key Files & Functions

| File | Key Locations | Role |
|------|---------------|------|
| `nanochat/dynamic_vocab.py` | `_build_fixed_prefetch_requests` (1306), `prefetch_step` (1494), `_compute_persistent_slot_state_for_meta` (1521), `_prepare_fixed_step` (1695+), `_start_grad_accum_window`, `_same_grad_accum_window_ids`, `PersistentSlotTransition` | The entire prefetch decision machinery and window state tracking |
| `scripts/base_train.py` | `fetch` timing block (1081-1099), `plan_sparse_batch_meta` (590), step loop around prefetch/prepare | Where the cost is measured and prefetch_step is invoked every microstep |
| `nanochat/dataloader.py` | Manifest iteration, `grad_accum_*` construction (712-787), `preview_next_transition`, step_meta emission | Source of window-level information that can be hoisted |
| `tests/test_dynamic_vocab_grad_accum.py` | Existing equivalence test + new isolated tests | Primary correctness oracle for any caching or decision changes |
| `dev/sparse_cpu_offload_bench.py` | Synthetic timing harness | Reproducible before/after measurement for the new counters |

---

## 6. Verification Strategy

Because we are changing when (and how often) decisions are made rather than the decisions themselves:

- **Primary signal**: Reduction in `prefetch_launch` / `sparse_next_fetch_ms` (and overall step time) in steady-state windows, measured via `--sparse-debug-timing` and the updated harness. Target: 5-10× drop inside long grad-accum regions.
- **Correctness**:
  - `PYTHONPATH=. pytest tests/test_dynamic_vocab_grad_accum.py -q` after every change (must continue to pass with both caching enabled and disabled).
  - Extend the test with explicit long-window scenarios (many microsteps inside `preserve_resident_grads`) and boundary transitions.
  - Short manifest runs (100-500 steps) comparing loss + core metrics against a no-caching baseline on the *same* manifest and random seed.
  - Manual inspection (or added assertions under debug) that prefetch plans are invalidated exactly on window boundaries.
- **Longer validation**: Full runs (thousands of steps) with the new timing counters logged to wandb; confirm no regression in final loss curves or val metrics vs. the post-Phase-3 baseline.
- **Race / consistency detection**: Keep the existing "missing live fixed-U rows" assertions. Add targeted checks that a cached plan used on microstep N inside a window would have produced the same result as a fresh computation.
- **Performance microbench**: Update `dev/sparse_cpu_offload_bench.py` (and any ad-hoc scripts) to report the new fine-grained timers and run it as part of every significant change.

---

## 7. Rollback & Safety

- Git remains the primary mechanism.
- Each phase should be reviewable and independently toggleable (via a small internal flag or early-return in the caching logic).
- Keep the "compute-every-time" path working and exercised in tests until the end of Phase 4.
- The existing `NANOCHAT_SPARSE_DEBUG` guard + rich timing give strong observability.

---

## 8. References & Prior Work

- Original plan `docs/plan-cpu-work-off-hot-path.md` (especially Phase 4 hoisting ideas and the verification strategy).
- Post-Phase-3 profiling data from d6 (20×) and d22 (26×) runs showing the shift in bottleneck from persistent slot compute to per-microstep prefetch decisions.
- Existing background executor pattern (`_stage_prefetch_executor`, `PersistentSlotTransition`, the two pending-future mechanisms) — proven safe and effective.
- Manifest dataloader precomputation for grad-accum windows and union locals (already does more global work than the runtime).

---

**Prioritization Rationale for Biggest Time Gains**

- **Phase 1 (window caching)** is expected to deliver the single largest reduction because it attacks the repetition factor (20-26×) directly in the common case (long resident windows).
- **Phase 2 (background decisions)** provides complementary gains on the (now much rarer) boundary computations and keeps the main thread as lean as possible.
- **Phase 3 (dataloader hoisting)** has the highest leverage long-term but higher implementation cost and is best done after the runtime caching pattern has been validated.

This plan is deliberately incremental: valuable wins are available in Phase 1 with relatively low risk, while still leaving the door open for deeper architectural hoisting later.

**Next step decision point** (for the implementer):

- Start with **Phase 0** (instrument the decision vs. launch split)?
- Jump straight to **Phase 1** (window-level caching of the prefetch plan), since it targets the repetition that dominates current profiles?
- Prototype the background decision task (Phase 2) in parallel because the executor infrastructure is already there?

This plan focuses on the highest-ROI algorithmic changes visible from current profiling and code structure while staying true to the "move CPU work off the hot path" philosophy of the original effort.