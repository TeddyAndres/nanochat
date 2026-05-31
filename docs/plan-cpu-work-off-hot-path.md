# Plan: Moving All CPU Work Off the Hot Path – Sparse + Manifest (Fixed-U + Grad-Accum)

**Branch**: `feature/upstream-sparse-runtime`  
**Primary file**: `nanochat/dynamic_vocab.py`  
**Date**: 2026-05 (reviewed & strengthened 2026-05 post-Phase 1)  
**Status**: Phases 2+3 core implemented (pure compute + apply split + guard + tests + prefetch re-enabled). Phase 1 baseline + new validation gates in place. Remaining: full CUDA prefetch test, longer loss-curve runs, Phase 4/5 polish.  
**Author context**: Captured from deep exploration + background subagent analysis + previously documented race condition. Updated with additional test recommendations from implementation review.

---

## 1. Context & Motivation

The sparse + manifest training path (`--sparse-mode --sparse-manifest <path>`) with `grad_accum_steps > 1` has a significant amount of Python/CPU work on the per-microstep critical path inside `DynamicVocabRuntime`.

This work sits directly in front of `torch.compile` + CUDA graph execution and is a major limiter of throughput.

Key sources of CPU work (exhaustively catalogued via code analysis):
- Repeated `.detach().to(device="cpu", dtype=torch.long)` + clones on every `step_meta` tensor coming from the dataloader.
- Padding logic when manifest `u_max` < runtime `fixed_u_max`.
- Heavy use of `torch.full((vocab_size,))`, `nonzero(as_tuple=False)`, `index_select`, `scatter_`, `fill_`, advanced indexing, and `torch.cat` on the large persistent slot maps.
- Arriving/leaving slot assignment logic (`_compute_persistent_slot_state_cpu`).
- Per-micro staging decisions, union IO remapping via `_grad_accum_wte_local_to_slot_cpu`, logit mask maintenance, and construction of `active_vocab`/`optimizer_state` dicts passed to the compiled model.
- Writeback queuing, future waits, and small tensor factories.

Existing infrastructure already moves *some* work off the path:
- Row data prefetch via `_stage_prefetch_executor`.
- Apply-stage prefetch.
- Background CPU writebacks via `_cpu_writeback_executor`.
- Scaffolding for persistent slot prefetch (`PersistentSlotPrefetchState`, `_compute_persistent_slot_state_for_meta`, `prefetched_persistent_slot_*` parameters, timers).

However, the most expensive remaining piece — the persistent stable slot / overlap-reuse map logic — is **intentionally kept on the foreground path** due to a known race condition.

### The Critical Race Condition (Documented in Workspace Memory)

See: `.grok/memory/teddy-831f2fbd/MEMORY.md` → section "Sparse Manifest / Fixed-U Runtime → Critical Race Condition — Persistent Slot Map Prefetch"

**Root cause**: `_compute_persistent_slot_state_cpu` mutates live structures (`_fixed_input_global_to_slot_cpu`, `fixed_input_slot_to_global_cpu`, `fixed_lm_head_slot_to_global_cpu`, `_grad_accum_wte_local_to_slot_cpu`, etc.). These are read by `apply_accumulated_gradients()` and continuing microsteps. Running this computation from the prefetch executor caused the next window’s slot assignments to leak in before the current window’s apply finished.

**Symptom**: "Sparse grad accumulation map is missing live fixed-U rows"

**Current safe state**: Persistent slot work is forced to the foreground (future is explicitly set to `None` in `prefetch_step`). Only non-mutating row prefetch runs async.

This race is easy to reintroduce. Any plan must solve the mutation problem rather than just "turning the prefetch back on."

---

## 2. Goals

1. **Primary**: Move as much CPU calculation as possible off the per-microstep hot path for the manifest fixed-U + `grad_accum_steps > 1` code path.
2. **Secondary**: Preserve (or improve) correctness, debuggability, and the existing rich timing instrumentation.
3. **Tertiary**: Make the persistent slot logic safely runnable in the background (the biggest single remaining win).
4. Enable higher sustained throughput without changing training semantics or loss curves.

**Non-goals** (per user): No requirement to keep intermediate states runnable. Rollback is acceptable if a phase fails.

---

## 3. Constraints & Risks

- **Mutation hazard** (highest risk): The persistent slot maps must never be mutated from a background thread while `apply_accumulated_gradients` or microstep logic may be reading them.
- `torch.compile` + CUDA graph sensitivity: Changes to shapes, Python object construction, or control flow crossing the `prepare_step` → `model()` boundary can cause recompiles or graph breaks.
- Grad-accum window semantics must remain exactly correct (`preserve_resident_grads` vs. new window, `is_grad_accum_boundary`, etc.).
- The dataloader already does substantial work producing `step_meta` (clones, transitions, union local maps). We should not duplicate effort.
- Existing timing breakdown (`sparse_prep_*_ms`, `prep_persistent_slot_ms`, `prep_active_vocab_build_ms`, etc.) must continue to work for measurement.

---

## 4. Recommended Phased Approach

### Phase 0: Instrumentation & Baseline (Low risk, high diagnostic value)

**Goal**: Make every remaining CPU cost visible before making structural changes.

**Work**:
- Audit and extend the existing timing points inside `_prepare_fixed_step` and helpers.
- Add a top-level timer around the entire body of `_prepare_fixed_step` (excluding future waits) if not already captured.
- Ensure the padding logic (lines ~1575-1582), `active_vocab` dict construction, and all per-micro `detach().to` / small tensor work have dedicated or clearly attributable timers.
- Add a one-time "CPU work summary" log (under `--sparse-debug-timing`) that prints a breakdown after the first few steps.
- Run controlled short manifests (1–5 optimizer steps) with `--sparse-debug-timing --sparse-debug-sync-after-backward` on the current branch to capture a precise before baseline.

**Deliverable**: A clear "before" profile showing the relative cost of persistent slot work vs. per-micro staging vs. dict building vs. padding vs. miscellaneous.

**Rollback risk**: Very low.

---

### Phase 1: Eliminate Cheap/Redundant Per-Micro CPU Work (High impact, low risk)

Focus on work that runs on **every microstep**, including inside `preserve_resident_grads`.

**Target items** (drawn from the exhaustive catalog):
- Repeated `step_meta[...].detach().to(device="cpu", dtype=torch.long)` calls. Consider accepting that the dataloader already produces CPU tensors and only clone when truly necessary.
- The manifest vs. runtime `u_max` padding copies (`torch.full` + `.copy_()`). Either enforce `u_max` alignment at manifest build time or allocate the runtime buffers to exactly the manifest size.
- `torch.full((vocab_size,))` allocations inside the hot path (leverage and extend the existing persistent `_fixed_input_global_to_slot_cpu` pattern).
- Small repeated factories (`_empty_long_cpu()`, `torch.arange(...)`, `torch.zeros(...)` for masks).
- Construction of the `step_active_vocab` and `step_optimizer_state` dicts (already timed; explore pre-building or flattening these structures).
- Union IO remapping indexing (when `_grad_accum_wte_local_to_slot_cpu` is active).
- Redundant `torch.equal` checks and flag recomputation when values are stable across a window.

**Approach**: Incremental, measurable changes. After each sub-item, re-run the timing baseline from Phase 0.

**Risk**: Low. Mostly local simplifications.

---

### Phase 2: Refactor Persistent Slot Logic to Be Side-Effect-Free (The Enabling Step)

This is the core of the documented race and the biggest remaining CPU cost on window boundaries.

**Current problem**:
- `_compute_persistent_slot_state_cpu` (and the `_for_meta` wrapper) mutates live maps in-place.
- It is called (via fallback) on every new grad-accum window inside `_prepare_fixed_step`.
- The scaffolding to run it in the background already exists but is deliberately disabled.

**Required change**:
1. Refactor `_compute_persistent_slot_state_cpu` into a pure function (or clearly separated compute + apply phases).
2. It should return a rich result object (e.g., `PersistentSlotUpdate` or `SlotTransitionResult`) containing:
   - All deferred writeback ids/slots (input + lm_head)
   - `input_stage_ids_cpu` / `input_stage_slot_ids_cpu`
   - The updated `local_to_slot` slice for the window (`_grad_accum_wte_local_to_slot_cpu` content)
   - Any other derived state needed by the rest of prepare
3. Remove (or make optional) all in-place mutations to the live maps inside the compute function.
4. Introduce an atomic "apply" step on the main thread that performs the mutations after the background work (or sync fallback) has completed.
5. Update call sites in `_prepare_fixed_step` and any readers in `apply_accumulated_gradients` / microstep logic.

**Design options** (choose one after review):
- **Option A (preferred for clarity)**: Return a complete next-state snapshot for the affected maps + a set of deltas. Apply is a small, fast function.
- **Option B**: Return only the decisions (arriving/leaving/assigned slots). The apply phase replays the decisions against the live maps.
- **Option C**: Use read-only snapshots of the maps at the start of window computation (cheaper than full copies for the big vocab-sized maps).

Update `PersistentSlotPrefetchState` dataclass (or introduce a dedicated `PersistentSlotTransition` / `SlotStateResult` dataclass) to serve as the actual typed carrier for the rich result object returned by the compute function. All call sites (prefetch consumption, fallback, `_prepare_fixed_step`) must be updated to use it.

**Mandatory hardening (Phase 2 deliverable)**:
- Add strong comments documenting the mutation contract and thread-safety expectations around every live map.
- Add a debug-mode runtime check (enabled by `NANOCHAT_SPARSE_DEBUG=1` or an equivalent internal flag, cheap when disabled) that asserts the mutating compute path is only ever entered from the main thread (or while no `apply_accumulated_gradients` / resident-grads readers are active). This directly prevents re-introduction of the original race.

**Isolated unit tests (Phase 2 deliverable)**:
- Add focused unit tests (in `tests/test_dynamic_vocab_grad_accum.py` or a new `test_persistent_slot_compute.py`) that exercise `_compute_persistent_slot_state_cpu` (and the `_for_meta` wrapper) in isolation. These tests supply controlled prior map state + a new `grad_accum_ids_cpu` set and assert the exact returned dict (or result object) contents plus any side effects on the input maps (before the refactor) or on the result object (after). This makes the compute-vs-apply split testable without running full windows.

**Deliverable**: A version of the persistent slot logic that is safe to run from `_stage_prefetch_executor`, accompanied by the typed result object, the debug guard, and the isolated unit tests.

**Risk**: Medium-High. This is the most invasive single change. Rollback is expected to be used if needed.

---

### Phase 3: Re-enable Persistent Slot Prefetch (The Big Win)

Once Phase 2 is complete:

- Re-enable submission of persistent slot work in `prefetch_step` (remove the `= None` guard and the "keep on foreground" comment).
- Ensure the consumption path in `prepare_step` / `_prepare_fixed_step` correctly handles hits, misses, and first-window cases using the already-wired `prefetched_persistent_slot_*` parameters.
- Measure the reduction in `prep_persistent_slot_ms` (and overall `sparse_prepare_ms`) on window boundaries.
- Validate that `apply_accumulated_gradients` and microstep logic still see consistent maps.

**Risk**: Medium (the mutation hazard should now be eliminated by Phase 2).

---

### Phase 4: Further Hoisting and Structural Improvements

- Push more transition / slot decision work into the dataloader or manifest builder (so the runtime receives more "already decided" data in `step_meta`).
- Explore per-window read-only snapshots of the critical maps instead of repeated index operations.
- Investigate whether the handoff of `active_vocab` / `optimizer_state` / `logit_mask` to the compiled model can be made cheaper (pre-built structures, different representation, etc.).
- Reduce or eliminate CPU work that occurs between the end of `prepare_step` and the `model()` call (the `sparse_boundary_overhead_ms` metric).

---

### Phase 5: Measurement, Hardening, Documentation, and Cleanup

- Full before/after timing comparison on realistic manifest runs (hundreds to thousands of steps), using the reproducible harness from the Verification Strategy (the new `dev/sparse_cpu_offload_bench.py` or equivalent).
- Ensure **all** tests pass, especially the extended `tests/test_dynamic_vocab_grad_accum.py` (and any new `test_*_prefetch.py` variant) run with `PYTHONPATH=.`. The prefetch-exercising variant must be part of the pre-Phase-3 gate.
- Add or update runtime assertions / debug modes that would have caught the original race (the mandatory background-mutation guard from Phase 2 is the primary artifact here).
- Create / land the isolated unit tests for the persistent slot compute logic (Phase 2 deliverable) and the CUDA + `prefetch_step` coverage test (Phase 3 gate).
- Update code comments (especially around `prefetch_step`, `_compute_persistent_slot_state_*`, and the live maps), the workspace `MEMORY.md` entry (or its committed equivalents in the source), and `docs/sparse-manifest-run.md` if user-visible behavior or flags change.
- Clean up any temporary scaffolding or duplicated logic introduced during the phases.
- Consider adding a "CPU offload level" or feature flag for future experiments.
- Ensure the `PersistentSlotPrefetchState` (or its successor result dataclass) is fully documented and used consistently.

---

## 5. Key Files & Functions

| File | Key Locations | Role |
|------|---------------|------|
| `nanochat/dynamic_vocab.py` | `_prepare_fixed_step`, `prepare_step`, `prefetch_step`, `_compute_persistent_slot_state_cpu` + `_for_meta` wrapper, `apply_accumulated_gradients`, `_start_grad_accum_window`, staging helpers, `__init__` | The entire hot path and prefetch machinery (note: line numbers in this table are historical; use `grep` for current locations) |
| `scripts/base_train.py` | prepare/prefetch/accumulate/apply call sites + timing accumulation around lines 1013–1180 | Training loop ordering and measurement |
| `nanochat/dataloader.py` | `tokenizing_distributed_data_loader_with_state_bos_bestfit_manifest`, `apply_manifest_transition`, step_meta construction | Source of `step_meta` tensors (already performs useful union remapping work) |
| `tests/test_dynamic_vocab_grad_accum.py` | Full file (especially `_build_runtime`, `_step_meta`, `_run_window`, and the final equivalence assertions on weights / optimizer state / counters) | Primary correctness oracle for slot assignment, overlap reuse, writebacks, and grad-accum window transitions. Must pass after every change; will be extended for prefetch coverage. |
| `dev/sparse_cpu_offload_bench.py` (to be added in Phase 5) | New small script | Reproducible before/after timing harness for the `sparse_prep_*_ms` family (see Verification Strategy) |
| `docs/plan-cpu-work-off-hot-path.md` | This file | The plan itself |
| `.grok/memory/teddy-831f2fbd/MEMORY.md` | Sparse Manifest / Fixed-U Runtime section | Durable record of the race condition (local to author; code comments in `prefetch_step` and `_compute_*` serve as the committed equivalent) |

---

## 6. Verification Strategy

Because intermediate runnable states are not required:

- **Primary signal**: The existing `--sparse-debug-timing` + per-category `sparse_prep_*_ms` breakdowns (including `prep_persistent_slot_ms`, `prep_cpu_reuse_map_ms`, `prep_active_vocab_build_ms`, and `sparse_boundary_overhead_ms`).
- **Secondary signal**: `sparse_boundary_overhead_ms` (time between end of `prepare_step` and start of the compiled `model()` call) and overall `sparse_prepare_ms`.

### Correctness (run after every significant change)
- `PYTHONPATH=. pytest tests/test_dynamic_vocab_grad_accum.py -q` (the core equivalence test against `disable_overlap_reuse`).
- Short (1–few optimizer step) manifest runs comparing loss / core metrics against a known-good baseline on the *same* manifest.
- Manual or assertion-based inspection of slot maps (`fixed_input_slot_to_global_cpu`, `_grad_accum_wte_local_to_slot_cpu`, etc.) at window boundaries during development.

### Longer validation (end of major phases)
- Hundreds to thousands of steps on realistic manifests; compare full loss curves + final metrics (weights, optimizer state, hot activation counts, etc.) against a no-prefetch / foreground-only baseline.

### Race detection & mutation hazard (mandatory)
- Keep/enhance the existing "Sparse grad accumulation map is missing live fixed-U rows" assertions (in both `_prepare_fixed_step` and `apply_accumulated_gradients`).
- The **mandatory** debug-mode thread/mutation guard added in Phase 2 (see Phase 2 section) must be present and pass in debug builds.
- A stress-oriented test (or extension of the grad-accum test under `NANOCHAT_SPARSE_DEBUG=1`) that exercises rapid window transitions + concurrent-looking apply calls (even if serialized) to ensure the guard would have fired on the original bug.

### Additional tests required before Phase 3 (the "prefetch re-enable" gate)

These close the coverage gaps that existed after Phase 1:

1. **Isolated compute-function unit tests (Phase 2 deliverable)**  
   Add direct tests for `_compute_persistent_slot_state_cpu` and `_compute_persistent_slot_state_for_meta`.  
   - Supply a controlled prior state of the fixed-slot maps + a realistic `grad_accum_ids_cpu` tensor.  
   - Assert the exact contents of the returned result object (deferred writebacks, stage ids/slots, `local_to_slot` slice, timing, flags).  
   - After the refactor, the tests must pass against both the pure-compute path *and* the foreground apply path.  
   - These tests become the primary regression suite for any future changes to arriving/leaving logic.

2. **CUDA + prefetch_step coverage test (must exist and pass before re-enabling submission in Phase 3)**  
   Extend `test_dynamic_vocab_grad_accum.py` (or add a sibling `test_dynamic_vocab_grad_accum_prefetch.py`, marked slow/CUDA) that:
   - Instantiates the runtime on CUDA when available (falls back to CPU-only behavior gracefully).
   - Mimics the real training loop: after each window (or at the appropriate micro-step boundary), calls `dynamic_vocab.prefetch_step(next_meta)`.
   - Runs multiple overlapping grad-accum windows with realistic arrival/leaving patterns.
   - Still asserts full numerical equivalence (weights, optimizer state, counters) against the `disable_overlap_reuse` baseline.
   - Explicitly exercises both *hit* (prefetched result consumed) and *miss* (fallback synchronous compute) paths for the persistent slot future.
   - Verifies that `apply_accumulated_gradients` and resident-grads microsteps see consistent maps in both hit and miss cases.

3. **Reproducible timing / measurement harness (Phase 5 deliverable)**  
   Add a small script (e.g. `dev/sparse_cpu_offload_bench.py`) that:
   - Runs a tiny but fixed manifest (1–5 optimizer steps, small vocab) under `--sparse-debug-timing --sparse-debug-sync-after-backward`.
   - Prints (and optionally writes JSON) a stable summary of the key counters: `prep_persistent_slot_ms` (per window and average), `sparse_boundary_overhead_ms`, `prep_cpu_reuse_map_ms`, etc.
   - Can be invoked with a `--baseline` or `--compare` mode for easy before/after diffs.
   - This script (or its output) is the canonical artifact for Phase 0 baseline capture and Phase 5 final measurement claims.

These three additions ensure that (a) the heavy logic is unit-tested, (b) the newly-enabled async path is actually exercised by CI/tests, and (c) performance wins are measured repeatably rather than via one-off log inspection.

---

## 7. Rollback & Safety

- Git is the primary rollback mechanism (feature branch is not yet pushed).
- Each phase should be reviewable as a logical unit.
- Keep the "safe" (foreground) path working until Phase 3 is validated.
- The workspace memory entry + comments in `prefetch_step` act as a permanent warning about the mutation hazard.

---

## 8. References & Prior Work

- Workspace memory entry on the persistent slot race (highly recommended reading before touching `prefetch_step` or `_compute_persistent_slot_state_*`; the committed comments in those functions are the durable source of truth for contributors).
- Existing `PersistentSlotPrefetchState` scaffolding and the `_compute_*_for_meta` / prefetched state wiring (Phase 1 partial implementation).
- Rich timing instrumentation + Phase 1 per-micro reductions (see recent commits on this branch and `tests/test_dynamic_vocab_grad_accum.py`).
- `docs/sparse-manifest-run.md` (user-facing instructions for the path being optimized).
- The three additional test artifacts mandated in Section 6 (isolated compute tests, CUDA prefetch coverage test, reproducible `sparse_cpu_offload_bench.py`) are considered first-class deliverables of Phases 2/3/5.

---

**Next step decision point** (for the implementer, post-Phase 1):

- The instrumentation baseline (Phase 0) and cheap per-micro wins (Phase 1) are largely complete on this branch. The next major step is detailed design + implementation of **Phase 2** (side-effect-free persistent slot refactor). This is the gate for everything else.
- When implementing Phase 2, the isolated unit tests for the compute function and the mandatory debug-mode mutation guard must be delivered together with the refactor.
- Before merging the Phase 3 change that re-enables `_pending_persistent_slot_future` submission, the CUDA + `prefetch_step`-exercising coverage test (Section 6) must exist and pass.
- Use the (to-be-added) `dev/sparse_cpu_offload_bench.py` for all before/after timing claims in Phases 3 and 5.

This plan is deliberately phased so that valuable progress can be made (and measured) even if later, higher-risk phases are rolled back. The strengthened test gates above exist precisely to make the high-risk phases safe to land.