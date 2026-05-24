# Plan: Moving All CPU Work Off the Hot Path – Sparse + Manifest (Fixed-U + Grad-Accum)

**Branch**: `feature/upstream-sparse-runtime`  
**Primary file**: `nanochat/dynamic_vocab.py`  
**Date**: 2026-05  
**Status**: Detailed plan recorded (ready for execution or further refinement)  
**Author context**: Captured from deep exploration + background subagent analysis + previously documented race condition.

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

Update `PersistentSlotPrefetchState` dataclass to hold the new result type.

Add strong comments and (optionally) a debug-mode runtime check that detects mutations from background threads.

**Deliverable**: A version of the persistent slot logic that is safe to run from `_stage_prefetch_executor`.

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

- Full before/after timing comparison on realistic manifest runs (hundreds to thousands of steps).
- Ensure all existing tests pass, especially `tests/test_dynamic_vocab_grad_accum.py` (run with `PYTHONPATH=.`).
- Add or update runtime assertions / debug modes that would have caught the original race.
- Update code comments, the workspace `MEMORY.md` entry, and `docs/sparse-manifest-run.md` if user-visible behavior or flags change.
- Clean up any temporary scaffolding or duplicated logic introduced during the phases.
- Consider adding a "CPU offload level" or feature flag for future experiments.

---

## 5. Key Files & Functions

| File | Key Locations | Role |
|------|---------------|------|
| `nanochat/dynamic_vocab.py` | `_prepare_fixed_step` (1504–1927), `prepare_step` (1928), `prefetch_step` (1399), `_compute_persistent_slot_state_cpu` (1260) + `_for_meta` wrapper, `apply_accumulated_gradients` (2179), `_start_grad_accum_window` (753), staging helpers, `__init__` | The entire hot path and prefetch machinery |
| `scripts/base_train.py` | Lines 1013 (prepare), 1093 (prefetch), 1115 (accumulate), 1180 (apply), timing accumulation | Training loop ordering and measurement |
| `nanochat/dataloader.py` | `tokenizing_distributed_data_loader_with_state_bos_bestfit_manifest` (~327), `apply_manifest_transition`, step_meta construction (~762) | Source of `step_meta` tensors |
| `docs/plan-cpu-work-off-hot-path.md` | This file | The plan itself |
| `.grok/memory/teddy-831f2fbd/MEMORY.md` | Sparse Manifest / Fixed-U Runtime section | Durable record of the race condition |

---

## 6. Verification Strategy

Because intermediate runnable states are not required:

- **Primary signal**: The existing `--sparse-debug-timing` + per-category `sparse_prep_*_ms` breakdowns (including the new `prep_persistent_slot_ms` and `prep_active_vocab_build_ms` timers).
- **Secondary signal**: `sparse_boundary_overhead_ms` (time between end of prepare and start of compiled model call).
- **Correctness**:
  - `PYTHONPATH=. pytest tests/test_dynamic_vocab_grad_accum.py -q` after every significant change.
  - Short (1–few optimizer step) manifest runs comparing loss / core metrics against a known-good baseline on the same manifest.
  - Manual or assertion-based inspection of slot maps at window boundaries during development.
- **Longer validation**: At the end of major phases, run hundreds of steps and compare loss curves + final metrics.
- **Race detection**: Keep or enhance the test that would have caught the original "missing live fixed-U rows" failure.

---

## 7. Rollback & Safety

- Git is the primary rollback mechanism (feature branch is not yet pushed).
- Each phase should be reviewable as a logical unit.
- Keep the "safe" (foreground) path working until Phase 3 is validated.
- The workspace memory entry + comments in `prefetch_step` act as a permanent warning about the mutation hazard.

---

## 8. References & Prior Work

- Workspace memory entry on the persistent slot race (highly recommended reading before touching `prefetch_step` or `_compute_persistent_slot_state_*`).
- Existing `PersistentSlotPrefetchState` dataclass and prefetch scaffolding (already partially implemented).
- Rich timing instrumentation added in recent commits on this branch.
- `docs/sparse-manifest-run.md` (user-facing instructions for the path being optimized).

---

**Implementation status** (branch `cursor/cpu-work-phase2-prefetch-58ac`):

- **Phase 1**: Done (merged via `cursor/cpu-work-phase1-58ac`).
- **Phase 2**: `PersistentSlotUpdate` + `_plan_persistent_slot_update_cpu` / `_apply_persistent_slot_update_cpu` (compute/apply split).
- **Phase 3**: Persistent-slot prefetch re-enabled in `prefetch_step` for next-window microstep 0.

**Next step decision point** (for the implementer):

- **Phase 0** baseline capture on a real manifest run with `--sparse-debug-timing`?
- **Phase 4** dataloader / manifest hoisting?

This plan is deliberately phased so that valuable progress can be made (and measured) even if later, higher-risk phases are rolled back.