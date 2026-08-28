#pragma once
#include <cstddef>

class IncumbentSink;
struct DispatchOutcome;
struct ExecutionContext;
struct HeuristicBudget;
struct ProblemView;

namespace fj {

// Patience: `mip_heuristic_fj_patience`, a multiple of `nnz << 10` — the
// same unit as this heuristic's effort option (issue #111, made an option
// by #106, put on the effort unit by #116).  FJ's counter is in step
// units, so its value is not comparable with the other three heuristics'
// — do not read equal numbers as equal tolerances.
//
// Scope: **per worker**, matching `mip_heuristic_fj_effort`, which sizes
// one worker's allowance rather than a whole dispatch
// (`budget_is_per_worker` in mode_dispatch's `kChain`).  The
// runner-level gate is therefore `num_workers` times this, and the
// worker-level gate is the value itself.  FJ is the only entry with that
// scope; the other three size a whole dispatch and are divided across
// the pool.
//
// The default 256 is `nnz << 8`, the value FJ has always used and the
// model the other three were moved onto: exactly a quarter of FJ's
// default per-worker budget (`nnz << 10`).  0 disables the gate
// entirely.  The default is registered in
// `third_party/highs_patch/apply_patch.cmake` and pinned by
// `tests/test_smoke.cpp`; `docs/PARAMETERS.md` carries the calibration
// notes.

// Runs N continuous `parallel::for_each` FjWorkers with per-worker
// self-termination, each seeded differently; a worker that finishes is
// rebuilt in place with a fresh seed.  Set `threads=1` for a single
// worker whose behaviour is reproducible under a fixed `random_seed`.
//
// Implements the uniform runner contract; see heuristic_context.h.
DispatchOutcome run(const ProblemView& problem, const HeuristicBudget& budget,
                    ExecutionContext& exec, IncumbentSink& sink);
}  // namespace fj
