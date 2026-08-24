#pragma once

#include <cstddef>

class IncumbentSink;
struct ExecutionContext;
struct HeuristicBudget;
struct ProblemView;

namespace scylla {

// Stall threshold: `mip_heuristic_scylla_stall`, in effort units (PDLP
// iterations x nnz) per constraint-matrix nonzero (issue #111, made an
// option by #106).  Scylla's counter is in a different unit from FPR's
// and LocalMIP's, so the values are not comparable across heuristics.
//
// Scope: **whole dispatch**, matching `mip_heuristic_scylla_effort` —
// with one documented deviation: Scylla hands its workers the
// *dispatch-level* value rather than `HeuristicBudget::worker_stale`
// (see scylla.cpp), because a Scylla worker's own counter is already
// charged the PDLP cost divided by the worker count, so dividing again
// would gate it N times too tightly.
//
// The default 512 reproduces the pre-#111 runner gate: at the default
// effort 0.0296 that gate was `heuristic_effort_budget(nnz, 0.0296) / 4`
// = 606 x nnz, and 512 is the neighbouring power of two (0.84x).  Small
// in absolute terms because a single PDLP solve charges `iters x nnz`,
// so this is a handful of unproductive pump rounds rather than hundreds
// of sweeps — and for the same reason Scylla cannot honour a threshold
// below the cost of one solve, whatever the option says.  0 disables the
// gate entirely.  The default is registered in
// `third_party/highs_patch/apply_patch.cmake` and pinned by
// `tests/test_smoke.cpp`; `docs/PARAMETERS.md` carries the calibration
// notes.

// Scylla feasibility-pump heuristic (Mexi et al. 2023, Algorithm 1.1):
// N workers sharing a single PDLP instance via a mutex-guarded
// `ContestedPdlp`.  Each worker owns its own warm-start, α_K decay,
// cycle history, RNG, and static FPR rounding strategy
// (`kFprConfigs[w % kNumFprConfigs]`).  Only one PDLP solve is in
// flight at a time, so cuPDLP GPU state is never contended.
//
// Workers run continuously (`run_opportunistic_loop`) with per-worker
// self-termination; a retired chain is rebuilt in place with a fresh
// seed.  Set `threads=1` for a single chain whose behaviour is
// reproducible under a fixed `random_seed`.
//
// Implements the uniform runner contract; see heuristic_context.h.
size_t run(const ProblemView& problem, const HeuristicBudget& budget, ExecutionContext& exec,
           IncumbentSink& sink);

}  // namespace scylla
