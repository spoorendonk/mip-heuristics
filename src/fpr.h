#pragma once

#include <cstddef>

class IncumbentSink;
struct ExecutionContext;
struct HeuristicBudget;
struct ProblemView;

namespace fpr {

// Stall threshold, in effort units (coefficient accesses) per
// constraint-matrix nonzero (issue #111) — roughly "this many full
// sweeps of the matrix without a solution".
//
// Scope: **whole dispatch**, matching `mip_heuristic_fpr_effort`.  Each
// worker's share is this divided by the worker count.
//
// 2048 reproduces the pre-#111 runner gate: at the default effort
// 0.0884 that gate was `heuristic_effort_budget(nnz, 0.0884) / 4`
// = 1810 x nnz, and 2048 is the neighbouring power of two (1.13x).
//
// PROVISIONAL, pending the per-heuristic budget calibration (#106).
// #106 sweeps each heuristic's effort option and will show where each
// one actually stops producing solutions; these values are placeholders
// chosen to reproduce the pre-#111 gate at the shipped default effort,
// not the result of a measurement.
inline constexpr size_t kStallPerNnzFpr = 2048;

// Runs N continuous `parallel::for_each` FprWorkers with per-worker
// self-termination.  Each attempt advances the worker's attempt index
// (new random init, shuffled variable order) and rotates its config
// through the paper-curated strategy x mode list.  Set `threads=1` for
// a single worker whose behaviour is reproducible under a fixed
// `random_seed`.
//
// Implements the uniform runner contract; see heuristic_context.h.
size_t run(const ProblemView& problem, const HeuristicBudget& budget, ExecutionContext& exec,
           IncumbentSink& sink);

#ifndef NDEBUG
// Test-only lifecycle counters for the issue #77 pause/resume path.
// Defined in fpr.cpp.  Tests assert these are non-zero after a
// solve at small `mip_heuristic_fpr_effort` to verify the kBudgetGate /
// multi-attempt-fill paths actually fired (objective equality alone
// is a proxy that misses lifecycle-path regressions where the
// rotation diverges but converges back to the same final objective).
size_t budget_gate_hits();
size_t multi_attempt_iters();
void reset_test_counters();
#endif

}  // namespace fpr
