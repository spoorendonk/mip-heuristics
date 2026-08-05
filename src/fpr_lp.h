#pragma once
#include <cstddef>
class HighsMipSolver;
namespace fpr_lp {
// Run LP-dependent FPR configs (paper Classes 2-3) using the root LP solution.
// Requires an optimal LP relaxation. Called during B&B dive (after RINS/RENS).
//
// Gating and budget are derived internally so fpr_lp participates in the
// same B&B heuristic budget as RENS/RINS (issue: pre-split it drew an
// unaccounted nnz-based budget per call):
//  - enabled iff heuristics::effective_flags(options).fpr — i.e. the
//    mip_heuristic_run_fpr flag as overridden by mip_heuristic_preset, so
//    preset=off really disables it;
//  - per-call effort budget = the remaining LP-iteration headroom of the
//    moreHeuristicsAllowed() envelope (total_lp_iterations *
//    mip_heuristic_effort + 10000 - heuristic_lp_iterations), converted
//    at nnz effort-units per LP iteration and capped at
//    heuristic_effort_budget(nnz, mip_heuristic_effort);
//  - all consumed work — the reference-LP solves in setup plus worker
//    effort / nnz — is charged back to heuristic_lp_iterations and
//    total_lp_iterations, mirroring how RENS/RINS book their sub-MIP LP
//    iterations, so the shared envelope depletes;
//  - skipped entirely while parallelLockActive() (multi-worker B&B
//    search under parallel=on): the counters above are shared and fpr_lp
//    has no worker-local flush infrastructure, so running there would
//    race.  Never fires on the default single-search-worker runs.
void run(HighsMipSolver &mipsolver);

// Test hook: counters incremented once per dispatch into each variant.
// fpr_lp is a single heuristic family, so it always runs arm-aligned
// parallel workers and only mip_heuristic_opportunistic selects between
// these two variants.  Process-global; reset before each test that inspects.
struct DispatchCounts {
    size_t seq_det = 0;
    size_t seq_opp = 0;
};
DispatchCounts dispatch_counts();
void reset_dispatch_counts();
}  // namespace fpr_lp
