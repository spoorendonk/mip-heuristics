#pragma once
#include <cstddef>
class HighsMipSolver;
namespace fpr_lp {
// Run LP-dependent FPR configs (paper Classes 2-3) using the root LP solution.
// Requires an optimal LP relaxation. Called during B&B dive (after RINS/RENS).
void run(HighsMipSolver &mipsolver, size_t max_effort);

// Test hook: counters incremented once per dispatch into each variant.
// fpr_lp has a single heuristic family (unlike the presolve portfolio),
// so it always runs arm-aligned parallel workers; the mip_heuristic_portfolio
// flag is ignored here and only mip_heuristic_opportunistic selects between
// these two variants.  Process-global; reset before each test that inspects.
struct DispatchCounts {
    size_t seq_det = 0;
    size_t seq_opp = 0;
};
DispatchCounts dispatch_counts();
void reset_dispatch_counts();
}  // namespace fpr_lp
