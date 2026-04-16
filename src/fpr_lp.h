#pragma once
#include <cstddef>
class HighsMipSolver;
namespace fpr_lp {
// Run LP-dependent FPR configs (paper Classes 2-3) using the root LP solution.
// Requires an optimal LP relaxation. Called during B&B dive (after RINS/RENS).
void run(HighsMipSolver &mipsolver, size_t max_effort);

// Test hook: counters incremented once per dispatch into each cell of
// the 2x2 execution matrix.  Tests assert these to verify fpr_lp
// actually entered the intended mode rather than HiGHS solving the
// instance via other heuristics first.  Process-global; reset before
// each test that inspects them.
struct DispatchCounts {
    size_t seq_det = 0;
    size_t seq_opp = 0;
    size_t port_det = 0;
    size_t port_opp = 0;
};
DispatchCounts dispatch_counts();
void reset_dispatch_counts();
}  // namespace fpr_lp
