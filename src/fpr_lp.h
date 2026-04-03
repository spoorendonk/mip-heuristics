#pragma once
#include <cstddef>
class HighsMipSolver;
namespace fpr_lp {
// Run LP-dependent FPR configs (paper Classes 2-3) using the root LP solution.
// Requires an optimal LP relaxation. Called during B&B dive (after RINS/RENS).
void run(HighsMipSolver &mipsolver, size_t max_effort);
}  // namespace fpr_lp
