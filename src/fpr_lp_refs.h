#pragma once

#include <cstdint>
#include <vector>

class HighsMipSolver;

// ---------------------------------------------------------------------------
// LP reference solutions
// ---------------------------------------------------------------------------

// Solve the LP relaxation without objective using barrier (no crossover)
// to obtain the analytic center. Returns col_value vector.  Adds the LP
// iterations spent (simplex + IPM + crossover) to `lp_iterations` so the
// caller can charge them against the shared B&B heuristic budget.
std::vector<double> compute_analytic_center(const HighsMipSolver& mipsolver, bool use_objective,
                                            int64_t& lp_iterations);

// Solve the LP relaxation without objective using simplex to obtain a vertex.
// Adds the LP iterations spent to `lp_iterations` as above.
std::vector<double> compute_zero_obj_vertex(const HighsMipSolver& mipsolver,
                                            int64_t& lp_iterations);
