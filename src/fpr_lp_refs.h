#pragma once

#include <vector>

class HighsMipSolver;

// ---------------------------------------------------------------------------
// LP reference solutions
// ---------------------------------------------------------------------------

// Solve the LP relaxation without objective using barrier (no crossover)
// to obtain the analytic center. Returns col_value vector.
std::vector<double> compute_analytic_center(const HighsMipSolver& mipsolver, bool use_objective);

// Solve the LP relaxation without objective using simplex to obtain a vertex.
std::vector<double> compute_zero_obj_vertex(const HighsMipSolver& mipsolver);
