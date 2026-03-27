#pragma once

#include <cstddef>
#include <random>

struct CscMatrix;
struct HeuristicResult;
class HighsMipSolver;

namespace local_mip {
void run(HighsMipSolver &mipsolver);

// Single-worker variant for portfolio mode. Returns result without submitting.
// If initial_solution is non-null, uses it as starting point.
// max_effort: effort budget (coefficient accesses); 0 = unlimited.
HeuristicResult worker(HighsMipSolver &mipsolver, const CscMatrix &csc,
                       std::mt19937 &rng, const double *initial_solution,
                       size_t max_effort = 0);
} // namespace local_mip
