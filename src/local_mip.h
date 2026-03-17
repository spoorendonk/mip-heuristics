#pragma once

#include <random>

struct CscMatrix;
struct HeuristicResult;
class HighsMipSolver;

namespace local_mip {
void run(HighsMipSolver& mipsolver);

// Single-worker variant for portfolio mode. Returns result without submitting.
// If initial_solution is non-null, uses it as starting point.
HeuristicResult worker(HighsMipSolver& mipsolver, const CscMatrix& csc,
                       std::mt19937& rng, const double* initial_solution);
}  // namespace local_mip
