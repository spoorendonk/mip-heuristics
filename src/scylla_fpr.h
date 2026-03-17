#pragma once

#include <random>

struct HeuristicResult;
class HighsMipSolver;

namespace scylla_fpr {
void run(HighsMipSolver& mipsolver);

// Single-attempt variant for portfolio mode. Returns result without submitting.
HeuristicResult attempt(HighsMipSolver& mipsolver, std::mt19937& rng);
}  // namespace scylla_fpr
