#pragma once

#include "util/HighsInt.h"

#include <random>
#include <vector>

class HighsMipSolver;

// ---------------------------------------------------------------------------
// Variable ranking strategies (paper Table 1)
// ---------------------------------------------------------------------------

enum class VarStrategy {
    kLR,          // formulation order
    kType,        // grouped by type: binary, integer, continuous
    kRandom,      // random shuffle within each type bucket
    kLocks,       // sorted by max(uplocks, downlocks) within type
    kTypecl,      // clique cover for binaries, then type
    kCliques,     // clique cover + analytic-center-weighted random sort
    kCliques2,    // dynamic clique cover using LP solution
    kDomainSize,  // dynamic: smallest domain first at each DFS node
};

// Does this variable strategy recompute ordering dynamically at each DFS node?
inline bool is_dynamic_var_strategy(VarStrategy s) {
    return s == VarStrategy::kDomainSize;
}

// Produce a variable ordering for the given strategy.
// Returns a permutation of [0, ncol) with integer variables first.
// For clique-based strategies, `lp_ref` is the LP/analytic-center solution
// (may be nullptr for LP-free strategies).
std::vector<HighsInt> compute_var_order(const HighsMipSolver& mipsolver, VarStrategy strategy,
                                        std::mt19937& rng, const double* lp_ref = nullptr);
