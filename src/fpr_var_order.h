#pragma once

#include "fpr_strategies.h"
#include "rng.h"
#include "util/HighsInt.h"

#include <vector>

class HighsMipSolver;

// Produce a variable ordering for the given strategy.
// Returns a permutation of [0, ncol) with integer variables first.
// For clique-based strategies, `lp_ref` is the LP/analytic-center solution
// (may be nullptr for LP-free strategies).
std::vector<HighsInt> compute_var_order(const HighsMipSolver& mipsolver, VarStrategy strategy,
                                        Rng& rng, const double* lp_ref = nullptr);
