#pragma once

#include <random>
#include <vector>

#include "util/HighsInt.h"

class PropEngine;

// Paper Fig. 5: RepairSearch with secondary propagation engine R.
// E: main propagation engine (has partial assignment from Phase 2).
// solution/lhs_cache: current complete assignment (may violate constraints).
// col_lb/col_ub: global column bounds (for initializing R).
// row_lo/row_hi: row bounds.
// Returns true if a feasible solution was found (solution modified in-place).
bool repair_search(PropEngine& E, std::vector<double>& solution,
                   std::vector<double>& lhs_cache, const double* col_lb,
                   const double* col_ub, const double* row_lo,
                   const double* row_hi, HighsInt repair_iterations,
                   double repair_noise, bool repair_track_best,
                   size_t max_effort, std::mt19937& rng, size_t& effort_out);
