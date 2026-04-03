#pragma once

#include <cstddef>
#include <random>

#include "util/HighsInt.h"

class PropEngine;

struct WalkSatMove {
  HighsInt var = -1;
  double val = 0.0;
};

// Select a WalkSAT repair move for the given violated row.
// Uses PropEngine for read-only matrix data, bounds, integrality, feastol.
// solution/lhs_cache are the current (possibly infeasible) assignment.
// col_lb/col_ub are clamping bounds (may differ from PropEngine's narrowed
// bounds — callers pass global bounds for repair moves).
// Returns {-1, 0.0} if no valid candidate found.
// Increments effort by coefficient accesses consumed.
WalkSatMove walksat_select_move(HighsInt row, const double* solution,
                                const double* lhs_cache, const double* col_lb,
                                const double* col_ub, const PropEngine& data,
                                double noise, std::mt19937& rng,
                                size_t& effort);
