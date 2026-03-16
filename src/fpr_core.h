#pragma once

#include <vector>

#include "util/HighsInt.h"

class HighsMipSolver;

struct FprConfig {
  int max_attempts;
  uint32_t rng_seed_offset;
  // Per-variable hint for choose_fix_value (nullable; length ncol if non-null)
  const double* hint;
  // Ranking scores per variable (length ncol; caller computes)
  const double* scores;
  // Fallback values for zero-cost continuous vars (length ncol)
  const double* cont_fallback;
};

void fpr_core(HighsMipSolver& mipsolver, const FprConfig& cfg);
