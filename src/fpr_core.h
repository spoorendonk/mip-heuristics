#pragma once

#include <cstdint>

struct CscMatrix;
class HighsMipSolver;

struct FprConfig {
  int max_attempts;
  uint32_t rng_seed_offset;
  // Per-variable hint for choose_fix_value (nullable; length ncol if non-null).
  // Used only on attempt 0 (papers: FPR uses incumbent, Scylla uses LP sol).
  const double* hint;
  // Ranking scores per variable (length ncol; caller computes)
  const double* scores;
  // Fallback values for zero-cost continuous vars (length ncol)
  const double* cont_fallback;
  // Optional pre-built CSC matrix (avoids redundant build if caller already has
  // one)
  const CscMatrix* csc;
};

void fpr_core(HighsMipSolver& mipsolver, const FprConfig& cfg);
