#pragma once

#include <cstdint>
#include <random>

struct CscMatrix;
struct HeuristicResult;
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

// Original: runs all attempts, submits solutions via trySolution.
void fpr_core(HighsMipSolver& mipsolver, const FprConfig& cfg);

// Single-attempt variant for portfolio mode. Returns result without submitting.
// Uses provided RNG and attempt index. If initial_solution is non-null and
// attempt_idx == 0, uses it as the starting point (like hint in FprConfig).
HeuristicResult fpr_attempt(HighsMipSolver& mipsolver, const FprConfig& cfg,
                            std::mt19937& rng, int attempt_idx,
                            const double* initial_solution);
