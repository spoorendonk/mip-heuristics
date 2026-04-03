#pragma once

#include <cstdint>
#include <random>
#include <vector>

struct CscMatrix;
struct HeuristicResult;
class HighsMipSolver;

// Variable state used during fix-and-propagate.
// Layout must remain compatible with fpr_strategies.cpp.
struct VarState {
  double lb, ub, val;
  bool fixed;
};

struct FprConfig {
  size_t max_effort; // effort budget (coefficient accesses)
  uint32_t rng_seed_offset;
  // Per-variable hint for choose_fix_value (nullable; length ncol if non-null).
  // Used only on attempt 0 (papers: FPR uses incumbent, Scylla uses LP sol).
  const double *hint;
  // Ranking scores per variable (length ncol; caller computes)
  const double *scores;
  // Fallback values for zero-cost continuous vars (length ncol)
  const double *cont_fallback;
  // Optional pre-built CSC matrix (avoids redundant build if caller already has
  // one)
  const CscMatrix *csc;
};

// Build a default FprConfig with degree*(1+|cost|) scores,
// zero continuous fallback, and incumbent hint.
FprConfig build_default_fpr_config(const HighsMipSolver &mipsolver,
                                   const CscMatrix &csc,
                                   std::vector<double> &scores_buf,
                                   std::vector<double> &cont_fallback_buf);

// Original: runs all attempts, submits solutions via trySolution.
// Returns total effort (coefficient accesses) consumed.
size_t fpr_core(HighsMipSolver &mipsolver, const FprConfig &cfg);

// Single-attempt variant for portfolio mode. Returns result without submitting.
// Uses provided RNG and attempt index. If initial_solution is non-null, uses it
// as the starting point (overriding cfg.hint). Otherwise falls back to cfg.hint
// on attempt 0, or random initialization on later attempts.
HeuristicResult fpr_attempt(HighsMipSolver &mipsolver, const FprConfig &cfg,
                            std::mt19937 &rng, int attempt_idx,
                            const double *initial_solution);
