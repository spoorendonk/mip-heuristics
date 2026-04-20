#pragma once

#include "fpr_strategies.h"
#include "util/HighsInt.h"

#include <random>

struct CscMatrix;
struct HeuristicResult;
class HighsMipSolver;

struct FprConfig {
    size_t max_effort;  // effort budget (coefficient accesses)
    // Per-variable hint for choose_fix_value (nullable; length ncol if non-null).
    // Used only on attempt 0 (papers: FPR uses incumbent, Scylla uses LP sol).
    const double *hint;
    // Ranking scores per variable (length ncol; caller computes).
    // Used only when strategy is null (legacy mode).
    const double *scores;
    // Fallback values for zero-cost continuous vars (length ncol)
    const double *cont_fallback;
    // Optional pre-built CSC matrix (avoids redundant build if caller already has
    // one)
    const CscMatrix *csc;

    // --- Framework mode (paper Section 3) ---
    FrameworkMode mode = FrameworkMode::kDiveprop;

    // --- Strategy (paper Table 3) ---
    // When non-null, uses the paper's variable ranking and value selection.
    // When null, falls back to legacy scores-based ranking + hint/goodobj.
    const FprStrategyConfig *strategy = nullptr;
    // LP reference solution for LP-based strategies (nullable).
    const double *lp_ref = nullptr;

    // --- Pre-computed variable order (avoids data races on cliquePartition) ---
    // When non-null, fpr_attempt uses this order instead of computing one.
    const HighsInt *precomputed_var_order = nullptr;
    HighsInt precomputed_var_order_size = 0;

    // --- Repair parameters (paper: Salvagnin et al. 2025, Section 5) ---
    // Noise parameter p: probability of random walk move (paper default: 0.75).
    // Greedy probability = 1 - repair_noise.
    double repair_noise = 0.75;
    // Iteration limit per repair call (paper default: 200).
    HighsInt repair_iterations = 200;
    // Track best total violation during walk and restore at end (paper: yes).
    bool repair_track_best = true;
};

// Single-attempt variant for portfolio mode. Returns result without submitting.
// Uses provided RNG and attempt index. If initial_solution is non-null, uses it
// as the starting point (overriding cfg.hint). Otherwise falls back to cfg.hint
// on attempt 0, or random initialization on later attempts.
HeuristicResult fpr_attempt(HighsMipSolver &mipsolver, const FprConfig &cfg, std::mt19937 &rng,
                            int attempt_idx, const double *initial_solution);
