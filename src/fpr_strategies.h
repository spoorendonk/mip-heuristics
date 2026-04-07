#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "heuristic_common.h"
#include "util/HighsInt.h"

class HighsMipSolver;

// ---------------------------------------------------------------------------
// Strategy enums (paper Table 1, Table 2, Section 3)
// ---------------------------------------------------------------------------

enum class VarStrategy {
  kLR,       // formulation order
  kType,     // grouped by type: binary, integer, continuous
  kRandom,   // random shuffle within each type bucket
  kLocks,    // sorted by max(uplocks, downlocks) within type
  kTypecl,   // clique cover for binaries, then type
  kCliques,  // clique cover + analytic-center-weighted random sort
  kCliques2, // dynamic clique cover using LP solution
};

enum class ValStrategy {
  kUp,       // always upper bound
  kRandom,   // random between lb and ub
  kGoodobj,  // fix toward objective
  kBadobj,   // fix against objective
  kLoosedyn, // dynamic locks based on current activities
  kZerocore, // zero-obj analytic center, fractional rounding
  kZerolp,   // zero-obj LP vertex, fractional rounding
  kCore,     // full-obj analytic center, fractional rounding
  kLp,       // full-obj LP solution, fractional rounding
};

enum class FrameworkMode {
  kDfs,           // propagation on, repair off, backtrack on infeasibility
  kDfsrep,        // propagation on, repair on, backtrack on infeasibility
  kDive,          // propagation off, repair on (at end only), no backtrack
  kDiveprop,      // propagation on, repair on, no backtrack
  kRepairSearch,  // propagation on, DFS repair with secondary engine R (Fig. 5)
};

// ---------------------------------------------------------------------------
// Strategy configuration
// ---------------------------------------------------------------------------

struct FprStrategyConfig {
  VarStrategy var_strategy = VarStrategy::kType;
  ValStrategy val_strategy = ValStrategy::kGoodobj;
};

// Framework mode properties
inline bool mode_propagates(FrameworkMode m) {
  return m != FrameworkMode::kDive;
}

inline bool mode_repairs(FrameworkMode m) {
  // RepairSearch has its own Phase 3 dispatch — not the WalkSAT path.
  return m == FrameworkMode::kDfsrep || m == FrameworkMode::kDive ||
         m == FrameworkMode::kDiveprop;
}

inline bool mode_backtracks(FrameworkMode m) {
  return m == FrameworkMode::kDfs || m == FrameworkMode::kDfsrep ||
         m == FrameworkMode::kRepairSearch;
}

// Does this strategy require an LP solution?
inline bool strategy_needs_lp(const FprStrategyConfig& cfg) {
  switch (cfg.val_strategy) {
    case ValStrategy::kZerocore:
    case ValStrategy::kZerolp:
    case ValStrategy::kCore:
    case ValStrategy::kLp:
      return true;
    default:
      break;
  }
  return cfg.var_strategy == VarStrategy::kCliques ||
         cfg.var_strategy == VarStrategy::kCliques2;
}

// ---------------------------------------------------------------------------
// Paper's 14 named strategy combinations (Table 3)
// Each is a (variable, value) pair; combined with a FrameworkMode separately.
// ---------------------------------------------------------------------------

// LP-free strategies
inline constexpr FprStrategyConfig kStratRandom{VarStrategy::kTypecl,
                                                ValStrategy::kRandom};
inline constexpr FprStrategyConfig kStratRandom2{VarStrategy::kRandom,
                                                 ValStrategy::kRandom};
inline constexpr FprStrategyConfig kStratBadobj{VarStrategy::kType,
                                                ValStrategy::kBadobj};
inline constexpr FprStrategyConfig kStratBadobjcl{VarStrategy::kTypecl,
                                                  ValStrategy::kBadobj};
inline constexpr FprStrategyConfig kStratGoodobj{VarStrategy::kType,
                                                 ValStrategy::kGoodobj};
inline constexpr FprStrategyConfig kStratGoodobjcl{VarStrategy::kTypecl,
                                                   ValStrategy::kGoodobj};
inline constexpr FprStrategyConfig kStratLocks{VarStrategy::kLR,
                                               ValStrategy::kLoosedyn};
inline constexpr FprStrategyConfig kStratLocks2{VarStrategy::kLocks,
                                                ValStrategy::kLoosedyn};
inline constexpr FprStrategyConfig kStratCliques{VarStrategy::kCliques,
                                                 ValStrategy::kUp};
inline constexpr FprStrategyConfig kStratCliques2{VarStrategy::kCliques2,
                                                  ValStrategy::kUp};

// LP-dependent strategies
inline constexpr FprStrategyConfig kStratZerocore{VarStrategy::kTypecl,
                                                  ValStrategy::kZerocore};
inline constexpr FprStrategyConfig kStratZerolp{VarStrategy::kTypecl,
                                                ValStrategy::kZerolp};
inline constexpr FprStrategyConfig kStratCore{VarStrategy::kTypecl,
                                              ValStrategy::kCore};
inline constexpr FprStrategyConfig kStratLp{VarStrategy::kTypecl,
                                            ValStrategy::kLp};

// ---------------------------------------------------------------------------
// Variable ranking
// ---------------------------------------------------------------------------

// Produce a variable ordering for the given strategy.
// Returns a permutation of [0, ncol) with integer variables first.
// For clique-based strategies, `lp_ref` is the LP/analytic-center solution
// (may be nullptr for LP-free strategies).
std::vector<HighsInt> compute_var_order(const HighsMipSolver& mipsolver,
                                        VarStrategy strategy,
                                        std::mt19937& rng,
                                        const double* lp_ref = nullptr);

// ---------------------------------------------------------------------------
// Value selection
// ---------------------------------------------------------------------------

// Choose a fixing value for variable j given the current domain [lb, ub].
// For LP-based strategies, `lp_ref[j]` provides the reference LP value.
// For loosedyn, precomputed row activities and bound arrays are needed.
double choose_value(HighsInt j, double lb, double ub, bool is_int,
                    bool minimize, double cost, ValStrategy strategy,
                    std::mt19937& rng, const double* lp_ref,
                    // loosedyn support: nullable pointers
                    const double* row_lo, const double* row_hi,
                    const double* min_act, const double* max_act,
                    const CscMatrix* csc);

// ---------------------------------------------------------------------------
// LP reference solutions
// ---------------------------------------------------------------------------

// Solve the LP relaxation without objective using barrier (no crossover)
// to obtain the analytic center. Returns col_value vector.
std::vector<double> compute_analytic_center(const HighsMipSolver& mipsolver,
                                            bool use_objective);

// Solve the LP relaxation without objective using simplex to obtain a vertex.
std::vector<double> compute_zero_obj_vertex(const HighsMipSolver& mipsolver);
