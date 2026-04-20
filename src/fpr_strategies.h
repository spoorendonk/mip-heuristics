#pragma once

// Umbrella header: re-exports the three split TUs so existing callers that
// included "fpr_strategies.h" keep working unchanged.
//
// The actual implementations now live in:
//   - fpr_var_order.{h,cpp}  — variable ranking (VarStrategy + compute_var_order)
//   - fpr_val_select.{h,cpp} — value selection (ValStrategy + choose_value)
//   - fpr_lp_refs.{h,cpp}    — LP reference solutions (only TU that pulls Highs.h)

#include "fpr_lp_refs.h"
#include "fpr_val_select.h"
#include "fpr_var_order.h"

// ---------------------------------------------------------------------------
// Strategy configuration (paper Section 3)
// ---------------------------------------------------------------------------

struct FprStrategyConfig {
    VarStrategy var_strategy = VarStrategy::kType;
    ValStrategy val_strategy = ValStrategy::kGoodobj;
};

enum class FrameworkMode {
    kDfs,           // propagation on, repair off, backtrack on infeasibility
    kDfsrep,        // propagation on, repair on, backtrack on infeasibility
    kDive,          // propagation off, repair on (at end only), no backtrack
    kDiveprop,      // propagation on, repair on, no backtrack
    kRepairSearch,  // propagation on, DFS repair with secondary engine R (Fig. 5)
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
    return cfg.var_strategy == VarStrategy::kCliques || cfg.var_strategy == VarStrategy::kCliques2;
}

// ---------------------------------------------------------------------------
// Paper's 14 named strategy combinations (Table 3)
// Each is a (variable, value) pair; combined with a FrameworkMode separately.
// ---------------------------------------------------------------------------

// LP-free strategies
inline constexpr FprStrategyConfig kStratRandom{VarStrategy::kTypecl, ValStrategy::kRandom};
inline constexpr FprStrategyConfig kStratRandom2{VarStrategy::kRandom, ValStrategy::kRandom};
inline constexpr FprStrategyConfig kStratBadobj{VarStrategy::kType, ValStrategy::kBadobj};
inline constexpr FprStrategyConfig kStratBadobjcl{VarStrategy::kTypecl, ValStrategy::kBadobj};
inline constexpr FprStrategyConfig kStratGoodobj{VarStrategy::kType, ValStrategy::kGoodobj};
inline constexpr FprStrategyConfig kStratGoodobjcl{VarStrategy::kTypecl, ValStrategy::kGoodobj};
inline constexpr FprStrategyConfig kStratLocks{VarStrategy::kLR, ValStrategy::kLoosedyn};
inline constexpr FprStrategyConfig kStratLocks2{VarStrategy::kLocks, ValStrategy::kLoosedyn};
inline constexpr FprStrategyConfig kStratCliques{VarStrategy::kCliques, ValStrategy::kUp};
inline constexpr FprStrategyConfig kStratCliques2{VarStrategy::kCliques2, ValStrategy::kUp};
inline constexpr FprStrategyConfig kStratDomsize{VarStrategy::kDomainSize, ValStrategy::kLoosedyn};

// LP-dependent strategies
inline constexpr FprStrategyConfig kStratZerocore{VarStrategy::kTypecl, ValStrategy::kZerocore};
inline constexpr FprStrategyConfig kStratZerolp{VarStrategy::kTypecl, ValStrategy::kZerolp};
inline constexpr FprStrategyConfig kStratCore{VarStrategy::kTypecl, ValStrategy::kCore};
inline constexpr FprStrategyConfig kStratLp{VarStrategy::kTypecl, ValStrategy::kLp};

// ---------------------------------------------------------------------------
// Named configuration (strategy + framework mode pair)
// ---------------------------------------------------------------------------

struct NamedConfig {
    FprStrategyConfig strat;
    FrameworkMode mode;
};
