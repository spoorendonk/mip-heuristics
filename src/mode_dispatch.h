#pragma once

#include <cstddef>

class HighsMipSolver;
class HighsOptions;

namespace heuristics {

// The effective per-heuristic enable flags after applying
// mip_heuristic_preset on top of the individual mip_heuristic_* options.
// A recognized non-empty preset overrides all six flags; an empty or
// unknown preset leaves the individual option values in place.
struct HeuristicFlags {
    bool fj;
    bool fpr;
    bool local_mip;
    bool scylla;
    bool portfolio;
    bool opportunistic;
};

// Derive the effective flag set from `options`.  Shared by the presolve
// dispatch (run_presolve) and the B&B-dive fpr_lp entry point so both
// honour the same preset semantics — in particular `preset=off` disables
// fpr_lp too, keeping a preset=off run comparable to vanilla HiGHS.
// If `preset_recognized` is non-null it is set to whether a non-empty
// preset matched a known name (the caller decides whether to warn; this
// helper never logs because fpr_lp calls it once per dive).
HeuristicFlags effective_flags(const HighsOptions &options, bool *preset_recognized = nullptr);

// Top-level presolve heuristic dispatch. Reads mip_heuristic_* options
// and routes to sequential, portfolio deterministic, or portfolio
// opportunistic mode.  Returns true if the model was proven infeasible.
bool run_presolve(HighsMipSolver &mipsolver, size_t budget);

}  // namespace heuristics
