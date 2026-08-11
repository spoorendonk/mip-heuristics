#pragma once

#include <cstddef>

class HighsMipSolver;
class HighsOptions;

namespace heuristics {

// The effective per-heuristic enable flags selected by
// mip_heuristic_suite (off | fj | fpr | local_mip | scylla | all).
struct HeuristicFlags {
    bool fj;
    bool fpr;
    bool local_mip;
    bool scylla;
};

// Derive the effective flag set from `options`.  Shared by the presolve
// dispatch (run_presolve) and the B&B-dive fpr_lp entry point so both
// honour the same suite semantics — in particular `suite=off` disables
// fpr_lp too, which is what makes a suite=off run comparable to vanilla
// HiGHS, and `suite=local_mip` / `suite=scylla` disable it as well.
// An unrecognised value fails open (all four on) and sets `*recognized`
// to false if non-null; the caller decides whether to warn, because this
// helper is called once per B&B dive and must not log.
//
// `fj` additionally honours upstream's own mip_heuristic_run_feasibility_jump:
// setting it false disables FeasibilityJump at every suite value, matching
// what it does to the native call site at suite=off.
HeuristicFlags effective_flags(const HighsOptions &options, bool *recognized = nullptr);

// Top-level presolve heuristic dispatch. Reads mip_heuristic_* options
// and runs the fixed FJ -> FPR -> LocalMIP -> Scylla chain, each on
// continuous parallel workers.  Returns true if the model was proven
// infeasible.
bool run_presolve(HighsMipSolver &mipsolver, size_t budget);

// Emit the once-per-solve cannibalization instrumentation (issue #95):
// the `[Native]` line (HiGHS's own RENS / RINS / root-reduced-cost call
// counts and the shared heuristic LP-iteration counters) and the `[Root]`
// line (when the root LP started, and how much wall time the presolve
// chain spent before it).  Both at `log_dev_level=3`, like `[Heur]`.
//
// Called from `HighsMipSolver::cleanupSolve` by the patch, so it fires on
// every exit path exactly once.  Reads only; it must stay a pure report,
// because it also runs at `mip_heuristic_suite=off`, which is the
// vanilla-equivalence row of the benchmark matrix — the counters it
// prints there are the reference the patched rows are compared against.
// Sub-MIP solves (RENS/RINS build their own `HighsMipSolver`) return
// immediately: their counters describe a different model.
void log_solve_summary(HighsMipSolver &mipsolver);

}  // namespace heuristics
