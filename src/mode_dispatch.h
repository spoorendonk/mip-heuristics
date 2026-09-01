#pragma once

#include <cstddef>
#include <string>

class HighsMipSolver;
class HighsOptions;

namespace heuristics {

// The effective per-heuristic enable flags selected by
// mip_heuristic_suite, which is either one of the two whole-value aliases
// `off` (no heuristic) and `all` (every one), or a comma-separated list of
// the heuristic names `fj`, `fpr`, `local_mip`, `scylla` — so `fj,fpr`
// enables exactly those two.
struct HeuristicFlags {
    bool fj;
    bool fpr;
    bool local_mip;
    bool scylla;
};

// What parsing mip_heuristic_suite rejected, for the warning run_presolve
// emits.  Callers that only want the flags pass nothing — fpr_lp calls
// effective_flags once per B&B dive and has nothing to log.  Every value
// that parses cleanly allocates nothing either way; only a rejected token
// does, first to collect it and then, if this struct is asked for, to
// format it.
struct SuiteDiagnosis {
    // Every token of the value that named no heuristic, quoted and
    // comma-joined (`"fpr2", "walksat"`).  Empty when the whole value was
    // understood.  The warning has to name the token rather than only the
    // value: one typo inside an otherwise valid list silently promotes the
    // run to all four heuristics, and "unknown value" alone does not say
    // which name to fix.
    std::string unknown_tokens;
    // How many of those there are, so the caller can pluralize.
    size_t unknown_count = 0;
};

// Derive the effective flag set from `options`.  Shared by the presolve
// dispatch (run_presolve) and the B&B-dive fpr_lp entry point so both
// honour the same suite semantics — in particular `suite=off` disables
// fpr_lp too, which is what makes `off` an ablation of every heuristic of
// ours rather than of the presolve chain alone, and so does any value that
// does not name `fpr`.
// A value carrying an unrecognised token fails open (all four on) and
// reports the offending tokens through `*diagnosis` if non-null; the caller
// decides whether to warn, because this helper is called once per B&B dive
// and must not log.
//
// `fj` additionally honours upstream's own mip_heuristic_run_feasibility_jump:
// setting it false disables FeasibilityJump at every suite value, matching
// what it does to the native call site at suite=off.
HeuristicFlags effective_flags(const HighsOptions& options, SuiteDiagnosis* diagnosis = nullptr);

// Top-level presolve heuristic dispatch. Reads mip_heuristic_* options
// and runs the fixed FJ -> FPR -> LocalMIP -> Scylla chain, each on
// continuous parallel workers.  Returns true if the model was proven
// infeasible.
//
// No budget parameter: each heuristic's budget comes from its own
// `mip_heuristic_<name>_effort` option and the model's nnz, both read here
// (#110).  The call site is a patch string in
// `third_party/highs_patch/apply_patch.cmake`, so keeping the arithmetic
// out of it keeps it out of a file no compiler in this repo checks.
bool run_presolve(HighsMipSolver& mipsolver);
}  // namespace heuristics
