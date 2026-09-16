#pragma once

class HighsMipSolver;
class HighsOptions;

namespace heuristics {

// The `[Heur] name=` tag the dive-time `fpr_lp` books its dispatch under.
// One spelling, for the reason the four presolve heuristics take theirs
// from `kChain`'s own `name` field: the name a user reads in the trace has
// exactly one definition.  `fpr_lp` cannot take it from that table because
// it is not a chain entry — it runs during the B&B dive, not in presolve —
// so this constant is the binding instead, and `fpr_lp.cpp` charges under
// it.
//
// A plain `constexpr const char*` rather than an `inline constexpr
// std::string_view`: this header is inserted into HiGHS's own
// `HighsMipSolver.cpp` by `apply_patch.cmake` and compiled at HiGHS's
// `CMAKE_CXX_STANDARD 11`, so nothing here may use a C++17 spelling and
// nothing it includes may either.  See the same note in `fpr_lp.h`.  A
// pointer and not an array because `modernize-avoid-c-arrays` rejects the
// array form and `std::array<char, 7>` would spell a name as a length.
constexpr const char* kFprLpName = "fpr_lp";

// Whether any heuristic of ours can produce a solution under `options` —
// that is, whether any of the five `mip_heuristic_<name>_effort` options is
// above zero, with FJ additionally honouring upstream's own
// `mip_heuristic_run_feasibility_jump`.
//
// It exists for one caller outside this translation unit: the patched
// `printSolutionSourceKey` drops the group advertising our five solution
// sources when none of them can appear, which is what keeps the printed
// legend byte-identical to an unpatched binary's and so lets
// `bench/check_vanilla_equivalence.py` diff whole logs rather than a
// filtered subset of them.  Declared here rather than recomputed in that
// patch string so the legend cannot disagree with the dispatcher about
// which heuristics are live.  Being reachable from a file compiled at
// HiGHS's `CMAKE_CXX_STANDARD 11`, it takes `HighsOptions` by reference and
// returns `bool` and nothing more.
bool any_enabled(const HighsOptions& options);

// Top-level presolve heuristic dispatch. Reads mip_heuristic_* options
// and runs the fixed FJ -> FPR -> LocalMIP -> Scylla chain, each on
// continuous parallel workers.  Returns true if the model was proven
// infeasible.
//
// No budget parameter and no suite parameter: each heuristic's budget comes
// from its own `mip_heuristic_<name>_effort` option and the model's nnz,
// both read here (#110), and that same option is what selects it — a value
// at or below zero skips the heuristic entirely (#167).  The call site is a
// patch string in `third_party/highs_patch/apply_patch.cmake`, so keeping
// the arithmetic out of it keeps it out of a file no compiler in this repo
// checks.
bool run_presolve(HighsMipSolver& mipsolver);
}  // namespace heuristics
