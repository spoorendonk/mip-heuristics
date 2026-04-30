#pragma once

// Construction phase for Local-MIP cold start (issue #75).  Produces a
// (possibly infeasible) starting assignment for the search loop when
// neither FJ nor FPR left an incumbent behind and the shared
// SolutionPool is empty.
//
// NOTE on paper fidelity.  Lin, Zou, Cai (CP 2024) and the public
// reference implementation at
// https://github.com/shaowei-cai-group/Local-MIP (src/local_search/
// start/start.cpp) specify only the trivial `zero_start` construction
// — every variable set to the value closest to 0 within its bounds,
// with no greedy sweep and no variable ordering.  The construction
// implemented here extends that: it starts from the paper's
// zero-init (Phase A), then runs one coverage-weighted greedy pass
// (Phase B) that reuses the paper's §3.2 mtm candidate generator to
// drive per-variable moves minimising weighted row violation.  This
// is an engineering extension motivated by issue #75's "greedy
// variable-ordering + per-variable minimise-weighted-violation"
// prose (which is richer than the paper actually prescribes) and
// was flagged by all three round-2 reviewers.  The sweep can be
// disabled (caller passes `max_effort == 0` → Phase A only) or the
// ordering key replaced by a plain Fisher-Yates shuffle without
// functional regressions on the small tests; its benefit on hard
// instances (`genus-sym-*`, `neos-4232544-orira`, the 4 Local-MIP
// new-record targets) is an empirical question that issue #75's
// acceptance criterion calls out for validation.
//
// See `local_mip_construction.cpp` for the algorithmic details and
// commentary on how closely this follows the paper.

#include "heuristic_common.h"
#include "lp_data/HighsLp.h"
#include "mip/HighsMipSolver.h"
#include "rng.h"
#include "util/HighsInt.h"

#include <cstddef>
#include <vector>

namespace local_mip_detail {

// Fraction of `max_effort` spent on construction.  Paper uses a small
// fixed budget (Algorithm 1 Line 1 is O(ncol) with no sweep); we spend
// a little more so the greedy sweep can tighten a few violated
// constraints before the search loop takes over.  10% is an
// engineering compromise: small enough that the search phase (which
// actually produces improving moves) retains the bulk of the budget,
// large enough that the construction pass can visit every variable
// once on typical MIPLIB-sized presolve instances (each variable costs
// O(col_nnz) coefficient accesses; total construction cost is
// O(nnz)).  Revisit after benchmark if construction consumes
// noticeably more than the configured fraction.
inline constexpr double kConstructionEffortFraction = 0.10;

// Convenience: how much of the outer `max_effort` to hand to the
// construction pass.  Callers should pass the result to
// `construct_initial_solution` and subtract it from the search
// budget.
inline size_t construction_effort_cap(size_t max_effort) {
    return static_cast<size_t>(static_cast<double>(max_effort) * kConstructionEffortFraction);
}

// Inputs required by the construction phase, assembled as a struct so
// the unit-test entry point can drive the function without a full
// `HighsMipSolver`.  The `HighsMipSolver&` overload below is a thin
// wrapper.
struct ConstructionInputs {
    HighsInt ncol;
    HighsInt nrow;
    const std::vector<HighsInt> *ARstart;
    const std::vector<HighsInt> *ARindex;
    const std::vector<double> *ARvalue;
    const std::vector<double> *col_lb;
    const std::vector<double> *col_ub;
    const std::vector<double> *row_lo;
    const std::vector<double> *row_hi;
    const std::vector<HighsVarType> *integrality;
    const CscMatrix *csc;
    double feastol;
};

// Produce a starting assignment for the local search.  Writes `ncol`
// entries to `out_solution`.  The assignment is *not* guaranteed to be
// feasible — the paper's search framework (Algorithm 2) is designed
// to progress from a violated starting point.
//
// `rng` is advanced by the RNG draws used to break ties in the
// variable-coverage ordering.  `max_effort` caps the coefficient
// accesses the construction is allowed to consume (callers should
// pass `construction_effort_cap(max_effort)`).
//
// Returns the number of coefficient accesses the construction actually
// charged.  Callers are expected to add this to
// `mipdata->heuristic_effort_used` so cold-start work counts against
// the global effort budget (R1-3 round-3 review).
size_t construct_initial_solution(const ConstructionInputs &inputs, Rng &rng, size_t max_effort,
                                  std::vector<double> &out_solution);

// Thin wrapper over the inputs form: extracts the refs from
// `mipsolver` (model + mipdata).  Used by `local_mip.cpp`'s
// cold-start fallback.  Returns the construction effort (see overload
// above); callers must book it into `mipdata->heuristic_effort_used`.
size_t construct_initial_solution(HighsMipSolver &mipsolver, const CscMatrix &csc, Rng &rng,
                                  size_t max_effort, std::vector<double> &out_solution);

}  // namespace local_mip_detail
