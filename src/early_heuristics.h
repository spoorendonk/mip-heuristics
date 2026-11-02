#pragma once

#include <memory>

class HighsMipSolver;

namespace heuristics {

// FeasibilityJump on the original, unpresolved model while HiGHS presolves
// it (#205; cuOpt's "run no-relaxation heuristics before presolve").
//
// Opt-in through `mip_heuristic_early_fj_effort` (default 0, off), never in
// a sub-MIP.  The constructor starts one FJ worker (`fj::run_until_stopped`,
// slot 0) on a thread of its own over a view of `*orig_model_` built by the
// core `make_problem`; the view itself is built on that thread, so presolve
// is not delayed by its O(nnz) setup.  It stops at the first of: `finish()`,
// its budget (the option times the original model's `nnz << 10`), the time
// limit.  Patience and the rebuild restart point are FJ's own
// (`mip_heuristic_fj_patience`, `mip_heuristic_fj_restart`).
//
// `finish()`, called on the solving thread right after presolve returns,
// stops and joins the worker, books it as one `[Heur] name=early_fj` line,
// and hands its best solution to HiGHS's own original-space incumbent,
// `HighsMipSolver::solution_`, when it is feasible on the original model and
// beats what is there (a user's MIP start may be).  That is the slot a MIP
// start occupies, so from there HiGHS does the rest: `runSetup` crushes it
// into the presolved space as the incumbent and takes its objective as the
// upper bound, and `solution_` stays the answer unless the search finds
// better — including when presolve's dual reductions cut the point itself
// off.  If presolve ended the solve on the time limit, the upper bound is set
// too, so the run reports it; after any other presolve verdict (optimal,
// infeasible, unbounded, out of memory) nothing is installed.
//
// The destructor stops and joins as well, so every early return of
// `HighsMipSolver::run` leaves no thread behind.
//
// Not reproducible: the worker stops when presolve returns, so what it finds
// depends on timing even at `threads=1` (docs/REPRODUCIBILITY.md).
//
// C++11, like the other headers inserted into HiGHS's own sources:
// `apply_patch.cmake` includes it into `HighsMipSolver.cpp`.
class EarlyFj {
public:
    explicit EarlyFj(HighsMipSolver& mipsolver);
    ~EarlyFj();
    EarlyFj(const EarlyFj&) = delete;
    EarlyFj& operator=(const EarlyFj&) = delete;

    // `run_to_budget` joins without stopping, so the worker ends on its own
    // budget, patience or the time limit.  The call site in HiGHS never
    // sets it; it is how a test takes the race with presolve out of what
    // it asserts.
    void finish(bool run_to_budget = false);

private:
    struct Run;
    HighsMipSolver& mipsolver_;
    std::unique_ptr<Run> run_;
};

}  // namespace heuristics
