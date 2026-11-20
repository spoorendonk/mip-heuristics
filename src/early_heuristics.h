#pragma once

#include <memory>
#include <vector>

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
// Each improving point that passes HiGHS's own feasibility test on the
// original model is logged from the worker when found (`objective ... found
// at ...s`, #240).  Nothing else reaches the solve before `finish()`: that
// includes the improving-solution callback, which `runSetup` fires.
//
// `finish()`, called on the solving thread right after presolve returns,
// stops and joins the worker, books it as one `[Heur] name=early_fj` line,
// and hands the best point it logged to HiGHS's own original-space incumbent,
// `HighsMipSolver::solution_`, when it is feasible on the original model and
// beats what is there (a user's MIP start may be).  That is the slot a MIP
// start occupies, so from there HiGHS does the rest: `runSetup` crushes it
// into the presolved space as the incumbent, which seeds the chain's pool,
// and takes the crushed point's objective as the upper bound, and
// `solution_` stays the answer unless the search finds better.
//
// What is guaranteed (#306): a handed-over point is feasible on the original
// model and, when `runSetup` follows, its crushed image is feasible on the
// presolved model by the same test, so the cutoff it sets is attained by a
// presolved point.  The crush applies only duplicate-column and linear
// transforms and drops removed columns, so a point a reduction (a dual one,
// say) needs another column to compensate for fails that check; it is then
// dropped, logged as `dropped: infeasible in the presolved model`, and
// booked `found=0`.  It is not repaired, and no longer the answer of a run
// whose search finds nothing.  Nor does the reported pair disagree (#343):
// `runSetup` takes `solution_objective_` from the crushed point but keeps
// `solution_`, so where presolve gave the crushed point another objective
// (a dominated column fixed at the bound the point does not take, say) the
// point handed over is the crushed point's postsolved image instead, logged
// as `postsolved from its presolved image`, provided it is feasible on the
// original model and its own crush, as `runSetup` will take it, is feasible
// on the presolved model with the image's objective; else the point is
// dropped, as above.  The check covers the first `runSetup` only:
// a root restart re-crushes `solution_` through the longer postsolve stack
// unchecked, as upstream does any start or incumbent, and the restarted
// chain's pool is seeded from that.  Its cutoff stays the first setup's,
// which a presolved point attained.  If presolve ended the solve on the
// time limit, nothing searches: the point is handed over unchecked in the
// presolved space and the upper bound set to its own original-space
// objective, so the run reports it.  After any other presolve verdict
// (optimal, infeasible, unbounded, out of memory) nothing is installed.
// A user's MIP start takes upstream's path, which has no such check.
//
// In a `MIP_HEURISTICS_CUDA` build the constructor also starts the GPU FJ
// race's CUDA context (`fj::warm_gpu`, #289) when `mip_heuristic_fj_gpu` is
// set and FJ is enabled, outside a sub-MIP, whatever the early FJ option.
//
// The destructor stops and joins as well, so every early return of
// `HighsMipSolver::run` leaves no thread behind.  That includes the context's
// creation: the destructor waits for it (`fj::wait_gpu_warmup`), so a solve
// shorter than the creation ends when the creation does.
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
    // FJ races the GPU in this solve: in a CUDA build the constructor starts
    // `fj::warm_gpu` and the destructor waits for it.
    bool gpu_race_ = false;
};

// Hands `point`, in the original columns, to `solution_` by the tests and
// with the log lines `finish()` hands the worker's best over by; whether it
// did.  On the solving thread, after presolve returns.  `finish()` is its
// caller in a solve; a test calls it with a point of its own choosing.
bool install_early_start(HighsMipSolver& mipsolver, std::vector<double>& point);

}  // namespace heuristics
