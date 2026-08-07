#pragma once

#include "rng.h"
#include "solution_pool.h"

#include <mutex>
#include <vector>

class HighsMipSolver;

// The one place a heuristic worker hands a solution back to the solver.
//
// Owns the shared `SolutionPool`, the mutex that serialises HiGHS's
// non-thread-safe `trySolution`, and the `kSolutionSource*` tag the
// running heuristic's entries are attributed with.  Before this class the
// pool + mutex + on_accept wiring was written out twice, verbatim, in
// `mode_dispatch.cpp` and `fpr_lp.cpp`, and every worker hard-coded the
// source constant of its own heuristic at its `try_add` call.
//
// Submission is immediate rather than batched: the accept callback runs
// as soon as the pool takes a solution, so a HiGHS incumbent timestamp
// reflects find time rather than end-of-dispatch flush time.
class IncumbentSink {
public:
    // Constructs the pool, seeds it from the current incumbent, and wires
    // the accept callback.  `source` tags everything offered until
    // `set_source` says otherwise.
    IncumbentSink(HighsMipSolver &mipsolver, int source);

    IncumbentSink(const IncumbentSink &) = delete;
    IncumbentSink &operator=(const IncumbentSink &) = delete;

    // Offer a candidate solution.  Returns true if the pool accepted it,
    // in which case HiGHS has already been told, from inside this call.
    // Safe to call concurrently from any worker.
    bool offer(double objective, const std::vector<double> &solution) {
        return pool_.try_add(objective, solution, source_);
    }

    // Retarget the attribution tag for subsequent offers.  Legal only
    // between heuristics, on the dispatching thread, with every parallel
    // region joined — `mode_dispatch::run_sequential` is the sole caller,
    // and that is the same invariant which lets it book effort without
    // synchronisation.
    void set_source(int source) { source_ = source; }

    // Restart material for a worker beginning a fresh attempt.  Both are
    // thread-safe (the pool takes its own lock).
    bool get_restart(Rng &rng, std::vector<double> &out) { return pool_.get_restart(rng, out); }
    bool copy_best(std::vector<double> &out) { return pool_.copy_best(out); }

private:
    SolutionPool pool_;
    // Serialises `trySolution`: `HighsMipSolverData::addIncumbent` is not
    // thread-safe and the accept callback fires on whichever worker
    // thread produced the solution.
    std::mutex highs_mtx_;
    int source_;
};
