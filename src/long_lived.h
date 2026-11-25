#pragma once

// Long-lived mode's plug-in sites in HiGHS (#332, part of #311).  Inserted
// into HiGHS's own sources, so this header stays C++11 and names nothing
// but the solver.
//
// With `mip_heuristic_long_lived` on, the main MIP's solve owns one
// `LongLivedOwner` (long_lived_owner.h) from just after setup to the start
// of `cleanupSolve`: heuristic threads of its own push solutions to its
// queue, and the solving thread drains that queue into `trySolution` at
// HiGHS's existing poll points.  `mip_heuristic_long_lived_threads` slots of
// FJ, LocalMIP, ruin and recombine are such threads (#356).  With the option
// off, or in a sub-MIP, nothing is allocated and every call here is a null
// test.

#include <memory>

class HighsMipSolver;

namespace heuristics {

class LongLivedOwner;

// Solve-scoped owner, a local of `HighsMipSolver::run` declared right after
// the master worker exists, before the `restart:` label, so one owner
// spans every restart.  Sets `HighsMipSolver::long_lived_`, which is how
// the drain sites in `HighsMipSolverData` find it, and clears it on
// destruction.
class LongLived {
public:
    explicit LongLived(HighsMipSolver& mipsolver);
    ~LongLived();
    LongLived(const LongLived&) = delete;
    LongLived& operator=(const LongLived&) = delete;

private:
    HighsMipSolver& mipsolver_;
    std::unique_ptr<LongLivedOwner> owner_;
};

// Install every queued solution that can still matter, best first.  On the
// solving thread only: the top of `queryExternalSolution`, which every one
// of HiGHS's external-solution poll points reaches.
void drain_long_lived(HighsMipSolver& mipsolver);

// Start the long-lived slots `mip_heuristic_long_lived_threads` asks for
// (#356), once the presolve chain has returned and before the root node.
// Nothing without an owner, with 0 threads, while the owner is paused, when
// the slots already run (after a restart's resume), or past the deadline.
void start_long_lived(HighsMipSolver& mipsolver);

// Pause the owner (every thread joined), drain what its slots queued while
// it is still in this space, then book their effort with
// `stopped_by=restart`.  The first line of `performRestart`, before it overwrites
// the presolved model the slots read and before `numRestarts` moves.
void restart_long_lived(HighsMipSolver& mipsolver);

// Resume a paused owner (#361): its slots start again over the new
// presolved model, each with a sink and restart source of the new epoch.
// In `performRestart` once `runSetup` has built the new model, only when
// the solve goes on (`modelstatus_` unset, not `terminate()`); one place
// for all four of its call sites.
void resume_long_lived(HighsMipSolver& mipsolver);

// Stop the owner (every thread joined), drain once more, so a solution
// pushed after the last poll point is still installed, then book its
// slots' effort with `stopped_by=end` (or `deadline`).  The first line of
// `cleanupSolve`, which every exit of `run` goes through after setup.
// Rethrows the first exception a producer thread ended with.
void stop_long_lived(HighsMipSolver& mipsolver);

}  // namespace heuristics
