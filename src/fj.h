#pragma once
#include <cstddef>

class HighsMipSolver;
class IncumbentSink;

namespace fj {

// Runs N continuous `parallel::for_each` FjWorkers with per-worker
// self-termination, each seeded differently; a worker that finishes is
// rebuilt in place with a fresh seed.  Set `threads=1` for a single
// worker whose behaviour is reproducible under a fixed `random_seed`.
//
// `sink` is owned by the caller (mode_dispatch::run_sequential), which
// also sets the source tag workers' solutions are attributed with.
//
// Returns the total effort consumed.  The caller is responsible for
// booking the effort into `mipdata->heuristic_effort_used` — same
// contract as `local_mip::run_parallel`, `fpr::run_parallel`, and
// `scylla::run_parallel` (issue #79).  This makes mode_dispatch.cpp the
// single point of effort accounting for the four presolve heuristics.
size_t run_parallel(HighsMipSolver &mipsolver, IncumbentSink &sink, size_t max_effort);
}  // namespace fj
