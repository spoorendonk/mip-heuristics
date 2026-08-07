#pragma once
#include <cstddef>

class IncumbentSink;
struct ExecutionContext;
struct HeuristicBudget;
struct ProblemView;

namespace fj {

// Runs N continuous `parallel::for_each` FjWorkers with per-worker
// self-termination, each seeded differently; a worker that finishes is
// rebuilt in place with a fresh seed.  Set `threads=1` for a single
// worker whose behaviour is reproducible under a fixed `random_seed`.
//
// Uniform runner contract, shared by all four presolve heuristics
// (issue #94).  `mode_dispatch::run_sequential` owns the problem view,
// this heuristic's slice of the effort envelope, the execution context
// and the incumbent sink — including the source tag the solutions found
// here are attributed with.  Returns the total effort consumed; the
// caller books it through `EffortLedger`, the single point of effort
// accounting.  No heuristic self-books.
size_t run(const ProblemView &problem, const HeuristicBudget &budget, ExecutionContext &exec,
           IncumbentSink &sink);
}  // namespace fj
