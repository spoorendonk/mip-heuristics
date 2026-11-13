#pragma once
#include "heuristic_context.h"

class SolutionSink;

namespace fj {

// FJ's chain entry in a `MIP_HEURISTICS_CUDA` build (#283, design on #210):
// with `budget.fj_gpu` set, a batch of GPU climbers (`gpu/climber.h`) runs
// on its own thread while `fj::run` runs unchanged on the dispatching
// thread, both offering into `sink`.  The GPU thread polls only its own
// stop flag and the deadline, offers as slots `num_workers + c`, and never
// reads `mipdata`; when `fj::run` returns it is stopped and joined, and its
// effort is added to the outcome, which the dispatcher charges once.  With
// the option off, or when the GPU side cannot start (it warns, the
// bench-contract `FeasibilityJump runs on the CPU only`), this is `fj::run`.
//
// Same runner contract as `fj::run`; see heuristic_context.h.
DispatchOutcome race(const ProblemView& problem, const HeuristicBudget& budget,
                     ExecutionContext& exec, SolutionSink& sink);

}  // namespace fj
