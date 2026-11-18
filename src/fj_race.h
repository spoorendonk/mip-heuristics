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
//
// The climbers need the process's CUDA context, which `warm_gpu` creates on
// a thread of its own; the race never waits for it (#289).  Until it exists
// the GPU thread only builds its host model and polls, so a race whose CPU
// workers stop first ends without the GPU, unwarned.
DispatchOutcome race(const ProblemView& problem, const HeuristicBudget& budget,
                     ExecutionContext& exec, SolutionSink& sink);

// Starts creating the CUDA context (`gpu_fj::create_context`, blocking
// sync) on a thread of its own, once per process; later calls do nothing.
// `EarlyFj`'s constructor calls it before presolve when the option is on,
// so the context is usually ready by FJ's dispatch; `race` calls it too.
// Any thread may call it: it reads nothing of the solve.
void warm_gpu();

// Waits for a creation `warm_gpu` started and joins its thread; returns at
// once when none is running, or when another caller is already joining it
// (two concurrent solves: the first one's wait holds that solve until the
// creation ends, so the process still cannot exit during it).  The solve
// that started it calls this at its end (`EarlyFj`'s destructor), so the
// creation never runs into the CUDA runtime's exit-time teardown.  That wait
// is the only one: it costs a solve shorter than the creation the rest of
// it, never a dispatch.
void wait_gpu_warmup();

}  // namespace fj
