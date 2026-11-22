#pragma once
#include "heuristic_context.h"

#include <cstddef>
#include <functional>

class SolutionSink;

namespace fj {

// Patience: `mip_heuristic_fj_patience`, a multiple of `nnz << 10` — the
// same unit as this heuristic's effort option (issue #111, made an option
// by #106, put on the effort unit by #116).  FJ's counter is in step
// units, so its value is not comparable with the other three heuristics'
// — do not read equal numbers as equal tolerances.
//
// Scope: **per worker**, matching `mip_heuristic_fj_effort`, which sizes
// one worker's allowance rather than a whole dispatch
// (`budget_is_per_worker` in mode_dispatch's `kChain`).  The
// runner-level gate is therefore `num_workers` times this, and the
// worker-level gate is the value itself.  FJ is the only entry with that
// scope; the others size a whole dispatch and are divided across
// the pool.
//
// The default is 0, which disables the gate entirely (#107).  `nnz << 8`
// (0.25), a quarter of FJ's per-worker budget unit, is what FJ used
// before and the model the other three were moved onto.  The
// worker-level gate reads FJ's own `effortSinceLastImprovement`, not
// effort since the incumbent moved (#191; see fj_worker.cpp).  The default is registered in
// `third_party/highs_patch/apply_patch.cmake` and pinned by
// `tests/test_smoke.cpp`; `docs/PARAMETERS.md` carries the calibration
// notes.

// Runs N continuous `parallel::for_each` FjWorkers with per-worker
// self-termination, each seeded differently; a worker that finishes is
// rebuilt in place with a fresh seed.  Set `threads=1` for a single
// worker whose behaviour is reproducible under a fixed `random_seed`.
//
// Implements the uniform runner contract; see heuristic_context.h.
//
// `found_elsewhere`, when set, is asked after every attempt whether
// something outside these workers moved the incumbent since it was last
// asked; a yes counts as that attempt's improvement for the runner-level
// patience gate (#309).  `race` passes the GPU climbers' finds: they raise
// the pool's best, after which no CPU offer moves it, so without this a
// race's patience would run out while the GPU is still improving.  It is
// called from the worker threads, so it must be thread-safe.  An overload
// rather than a default argument, so the chain can take `&fj::run` as a
// runner.
DispatchOutcome run(const ProblemView& problem, const HeuristicBudget& budget,
                    ExecutionContext& exec, SolutionSink& sink);
DispatchOutcome run(const ProblemView& problem, const HeuristicBudget& budget,
                    ExecutionContext& exec, SolutionSink& sink,
                    const std::function<bool()>& found_elsewhere);

// One FjWorker slot on the calling thread until `exec` says stop (#171),
// rebuilt whenever it stalls, from `source` first.  The rebuilt worker
// starts from that point unperturbed.  See `run_until_stopped` in
// heuristic_context.h for the contract.
[[nodiscard]] size_t run_until_stopped(const ProblemView& problem, const HeuristicBudget& budget,
                                       const ExecutionContext& exec, int worker,
                                       const RestartSource& source, SolutionSink& sink);

// A slot keeps its `FjWorker`, and so FJ's model, across rebuilds and
// resets it (#323).  Test hook: `true` builds a new worker per rebuild
// instead, as before #323, so a test can check that a reset slot searches
// as a rebuilt one.  Process-global; set it only with no dispatch running.
void set_fresh_rebuilds_for_test(bool fresh);

}  // namespace fj
