#pragma once
#include "heuristic_context.h"
#include "worker_base.h"

#include <atomic>
#include <cstddef>
#include <cstdint>

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
DispatchOutcome run(const ProblemView& problem, const HeuristicBudget& budget,
                    ExecutionContext& exec, SolutionSink& sink);

// Whether slot `slot` of `num_workers` is the CDCL lane under
// `HeuristicBudget::fj_cdcl` (#237): the last, when there are two or more,
// since slot 0 carries vanilla FJ's seed.
constexpr bool is_cdcl_slot(size_t slot, size_t num_workers) {
    return num_workers >= 2 && slot + 1 == num_workers;
}

// The CDCL lane's one attempt (#237): `sat::sat_search` on `problem`, its
// point offered to `sink` under `trace`.  It ignores the objective, so it
// runs only while no feasible point is known (`problem.incumbent` empty and
// `sink` holding nothing), and it stops as soon as the sink holds one, the
// deadline passes, `loop_stop` (the runner's shared stop flag, when given)
// is set or `max_effort` (search effort, encoding included) is spent.
// Returns the effort spent, exempt from the runner's stale counter, and
// whether the offer improved the incumbent; zero effort when it did not run
// or was refused at once (`sat::SatRefusal::kNotPureBinary`).  Logs one
// `[FjCdcl]` line at `kVerbose`.  Polls `past_deadline()` only and charges
// nothing: safe on a worker thread.
[[nodiscard]] AttemptResult run_cdcl_lane(const ProblemView& problem, size_t max_effort,
                                          const ExecutionContext& exec, SolutionSink& sink,
                                          const WorkerTrace& trace, uint64_t seed,
                                          const std::atomic<bool>* loop_stop = nullptr);

// One FjWorker slot on the calling thread until `exec` says stop (#171),
// rebuilt whenever it stalls, from `source` first.  The rebuilt worker
// starts from that point unperturbed.  See `run_until_stopped` in
// heuristic_context.h for the contract.
[[nodiscard]] size_t run_until_stopped(const ProblemView& problem, const HeuristicBudget& budget,
                                       const ExecutionContext& exec, int worker,
                                       const RestartSource& source, SolutionSink& sink);
}  // namespace fj
