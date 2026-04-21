#pragma once

#include "continuous_loop.h"
#include "heuristic_common.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "parallel/HighsParallel.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>

// Generic opportunistic (continuous) parallel loop.
//
// Thin adapter over `ContinuousLoopState` (see `continuous_loop.h`).  Each
// worker runs in a tight loop calling `run_attempt` until the global
// budget, stale budget, or external termination signal is reached.
//
// Template parameters:
//   MakeState(int worker_idx, std::mt19937&) -> State
//     Called once per worker (inside the parallel region) to create
//     initial per-worker state.
//
//   RunAttempt(State&, std::mt19937&, size_t run_cap) -> HeuristicResult
//     Called repeatedly.  Should execute one heuristic attempt with at
//     most `run_cap` effort.  When the underlying worker is finished
//     (stalled), the callback should rebuild/restart the worker in-place.
//
// Thread-safety constraints:
//   - run_attempt must NOT spawn nested `parallel::for_each` regions.
//     (Scylla uses ScyllaWorker directly and does not go through this template.)
//   - Worker-0-only terminator polling: see `ContinuousLoopState`.
//   - Budget overshoot: concurrent workers can overshoot `budget` by
//     up to `N * default_run_cap` effort because each worker checks
//     the atomic total before starting an attempt.  Bounded overshoot
//     is acceptable for heuristic effort accounting.
//
// Returns total effort consumed across all workers.
template <typename MakeState, typename RunAttempt>
[[nodiscard]] size_t run_opportunistic_loop(HighsMipSolver &mipsolver, int num_workers,
                                            size_t budget, size_t stale_budget,
                                            size_t default_run_cap, uint32_t base_seed,
                                            MakeState make_state, RunAttempt run_attempt) {
    if (num_workers <= 0 || budget == 0) {
        return 0;
    }

    const int N = num_workers;
    ContinuousLoopState loop;

    highs::parallel::for_each(
        0, static_cast<HighsInt>(N),
        [&](HighsInt lo, HighsInt hi) {
            for (HighsInt w = lo; w < hi; ++w) {
                std::mt19937 rng(base_seed + static_cast<uint32_t>(w) * kSeedStride);
                int attempt_counter = 0;

                auto state = make_state(static_cast<int>(w), rng);

                while (!loop.stopped()) {
                    if (w == 0 && attempt_counter % 8 == 0) {
                        loop.poll_termination(mipsolver);
                    }
                    if (loop.stopped()) {
                        break;
                    }

                    size_t current = loop.total_effort.load(std::memory_order_relaxed);
                    size_t remaining = budget - std::min(budget, current);
                    size_t run_cap = std::min(default_run_cap, remaining);
                    if (run_cap == 0) {
                        loop.request_stop();
                        break;
                    }

                    auto result = run_attempt(state, rng, run_cap);
                    ++attempt_counter;

                    // Guard against workers that make no progress: a zero-effort
                    // return means this worker is done.  Request global stop so
                    // peers don't keep running past a solver timeout between
                    // worker-0's next polling tick (worker 0 is the only one that
                    // polls `mipsolver_.timer_` / `terminatorTerminated`).
                    if (result.effort == 0) {
                        loop.request_stop();
                        break;
                    }

                    loop.note_staleness(result.effort, result.found_feasible, stale_budget);
                    loop.add_effort(result.effort, budget);
                }
            }
        },
        1);

    return loop.total_effort.load(std::memory_order_relaxed);
}
