#pragma once

#include "heuristic_common.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "parallel/HighsParallel.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <random>

// Generic opportunistic (continuous) parallel loop.
//
// Encapsulates the `parallel::for_each` + `while(!stop)` + atomic
// budget/staleness/termination pattern.  Each worker runs in a tight
// loop calling `run_attempt` until the global budget, stale budget,
// or external termination signal is reached.
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
//   - Worker-0-only terminator polling: `terminatorTerminated()` mutates
//     `mipsolver.termination_status_` as a side effect, so concurrent
//     calls from multiple workers would race.  Only worker 0 polls it,
//     and it batches the check to once every 8 attempts.  Other workers
//     observe the atomic `stop` flag set by worker 0 within one
//     `default_run_cap` worth of effort.  Worst-case detection latency
//     is ~8 worker-0 attempts plus one `default_run_cap` per peer worker.
//   - `timer_.read()` reads stopped clock state during the presolve hook
//     and is safe to call from any worker (the benign-stale-read
//     race is documented in scylla_worker.cpp, which polls directly
//     from every worker).  The runner still gates it behind worker 0
//     simply because it is cheaper to pair with the terminator poll.
//   - Budget overshoot: concurrent workers can overshoot `budget` by
//     up to `N * default_run_cap` effort because each worker checks
//     the atomic total before starting an attempt.  This bounded
//     overshoot is acceptable for heuristic effort accounting.
//
// Returns total effort consumed across all workers.
template <typename MakeState, typename RunAttempt>
size_t run_opportunistic_loop(HighsMipSolver &mipsolver, int num_workers, size_t budget,
                              size_t stale_budget, size_t default_run_cap, uint32_t base_seed,
                              MakeState make_state, RunAttempt run_attempt) {
    if (num_workers <= 0 || budget == 0) {
        return 0;
    }

    auto *mipdata = mipsolver.mipdata_.get();
    const double time_limit = mipsolver.options_mip_->time_limit;
    const int N = num_workers;

    std::atomic<size_t> total_effort{0};
    std::atomic<size_t> effort_since_improvement{0};
    std::atomic<bool> stop{false};

    highs::parallel::for_each(
        0, static_cast<HighsInt>(N),
        [&](HighsInt lo, HighsInt hi) {
            for (HighsInt w = lo; w < hi; ++w) {
                std::mt19937 rng(base_seed + static_cast<uint32_t>(w) * kSeedStride);
                int attempt_counter = 0;

                auto state = make_state(static_cast<int>(w), rng);

                while (!stop.load(std::memory_order_relaxed)) {
                    // Worker 0 polls termination every 8 attempts to amortize
                    // the (non-thread-safe) timer/terminator query.
                    if (w == 0 && attempt_counter % 8 == 0) {
                        if (mipdata->terminatorTerminated() ||
                            mipsolver.timer_.read() >= time_limit) {
                            stop.store(true, std::memory_order_relaxed);
                        }
                    }
                    if (stop.load(std::memory_order_relaxed)) {
                        break;
                    }

                    // Compute remaining budget and per-attempt cap.
                    size_t current = total_effort.load(std::memory_order_relaxed);
                    size_t remaining = budget - std::min(budget, current);
                    size_t run_cap = std::min(default_run_cap, remaining);
                    if (run_cap == 0) {
                        stop.store(true, std::memory_order_relaxed);
                        break;
                    }

                    auto result = run_attempt(state, rng, run_cap);
                    ++attempt_counter;

                    // Guard against workers that make no progress: a zero-effort
                    // return means this worker is done, stop it rather than spin.
                    if (result.effort == 0) {
                        break;
                    }

                    // Update staleness tracking.
                    if (result.found_feasible) {
                        effort_since_improvement.store(0, std::memory_order_relaxed);
                    } else {
                        effort_since_improvement.fetch_add(result.effort,
                                                           std::memory_order_relaxed);
                    }

                    if (effort_since_improvement.load(std::memory_order_relaxed) >= stale_budget) {
                        stop.store(true, std::memory_order_relaxed);
                    }

                    // Update total effort and check budget.
                    size_t new_total = total_effort.fetch_add(result.effort) + result.effort;
                    if (new_total >= budget) {
                        stop.store(true, std::memory_order_relaxed);
                    }
                }
            }
        },
        1);

    return total_effort.load(std::memory_order_relaxed);
}
