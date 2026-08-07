#pragma once

#include "continuous_loop.h"
#include "heuristic_common.h"
#include "heuristic_context.h"
#include "parallel/HighsParallel.h"
#include "worker_base.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

// Generic opportunistic (continuous) parallel loop.
//
// Thin adapter over `ContinuousLoopState` (see `continuous_loop.h`).  Each
// worker runs in a tight loop calling `run_attempt` until the global
// budget, stale budget, or external termination signal is reached.
//
// Template parameters:
//   MakeState(int worker_idx, Rng&) -> State
//     Called once per worker (inside the parallel region) to create
//     initial per-worker state.
//
//   RunAttempt(State&, Rng&, size_t run_cap) -> AttemptResult
//     Called repeatedly.  Should execute one heuristic attempt with at
//     most `run_cap` effort.  When the underlying worker is finished
//     (stalled), the callback should rebuild/restart the worker in-place.
//
// Thread-safety constraints:
//   - run_attempt must NOT spawn nested `parallel::for_each` regions.
//   - Terminator polling is done by whichever worker holds the claimable
//     seat; at most one at a time.  See `ContinuousLoopState`.
//   - Budget overshoot: concurrent workers can overshoot `budget.total`
//     by up to `N * budget.attempt_cap` because each worker checks
//     the atomic total before starting an attempt.  Bounded overshoot
//     is acceptable for heuristic effort accounting.
//
// Returns total effort consumed across all workers.
template <typename MakeState, typename RunAttempt>
[[nodiscard]] size_t run_opportunistic_loop(const ExecutionContext &exec,
                                            const HeuristicBudget &budget, MakeState make_state,
                                            RunAttempt run_attempt) {
    const int N = static_cast<int>(exec.num_workers);
    if (N <= 0 || budget.total == 0) {
        return 0;
    }

    ContinuousLoopState loop;

    highs::parallel::for_each(
        0, static_cast<HighsInt>(N),
        [&](HighsInt lo, HighsInt hi) {
            for (HighsInt w = lo; w < hi; ++w) {
                Rng rng(exec.base_seed + static_cast<uint32_t>(w) * kSeedStride);
                int attempt_counter = 0;

                auto state = make_state(static_cast<int>(w), rng);

                while (!loop.stopped()) {
                    if ((attempt_counter & 1) == 0 && loop.claim_poller(static_cast<int>(w))) {
                        loop.poll_termination(exec);
                    }
                    if (loop.stopped()) {
                        break;
                    }

                    size_t current = loop.total_effort.load(std::memory_order_relaxed);
                    size_t remaining = budget.total - std::min(budget.total, current);
                    size_t run_cap = std::min(budget.attempt_cap, remaining);
                    if (run_cap == 0) {
                        loop.request_stop();
                        break;
                    }

                    auto result = run_attempt(state, rng, run_cap);
                    ++attempt_counter;

                    // Guard against workers that make no progress: a zero-effort
                    // return means this worker is done.  Retire it on its own —
                    // it used to request a *global* stop, on the grounds that
                    // worker 0 was the sole termination poller and peers would
                    // otherwise run past a solver timeout.  The claimable poller
                    // seat removes that constraint, so one exhausted chain no
                    // longer ends the dispatch for its peers.  A callback that
                    // can rebuild its worker should do so and report the retry's
                    // effort rather than returning 0 (see scylla.cpp, fpr_lp.cpp).
                    if (result.effort == 0) {
                        break;
                    }

                    loop.note_staleness(result.effort, result.found_improvement, budget.stale);
                    loop.add_effort(result.effort, budget.total);
                }

                loop.release_poller(static_cast<int>(w));
            }
        },
        1);

    return loop.total_effort.load(std::memory_order_relaxed);
}
