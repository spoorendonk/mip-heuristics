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
//     seat; at most one at a time.  The wall-clock deadline is polled by
//     every worker on every iteration — see `ContinuousLoopState`.
//   - Budget overshoot: concurrent workers can overshoot `budget.total`
//     by up to `n * budget.attempt_cap` because each worker checks
//     the atomic total before starting an attempt.  Bounded overshoot
//     is acceptable for heuristic effort accounting.
//
// (`run_opportunistic_loop` itself is declared below `attempt_with_rebuild`.)
//
// ---------------------------------------------------------------------------
//
// One attempt from a worker that retires when it stalls.
//
// Scylla and fpr_lp both rebuild a retired worker in place, and both need
// the same three steps — the third of which is easy to leave out and was
// what made the rebuild path dead code once already.  Replace the worker
// *before* the attempt if it has already retired, and again *after* an
// attempt that both retired the worker and recorded no effort: returning 0
// there would trip `run_opportunistic_loop`'s zero-effort guard and retire
// the slot for the rest of the dispatch.
//
// `worker` is the slot to refill; `rebuild()` must leave it holding a fresh
// one.  `run_until_stopped` drives both FJ and LocalMIP through it, since a
// slot there must never retire on a stall.  (The presolve dispatch's FJ and
// LocalMIP do not use it: they rebuild before an attempt only, and a
// fresh worker that charges nothing retires its slot.)
template <typename WorkerPtr, typename Rebuild>
AttemptResult attempt_with_rebuild(WorkerPtr& worker, size_t run_cap, Rebuild rebuild) {
    if (worker->finished()) {
        rebuild();
    }
    AttemptResult attempt = worker->run_attempt(run_cap);
    if (attempt.effort == 0 && worker->finished()) {
        rebuild();
        attempt = worker->run_attempt(run_cap);
    }
    return attempt;
}

// One worker slot's loop, on the calling thread: build the slot's state,
// then run attempts until `loop` says stop, the budget is spent or an
// attempt charges nothing.  `run_opportunistic_loop` runs one per worker
// inside a parallel region and `run_on_caller_thread` runs one on the
// caller's thread; this is the whole of what they share, so neither has a
// loop of its own.
template <typename MakeState, typename RunAttempt>
void run_slot(ContinuousLoopState& loop, const ExecutionContext& exec,
              const HeuristicBudget& budget, int w, MakeState& make_state,
              RunAttempt& run_attempt) {
    Rng rng(exec.worker_seed(w));
    int attempt_counter = 0;

    auto state = make_state(w, rng);

    while (!loop.stopped()) {
        // Flag, deadline and terminator in one predicate — see
        // `ContinuousLoopState::should_stop`.  The deadline half runs on
        // every iteration and from this thread (issue #114), which is what
        // keeps a worker that stopped its attempt *on* the deadline from
        // re-entering `run_attempt` and spinning on immediately-terminating
        // attempts until whoever holds the poller seat next looks at the
        // clock.
        if (loop.should_stop(exec, w, attempt_counter)) {
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
        // return means this worker is done.  Retire it on its own — it used
        // to request a *global* stop, on the grounds that worker 0 was the
        // sole termination poller and peers would otherwise run past a
        // solver timeout.  The claimable poller seat removes that
        // constraint, so one exhausted chain no longer ends the dispatch for
        // its peers.  A callback that can rebuild its worker should do so
        // and report the retry's effort rather than returning 0 (see
        // `attempt_with_rebuild`).
        if (result.effort == 0) {
            break;
        }

        loop.note_staleness(result.effort, result.found_improvement, budget.stale);
        loop.add_effort(result.effort, budget.total);
    }

    loop.release_poller(w);
}

// Returns total effort consumed across all workers.
template <typename MakeState, typename RunAttempt>
[[nodiscard]] size_t run_opportunistic_loop(const ExecutionContext& exec,
                                            const HeuristicBudget& budget, MakeState make_state,
                                            RunAttempt run_attempt) {
    const int n = static_cast<int>(exec.num_workers);
    if (n <= 0 || budget.total == 0) {
        return 0;
    }

    ContinuousLoopState loop;

    highs::parallel::for_each(
        0, static_cast<HighsInt>(n),
        [&](HighsInt lo, HighsInt hi) {
            for (HighsInt w = lo; w < hi; ++w) {
                run_slot(loop, exec, budget, static_cast<int>(w), make_state, run_attempt);
            }
        },
        1);

    return loop.total_effort.load(std::memory_order_relaxed);
}

// Worker slot `worker` alone, on the calling thread rather than in a
// parallel region (#171): the loop `run_until_stopped` drives FJ and
// LocalMIP with on a thread the caller owns.  This thread is the slot's
// only one, so it holds the poller seat from the start and the terminator
// is polled here, every other attempt.  Returns the effort the attempts
// charged.
template <typename MakeState, typename RunAttempt>
[[nodiscard]] size_t run_on_caller_thread(const ExecutionContext& exec,
                                          const HeuristicBudget& budget, int worker,
                                          MakeState make_state, RunAttempt run_attempt) {
    if (budget.total == 0) {
        return 0;
    }
    ContinuousLoopState loop(worker);
    run_slot(loop, exec, budget, worker, make_state, run_attempt);
    return loop.total_effort.load(std::memory_order_relaxed);
}
