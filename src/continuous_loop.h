#pragma once

#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "parallel/HighsParallel.h"

#include <atomic>
#include <cstddef>
#include <cstdint>

// Shared scaffold for continuous-parallel heuristic runners.
//
// `run_opportunistic_loop` (in `opportunistic_runner.h`) spins a fixed
// worker pool that calls `parallel::for_each` once, then each worker loops
// on its own calling an attempt function until a global stop condition is
// hit.  Its stop conditions:
//
//   - Exactly one worker at a time polls `terminatorTerminated()` and
//     `timer_.read() >= time_limit` on every other attempt (the HiGHS
//     timer and terminator are not thread-safe for concurrent callers)
//     and sets the shared atomic `stop` flag; other workers observe it
//     within one attempt.  Since inner-loop timer polling was removed
//     from the workers themselves, the overshoot on time_limit trip is
//     bounded by ~2 attempts per worker.  The duty is a claimable seat
//     rather than a fixed worker index — see `poller` below.
//
//   - `total_effort >= budget` — may be overshot by up to
//     `N * per_attempt_cap` due to the lock-free increment.
//
//   - `effort_since_improvement >= stale_budget`.
//
// `ContinuousLoopState` exposes the atomic counters and the termination
// poll as plain helpers rather than a single `note_attempt`, so a runner
// keeps its own ordering of the per-attempt updates inline (the
// opportunistic runner bails immediately on zero effort).
struct ContinuousLoopState {
    std::atomic<size_t> total_effort{0};
    std::atomic<size_t> effort_since_improvement{0};
    std::atomic<bool> stop{false};

    // Index of the worker currently responsible for termination polling,
    // or `kNoPoller` when the seat is vacant.
    //
    // This is a claimable seat rather than a hardcoded worker 0 so that a
    // worker which finishes early can retire on its own.  When polling was
    // pinned to worker 0, the only way to guarantee someone was still
    // watching the clock was to stop *every* worker as soon as any one of
    // them had nothing left to do — so a single retiring chain tore down
    // the whole team, and the rebuild paths that would have replaced it
    // were unreachable.  Handing the seat on keeps the timeout guarantee
    // without that coupling.
    static constexpr int kNoPoller = -1;
    std::atomic<int> poller{0};

    [[nodiscard]] bool stopped() const { return stop.load(std::memory_order_relaxed); }

    void request_stop() { stop.store(true, std::memory_order_relaxed); }

    // True when `w` should poll: it already holds the seat, or the seat is
    // vacant and `w` wins the race to take it.  The compare-exchange is
    // what keeps the HiGHS timer/terminator single-caller — at most one
    // worker can hold the seat, and the previous holder has already left
    // its loop before releasing.
    //
    // acquire/release, not relaxed: what the seat guards is not race-free.
    // `terminatorTerminated()` *writes* the non-atomic
    // `mipsolver.termination_status_` when a terminator is attached, so a
    // relaxed handoff would leave the outgoing holder's write unordered
    // against the incoming holder's — a data race by the memory model,
    // caught by ThreadSanitizer, and benign only because x86 `lock
    // cmpxchg` happens to be a full barrier.  The seat exists to make
    // these calls single-caller; it has to publish like one.
    //
    // The `cur == w` fast path needs no acquire: the only holder that
    // never CASed is worker 0 via the initialiser below, ordered by
    // thread creation, and a worker never re-claims after releasing.
    bool claim_poller(int w) {
        int cur = poller.load(std::memory_order_relaxed);
        if (cur == w) {
            return true;
        }
        if (cur != kNoPoller) {
            return false;
        }
        return poller.compare_exchange_strong(cur, w, std::memory_order_acquire,
                                             std::memory_order_relaxed);
    }

    // Called by a worker on its way out of the loop: vacate the seat if it
    // holds it, so a surviving peer can take over the polling duty.
    void release_poller(int w) {
        int expected = w;
        poller.compare_exchange_strong(expected, kNoPoller, std::memory_order_release,
                                       std::memory_order_relaxed);
    }

    // Seat-holder only — the underlying HiGHS calls are not thread-safe
    // for concurrent callers.  Callers batch the poll to every other
    // attempt.  Peers observe the `stop` flag atomically.
    void poll_termination(HighsMipSolver &mipsolver) {
        auto *mipdata = mipsolver.mipdata_.get();
        const double time_limit = mipsolver.options_mip_->time_limit;
        if (mipdata->terminatorTerminated() || mipsolver.timer_.read() >= time_limit) {
            request_stop();
        }
    }

    // Bump the cumulative-effort atomic and set `stop` if the cumulative
    // total crossed `budget`.  Returns the new total.
    size_t add_effort(size_t effort, size_t budget) {
        size_t new_total = total_effort.fetch_add(effort) + effort;
        if (new_total >= budget) {
            request_stop();
        }
        return new_total;
    }

    // Update the staleness atomic and set `stop` if it crossed
    // `stale_budget`.  Relaxed ordering — staleness is advisory.
    void note_staleness(size_t effort, bool improved, size_t stale_budget) {
        if (improved) {
            effort_since_improvement.store(0, std::memory_order_relaxed);
        } else {
            effort_since_improvement.fetch_add(effort, std::memory_order_relaxed);
        }
        if (effort_since_improvement.load(std::memory_order_relaxed) >= stale_budget) {
            request_stop();
        }
    }
};
