#pragma once

#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "parallel/HighsParallel.h"

#include <atomic>
#include <cstddef>
#include <cstdint>

// Shared scaffold for continuous-parallel heuristic runners.
//
// `run_opportunistic_loop` (in `opportunistic_runner.h`) and
// `run_bandit_opportunistic_loop` (in `bandit_runner.h`) both spin a fixed
// worker pool that calls `parallel::for_each` once, then each worker loops
// on its own calling an attempt function until a global stop condition is
// hit.  The stop conditions they share:
//
//   - Worker 0 polls `terminatorTerminated()` and `timer_.read() >=
//     time_limit` once every 8 attempts (the HiGHS timer and terminator
//     are not thread-safe for concurrent callers) and sets the shared
//     atomic `stop` flag; other workers observe it within one attempt.
//
//   - `total_effort >= budget` — may be overshot by up to
//     `N * per_attempt_cap` due to the lock-free increment.
//
//   - `effort_since_improvement >= stale_budget`.
//
// The two runners differ slightly in the order of per-attempt updates
// (bandit updates staleness even on zero-effort attempts before bailing;
// opportunistic bails immediately on zero effort), so `ContinuousLoopState`
// exposes the atomic counters and the termination poll as plain helpers
// rather than a single `note_attempt` that would need conditional
// semantics.  Each runner keeps its own ordering inline.
struct ContinuousLoopState {
    std::atomic<size_t> total_effort{0};
    std::atomic<size_t> effort_since_improvement{0};
    std::atomic<bool> stop{false};

    [[nodiscard]] bool stopped() const { return stop.load(std::memory_order_relaxed); }

    void request_stop() { stop.store(true, std::memory_order_relaxed); }

    // Worker 0 only — the underlying HiGHS calls are not thread-safe for
    // concurrent callers.  Callers batch the poll to once every 8
    // attempts.  Peers observe the `stop` flag atomically.
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
