#pragma once

#include <cstddef>

// Result of a single attempt for one worker.
struct AttemptResult {
    size_t effort = 0;
    bool found_improvement = false;
};

// Shared per-worker bookkeeping for the heuristic workers.
//
// Embed (composition, NOT inheritance) into workers that track cumulative
// effort + staleness + a hard total budget — `FjWorker`, `FprWorker`,
// `LocalMipWorker` and `ScyllaWorker`.  `FprWorker` joined them in issue
// #111 and arms only the staleness half (its `total_budget` stays at
// SIZE_MAX; see fpr.cpp).  `LpFprWorker` keeps a private stale counter
// and `finished_` flag without this struct.
//
// `stale_budget` is set by the caller from an absolute, instance-scaled
// constant (`stall_threshold` in heuristic_common.h), never from
// `total_budget`.  A threshold derived as a fraction of the allowance
// grows with the allowance, so it can only ever report "I have spent
// that fraction" — it cannot detect that the search stopped producing,
// which is the entire job of this counter (issue #111).
//
// Fields are plain (non-atomic) because each worker's inner loop accesses
// them single-threaded.  The continuous-parallel runner owns its own
// atomic counters; see `ContinuousLoopState` in `continuous_loop.h`.
struct WorkerBudgetState {
    size_t total_budget = 0;
    size_t stale_budget = 0;
    size_t total_effort = 0;
    size_t effort_since_improvement = 0;
    bool finished = false;

    // True when this worker has exceeded its staleness budget.
    [[nodiscard]] bool stale() const { return effort_since_improvement > stale_budget; }

    // True when already stale, or would become stale after `extra` more
    // effort.  Used for prospective inner-loop checks that avoid one attempt
    // of overshoot (see LocalMipWorker).
    [[nodiscard]] bool stale(size_t extra) const {
        return effort_since_improvement + extra > stale_budget;
    }

    // True when this worker has consumed its total budget.
    [[nodiscard]] bool exhausted() const { return total_effort >= total_budget; }

    // True when already exhausted, or would become exhausted after `extra`
    // more effort.  Mirrors the prospective overload of `stale`.
    [[nodiscard]] bool exhausted(size_t extra) const {
        return total_effort + extra >= total_budget;
    }

    // Clear the staleness counter; called by the worker itself on the
    // improvement path (and, for Scylla, when a peer broadcasts one via
    // `improvement_gen_`).
    void reset_staleness() { effort_since_improvement = 0; }

    // Accumulate effort when the worker found an improvement.  Resets
    // staleness and marks finished if total budget is exceeded.
    void charge_improvement(size_t effort) {
        total_effort += effort;
        effort_since_improvement = 0;
        if (exhausted()) {
            finished = true;
        }
    }

    // Accumulate effort when the worker found NO improvement.  Advances the
    // staleness counter and marks finished if either budget is exceeded.
    void charge_no_improvement(size_t effort) {
        total_effort += effort;
        effort_since_improvement += effort;
        if (exhausted() || stale()) {
            finished = true;
        }
    }
};
