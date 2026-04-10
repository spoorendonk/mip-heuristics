#pragma once

#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "parallel/HighsParallel.h"

#include <concepts>
#include <cstddef>
#include <memory>
#include <vector>

// Result of a single epoch for one worker.
struct EpochResult {
    size_t effort = 0;
    bool found_improvement = false;
};

// Interface for workers that can be paused/resumed at epoch boundaries.
template <typename T>
concept EpochWorker = requires(T w, size_t budget) {
    { w.run_epoch(budget) } -> std::same_as<EpochResult>;
    { w.finished() } -> std::convertible_to<bool>;
    { w.reset_staleness() } -> std::same_as<void>;
};

// Generic epoch loop shared by sequential parallel modes and portfolio
// deterministic mode.  Workers run in parallel within each epoch and
// synchronize at the barrier.  Finished workers are restarted via the
// caller-provided callback.
//
// Thread-safety contract: workers run concurrently and must only perform
// read-only access to mipsolver fields (model, options, timer, mipdata
// arrays).  Pool access must be mutex-protected (SolutionPool uses
// HighsSpinMutex).  These invariants hold during presolve heuristic
// execution when HiGHS internals are immutable.
//
// Returns total effort consumed.
template <EpochWorker W, typename RestartFn>
size_t run_epoch_loop(HighsMipSolver &mipsolver, std::vector<std::unique_ptr<W>> &workers,
                      size_t budget, size_t epoch_budget, RestartFn restart_finished,
                      size_t stale_budget = 0) {
    if (stale_budget == 0) {
        stale_budget = budget >> 2;
    }
    const int N = static_cast<int>(workers.size());
    if (N == 0) {
        return 0;
    }

    auto *mipdata = mipsolver.mipdata_.get();
    const double time_limit = mipsolver.options_mip_->time_limit;

    size_t total_effort = 0;
    size_t effort_since_improvement = 0;

    std::vector<EpochResult> epoch_results(N);

    while (total_effort < budget) {
        // Pre-epoch (sequential): termination checks
        if (mipdata->terminatorTerminated() || mipsolver.timer_.read() >= time_limit) {
            break;
        }
        if (effort_since_improvement > stale_budget) {
            break;
        }

        // Restart finished workers; check if all are done.
        bool all_finished = true;
        for (int w = 0; w < N; ++w) {
            if (workers[w]->finished()) {
                restart_finished(w);
            }
            if (!workers[w]->finished()) {
                all_finished = false;
            }
        }
        if (all_finished) {
            break;
        }

        // Epoch (parallel): all workers run.
        highs::parallel::for_each(
            0, static_cast<HighsInt>(N),
            [&](HighsInt lo, HighsInt hi) {
                for (HighsInt w = lo; w < hi; ++w) {
                    epoch_results[w] = workers[w]->run_epoch(epoch_budget);
                }
            },
            1);

        // Post-epoch (sequential): merge results.
        size_t epoch_effort = 0;
        bool any_improved = false;
        for (int w = 0; w < N; ++w) {
            epoch_effort += epoch_results[w].effort;
            if (epoch_results[w].found_improvement) {
                any_improved = true;
            }
        }
        total_effort += epoch_effort;

        if (any_improved) {
            effort_since_improvement = 0;
            for (int w = 0; w < N; ++w) {
                workers[w]->reset_staleness();
            }
        } else {
            effort_since_improvement += epoch_effort;
        }
    }

    return total_effort;
}
