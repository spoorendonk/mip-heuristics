#include "fj.h"

#include "epoch_runner.h"
#include "fj_worker.h"
#include "heuristic_common.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "opportunistic_runner.h"
#include "solution_pool.h"

#include <memory>
#include <vector>

namespace fj {

bool run(HighsMipSolver &mipsolver, size_t max_effort) {
    auto *mipdata = mipsolver.mipdata_.get();
    std::vector<double> sol;
    double obj = 0.0;
    size_t effort = 0;
    auto status = mipdata->feasibilityJumpCapture(sol, obj, effort, max_effort, nullptr);
    if (!sol.empty()) {
        mipdata->trySolution(sol, kSolutionSourceFJ);
    }
    mipdata->heuristic_effort_used += effort;
    return status == HighsModelStatus::kInfeasible;
}

namespace {

bool run_parallel_deterministic(HighsMipSolver &mipsolver, size_t max_effort) {
    const auto *model = mipsolver.model_;
    auto *mipdata = mipsolver.mipdata_.get();

    const bool minimize = (model->sense_ == ObjSense::kMinimize);
    const int N = highs::parallel::num_threads();

    SolutionPool pool(kPoolCapacity, minimize);
    seed_pool(pool, mipsolver);

    const size_t worker_budget = max_effort / static_cast<size_t>(N);
    constexpr int kEpochsPerWorker = 20;
    const size_t epoch_budget = std::max<size_t>(worker_budget / kEpochsPerWorker, 1);

    uint32_t base_seed = static_cast<uint32_t>(mipdata->numImprovingSols + kBaseSeedOffset);

    // Restart counter for generating fresh seeds when workers finish.
    uint32_t restart_counter = 0;

    std::vector<std::unique_ptr<FjWorker>> workers;
    workers.reserve(N);
    for (int w = 0; w < N; ++w) {
        uint32_t seed = base_seed + static_cast<uint32_t>(w) * kSeedStride;
        workers.push_back(std::make_unique<FjWorker>(mipsolver, pool, worker_budget, seed));
    }

    size_t total_effort = run_epoch_loop(
        mipsolver, workers, max_effort, epoch_budget,
        [&](int w) {
            // Restart finished FjWorker with a new seed.
            ++restart_counter;
            uint32_t seed = base_seed + (static_cast<uint32_t>(N) + restart_counter) * kSeedStride;
            workers[w] = std::make_unique<FjWorker>(mipsolver, pool, worker_budget, seed);
        },
        max_effort >> 2);

    mipdata->heuristic_effort_used += total_effort;

    for (auto &entry : pool.sorted_entries()) {
        mipdata->trySolution(entry.solution, kSolutionSourceFJ);
    }

    return false;
}

bool run_parallel_opportunistic(HighsMipSolver &mipsolver, size_t max_effort) {
    const auto *model = mipsolver.model_;
    auto *mipdata = mipsolver.mipdata_.get();

    const bool minimize = (model->sense_ == ObjSense::kMinimize);
    const int N = highs::parallel::num_threads();

    SolutionPool pool(kPoolCapacity, minimize);
    seed_pool(pool, mipsolver);

    uint32_t base_seed = static_cast<uint32_t>(mipdata->numImprovingSols + kBaseSeedOffset);
    const size_t worker_budget = max_effort / static_cast<size_t>(N);
    const size_t default_run_cap = std::max<size_t>(max_effort / (static_cast<size_t>(N) * 10), 1);

    struct FjState {
        std::unique_ptr<FjWorker> worker;
    };

    size_t total_effort = run_opportunistic_loop(
        mipsolver, N, max_effort, /*stale_budget=*/max_effort >> 2, default_run_cap, base_seed,
        [](int /*worker_idx*/, std::mt19937 & /*rng*/) -> FjState {
            // Lazy init: worker is created on first run_attempt call.
            return FjState{};
        },
        [&](FjState &state, std::mt19937 &rng, size_t run_cap) -> HeuristicResult {
            if (!state.worker || state.worker->finished()) {
                uint32_t seed = static_cast<uint32_t>(rng());
                state.worker = std::make_unique<FjWorker>(mipsolver, pool, worker_budget, seed);
            }
            auto epoch = state.worker->run_epoch(run_cap);
            HeuristicResult result;
            result.effort = epoch.effort;
            if (epoch.found_improvement) {
                result.found_feasible = true;
                result.objective = pool.snapshot().best_objective;
            }
            return result;
        });

    mipdata->heuristic_effort_used += total_effort;

    for (auto &entry : pool.sorted_entries()) {
        mipdata->trySolution(entry.solution, kSolutionSourceFJ);
    }

    return false;
}

}  // namespace

bool run_parallel(HighsMipSolver &mipsolver, size_t max_effort, bool opportunistic) {
    const auto *model = mipsolver.model_;
    if (model->num_col_ == 0 || model->num_row_ == 0) {
        return false;
    }

    if (opportunistic) {
        return run_parallel_opportunistic(mipsolver, max_effort);
    }
    return run_parallel_deterministic(mipsolver, max_effort);
}

}  // namespace fj
