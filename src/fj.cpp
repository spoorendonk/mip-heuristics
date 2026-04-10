#include "fj.h"

#include "epoch_runner.h"
#include "fj_worker.h"
#include "heuristic_common.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
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

// TODO(#61): when opportunistic=true, dispatch to a new
// run_parallel_opportunistic helper.  Currently the flag is accepted
// but ignored; seq/opp falls through to seq/det.
bool run_parallel(HighsMipSolver &mipsolver, size_t max_effort,
                  [[maybe_unused]] bool opportunistic) {
    const auto *model = mipsolver.model_;
    auto *mipdata = mipsolver.mipdata_.get();
    if (model->num_col_ == 0 || model->num_row_ == 0) {
        return false;
    }

    const bool minimize = (model->sense_ == ObjSense::kMinimize);
    const int mem_cap = max_workers_for_memory(
        estimate_worker_memory_fj(model->num_col_, model->num_row_, mipdata->ARindex_.size()));
    const int N = std::min(highs::parallel::num_threads(), mem_cap);

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

}  // namespace fj
