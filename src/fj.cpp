#include "fj.h"

#include "epoch_runner.h"
#include "fj_worker.h"
#include "heuristic_common.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "opportunistic_runner.h"
#include "parallel_setup.h"
#include "solution_pool.h"

#include <memory>
#include <vector>

namespace fj {

namespace {

size_t run_parallel_deterministic(HighsMipSolver &mipsolver, SolutionPool &pool,
                                  size_t max_effort) {
    ParallelSetup setup(mipsolver, max_effort);

    // Restart counter for generating fresh seeds when workers finish.
    uint32_t restart_counter = 0;

    std::vector<std::unique_ptr<FjWorker>> workers;
    workers.reserve(setup.N);
    const uint32_t random_seed = static_cast<uint32_t>(mipsolver.options_mip_->random_seed);
    for (size_t w = 0; w < setup.N; ++w) {
        uint32_t seed = random_seed + static_cast<uint32_t>(w);
        workers.push_back(std::make_unique<FjWorker>(mipsolver, pool, setup.worker_budget, seed));
    }

    return run_epoch_loop(
        mipsolver, workers, max_effort, setup.epoch_budget(kEpochsPerWorkerFj),
        [&](int w) {
            // Restart finished FjWorker: continue the linear seed sequence past N.
            ++restart_counter;
            uint32_t seed = random_seed + static_cast<uint32_t>(setup.N) + restart_counter;
            workers[w] = std::make_unique<FjWorker>(mipsolver, pool, setup.worker_budget, seed);
        },
        setup.stale_budget);
}

size_t run_parallel_opportunistic(HighsMipSolver &mipsolver, SolutionPool &pool,
                                  size_t max_effort) {
    ParallelSetup setup(mipsolver, max_effort);

    const uint32_t random_seed_opp = static_cast<uint32_t>(mipsolver.options_mip_->random_seed);

    struct FjState {
        std::unique_ptr<FjWorker> worker;
        uint32_t initial_seed = 0;
        bool first_creation = true;
    };

    return run_opportunistic_loop(
        mipsolver, static_cast<int>(setup.N), max_effort, setup.stale_budget, setup.default_run_cap,
        setup.base_seed,
        [random_seed_opp](int worker_idx, Rng & /*rng*/) -> FjState {
            // Pin first attempt to random_seed + w; worker 0 matches vanilla FJ's seed.
            return FjState{nullptr, random_seed_opp + static_cast<uint32_t>(worker_idx), true};
        },
        [&](FjState &state, Rng &rng, size_t run_cap) -> HeuristicResult {
            if (!state.worker || state.worker->finished()) {
                uint32_t seed;
                if (state.first_creation) {
                    seed = state.initial_seed;
                    state.first_creation = false;
                } else {
                    seed = static_cast<uint32_t>(rng());
                }
                state.worker =
                    std::make_unique<FjWorker>(mipsolver, pool, setup.worker_budget, seed);
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
}

}  // namespace

size_t run_parallel(HighsMipSolver &mipsolver, SolutionPool &pool, size_t max_effort,
                    bool opportunistic) {
    const auto *model = mipsolver.model_;
    if (model->num_col_ == 0 || model->num_row_ == 0) {
        return 0;
    }
    if (opportunistic) {
        return run_parallel_opportunistic(mipsolver, pool, max_effort);
    }
    return run_parallel_deterministic(mipsolver, pool, max_effort);
}

}  // namespace fj
