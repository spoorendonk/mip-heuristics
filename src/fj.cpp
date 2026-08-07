#include "fj.h"

#include "fj_worker.h"
#include "heuristic_common.h"
#include "heuristic_context.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "opportunistic_runner.h"
#include "solution_pool.h"

#include <memory>
#include <vector>

namespace fj {

namespace {

size_t run_parallel_workers(HighsMipSolver &mipsolver, SolutionPool &pool, size_t max_effort) {
    HeuristicContext ctx(mipsolver);
    const ExecutionContext &exec = ctx.exec();
    const HeuristicBudget budget = ctx.budget(max_effort);

    const uint32_t random_seed_opp = static_cast<uint32_t>(mipsolver.options_mip_->random_seed);

    struct FjState {
        std::unique_ptr<FjWorker> worker;
        uint32_t initial_seed = 0;
        bool first_creation = true;
    };

    return run_opportunistic_loop(
        mipsolver, static_cast<int>(exec.num_workers), budget.total, budget.stale,
        budget.attempt_cap, exec.base_seed,
        [random_seed_opp](int worker_idx, Rng & /*rng*/) -> FjState {
            // Pin first attempt to random_seed + w; worker 0 matches vanilla FJ's seed.
            return FjState{nullptr, random_seed_opp + static_cast<uint32_t>(worker_idx), true};
        },
        [&](FjState &state, Rng &rng, size_t run_cap) -> AttemptResult {
            if (!state.worker || state.worker->finished()) {
                uint32_t seed;
                if (state.first_creation) {
                    seed = state.initial_seed;
                    state.first_creation = false;
                } else {
                    seed = static_cast<uint32_t>(rng());
                }
                state.worker =
                    std::make_unique<FjWorker>(mipsolver, pool, budget.per_worker, seed);
            }
            return state.worker->run_attempt(run_cap);
        });
}

}  // namespace

size_t run_parallel(HighsMipSolver &mipsolver, SolutionPool &pool, size_t max_effort) {
    const auto *model = mipsolver.model_;
    if (model->num_col_ == 0 || model->num_row_ == 0) {
        return 0;
    }
    return run_parallel_workers(mipsolver, pool, max_effort);
}

}  // namespace fj
