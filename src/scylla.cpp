#include "scylla.h"

#include "contested_pdlp.h"
#include "epoch_runner.h"
#include "heuristic_common.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "opportunistic_runner.h"
#include "parallel/HighsParallel.h"
#include "parallel_setup.h"
#include "scylla_worker.h"
#include "solution_pool.h"

#include <algorithm>
#include <memory>

namespace scylla {

namespace {

HighsInt compute_pdlp_iter_cap(size_t max_effort, size_t nnz_lp) {
    if (nnz_lp == 0) {
        return 100;
    }
    auto cap = static_cast<HighsInt>((max_effort >> 2) / nnz_lp);
    return cap < 100 ? 100 : cap;
}

void run_parallel_deterministic(HighsMipSolver &mipsolver, SolutionPool &pool, size_t max_effort) {
    ParallelSetup setup(mipsolver, max_effort);

    const HighsInt pdlp_iter_cap =
        compute_pdlp_iter_cap(max_effort, setup.mipdata->ARindex_.size());
    ContestedPdlp pdlp(mipsolver, pdlp_iter_cap);
    if (!pdlp.initialized()) {
        return;
    }

    std::atomic<uint64_t> improvement_gen{0};

    const int N = static_cast<int>(setup.N);
    std::vector<std::unique_ptr<ScyllaWorker>> workers;
    workers.reserve(setup.N);
    for (int w = 0; w < N; ++w) {
        uint32_t seed = setup.base_seed + static_cast<uint32_t>(w) * kSeedStride;
        workers.push_back(std::make_unique<ScyllaWorker>(mipsolver, pdlp, setup.csc, pool,
                                                         max_effort, seed, w, N, &improvement_gen));
    }

    size_t total_effort = run_epoch_loop(
        mipsolver, workers, max_effort, setup.epoch_budget(kEpochsPerWorker),
        [](int) { /* no restart */ }, setup.stale_budget);

    setup.mipdata->heuristic_effort_used += total_effort;
}

void run_parallel_opportunistic(HighsMipSolver &mipsolver, SolutionPool &pool, size_t max_effort) {
    ParallelSetup setup(mipsolver, max_effort);

    const HighsInt pdlp_iter_cap =
        compute_pdlp_iter_cap(max_effort, setup.mipdata->ARindex_.size());
    ContestedPdlp pdlp(mipsolver, pdlp_iter_cap);
    if (!pdlp.initialized()) {
        return;
    }

    std::atomic<uint64_t> improvement_gen{0};

    // Pre-construct workers outside the parallel region so MakeState
    // can hand them back by index without racing on std::make_unique.
    const int N = static_cast<int>(setup.N);
    std::vector<std::unique_ptr<ScyllaWorker>> workers;
    workers.reserve(setup.N);
    for (int w = 0; w < N; ++w) {
        uint32_t seed = setup.base_seed + static_cast<uint32_t>(w) * kSeedStride;
        workers.push_back(std::make_unique<ScyllaWorker>(mipsolver, pdlp, setup.csc, pool,
                                                         max_effort, seed, w, N, &improvement_gen));
    }

    struct ScyllaOppState {
        int worker_idx;
    };

    size_t total_effort = run_opportunistic_loop(
        mipsolver, N, max_effort, setup.stale_budget, setup.default_run_cap, setup.base_seed,
        [](int worker_idx, Rng & /*rng*/) -> ScyllaOppState { return ScyllaOppState{worker_idx}; },
        [&](ScyllaOppState &state, Rng &rng, size_t run_cap) -> HeuristicResult {
            auto &worker = workers[state.worker_idx];
            HeuristicResult result;
            if (worker->finished()) {
                // Rebuild stale worker with a fresh seed so the opportunistic
                // loop doesn't lose parallelism over time (mirrors the
                // fpr_lp opp path).  `pdlp` is shared, so warm-start etc. are
                // reinitialized from scratch but the underlying LP stays.
                uint32_t new_seed = static_cast<uint32_t>(rng());
                worker =
                    std::make_unique<ScyllaWorker>(mipsolver, pdlp, setup.csc, pool, max_effort,
                                                   new_seed, state.worker_idx, N, &improvement_gen);
            }
            auto epoch = worker->run_epoch(run_cap);
            // Report a nominal 1 unit when the chain is still alive but the
            // epoch produced no measurable effort (e.g. a PDLP stall that has
            // not yet hit kMaxPdlpStalls). Prevents run_opportunistic_loop's
            // zero-effort guard from permanently retiring a live chain.
            result.effort = (epoch.effort == 0 && !worker->finished()) ? 1 : epoch.effort;
            if (epoch.found_improvement) {
                result.found_feasible = true;
                result.objective = pool.snapshot().best_objective;
            }
            return result;
        });

    setup.mipdata->heuristic_effort_used += total_effort;
}

}  // namespace

void run_parallel(HighsMipSolver &mipsolver, SolutionPool &pool, size_t max_effort,
                  bool opportunistic) {
    const auto *model = mipsolver.model_;
    if (model->num_col_ == 0 || model->num_row_ == 0) {
        return;
    }
    if (opportunistic) {
        run_parallel_opportunistic(mipsolver, pool, max_effort);
    } else {
        run_parallel_deterministic(mipsolver, pool, max_effort);
    }
}

}  // namespace scylla
