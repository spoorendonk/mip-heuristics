#include "scylla.h"

#include "contested_pdlp.h"
#include "epoch_runner.h"
#include "heuristic_common.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "opportunistic_runner.h"
#include "parallel/HighsParallel.h"
#include "scylla_worker.h"
#include "solution_pool.h"

#include <algorithm>
#include <memory>

namespace scylla {

namespace {

// Epoch granularity for deterministic mode: each worker takes ~10 turns
// inside the total budget, matching the other heuristics.
constexpr int kEpochsPerWorker = 10;

HighsInt compute_pdlp_iter_cap(size_t max_effort, size_t nnz_lp) {
    if (nnz_lp == 0) {
        return 100;
    }
    auto cap = static_cast<HighsInt>((max_effort >> 2) / nnz_lp);
    return cap < 100 ? 100 : cap;
}

struct ScyllaSetup {
    CscMatrix csc;
    SolutionPool pool;
    int num_workers;
    size_t stale_budget;
    uint32_t base_seed;
    HighsInt pdlp_iter_cap;

    ScyllaSetup(HighsMipSolver &mipsolver, size_t max_effort, bool minimize)
        : pool(kPoolCapacity, minimize) {
        auto *mipdata = mipsolver.mipdata_.get();
        const auto *model = mipsolver.model_;
        csc = build_csc(model->num_col_, model->num_row_, mipdata->ARstart_, mipdata->ARindex_,
                        mipdata->ARvalue_);
        seed_pool(pool, mipsolver);

        num_workers = highs::parallel::num_threads();

        stale_budget = max_effort >> 2;
        base_seed =
            compute_base_seed(mipdata->numImprovingSols, mipsolver.options_mip_->random_seed);
        pdlp_iter_cap = compute_pdlp_iter_cap(max_effort, mipdata->ARindex_.size());
    }
};

void run_parallel_deterministic(HighsMipSolver &mipsolver, size_t max_effort) {
    const auto *model = mipsolver.model_;
    auto *mipdata = mipsolver.mipdata_.get();
    if (model->num_col_ == 0 || model->num_row_ == 0) {
        return;
    }
    const bool minimize = (model->sense_ == ObjSense::kMinimize);

    ScyllaSetup setup(mipsolver, max_effort, minimize);

    ContestedPdlp pdlp(mipsolver, setup.pdlp_iter_cap);
    if (!pdlp.initialized()) {
        return;
    }

    std::atomic<uint64_t> improvement_gen{0};

    std::vector<std::unique_ptr<ScyllaWorker>> workers;
    workers.reserve(setup.num_workers);
    for (int w = 0; w < setup.num_workers; ++w) {
        uint32_t seed = setup.base_seed + static_cast<uint32_t>(w) * kSeedStride;
        workers.push_back(std::make_unique<ScyllaWorker>(mipsolver, pdlp, setup.csc, setup.pool,
                                                         max_effort, seed, w, setup.num_workers,
                                                         &improvement_gen));
    }

    const size_t epoch_budget = std::max<size_t>(
        max_effort / (static_cast<size_t>(setup.num_workers) * kEpochsPerWorker), 1);

    size_t total_effort = run_epoch_loop(
        mipsolver, workers, max_effort, epoch_budget, [](int) { /* no restart */ },
        setup.stale_budget);

    mipdata->heuristic_effort_used += total_effort;

    for (auto &entry : setup.pool.sorted_entries()) {
        mipdata->trySolution(entry.solution, kSolutionSourceScylla);
    }
}

void run_parallel_opportunistic(HighsMipSolver &mipsolver, size_t max_effort) {
    const auto *model = mipsolver.model_;
    auto *mipdata = mipsolver.mipdata_.get();
    if (model->num_col_ == 0 || model->num_row_ == 0) {
        return;
    }
    const bool minimize = (model->sense_ == ObjSense::kMinimize);

    ScyllaSetup setup(mipsolver, max_effort, minimize);
    const int N = setup.num_workers;

    ContestedPdlp pdlp(mipsolver, setup.pdlp_iter_cap);
    if (!pdlp.initialized()) {
        return;
    }

    const size_t default_run_cap = std::max<size_t>(max_effort / (static_cast<size_t>(N) * 10), 1);

    std::atomic<uint64_t> improvement_gen{0};

    // Pre-construct workers outside the parallel region so MakeState
    // can hand them back by index without racing on std::make_unique.
    std::vector<std::unique_ptr<ScyllaWorker>> workers;
    workers.reserve(N);
    for (int w = 0; w < N; ++w) {
        uint32_t seed = setup.base_seed + static_cast<uint32_t>(w) * kSeedStride;
        workers.push_back(std::make_unique<ScyllaWorker>(mipsolver, pdlp, setup.csc, setup.pool,
                                                         max_effort, seed, w, N, &improvement_gen));
    }

    struct ScyllaOppState {
        int worker_idx;
    };

    size_t total_effort = run_opportunistic_loop(
        mipsolver, N, max_effort, setup.stale_budget, default_run_cap, setup.base_seed,
        [](int worker_idx, std::mt19937 & /*rng*/) -> ScyllaOppState {
            return ScyllaOppState{worker_idx};
        },
        [&](ScyllaOppState &state, std::mt19937 &rng, size_t run_cap) -> HeuristicResult {
            auto &worker = workers[state.worker_idx];
            HeuristicResult result;
            if (worker->finished()) {
                // Rebuild stale worker with a fresh seed so the opportunistic
                // loop doesn't lose parallelism over time (mirrors the
                // fpr_lp opp path).  `pdlp` is shared, so warm-start etc. are
                // reinitialized from scratch but the underlying LP stays.
                uint32_t new_seed = static_cast<uint32_t>(rng());
                worker = std::make_unique<ScyllaWorker>(mipsolver, pdlp, setup.csc, setup.pool,
                                                        max_effort, new_seed, state.worker_idx, N,
                                                        &improvement_gen);
            }
            auto epoch = worker->run_epoch(run_cap);
            // Report a nominal 1 unit when the chain is still alive but the
            // epoch produced no measurable effort (e.g. a PDLP stall that has
            // not yet hit kMaxPdlpStalls). Prevents run_opportunistic_loop's
            // zero-effort guard from permanently retiring a live chain.
            result.effort = (epoch.effort == 0 && !worker->finished()) ? 1 : epoch.effort;
            if (epoch.found_improvement) {
                result.found_feasible = true;
                result.objective = setup.pool.snapshot().best_objective;
            }
            return result;
        });

    mipdata->heuristic_effort_used += total_effort;

    for (auto &entry : setup.pool.sorted_entries()) {
        mipdata->trySolution(entry.solution, kSolutionSourceScylla);
    }
}

}  // namespace

void run_parallel(HighsMipSolver &mipsolver, size_t max_effort, bool opportunistic) {
    const auto *model = mipsolver.model_;
    if (model->num_col_ == 0 || model->num_row_ == 0) {
        return;
    }
    if (opportunistic) {
        run_parallel_opportunistic(mipsolver, max_effort);
    } else {
        run_parallel_deterministic(mipsolver, max_effort);
    }
}

}  // namespace scylla
