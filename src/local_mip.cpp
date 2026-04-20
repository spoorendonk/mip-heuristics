#include "local_mip.h"

#include "epoch_runner.h"
#include "heuristic_common.h"
#include "local_mip_worker.h"
#include "lp_data/HConst.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "opportunistic_runner.h"
#include "parallel/HighsParallel.h"
#include "solution_pool.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

namespace local_mip {

using local_mip_detail::kEpochsPerWorker;
using local_mip_detail::LocalMipWorker;
using local_mip_detail::perturb_solution;

HeuristicResult worker(HighsMipSolver &mipsolver, const CscMatrix &csc, uint32_t seed,
                       const double *initial_solution, size_t max_effort) {
    const HighsInt ncol = mipsolver.model_->num_col_;
    const HighsInt nrow = mipsolver.model_->num_row_;

    HeuristicResult result;
    if (ncol == 0 || nrow == 0) {
        return result;
    }

    const bool minimize = (mipsolver.model_->sense_ == ObjSense::kMinimize);
    SolutionPool pool(1, minimize);

    // Disable stale_budget: pass max_effort so it can never fire before
    // the total budget is exhausted (original worker() had no stale_budget).
    LocalMipWorker w(mipsolver, csc, pool, max_effort, seed, initial_solution,
                     /*stale_budget=*/max_effort);

    size_t total_effort = 0;
    while (!w.finished()) {
        auto epoch = w.run_epoch(max_effort);
        total_effort += epoch.effort;
        if (epoch.effort == 0) {
            break;
        }
    }

    result.effort = total_effort;
    auto entries = pool.sorted_entries();
    if (!entries.empty()) {
        result.found_feasible = true;
        result.objective = entries[0].objective;
        result.solution = std::move(entries[0].solution);
    }

    return result;
}

namespace {

void run_parallel_deterministic(HighsMipSolver &mipsolver, size_t max_effort) {
    const auto *model = mipsolver.model_;
    auto *mipdata = mipsolver.mipdata_.get();
    const HighsInt ncol = model->num_col_;
    const HighsInt nrow = model->num_row_;

    const bool minimize = (model->sense_ == ObjSense::kMinimize);
    const int N = highs::parallel::num_threads();

    auto csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_, mipdata->ARvalue_);

    SolutionPool pool(kPoolCapacity, minimize);
    seed_pool(pool, mipsolver);

    const size_t worker_budget = max_effort / static_cast<size_t>(N);
    const size_t epoch_budget = std::max<size_t>(worker_budget / kEpochsPerWorker, 1);

    uint32_t base_seed = heuristic_base_seed(mipsolver.options_mip_->random_seed);

    // Create per-worker LocalMipWorker instances.
    // Worker 0: unperturbed incumbent.
    // Workers 1..N-1: perturbed incumbent.
    std::vector<std::unique_ptr<LocalMipWorker>> workers;
    workers.reserve(N);

    for (int w = 0; w < N; ++w) {
        uint32_t seed = base_seed + static_cast<uint32_t>(w) * kSeedStride;

        if (w == 0) {
            // Worker 0: unperturbed incumbent
            workers.push_back(std::make_unique<LocalMipWorker>(mipsolver, csc, pool, worker_budget,
                                                               seed, nullptr));
        } else {
            // Workers 1..N-1: perturbed incumbent
            std::vector<double> perturbed = mipdata->incumbent;
            std::mt19937 perturb_rng(seed);
            perturb_solution(perturbed, *mipdata, model->integrality_, model->col_lower_,
                             model->col_upper_, ncol, perturb_rng);
            workers.push_back(std::make_unique<LocalMipWorker>(mipsolver, csc, pool, worker_budget,
                                                               seed, perturbed.data()));
        }
    }

    // Track restart seed counter for deterministic restart seeding.
    uint32_t restart_seed_counter = static_cast<uint32_t>(N);

    size_t total_effort = run_epoch_loop(
        mipsolver, workers, max_effort, epoch_budget,
        [&](int w) {
            // Restart stalled worker from pool's best solution + new
            // perturbation + new seed.
            uint32_t new_seed =
                base_seed + static_cast<uint32_t>(restart_seed_counter++) * kSeedStride;

            std::vector<double> restart_sol;
            std::mt19937 restart_rng(new_seed);
            if (!pool.get_restart(restart_rng, restart_sol)) {
                // No pool solution; use incumbent
                restart_sol = mipdata->incumbent;
            }
            perturb_solution(restart_sol, *mipdata, model->integrality_, model->col_lower_,
                             model->col_upper_, ncol, restart_rng);
            workers[w] = std::make_unique<LocalMipWorker>(mipsolver, csc, pool, worker_budget,
                                                          new_seed, restart_sol.data());
        },
        max_effort >> 2);

    mipdata->heuristic_effort_used += total_effort;

    for (auto &entry : pool.sorted_entries()) {
        mipdata->trySolution(entry.solution, kSolutionSourceLocalMIP);
    }
}

void run_parallel_opportunistic(HighsMipSolver &mipsolver, size_t max_effort) {
    const auto *model = mipsolver.model_;
    auto *mipdata = mipsolver.mipdata_.get();
    const HighsInt ncol = model->num_col_;
    const HighsInt nrow = model->num_row_;

    const bool minimize = (model->sense_ == ObjSense::kMinimize);
    const int N = highs::parallel::num_threads();

    auto csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_, mipdata->ARvalue_);

    SolutionPool pool(kPoolCapacity, minimize);
    seed_pool(pool, mipsolver);

    uint32_t base_seed = heuristic_base_seed(mipsolver.options_mip_->random_seed);
    const size_t worker_budget = max_effort / static_cast<size_t>(N);
    const size_t default_run_cap = std::max<size_t>(max_effort / (static_cast<size_t>(N) * 10), 1);

    struct LmState {
        std::unique_ptr<LocalMipWorker> worker;
    };

    size_t total_effort = run_opportunistic_loop(
        mipsolver, N, max_effort, /*stale_budget=*/max_effort >> 2, default_run_cap, base_seed,
        [&](int worker_idx, std::mt19937 &rng) -> LmState {
            // Worker 0: unperturbed incumbent; others: perturbed.
            if (worker_idx == 0) {
                uint32_t seed = static_cast<uint32_t>(rng());
                return LmState{std::make_unique<LocalMipWorker>(mipsolver, csc, pool, worker_budget,
                                                                seed, nullptr)};
            }
            std::vector<double> perturbed = mipdata->incumbent;
            perturb_solution(perturbed, *mipdata, model->integrality_, model->col_lower_,
                             model->col_upper_, ncol, rng);
            uint32_t seed = static_cast<uint32_t>(rng());
            return LmState{std::make_unique<LocalMipWorker>(mipsolver, csc, pool, worker_budget,
                                                            seed, perturbed.data())};
        },
        [&](LmState &state, std::mt19937 &rng, size_t run_cap) -> HeuristicResult {
            if (!state.worker || state.worker->finished()) {
                // Restart from pool or incumbent with fresh perturbation.
                std::vector<double> restart_sol;
                if (!pool.get_restart(rng, restart_sol)) {
                    restart_sol = mipdata->incumbent;
                }
                perturb_solution(restart_sol, *mipdata, model->integrality_, model->col_lower_,
                                 model->col_upper_, ncol, rng);
                uint32_t seed = static_cast<uint32_t>(rng());
                state.worker = std::make_unique<LocalMipWorker>(mipsolver, csc, pool, worker_budget,
                                                                seed, restart_sol.data());
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
        mipdata->trySolution(entry.solution, kSolutionSourceLocalMIP);
    }
}

}  // namespace

void run_parallel(HighsMipSolver &mipsolver, size_t max_effort, bool opportunistic) {
    const auto *model = mipsolver.model_;
    auto *mipdata = mipsolver.mipdata_.get();
    const HighsInt ncol = model->num_col_;
    const HighsInt nrow = model->num_row_;
    if (ncol == 0 || nrow == 0) {
        return;
    }
    if (mipdata->incumbent.empty()) {
        return;
    }

    if (opportunistic) {
        run_parallel_opportunistic(mipsolver, max_effort);
    } else {
        run_parallel_deterministic(mipsolver, max_effort);
    }
}

}  // namespace local_mip
