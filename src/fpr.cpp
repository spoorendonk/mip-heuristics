#include "fpr.h"

#include "epoch_runner.h"
#include "fpr_core.h"
#include "fpr_strategies.h"
#include "heuristic_common.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "opportunistic_runner.h"
#include "parallel/HighsParallel.h"
#include "solution_pool.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace fpr {

// Worker that wraps a single fpr_attempt call per epoch.  Satisfies
// the EpochWorker concept from epoch_runner.h.
class FprWorker {
public:
    FprWorker(HighsMipSolver &mipsolver, const CscMatrix &csc, SolutionPool &pool,
              FprStrategyConfig strat, FrameworkMode mode, uint32_t seed);

    EpochResult run_epoch(size_t epoch_budget);

    // Returns true when the worker has exceeded the hard stale threshold
    // (used by the opportunistic path to trigger worker replacement).
    // The deterministic path never hits this in practice because the
    // epoch loop's stale_budget fires first.
    bool finished() const { return finished_; }

    void reset_staleness() { epochs_without_improvement_ = 0; }

private:
    void randomize_config();
    void recompute_var_order();

    HighsMipSolver &mipsolver_;
    const CscMatrix &csc_;
    SolutionPool &pool_;

    FprStrategyConfig strat_;
    FrameworkMode mode_;

    int attempt_idx_ = 0;
    int epochs_without_improvement_ = 0;
    bool finished_ = false;

    // Hard stale threshold for opportunistic mode: after this many
    // consecutive epochs without improvement the worker signals finished.
    // The soft threshold (kStaleEpochThreshold = 3) still applies for
    // config randomization before this point.
    static constexpr int kHardStaleThreshold = 15;

    std::vector<HighsInt> var_order_;
    std::mt19937 rng_;
};

static_assert(EpochWorker<FprWorker>, "FprWorker must satisfy EpochWorker concept");

namespace {

// Paper Section 6.3, Class 1 — LP-free strategies.
// These are the strategies selected in the paper's portfolio for the LP-free
// class (run before any LP is solved).
//
// Paper's 6 LP-free configs + RepairSearch + dynamic domain-size:
//   dfs-badobjcl, dfs-locks2, dive-locks2,
//   dfsrep-locks, dfsrep-badobjcl, diveprop-random, repairsearch-locks,
//   dfs-domsize (not in paper — experimental, O(ncol²) cost excludes it
//   from portfolio bandit arms where repeated pulls would be too expensive)
constexpr NamedConfig kLpFreeConfigs[] = {
    {kStratBadobjcl, FrameworkMode::kDfs},       {kStratLocks2, FrameworkMode::kDfs},
    {kStratLocks2, FrameworkMode::kDive},        {kStratLocks, FrameworkMode::kDfsrep},
    {kStratBadobjcl, FrameworkMode::kDfsrep},    {kStratRandom, FrameworkMode::kDiveprop},
    {kStratLocks, FrameworkMode::kRepairSearch}, {kStratDomsize, FrameworkMode::kDfs},
};

constexpr int kNumLpFreeConfigs =
    static_cast<int>(sizeof(kLpFreeConfigs) / sizeof(kLpFreeConfigs[0]));

// Thread-safe LP-free strategies for config randomization during parallel
// epochs.  Strategies whose VarStrategy uses HighsCliqueTable::cliquePartition
// (kTypecl, kCliques, kCliques2) are excluded because cliquePartition mutates
// internal state and is not thread-safe.  This leaves 6 of the 11 named
// LP-free strategies; combined with 5 modes = 30 thread-safe configs.
constexpr FprStrategyConfig kSafeLpFreeStrategies[] = {
    kStratRandom2,  // random / random
    kStratBadobj,   // type / badobj
    kStratGoodobj,  // type / goodobj
    kStratLocks,    // LR / loosedyn
    kStratLocks2,   // locks / loosedyn
    kStratDomsize,  // domainSize / loosedyn
};

constexpr int kNumSafeStrategies =
    static_cast<int>(sizeof(kSafeLpFreeStrategies) / sizeof(kSafeLpFreeStrategies[0]));

constexpr FrameworkMode kAllModes[] = {
    FrameworkMode::kDfs,      FrameworkMode::kDfsrep,       FrameworkMode::kDive,
    FrameworkMode::kDiveprop, FrameworkMode::kRepairSearch,
};

constexpr int kNumAllModes = static_cast<int>(sizeof(kAllModes) / sizeof(kAllModes[0]));

// Number of stale epochs before a worker randomizes its config.
constexpr int kStaleEpochThreshold = 3;

// Max workers for parallel mode.
constexpr int kMaxFprWorkers = 8;

}  // namespace

// ---------------------------------------------------------------------------
// Original sequential run (unchanged)
// ---------------------------------------------------------------------------

void run(HighsMipSolver &mipsolver, size_t max_effort) {
    const auto *model = mipsolver.model_;
    auto *mipdata = mipsolver.mipdata_.get();
    const HighsInt ncol = model->num_col_;
    const HighsInt nrow = model->num_row_;
    if (ncol == 0 || nrow == 0) {
        return;
    }

    const bool minimize = (model->sense_ == ObjSense::kMinimize);
    SolutionPool pool(kPoolCapacity, minimize);

    seed_pool(pool, mipsolver);

    // Build CSC once for all workers
    auto csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_, mipdata->ARvalue_);

    const double *hint = mipdata->incumbent.empty() ? nullptr : mipdata->incumbent.data();

    size_t total_effort = 0;
    const int num_configs = kNumLpFreeConfigs;

    // Pre-compute variable orders sequentially to avoid data races on
    // HighsCliqueTable::cliquePartition (which mutates internal state).
    std::vector<std::vector<HighsInt>> var_orders(num_configs);
    for (int w = 0; w < num_configs; ++w) {
        std::mt19937 rng(42 + static_cast<uint32_t>(w));
        var_orders[w] =
            compute_var_order(mipsolver, kLpFreeConfigs[w].strat.var_strategy, rng, nullptr);
    }

    // Run all LP-free configs in parallel (paper Section 6.3, Class 1)
    std::vector<HeuristicResult> results(num_configs);

    highs::parallel::for_each(
        0, static_cast<HighsInt>(num_configs),
        [&](HighsInt lo, HighsInt hi) {
            for (HighsInt w = lo; w < hi; ++w) {
                std::mt19937 rng(42 + static_cast<uint32_t>(w));

                FprConfig cfg{};
                cfg.max_effort = max_effort;
                cfg.hint = hint;
                cfg.scores = nullptr;
                cfg.cont_fallback = nullptr;
                cfg.csc = &csc;
                cfg.mode = kLpFreeConfigs[w].mode;
                cfg.strategy = &kLpFreeConfigs[w].strat;
                cfg.lp_ref = nullptr;
                cfg.precomputed_var_order = var_orders[w].data();
                cfg.precomputed_var_order_size = static_cast<HighsInt>(var_orders[w].size());

                results[w] = fpr_attempt(mipsolver, cfg, rng, 0, nullptr);
            }
        },
        1);

    for (int w = 0; w < num_configs; ++w) {
        total_effort += results[w].effort;
        if (results[w].found_feasible) {
            pool.try_add(results[w].objective, results[w].solution);
        }
    }

    mipdata->heuristic_effort_used += total_effort;

    // Submit best solutions to solver
    for (auto &entry : pool.sorted_entries()) {
        mipdata->trySolution(entry.solution, kSolutionSourceFPR);
    }
}

// ---------------------------------------------------------------------------
// FprWorker implementation
// ---------------------------------------------------------------------------

FprWorker::FprWorker(HighsMipSolver &mipsolver, const CscMatrix &csc, SolutionPool &pool,
                     FprStrategyConfig strat, FrameworkMode mode, uint32_t seed)
    : mipsolver_(mipsolver), csc_(csc), pool_(pool), strat_(strat), mode_(mode), rng_(seed) {
    recompute_var_order();
}

void FprWorker::randomize_config() {
    int s_idx = std::uniform_int_distribution<int>(0, kNumSafeStrategies - 1)(rng_);
    int m_idx = std::uniform_int_distribution<int>(0, kNumAllModes - 1)(rng_);
    strat_ = kSafeLpFreeStrategies[s_idx];
    mode_ = kAllModes[m_idx];
    recompute_var_order();
}

void FprWorker::recompute_var_order() {
    var_order_ = compute_var_order(mipsolver_, strat_.var_strategy, rng_, nullptr);
}

EpochResult FprWorker::run_epoch(size_t epoch_budget) {
    EpochResult epoch{};

    // After K stale epochs, randomize config from full space.
    if (epochs_without_improvement_ >= kStaleEpochThreshold) {
        randomize_config();
        epochs_without_improvement_ = 0;
    }

    // Get pool restart solution if available.
    std::vector<double> initial_solution;
    const double *init_ptr = nullptr;
    if (pool_.get_restart(rng_, initial_solution)) {
        init_ptr = initial_solution.data();
    }

    FprConfig cfg{};
    cfg.max_effort = epoch_budget;
    cfg.hint = nullptr;
    cfg.scores = nullptr;
    cfg.cont_fallback = nullptr;
    cfg.csc = &csc_;
    cfg.mode = mode_;
    cfg.strategy = &strat_;
    cfg.lp_ref = nullptr;
    cfg.precomputed_var_order = var_order_.data();
    cfg.precomputed_var_order_size = static_cast<HighsInt>(var_order_.size());

    auto result = fpr_attempt(mipsolver_, cfg, rng_, attempt_idx_, init_ptr);
    epoch.effort = result.effort;

    if (result.found_feasible) {
        pool_.try_add(result.objective, result.solution);
        epoch.found_improvement = true;
        epochs_without_improvement_ = 0;
    } else {
        ++epochs_without_improvement_;
        if (epochs_without_improvement_ >= kHardStaleThreshold) {
            finished_ = true;
        }
    }

    ++attempt_idx_;

    return epoch;
}

// ---------------------------------------------------------------------------
// Parallel epoch-gated FPR
// ---------------------------------------------------------------------------

namespace {

void run_parallel_deterministic(HighsMipSolver &mipsolver, size_t max_effort) {
    const auto *model = mipsolver.model_;
    auto *mipdata = mipsolver.mipdata_.get();
    const HighsInt ncol = model->num_col_;
    const HighsInt nrow = model->num_row_;

    const bool minimize = (model->sense_ == ObjSense::kMinimize);
    // FPR workers are lightweight (CscMatrix ref + var_order + RNG), but
    // still cap at kMaxFprWorkers and by available memory.
    const size_t fpr_worker_mem = static_cast<size_t>(ncol) * sizeof(HighsInt);  // var_order vector
    const int mem_cap = max_workers_for_memory(fpr_worker_mem);
    const int N = std::min({highs::parallel::num_threads(), kMaxFprWorkers, mem_cap});

    auto csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_, mipdata->ARvalue_);

    SolutionPool pool(kPoolCapacity, minimize);
    seed_pool(pool, mipsolver);

    // Per-worker epoch budget: divide total across workers and target
    // ~10 epochs per worker for meaningful synchronization.
    constexpr int kEpochsPerWorker = 10;
    const size_t per_worker = max_effort / static_cast<size_t>(N);
    const size_t epoch_budget = std::max<size_t>(per_worker / kEpochsPerWorker, 1);

    uint32_t base_seed = static_cast<uint32_t>(mipdata->numImprovingSols + kBaseSeedOffset);

    // Create workers sequentially — construction MUST stay sequential because
    // initial curated configs may use clique-based VarStrategy (e.g.,
    // kStratBadobjcl) whose compute_var_order calls cliquePartition (not
    // thread-safe).  Subsequent randomize_config() in run_epoch() only picks
    // from kSafeLpFreeStrategies which excludes clique strategies.
    std::vector<std::unique_ptr<FprWorker>> workers;
    workers.reserve(N);
    for (int w = 0; w < N; ++w) {
        int cfg_idx = w % kNumLpFreeConfigs;
        uint32_t seed = base_seed + static_cast<uint32_t>(w) * kSeedStride;
        workers.push_back(std::make_unique<FprWorker>(mipsolver, csc, pool,
                                                      kLpFreeConfigs[cfg_idx].strat,
                                                      kLpFreeConfigs[cfg_idx].mode, seed));
    }

    size_t total_effort = run_epoch_loop(
        mipsolver, workers, max_effort, epoch_budget,
        [](int) { /* FprWorkers rarely hit hard stale threshold in det mode */ }, max_effort >> 2);

    mipdata->heuristic_effort_used += total_effort;

    for (auto &entry : pool.sorted_entries()) {
        mipdata->trySolution(entry.solution, kSolutionSourceFPR);
    }
}

void run_parallel_opportunistic(HighsMipSolver &mipsolver, size_t max_effort) {
    const auto *model = mipsolver.model_;
    auto *mipdata = mipsolver.mipdata_.get();
    const HighsInt ncol = model->num_col_;
    const HighsInt nrow = model->num_row_;

    const bool minimize = (model->sense_ == ObjSense::kMinimize);
    const size_t fpr_worker_mem = static_cast<size_t>(ncol) * sizeof(HighsInt);
    const int mem_cap = max_workers_for_memory(fpr_worker_mem);
    const int N = std::min({highs::parallel::num_threads(), kMaxFprWorkers, mem_cap});

    auto csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_, mipdata->ARvalue_);

    SolutionPool pool(kPoolCapacity, minimize);
    seed_pool(pool, mipsolver);

    uint32_t base_seed = static_cast<uint32_t>(mipdata->numImprovingSols + kBaseSeedOffset);
    const size_t default_run_cap = std::max<size_t>(max_effort / (static_cast<size_t>(N) * 10), 1);

    // Create workers sequentially — construction MUST stay sequential because
    // initial curated configs may use clique-based VarStrategy (e.g.,
    // kStratBadobjcl) whose compute_var_order calls cliquePartition (not
    // thread-safe).
    std::vector<std::unique_ptr<FprWorker>> workers;
    workers.reserve(N);
    for (int w = 0; w < N; ++w) {
        int cfg_idx = w % kNumLpFreeConfigs;
        uint32_t seed = base_seed + static_cast<uint32_t>(w) * kSeedStride;
        workers.push_back(std::make_unique<FprWorker>(mipsolver, csc, pool,
                                                      kLpFreeConfigs[cfg_idx].strat,
                                                      kLpFreeConfigs[cfg_idx].mode, seed));
    }

    struct FprOppState {
        int worker_idx;
    };

    size_t total_effort = run_opportunistic_loop(
        mipsolver, N, max_effort, /*stale_budget=*/max_effort >> 2, default_run_cap, base_seed,
        [](int worker_idx, std::mt19937 & /*rng*/) -> FprOppState {
            return FprOppState{worker_idx};
        },
        [&](FprOppState &state, std::mt19937 &rng, size_t run_cap) -> HeuristicResult {
            auto &worker = workers[state.worker_idx];
            if (worker->finished()) {
                // Replace with a new worker using a safe (non-clique) strategy
                // and a random mode.  Thread-safe because kSafeLpFreeStrategies
                // excludes clique-based strategies.
                int s_idx = std::uniform_int_distribution<int>(0, kNumSafeStrategies - 1)(rng);
                int m_idx = std::uniform_int_distribution<int>(0, kNumAllModes - 1)(rng);
                uint32_t seed = static_cast<uint32_t>(rng());
                worker = std::make_unique<FprWorker>(
                    mipsolver, csc, pool, kSafeLpFreeStrategies[s_idx], kAllModes[m_idx], seed);
            }
            auto epoch = worker->run_epoch(run_cap);
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
        mipdata->trySolution(entry.solution, kSolutionSourceFPR);
    }
}

}  // namespace

void run_parallel(HighsMipSolver &mipsolver, size_t max_effort, bool opportunistic) {
    const auto *model = mipsolver.model_;
    const HighsInt ncol = model->num_col_;
    const HighsInt nrow = model->num_row_;
    if (ncol == 0 || nrow == 0) {
        return;
    }

    if (opportunistic) {
        run_parallel_opportunistic(mipsolver, max_effort);
    } else {
        run_parallel_deterministic(mipsolver, max_effort);
    }
}

}  // namespace fpr
