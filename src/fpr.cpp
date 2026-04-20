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

// Precomputed variable orders indexed by kFprStrategies position.  Built
// once sequentially before any parallel region (some strategies call
// HighsCliqueTable::cliquePartition which is not thread-safe).
using VarOrderTable = std::vector<std::vector<HighsInt>>;

// Worker that wraps a single fpr_attempt call per epoch.  Satisfies
// the EpochWorker concept from epoch_runner.h.
class FprWorker {
public:
    FprWorker(HighsMipSolver &mipsolver, const CscMatrix &csc, SolutionPool &pool,
              const VarOrderTable &var_orders, int strat_idx, FrameworkMode mode, uint32_t seed);

    EpochResult run_epoch(size_t epoch_budget);

    // Returns true when the worker has exceeded the hard stale threshold
    // (used by the opportunistic path to trigger worker replacement).
    // The deterministic path never hits this in practice because the
    // epoch loop's stale_budget fires first.
    bool finished() const { return finished_; }

    void reset_staleness() { epochs_without_improvement_ = 0; }

private:
    void randomize_config();

    HighsMipSolver &mipsolver_;
    const CscMatrix &csc_;
    SolutionPool &pool_;
    const VarOrderTable &var_orders_;

    int strat_idx_;
    FrameworkMode mode_;

    int attempt_idx_ = 0;
    int epochs_without_improvement_ = 0;
    bool finished_ = false;

    // Hard stale threshold for opportunistic mode: after this many
    // consecutive epochs without improvement the worker signals finished.
    // The soft threshold (kStaleEpochThreshold = 3) still applies for
    // config randomization before this point.
    static constexpr int kHardStaleThreshold = 15;

    std::mt19937 rng_;
};

static_assert(EpochWorker<FprWorker>, "FprWorker must satisfy EpochWorker concept");

namespace {

// Master strategy pool for all FPR parallel paths.  var_orders are
// precomputed for each entry (see precompute_var_orders) so any strategy
// — including clique-based ones like kStratBadobjcl whose compute_var_order
// calls HighsCliqueTable::cliquePartition — can be used inside a parallel
// region without racing on cliquePartition's internal state.
constexpr FprStrategyConfig kFprStrategies[] = {
    // Strategies used by the paper's curated initial configs.
    kStratBadobjcl,  // 0: type+cliques / badobj
    kStratLocks2,    // 1: locks / loosedyn
    kStratLocks,     // 2: LR / loosedyn
    kStratRandom,    // 3: type+cliques / random
    kStratDomsize,   // 4: domainSize / loosedyn
    // Extra strategies kept for randomization diversity at restart.
    kStratRandom2,  // 5: random / random
    kStratBadobj,   // 6: type / badobj
    kStratGoodobj,  // 7: type / goodobj
};
constexpr int kNumFprStrategies = static_cast<int>(std::size(kFprStrategies));

// Paper Section 6.3, Class 1 — LP-free initial configs.  Each entry gives
// a worker its starting (strategy, mode); strat_idx is an index into
// kFprStrategies.
struct InitialFprConfig {
    int strat_idx;
    FrameworkMode mode;
};
constexpr InitialFprConfig kInitialFprConfigs[] = {
    {0, FrameworkMode::kDfs},           // kStratBadobjcl, dfs
    {1, FrameworkMode::kDfs},           // kStratLocks2, dfs
    {1, FrameworkMode::kDive},          // kStratLocks2, dive
    {2, FrameworkMode::kDfsrep},        // kStratLocks, dfsrep
    {0, FrameworkMode::kDfsrep},        // kStratBadobjcl, dfsrep
    {3, FrameworkMode::kDiveprop},      // kStratRandom, diveprop
    {2, FrameworkMode::kRepairSearch},  // kStratLocks, repairsearch
    {4, FrameworkMode::kDfs},           // kStratDomsize, dfs
};
constexpr int kNumInitialFprConfigs = static_cast<int>(std::size(kInitialFprConfigs));

constexpr FrameworkMode kAllModes[] = {
    FrameworkMode::kDfs,      FrameworkMode::kDfsrep,       FrameworkMode::kDive,
    FrameworkMode::kDiveprop, FrameworkMode::kRepairSearch,
};

constexpr int kNumAllModes = static_cast<int>(std::size(kAllModes));

// Number of stale epochs before a worker randomizes its config.
constexpr int kStaleEpochThreshold = 3;

// Compute variable orders for every strategy in kFprStrategies.  MUST be
// called from a sequential context: clique-based var_strategies invoke
// HighsCliqueTable::cliquePartition which mutates internal state and is
// not thread-safe.
VarOrderTable precompute_var_orders(HighsMipSolver &mipsolver) {
    VarOrderTable orders(kNumFprStrategies);
    const uint32_t base = heuristic_base_seed(mipsolver.options_mip_->random_seed);
    for (int i = 0; i < kNumFprStrategies; ++i) {
        std::mt19937 rng(base + static_cast<uint32_t>(i));
        orders[i] = compute_var_order(mipsolver, kFprStrategies[i].var_strategy, rng, nullptr);
    }
    return orders;
}

}  // namespace

// ---------------------------------------------------------------------------
// FprWorker implementation
// ---------------------------------------------------------------------------

FprWorker::FprWorker(HighsMipSolver &mipsolver, const CscMatrix &csc, SolutionPool &pool,
                     const VarOrderTable &var_orders, int strat_idx, FrameworkMode mode,
                     uint32_t seed)
    : mipsolver_(mipsolver),
      csc_(csc),
      pool_(pool),
      var_orders_(var_orders),
      strat_idx_(strat_idx),
      mode_(mode),
      rng_(seed) {}

void FprWorker::randomize_config() {
    strat_idx_ = std::uniform_int_distribution<int>(0, kNumFprStrategies - 1)(rng_);
    int m_idx = std::uniform_int_distribution<int>(0, kNumAllModes - 1)(rng_);
    mode_ = kAllModes[m_idx];
    // var_order lookup via var_orders_[strat_idx_] — no recomputation needed.
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

    const auto &strat = kFprStrategies[strat_idx_];
    const auto &var_order = var_orders_[strat_idx_];

    FprConfig cfg{};
    cfg.max_effort = epoch_budget;
    cfg.hint = nullptr;
    cfg.scores = nullptr;
    cfg.cont_fallback = nullptr;
    cfg.csc = &csc_;
    cfg.mode = mode_;
    cfg.strategy = &strat;
    cfg.lp_ref = nullptr;
    cfg.precomputed_var_order = var_order.data();
    cfg.precomputed_var_order_size = static_cast<HighsInt>(var_order.size());

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
    const int N = highs::parallel::num_threads();

    auto csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_, mipdata->ARvalue_);

    // Precompute var_orders sequentially before any parallel region.
    VarOrderTable var_orders = precompute_var_orders(mipsolver);

    SolutionPool pool(kPoolCapacity, minimize);
    seed_pool(pool, mipsolver);

    // Per-worker epoch budget: divide total across workers and target
    // ~10 epochs per worker for meaningful synchronization.
    constexpr int kEpochsPerWorker = 10;
    const size_t per_worker = max_effort / static_cast<size_t>(N);
    const size_t epoch_budget = std::max<size_t>(per_worker / kEpochsPerWorker, 1);

    uint32_t base_seed = heuristic_base_seed(mipsolver.options_mip_->random_seed);

    std::vector<std::unique_ptr<FprWorker>> workers;
    workers.reserve(N);
    for (int w = 0; w < N; ++w) {
        int cfg_idx = w % kNumInitialFprConfigs;
        uint32_t seed = base_seed + static_cast<uint32_t>(w) * kSeedStride;
        workers.push_back(std::make_unique<FprWorker>(mipsolver, csc, pool, var_orders,
                                                      kInitialFprConfigs[cfg_idx].strat_idx,
                                                      kInitialFprConfigs[cfg_idx].mode, seed));
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
    const int N = highs::parallel::num_threads();

    auto csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_, mipdata->ARvalue_);

    // Precompute var_orders sequentially before any parallel region.
    VarOrderTable var_orders = precompute_var_orders(mipsolver);

    SolutionPool pool(kPoolCapacity, minimize);
    seed_pool(pool, mipsolver);

    uint32_t base_seed = heuristic_base_seed(mipsolver.options_mip_->random_seed);
    const size_t default_run_cap = std::max<size_t>(max_effort / (static_cast<size_t>(N) * 10), 1);

    std::vector<std::unique_ptr<FprWorker>> workers;
    workers.reserve(N);
    for (int w = 0; w < N; ++w) {
        int cfg_idx = w % kNumInitialFprConfigs;
        uint32_t seed = base_seed + static_cast<uint32_t>(w) * kSeedStride;
        workers.push_back(std::make_unique<FprWorker>(mipsolver, csc, pool, var_orders,
                                                      kInitialFprConfigs[cfg_idx].strat_idx,
                                                      kInitialFprConfigs[cfg_idx].mode, seed));
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
                // Replace with a random (strategy, mode) from the full pool.
                // Safe because var_orders are precomputed for every strategy.
                int strat_idx = std::uniform_int_distribution<int>(0, kNumFprStrategies - 1)(rng);
                int m_idx = std::uniform_int_distribution<int>(0, kNumAllModes - 1)(rng);
                uint32_t seed = static_cast<uint32_t>(rng());
                worker = std::make_unique<FprWorker>(mipsolver, csc, pool, var_orders, strat_idx,
                                                     kAllModes[m_idx], seed);
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
