#include "fpr.h"

#include <random>
#include <vector>

#include "fpr_core.h"
#include "fpr_strategies.h"
#include "heuristic_common.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "parallel/HighsParallel.h"
#include "solution_pool.h"

namespace fpr {

namespace {

constexpr int kPoolCapacity = 10;

// Paper Section 6.3, Class 1 — LP-free strategies.
// These are the strategies selected in the paper's portfolio for the LP-free
// class (run before any LP is solved).
struct NamedConfig {
  FprStrategyConfig strat;
  FrameworkMode mode;
};

// Paper's 6 LP-free configs:
//   dfs-badobjcl, dfs-locks2, dive-locks2,
//   dfsrep-locks, dfsrep-badobjcl, diveprop-random
constexpr NamedConfig kLpFreeConfigs[] = {
    {kStratBadobjcl, FrameworkMode::kDfs},
    {kStratLocks2, FrameworkMode::kDfs},
    {kStratLocks2, FrameworkMode::kDive},
    {kStratLocks, FrameworkMode::kDfsrep},
    {kStratBadobjcl, FrameworkMode::kDfsrep},
    {kStratRandom, FrameworkMode::kDiveprop},
};

constexpr int kNumLpFreeConfigs =
    static_cast<int>(sizeof(kLpFreeConfigs) / sizeof(kLpFreeConfigs[0]));

}  // namespace

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

  // Seed pool with incumbent if available
  if (!mipdata->incumbent.empty()) {
    double obj = model->offset_;
    for (HighsInt j = 0; j < ncol; ++j) {
      obj += model->col_cost_[j] * mipdata->incumbent[j];
    }
    pool.try_add(obj, mipdata->incumbent);
  }

  // Build CSC once for all workers
  auto csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_,
                       mipdata->ARvalue_);

  const double *hint =
      mipdata->incumbent.empty() ? nullptr : mipdata->incumbent.data();

  size_t total_effort = 0;
  const int num_configs = kNumLpFreeConfigs;

  // Pre-compute variable orders sequentially to avoid data races on
  // HighsCliqueTable::cliquePartition (which mutates internal state).
  std::vector<std::vector<HighsInt>> var_orders(num_configs);
  for (int w = 0; w < num_configs; ++w) {
    std::mt19937 rng(42 + static_cast<uint32_t>(w));
    var_orders[w] = compute_var_order(
        mipsolver, kLpFreeConfigs[w].strat.var_strategy, rng, nullptr);
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
          cfg.precomputed_var_order_size =
              static_cast<HighsInt>(var_orders[w].size());

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

}  // namespace fpr
