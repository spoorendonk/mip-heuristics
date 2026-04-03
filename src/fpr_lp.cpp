#include "fpr_lp.h"

#include <random>
#include <vector>

#include "fpr_core.h"
#include "fpr_strategies.h"
#include "heuristic_common.h"
#include "mip/HighsLpRelaxation.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "parallel/HighsParallel.h"
#include "solution_pool.h"

namespace fpr_lp {

namespace {

constexpr int kPoolCapacity = 10;

struct NamedConfig {
  FprStrategyConfig strat;
  FrameworkMode mode;
};

// Paper Section 6.3, Class 2 — zero-obj LP strategies
constexpr NamedConfig kClass2Configs[] = {
    {kStratZerocore, FrameworkMode::kDfs},
    {kStratZerocore, FrameworkMode::kDive},
    {kStratZerocore, FrameworkMode::kDiveprop},
    {kStratCliques, FrameworkMode::kDfs},  // paper: "if predominant clique
                                            // structure"; run unconditionally,
                                            // degrades gracefully on non-clique
                                            // models
};
constexpr int kNumClass2 =
    static_cast<int>(sizeof(kClass2Configs) / sizeof(kClass2Configs[0]));

// Paper Section 6.3, Class 3 — full-obj LP strategies
constexpr NamedConfig kClass3Configs[] = {
    {kStratZerolp, FrameworkMode::kDfs},
    {kStratZerolp, FrameworkMode::kDiveprop},
    {kStratCliques2, FrameworkMode::kDiveprop},
    {kStratLp, FrameworkMode::kDfs},
    {kStratLp, FrameworkMode::kDive},
    {kStratLp, FrameworkMode::kDiveprop},
};
constexpr int kNumClass3 =
    static_cast<int>(sizeof(kClass3Configs) / sizeof(kClass3Configs[0]));

// Run a set of configs in parallel, collecting results into a pool.
// Returns total effort consumed.
size_t run_configs(HighsMipSolver &mipsolver, const CscMatrix &csc,
                   const NamedConfig *configs, int num_configs,
                   const double *hint, const double *lp_ref,
                   SolutionPool &pool, size_t budget) {
  // Pre-compute variable orders sequentially to avoid data races on
  // HighsCliqueTable::cliquePartition (which mutates internal state).
  std::vector<std::vector<HighsInt>> var_orders(num_configs);
  for (int w = 0; w < num_configs; ++w) {
    std::mt19937 rng(42 + static_cast<uint32_t>(w) + 100);
    var_orders[w] = compute_var_order(mipsolver, configs[w].strat.var_strategy,
                                      rng, lp_ref);
  }

  std::vector<HeuristicResult> results(num_configs);

  highs::parallel::for_each(
      0, static_cast<HighsInt>(num_configs),
      [&](HighsInt lo, HighsInt hi) {
        for (HighsInt w = lo; w < hi; ++w) {
          std::mt19937 rng(42 + static_cast<uint32_t>(w) + 100);

          FprConfig cfg{};
          cfg.max_effort = budget;
          cfg.hint = hint;
          cfg.scores = nullptr;
          cfg.cont_fallback = nullptr;
          cfg.csc = &csc;
          cfg.mode = configs[w].mode;
          cfg.strategy = &configs[w].strat;
          cfg.lp_ref = lp_ref;
          cfg.precomputed_var_order = var_orders[w].data();
          cfg.precomputed_var_order_size =
              static_cast<HighsInt>(var_orders[w].size());

          results[w] = fpr_attempt(mipsolver, cfg, rng, 0, nullptr);
        }
      },
      1);

  size_t total_effort = 0;
  for (int w = 0; w < num_configs; ++w) {
    total_effort += results[w].effort;
    if (results[w].found_feasible) {
      pool.try_add(results[w].objective, results[w].solution);
    }
  }
  return total_effort;
}

}  // namespace

void run(HighsMipSolver &mipsolver, size_t max_effort) {
  const auto *model = mipsolver.model_;
  auto *mipdata = mipsolver.mipdata_.get();
  const HighsInt ncol = model->num_col_;
  const HighsInt nrow = model->num_row_;
  if (ncol == 0 || nrow == 0) {
    return;
  }

  // Guard: need an optimal LP relaxation
  auto lp_status = mipdata->lp.getStatus();
  if (!HighsLpRelaxation::scaledOptimal(lp_status)) {
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

  // Build CSC once
  auto csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_,
                       mipdata->ARvalue_);

  const double *hint =
      mipdata->incumbent.empty() ? nullptr : mipdata->incumbent.data();

  // Get LP solution
  const auto &lp_sol = mipdata->lp.getLpSolver().getSolution().col_value;
  const double *lp_ptr = lp_sol.data();

  // Compute zero-obj analytic center (for zerocore strategies)
  auto analytic_center = compute_analytic_center(mipsolver, false);
  const double *ac_ptr =
      analytic_center.empty() ? lp_ptr : analytic_center.data();

  // Compute zero-obj LP vertex (for zerolp strategies)
  auto zero_vertex = compute_zero_obj_vertex(mipsolver);
  const double *zv_ptr = zero_vertex.empty() ? lp_ptr : zero_vertex.data();

  size_t total_effort = 0;
  size_t remaining = max_effort;

  // Class 2: zero-obj LP strategies (use analytic center / zero vertex)
  auto snap_before_c2 = pool.snapshot();
  total_effort += run_configs(mipsolver, csc, kClass2Configs, kNumClass2, hint,
                              ac_ptr, pool, remaining);
  remaining = (total_effort < max_effort) ? max_effort - total_effort : 0;

  // Check if Class 2 found a new solution — if so, skip Class 3
  // (paper: stop if feasible found between classes).
  // Compare pool snapshot before/after to detect new entries (pool was
  // seeded with incumbent, so non-empty doesn't mean Class 2 found anything).
  auto snap_after_c2 = pool.snapshot();
  bool class2_found_new =
      snap_after_c2.has_solution &&
      (!snap_before_c2.has_solution ||
       (minimize ? snap_after_c2.best_objective < snap_before_c2.best_objective
                 : snap_after_c2.best_objective >
                       snap_before_c2.best_objective));

  if (!class2_found_new && remaining > 0) {
    // Class 3: full-obj LP strategies (use LP solution as lp_ref)
    total_effort += run_configs(mipsolver, csc, kClass3Configs, kNumClass3,
                                hint, lp_ptr, pool, remaining);
  }

  mipdata->heuristic_effort_used += total_effort;

  // Submit best solutions to solver
  for (auto &entry : pool.sorted_entries()) {
    mipdata->trySolution(entry.solution, kSolutionSourceFPR);
  }
}

}  // namespace fpr_lp
