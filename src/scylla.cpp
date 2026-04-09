#include "scylla.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "Highs.h"
#include "fpr_core.h"
#include "fpr_strategies.h"
#include "heuristic_common.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "parallel/HighsParallel.h"
#include "pump_common.h"
#include "pump_worker.h"
#include "solution_pool.h"

namespace scylla {

namespace {

// LP-free FPR configs for parallel rounding (subset of fpr.cpp's configs).
constexpr NamedConfig kFprConfigs[] = {
    {kStratBadobjcl, FrameworkMode::kDfs},
    {kStratLocks2, FrameworkMode::kDfs},
    {kStratLocks2, FrameworkMode::kDive},
    {kStratLocks, FrameworkMode::kDfsrep},
};

constexpr int kNumFprConfigs =
    static_cast<int>(sizeof(kFprConfigs) / sizeof(kFprConfigs[0]));

// Convenience wrapper for the sequential (single-worker) code path.
// Runs a single pump chain to completion within the given budget.
size_t pump_worker(HighsMipSolver &mipsolver, const CscMatrix &csc,
                   SolutionPool &pool, size_t worker_budget, uint32_t seed) {
  PumpWorker worker(mipsolver, csc, pool, worker_budget, seed);
  auto result = worker.run_epoch(worker_budget);
  return result.effort;
}

} // namespace

void run(HighsMipSolver &mipsolver, size_t max_effort) {
  const auto *model = mipsolver.model_;
  auto *mipdata = mipsolver.mipdata_.get();
  const HighsInt ncol = model->num_col_;
  const HighsInt nrow = model->num_row_;
  if (ncol == 0 || nrow == 0) return;

  const bool minimize = (model->sense_ == ObjSense::kMinimize);

  auto csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_,
                       mipdata->ARvalue_);

  SolutionPool pool(kPoolCapacity, minimize);
  seed_pool(pool, mipsolver);

  size_t effort = pump_worker(mipsolver, csc, pool, max_effort, 42);

  mipdata->heuristic_effort_used += effort;

  for (auto &entry : pool.sorted_entries()) {
    mipdata->trySolution(entry.solution, kSolutionSourceScylla);
  }
}

void run_parallel(HighsMipSolver &mipsolver, size_t max_effort) {
  const auto *model = mipsolver.model_;
  auto *mipdata = mipsolver.mipdata_.get();
  const HighsInt ncol = model->num_col_;
  const HighsInt nrow = model->num_row_;
  if (ncol == 0 || nrow == 0) return;

  const bool minimize = (model->sense_ == ObjSense::kMinimize);
  const auto &integrality = model->integrality_;
  const auto &orig_cost = model->col_cost_;
  const double time_limit = mipsolver.options_mip_->time_limit;

  auto csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_,
                       mipdata->ARvalue_);

  SolutionPool pool(kPoolCapacity, minimize);
  seed_pool(pool, mipsolver);

  // --- Count integers and compute cost scale ---
  HighsInt num_integers = 0;
  double norm_c_sq = 0.0;
  for (HighsInt j = 0; j < ncol; ++j) {
    if (is_integer(integrality, j)) ++num_integers;
    norm_c_sq += orig_cost[j] * orig_cost[j];
  }
  if (num_integers == 0) return;
  double norm_c = std::sqrt(norm_c_sq);
  double cost_scale = (norm_c > 1e-15) ? std::sqrt(num_integers) / norm_c : 1.0;

  // --- Set up single PDLP instance ---
  size_t nnz_lp = mipdata->ARindex_.size();
  if (nnz_lp == 0) return;

  Highs highs;
  highs.setOptionValue("solver", "pdlp");
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("pdlp_scaling", true);
  highs.setOptionValue("pdlp_e_restart_method", 2);

  auto pdlp_iter_cap =
      static_cast<HighsInt>((max_effort >> 2) / nnz_lp);
  if (pdlp_iter_cap < 100) pdlp_iter_cap = 100;
  highs.setOptionValue("pdlp_iteration_limit", pdlp_iter_cap);

  auto lp = pump::build_lp_relaxation(*model, *mipdata);
  highs.passModel(std::move(lp));

  HighsSolution warm_start;

  // --- Parallel FPR config setup ---
  const int M = highs::parallel::num_threads();

  // Pre-compute variable orders sequentially (avoids data races on
  // HighsCliqueTable::cliquePartition).  Only kNumFprConfigs unique
  // configs exist; workers beyond that reuse orders via modulo.
  std::vector<std::vector<HighsInt>> var_orders(kNumFprConfigs);
  for (int w = 0; w < kNumFprConfigs; ++w) {
    std::mt19937 rng(kBaseSeedOffset + static_cast<uint32_t>(w));
    var_orders[w] = compute_var_order(
        mipsolver, kFprConfigs[w].strat.var_strategy, rng, nullptr);
  }

  // --- Pump loop state ---
  double epsilon = pump::kEpsilonInit;
  double alpha_K = 1.0;
  int K = 0;
  int pdlp_stall_count = 0;
  size_t total_effort = 0;
  size_t effort_since_improvement = 0;
  const size_t stale_budget = max_effort >> 2;
  std::mt19937 rng(kBaseSeedOffset);

  std::vector<double> scores(ncol);
  std::vector<double> modified_cost(ncol);
  std::vector<HeuristicResult> results(M);
  std::vector<std::vector<double>> cycle_history;
  cycle_history.reserve(pump::kCycleWindow);

  // --- Main pump loop: serial PDLP + parallel FPR ---
  while (total_effort < max_effort) {
    if (mipsolver.timer_.read() >= time_limit) break;
    if (effort_since_improvement > stale_budget) break;

    ++K;

    highs.setOptionValue("pdlp_optimality_tolerance", epsilon);
    double remaining = time_limit - mipsolver.timer_.read();
    if (remaining <= 0.0) break;
    highs.setOptionValue("time_limit", remaining);

    if (warm_start.value_valid && warm_start.dual_valid) {
      highs.setSolution(warm_start);
    }

    HighsStatus status = highs.run();

    HighsInt pdlp_iters = 0;
    highs.getInfoValue("pdlp_iteration_count", pdlp_iters);
    size_t iter_effort = static_cast<size_t>(pdlp_iters) * nnz_lp;
    total_effort += iter_effort;
    effort_since_improvement += iter_effort;

    if (status == HighsStatus::kError) break;
    if (highs.getModelStatus() == HighsModelStatus::kInfeasible) break;

    if (pdlp_iters == 0) {
      ++pdlp_stall_count;
      if (pdlp_stall_count >= pump::kMaxPdlpStalls) break;
    } else {
      pdlp_stall_count = 0;
    }

    const auto &sol = highs.getSolution();
    if (sol.col_value.empty()) break;

    warm_start.col_value = sol.col_value;
    warm_start.row_dual = sol.row_dual;
    warm_start.value_valid = sol.value_valid;
    warm_start.dual_valid = sol.dual_valid;

    const auto &x_bar = sol.col_value;

    // Check if PDLP solution is already MIP-feasible (fast path)
    {
      bool mip_feasible = true;
      const double feastol =
          mipsolver.options_mip_->mip_feasibility_tolerance;
      for (HighsInt j = 0; j < ncol; ++j) {
        if (!is_integer(integrality, j)) continue;
        if (std::abs(x_bar[j] - std::round(x_bar[j])) > feastol) {
          mip_feasible = false;
          break;
        }
      }
      if (mip_feasible) {
        for (HighsInt i = 0; i < nrow; ++i) {
          double lhs = 0.0;
          for (HighsInt k = mipdata->ARstart_[i];
               k < mipdata->ARstart_[i + 1]; ++k) {
            lhs += mipdata->ARvalue_[k] * x_bar[mipdata->ARindex_[k]];
          }
          if (lhs > model->row_upper_[i] + feastol ||
              lhs < model->row_lower_[i] - feastol) {
            mip_feasible = false;
            break;
          }
        }
      }
      if (mip_feasible) {
        double obj = model->offset_;
        for (HighsInt j = 0; j < ncol; ++j) {
          obj += orig_cost[j] * x_bar[j];
        }
        pool.try_add(obj, x_bar);
        effort_since_improvement = 0;
        continue;
      }
    }

    // Compute fractionality scores for legacy FPR ranking
    for (HighsInt j = 0; j < ncol; ++j) {
      if (!is_integer(integrality, j)) {
        scores[j] = -1.0;
      } else {
        scores[j] = std::abs(x_bar[j] - std::round(x_bar[j]));
      }
    }

    // --- Parallel FPR rounding: try M configs on same x_bar ---
    size_t remaining_budget =
        max_effort > total_effort ? max_effort - total_effort : 0;
    if (remaining_budget == 0) break;
    size_t per_config_budget = remaining_budget / std::max(M, 1);

    // Clear results without deallocating solution vectors.
    for (auto &r : results) {
      r.found_feasible = false;
      r.effort = 0;
      r.solution.clear();
    }

    highs::parallel::for_each(
        0, static_cast<HighsInt>(M),
        [&](HighsInt lo, HighsInt hi) {
          for (HighsInt w = lo; w < hi; ++w) {
            const int ci = static_cast<int>(w) % kNumFprConfigs;

            std::mt19937 w_rng(kBaseSeedOffset +
                               static_cast<uint32_t>(w) * kSeedStride +
                               static_cast<uint32_t>(K));

            FprConfig cfg{};
            cfg.max_effort = per_config_budget;
            cfg.rng_seed_offset =
                kBaseSeedOffset + static_cast<uint32_t>(w) + K;
            cfg.hint = x_bar.data();
            cfg.scores = scores.data();
            cfg.cont_fallback = x_bar.data();
            cfg.csc = &csc;
            cfg.mode = kFprConfigs[ci].mode;
            cfg.strategy = &kFprConfigs[ci].strat;
            cfg.lp_ref = nullptr;
            cfg.precomputed_var_order = var_orders[ci].data();
            cfg.precomputed_var_order_size =
                static_cast<HighsInt>(var_orders[ci].size());

            results[w] = fpr_attempt(mipsolver, cfg, w_rng, 0, nullptr);
          }
        },
        1);

    // Aggregate FPR effort and pick best feasible result
    int best_idx = -1;
    double best_obj = minimize ? std::numeric_limits<double>::infinity()
                               : -std::numeric_limits<double>::infinity();
    size_t fpr_effort = 0;

    for (int w = 0; w < M; ++w) {
      fpr_effort += results[w].effort;
      if (results[w].found_feasible && !results[w].solution.empty()) {
        pool.try_add(results[w].objective, results[w].solution);
        bool is_better = minimize ? (results[w].objective < best_obj)
                                  : (results[w].objective > best_obj);
        if (is_better) {
          best_obj = results[w].objective;
          best_idx = w;
        }
      }
    }

    total_effort += fpr_effort;
    effort_since_improvement += fpr_effort;

    if (best_idx >= 0) {
      effort_since_improvement = 0;
    }

    // Use best x_hat for cycling detection and objective update.
    // Fall back to first non-empty result if no feasible solution found.
    std::vector<double> *x_hat = nullptr;
    if (best_idx >= 0) {
      x_hat = &results[best_idx].solution;
    } else {
      for (int w = 0; w < M; ++w) {
        if (!results[w].solution.empty()) {
          x_hat = &results[w].solution;
          break;
        }
      }
    }
    if (x_hat == nullptr) continue;

    // Cycling detection + perturbation
    if (pump::detect_cycling(cycle_history, *x_hat, integrality, ncol)) {
      pump::perturb(*x_hat, *model, rng);
    }
    if (static_cast<int>(cycle_history.size()) < pump::kCycleWindow) {
      cycle_history.push_back(*x_hat);
    } else {
      cycle_history[(K - 1) % pump::kCycleWindow] = *x_hat;
    }

    // Objective update
    alpha_K *= pump::kAlpha;
    pump::compute_pump_objective(orig_cost, *x_hat, x_bar, integrality,
                           model->col_lower_, model->col_upper_, alpha_K,
                           cost_scale, ncol, modified_cost);
    highs.changeColsCost(0, ncol - 1, modified_cost.data());

    epsilon = std::max(pump::kBeta * epsilon, pump::kEpsilonFloor);
  }

  mipdata->heuristic_effort_used += total_effort;

  for (auto &entry : pool.sorted_entries()) {
    mipdata->trySolution(entry.solution, kSolutionSourceScylla);
  }
}

} // namespace scylla
