#include "scylla.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <random>
#include <vector>

#include "Highs.h"
#include "fpr_core.h"
#include "heuristic_common.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "parallel/HighsParallel.h"
#include "solution_pool.h"

namespace scylla {

namespace {

constexpr int kPoolCapacity = 10;

// Algorithm 1.1 parameters (Mexi et al. 2023, §2)
constexpr double kAlpha = 0.9;
constexpr double kEpsilonInit = 0.01;
constexpr double kBeta = 0.98;
constexpr double kEpsilonFloor = 1e-8;
constexpr int kCycleWindow = 3;
constexpr double kPerturbFraction = 0.2;
constexpr double kCycleTol = 0.5; // integer values differ by >= 1.0
constexpr int kMaxPdlpStalls = 3;

void seed_pool(SolutionPool &pool, const HighsMipSolver &mipsolver) {
  const auto *model = mipsolver.model_;
  auto *mipdata = mipsolver.mipdata_.get();
  if (mipdata->incumbent.empty()) {
    return;
  }
  const HighsInt ncol = model->num_col_;
  double obj = model->offset_;
  for (HighsInt j = 0; j < ncol; ++j) {
    obj += model->col_cost_[j] * mipdata->incumbent[j];
  }
  pool.try_add(obj, mipdata->incumbent);
}

// Build LP relaxation from the presolved MIP model (strip integrality).
// Uses row-wise storage from mipdata since that's readily available.
HighsLp build_lp_relaxation(const HighsLp &model,
                            const HighsMipSolverData &mipdata) {
  HighsLp lp;
  lp.num_col_ = model.num_col_;
  lp.num_row_ = model.num_row_;
  lp.col_cost_ = model.col_cost_;
  lp.col_lower_ = model.col_lower_;
  lp.col_upper_ = model.col_upper_;
  lp.row_lower_ = model.row_lower_;
  lp.row_upper_ = model.row_upper_;
  lp.sense_ = model.sense_;
  lp.offset_ = model.offset_;
  // Row-wise sparse matrix from mipdata
  lp.a_matrix_.format_ = MatrixFormat::kRowwise;
  lp.a_matrix_.num_col_ = model.num_col_;
  lp.a_matrix_.num_row_ = model.num_row_;
  lp.a_matrix_.start_ = mipdata.ARstart_;
  lp.a_matrix_.index_ = mipdata.ARindex_;
  lp.a_matrix_.value_ = mipdata.ARvalue_;
  // No integrality — this is an LP relaxation
  return lp;
}

// Compute modified objective (Algorithm 1.1, line 15).
//   ĉ_j = α^K · (√|I|/‖c‖) · c_j + (1-α^K) · Δ_j    for j ∈ I
//   ĉ_j = α^K · (√|I|/‖c‖) · c_j                      for j ∉ I
// where Δ_j = 1 - 2·x̂_j for binary, sign(x̄_j - x̂_j) for general integer.
void compute_pump_objective(
    const std::vector<double> &orig_cost,
    const std::vector<double> &x_rounded,
    const std::vector<double> &x_lp,
    const std::vector<HighsVarType> &integrality,
    const std::vector<double> &col_lb, const std::vector<double> &col_ub,
    double alpha_K, double cost_scale, HighsInt ncol,
    std::vector<double> &modified_cost) {
  for (HighsInt j = 0; j < ncol; ++j) {
    double scaled_cost = alpha_K * cost_scale * orig_cost[j];
    if (is_integer(integrality, j)) {
      // L1 proximity subgradient
      double delta;
      if (col_lb[j] == 0.0 && col_ub[j] == 1.0) {
        // Binary: standard FP trick
        delta = 1.0 - 2.0 * x_rounded[j];
      } else {
        // General integer: subgradient direction from LP solution
        double diff = x_lp[j] - x_rounded[j];
        delta = (diff >= 0.0) ? 1.0 : -1.0;
      }
      modified_cost[j] = scaled_cost + (1.0 - alpha_K) * delta;
    } else {
      modified_cost[j] = scaled_cost;
    }
  }
}

// Detect cycling: check if x_rounded matches any solution in history.
bool detect_cycling(
    const std::vector<std::vector<double>> &history,
    const std::vector<double> &x_rounded,
    const std::vector<HighsVarType> &integrality, HighsInt ncol) {
  for (const auto &prev : history) {
    if (prev.empty()) continue;
    bool match = true;
    for (HighsInt j = 0; j < ncol; ++j) {
      if (!is_integer(integrality, j)) continue;
      if (std::abs(x_rounded[j] - prev[j]) > kCycleTol) {
        match = false;
        break;
      }
    }
    if (match) return true;
  }
  return false;
}

// Perturb a rounded solution to break cycling (Algorithm 1.1, line 14).
void perturb(std::vector<double> &x, const HighsLp &model,
             std::mt19937 &rng) {
  const HighsInt ncol = model.num_col_;
  const auto &integrality = model.integrality_;
  const auto &lb = model.col_lower_;
  const auto &ub = model.col_upper_;

  for (HighsInt j = 0; j < ncol; ++j) {
    if (!is_integer(integrality, j)) continue;
    if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) > kPerturbFraction) {
      continue;
    }
    double lo = std::ceil(lb[j]);
    double hi = std::floor(ub[j]);
    if (hi <= lo) continue;
    // Pick a different integer value uniformly via modular shift
    double current = std::round(x[j]);
    auto irange = static_cast<int64_t>(hi - lo);
    int64_t shift = std::uniform_int_distribution<int64_t>(1, irange)(rng);
    x[j] = lo + std::fmod(current - lo + shift, irange + 1.0);
  }
}

// Run a single pump chain. Each worker has its own PDLP solver, K counter,
// cycling history, and RNG. Solutions are collected in the shared pool.
// Returns total effort consumed by this worker.
size_t pump_worker(HighsMipSolver &mipsolver, const CscMatrix &csc,
                   SolutionPool &pool, size_t worker_budget,
                   uint32_t seed,
                   // Parallel mode: shared atomics for global effort tracking.
                   // When null, runs in sequential (single-worker) mode.
                   std::atomic<size_t> *global_effort = nullptr,
                   std::atomic<bool> *stop_flag = nullptr) {
  const auto *model = mipsolver.model_;
  auto *mipdata = mipsolver.mipdata_.get();
  const HighsInt ncol = model->num_col_;
  const HighsInt nrow = model->num_row_;
  const auto &integrality = model->integrality_;
  const auto &orig_cost = model->col_cost_;

  // Pre-compute cost scaling: sqrt(|I|) / ||c||_2
  HighsInt num_integers = 0;
  double norm_c_sq = 0.0;
  for (HighsInt j = 0; j < ncol; ++j) {
    if (is_integer(integrality, j)) ++num_integers;
    norm_c_sq += orig_cost[j] * orig_cost[j];
  }
  if (num_integers == 0) return 0;

  double norm_c = std::sqrt(norm_c_sq);
  double cost_scale =
      (norm_c > 1e-15) ? std::sqrt(num_integers) / norm_c : 1.0;

  // Build LP relaxation and configure PDLP solver
  auto lp = build_lp_relaxation(*model, *mipdata);
  Highs highs;
  highs.setOptionValue("solver", "pdlp");
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("pdlp_scaling", true);
  highs.setOptionValue("pdlp_e_restart_method", 2);
  size_t nnz_lp = mipdata->ARindex_.size();
  HighsInt pdlp_iter_cap =
      (nnz_lp > 0)
          ? static_cast<HighsInt>((worker_budget >> 2) / nnz_lp)
          : 10000;
  if (pdlp_iter_cap < 100) pdlp_iter_cap = 100;
  highs.setOptionValue("pdlp_iteration_limit", pdlp_iter_cap);
  highs.passModel(std::move(lp));

  // Algorithm state
  double epsilon = kEpsilonInit;
  double alpha_K = 1.0;
  const double time_limit = mipsolver.options_mip_->time_limit;
  size_t total_effort = 0;
  size_t effort_since_improvement = 0;
  const size_t stale_budget = worker_budget >> 2;

  std::vector<std::vector<double>> cycle_history;
  cycle_history.reserve(kCycleWindow);

  std::vector<double> scores(ncol);
  std::vector<double> modified_cost(ncol);
  std::mt19937 rng(seed);

  auto should_stop = [&]() {
    if (stop_flag && stop_flag->load(std::memory_order_relaxed)) return true;
    if (mipsolver.timer_.read() >= time_limit) return true;
    return false;
  };

  // Warm-start state: store the previous PDLP primal-dual iterate so the
  // next solve starts from a nearby point (Mexi et al. 2023, §2.1).
  // Between pump iterations only the objective changes — the constraint
  // matrix is identical — so the previous iterate is a strong warm-start.
  HighsSolution warm_start;
  int pdlp_stall_count = 0;
  for (int K = 1; total_effort < worker_budget; ++K) {
    if (should_stop()) break;
    if (mipdata->terminatorTerminated()) break;
    if (effort_since_improvement > stale_budget) break;

    // Configure PDLP tolerances -- progressive refinement
    highs.setOptionValue("pdlp_optimality_tolerance", epsilon);
    double remaining = time_limit - mipsolver.timer_.read();
    if (remaining <= 0.0) break;
    highs.setOptionValue("time_limit", remaining);

    // Warm-start PDLP from the previous iteration's primal-dual iterate.
    if (warm_start.value_valid && warm_start.dual_valid) {
      highs.setSolution(warm_start);
    }

    HighsStatus status = highs.run();

    HighsInt pdlp_iters = 0;
    highs.getInfoValue("pdlp_iteration_count", pdlp_iters);
    size_t iter_effort = static_cast<size_t>(pdlp_iters) * nnz_lp;
    total_effort += iter_effort;
    effort_since_improvement += iter_effort;
    if (global_effort) {
      global_effort->fetch_add(iter_effort, std::memory_order_relaxed);
    }

    if (status == HighsStatus::kError) break;
    if (highs.getModelStatus() == HighsModelStatus::kInfeasible) break;

    if (pdlp_iters == 0) {
      ++pdlp_stall_count;
      if (pdlp_stall_count >= kMaxPdlpStalls) break;
    } else {
      pdlp_stall_count = 0;
    }
    const auto &sol = highs.getSolution();
    if (sol.col_value.empty()) break;

    // Capture the primal-dual iterate for warm-starting the next PDLP solve.
    // We need col_value (primal) and row_dual (dual); setSolution() computes
    // the derived row_value and col_dual from these two.
    warm_start.col_value = sol.col_value;
    warm_start.row_dual = sol.row_dual;
    warm_start.value_valid = sol.value_valid;
    warm_start.dual_valid = sol.dual_valid;

    const auto &x_bar = sol.col_value;

    // Check if PDLP solution is already MIP-feasible (fast path)
    {
      bool mip_feasible = true;
      const double feastol = mipsolver.options_mip_->mip_feasibility_tolerance;
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

    // Compute fractionality scores for fix-and-propagate ranking
    for (HighsInt j = 0; j < ncol; ++j) {
      if (!is_integer(integrality, j)) {
        scores[j] = -1.0;
      } else {
        scores[j] = std::abs(x_bar[j] - std::round(x_bar[j]));
      }
    }

    // Fix-and-propagate to round PDLP solution
    FprConfig cfg{};
    cfg.max_effort = worker_budget - std::min(worker_budget, total_effort);
    cfg.rng_seed_offset = seed + K;
    cfg.hint = x_bar.data();
    cfg.scores = scores.data();
    cfg.cont_fallback = x_bar.data();
    cfg.csc = &csc;

    auto result = fpr_attempt(mipsolver, cfg, rng, 0, nullptr);
    total_effort += result.effort;
    effort_since_improvement += result.effort;
    if (global_effort) {
      global_effort->fetch_add(result.effort, std::memory_order_relaxed);
    }

    if (result.found_feasible && !result.solution.empty()) {
      pool.try_add(result.objective, result.solution);
      effort_since_improvement = 0;
    }

    auto &x_hat = result.solution;
    if (x_hat.empty()) continue;

    // Cycling detection + perturbation
    if (detect_cycling(cycle_history, x_hat, integrality, ncol)) {
      perturb(x_hat, *model, rng);
    }
    if (static_cast<int>(cycle_history.size()) < kCycleWindow) {
      cycle_history.push_back(x_hat);
    } else {
      cycle_history[(K - 1) % kCycleWindow] = x_hat;
    }

    // Objective update
    alpha_K = std::pow(kAlpha, K);
    compute_pump_objective(orig_cost, x_hat, x_bar, integrality,
                           model->col_lower_, model->col_upper_, alpha_K,
                           cost_scale, ncol, modified_cost);
    highs.changeColsCost(0, ncol - 1, modified_cost.data());

    epsilon = std::max(kBeta * epsilon, kEpsilonFloor);
  }

  return total_effort;
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
  const int N = highs::parallel::num_threads();

  auto csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_,
                       mipdata->ARvalue_);

  SolutionPool pool(kPoolCapacity, minimize);
  seed_pool(pool, mipsolver);

  const size_t worker_budget = max_effort / static_cast<size_t>(N);
  std::atomic<size_t> global_effort{0};

  uint32_t base_seed =
      static_cast<uint32_t>(mipdata->numImprovingSols + 42);

  highs::parallel::for_each(
      0, static_cast<HighsInt>(N),
      [&](HighsInt lo, HighsInt hi) {
        for (HighsInt w = lo; w < hi; ++w) {
          uint32_t seed = base_seed + static_cast<uint32_t>(w) * 997;
          pump_worker(mipsolver, csc, pool, worker_budget, seed,
                      &global_effort);
        }
      },
      1);

  mipdata->heuristic_effort_used +=
      global_effort.load(std::memory_order_relaxed);

  for (auto &entry : pool.sorted_entries()) {
    mipdata->trySolution(entry.solution, kSolutionSourceScylla);
  }
}

} // namespace scylla
