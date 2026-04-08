#include "scylla.h"

#include <algorithm>
#include <cmath>
#include <memory>
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

// Encapsulates per-worker pump chain state so that execution can be
// paused at epoch boundaries and resumed.  Each worker owns its own
// PDLP solver instance, warm-start vectors, cycling history, and RNG.
class PumpWorker {
 public:
  PumpWorker(HighsMipSolver &mipsolver, const CscMatrix &csc,
             SolutionPool &pool, size_t total_budget, uint32_t seed)
      : mipsolver_(mipsolver),
        csc_(csc),
        pool_(pool),
        total_budget_(total_budget),
        seed_(seed),
        rng_(seed) {
    const auto *model = mipsolver_.model_;
    auto *mipdata = mipsolver_.mipdata_.get();
    ncol_ = model->num_col_;
    nrow_ = model->num_row_;

    // Pre-compute cost scaling: sqrt(|I|) / ||c||_2
    const auto &orig_cost = model->col_cost_;
    const auto &integrality = model->integrality_;
    HighsInt num_integers = 0;
    double norm_c_sq = 0.0;
    for (HighsInt j = 0; j < ncol_; ++j) {
      if (is_integer(integrality, j)) ++num_integers;
      norm_c_sq += orig_cost[j] * orig_cost[j];
    }
    if (num_integers == 0 || ncol_ == 0) {
      finished_ = true;
      return;
    }

    double norm_c = std::sqrt(norm_c_sq);
    cost_scale_ = (norm_c > 1e-15) ? std::sqrt(num_integers) / norm_c : 1.0;

    // Build LP relaxation and configure PDLP solver
    auto lp = build_lp_relaxation(*model, *mipdata);
    highs_.setOptionValue("solver", "pdlp");
    highs_.setOptionValue("output_flag", false);
    highs_.setOptionValue("pdlp_scaling", true);
    highs_.setOptionValue("pdlp_e_restart_method", 2);
    nnz_lp_ = mipdata->ARindex_.size();
    if (nnz_lp_ == 0) {
      finished_ = true;
      return;
    }
    HighsInt pdlp_iter_cap =
        (nnz_lp_ > 0)
            ? static_cast<HighsInt>((total_budget_ >> 2) / nnz_lp_)
            : 10000;
    if (pdlp_iter_cap < 100) pdlp_iter_cap = 100;
    highs_.setOptionValue("pdlp_iteration_limit", pdlp_iter_cap);
    highs_.passModel(std::move(lp));

    stale_budget_ = total_budget_ >> 2;
    scores_.resize(ncol_);
    modified_cost_.resize(ncol_);
    cycle_history_.reserve(kCycleWindow);
  }

  // Run pump chain iterations until epoch_budget effort is consumed.
  // Returns effort spent in this epoch.  Sets finished_ when the
  // worker cannot make further progress (stall, infeasibility, etc.).
  struct EpochResult {
    size_t effort = 0;
    bool found_improvement = false;
  };

  EpochResult run_epoch(size_t epoch_budget) {
    if (finished_) return {};

    const auto *model = mipsolver_.model_;
    auto *mipdata = mipsolver_.mipdata_.get();
    const auto &integrality = model->integrality_;
    const auto &orig_cost = model->col_cost_;
    const double time_limit = mipsolver_.options_mip_->time_limit;

    EpochResult epoch{};

    while (epoch.effort < epoch_budget && total_effort_ < total_budget_) {
      if (mipsolver_.timer_.read() >= time_limit) {
        finished_ = true;
        break;
      }
      if (effort_since_improvement_ > stale_budget_) {
        finished_ = true;
        break;
      }

      ++K_;

      // Configure PDLP tolerances -- progressive refinement
      highs_.setOptionValue("pdlp_optimality_tolerance", epsilon_);
      double remaining = time_limit - mipsolver_.timer_.read();
      if (remaining <= 0.0) {
        finished_ = true;
        break;
      }
      highs_.setOptionValue("time_limit", remaining);

      // Warm-start PDLP from the previous iteration's primal-dual iterate.
      if (warm_start_.value_valid && warm_start_.dual_valid) {
        highs_.setSolution(warm_start_);
      }

      HighsStatus status = highs_.run();

      HighsInt pdlp_iters = 0;
      highs_.getInfoValue("pdlp_iteration_count", pdlp_iters);
      size_t iter_effort = static_cast<size_t>(pdlp_iters) * nnz_lp_;
      total_effort_ += iter_effort;
      effort_since_improvement_ += iter_effort;
      epoch.effort += iter_effort;

      if (status == HighsStatus::kError) {
        finished_ = true;
        break;
      }
      if (highs_.getModelStatus() == HighsModelStatus::kInfeasible) {
        finished_ = true;
        break;
      }

      if (pdlp_iters == 0) {
        ++pdlp_stall_count_;
        if (pdlp_stall_count_ >= kMaxPdlpStalls) {
          finished_ = true;
          break;
        }
      } else {
        pdlp_stall_count_ = 0;
      }
      const auto &sol = highs_.getSolution();
      if (sol.col_value.empty()) {
        finished_ = true;
        break;
      }

      // Capture primal-dual iterate for warm-starting next PDLP solve.
      warm_start_.col_value = sol.col_value;
      warm_start_.row_dual = sol.row_dual;
      warm_start_.value_valid = sol.value_valid;
      warm_start_.dual_valid = sol.dual_valid;

      const auto &x_bar = sol.col_value;

      // Check if PDLP solution is already MIP-feasible (fast path)
      {
        bool mip_feasible = true;
        const double feastol =
            mipsolver_.options_mip_->mip_feasibility_tolerance;
        for (HighsInt j = 0; j < ncol_; ++j) {
          if (!is_integer(integrality, j)) continue;
          if (std::abs(x_bar[j] - std::round(x_bar[j])) > feastol) {
            mip_feasible = false;
            break;
          }
        }
        if (mip_feasible) {
          for (HighsInt i = 0; i < nrow_; ++i) {
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
          for (HighsInt j = 0; j < ncol_; ++j) {
            obj += orig_cost[j] * x_bar[j];
          }
          pool_.try_add(obj, x_bar);
          effort_since_improvement_ = 0;
          epoch.found_improvement = true;
          continue;
        }
      }

      // Compute fractionality scores for fix-and-propagate ranking
      for (HighsInt j = 0; j < ncol_; ++j) {
        if (!is_integer(integrality, j)) {
          scores_[j] = -1.0;
        } else {
          scores_[j] = std::abs(x_bar[j] - std::round(x_bar[j]));
        }
      }

      // Fix-and-propagate to round PDLP solution
      FprConfig cfg{};
      cfg.max_effort =
          std::min(epoch_budget - std::min(epoch_budget, epoch.effort),
                   total_budget_ - std::min(total_budget_, total_effort_));
      cfg.rng_seed_offset = seed_ + K_;
      cfg.hint = x_bar.data();
      cfg.scores = scores_.data();
      cfg.cont_fallback = x_bar.data();
      cfg.csc = &csc_;

      auto result = fpr_attempt(mipsolver_, cfg, rng_, 0, nullptr);
      total_effort_ += result.effort;
      effort_since_improvement_ += result.effort;
      epoch.effort += result.effort;

      if (result.found_feasible && !result.solution.empty()) {
        pool_.try_add(result.objective, result.solution);
        effort_since_improvement_ = 0;
        epoch.found_improvement = true;
      }

      auto &x_hat = result.solution;
      if (x_hat.empty()) continue;

      // Cycling detection + perturbation
      if (detect_cycling(cycle_history_, x_hat, integrality, ncol_)) {
        perturb(x_hat, *model, rng_);
      }
      if (static_cast<int>(cycle_history_.size()) < kCycleWindow) {
        cycle_history_.push_back(x_hat);
      } else {
        cycle_history_[(K_ - 1) % kCycleWindow] = x_hat;
      }

      // Objective update
      alpha_K_ *= kAlpha;
      compute_pump_objective(orig_cost, x_hat, x_bar, integrality,
                             model->col_lower_, model->col_upper_, alpha_K_,
                             cost_scale_, ncol_, modified_cost_);
      highs_.changeColsCost(0, ncol_ - 1, modified_cost_.data());

      epsilon_ = std::max(kBeta * epsilon_, kEpsilonFloor);
    }

    return epoch;
  }

  bool finished() const { return finished_; }
  size_t total_effort() const { return total_effort_; }

  // Reset the improvement staleness counter (called at epoch boundary
  // when another worker found an improvement).
  void reset_staleness() { effort_since_improvement_ = 0; }

 private:
  HighsMipSolver &mipsolver_;
  const CscMatrix &csc_;
  SolutionPool &pool_;
  const size_t total_budget_;
  const uint32_t seed_;

  HighsInt ncol_ = 0;
  HighsInt nrow_ = 0;
  double cost_scale_ = 1.0;
  size_t nnz_lp_ = 0;
  size_t stale_budget_ = 0;

  Highs highs_;
  HighsSolution warm_start_;
  int pdlp_stall_count_ = 0;

  double epsilon_ = kEpsilonInit;
  double alpha_K_ = 1.0;
  int K_ = 0;

  size_t total_effort_ = 0;
  size_t effort_since_improvement_ = 0;
  bool finished_ = false;

  std::vector<std::vector<double>> cycle_history_;
  std::vector<double> scores_;
  std::vector<double> modified_cost_;
  std::mt19937 rng_;
};

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
  const int N = highs::parallel::num_threads();

  auto csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_,
                       mipdata->ARvalue_);

  SolutionPool pool(kPoolCapacity, minimize);
  seed_pool(pool, mipsolver);

  // Per-worker total budget and epoch budget.
  // Each epoch gives every worker a small slice; ~10 epochs per worker
  // provides meaningful synchronization without excessive overhead.
  const size_t worker_budget = max_effort / static_cast<size_t>(N);
  constexpr int kEpochsPerWorker = 10;
  const size_t epoch_budget =
      std::max<size_t>(worker_budget / kEpochsPerWorker, 1);

  uint32_t base_seed =
      static_cast<uint32_t>(mipdata->numImprovingSols + 42);

  // Create per-worker PumpWorker instances (sequential — deterministic
  // initialization order with deterministic seeds).
  std::vector<std::unique_ptr<PumpWorker>> workers;
  workers.reserve(N);
  for (int w = 0; w < N; ++w) {
    uint32_t seed = base_seed + static_cast<uint32_t>(w) * 997;
    workers.push_back(
        std::make_unique<PumpWorker>(mipsolver, csc, pool, worker_budget, seed));
  }

  // Pre-allocate per-worker epoch results outside the loop.
  std::vector<PumpWorker::EpochResult> epoch_results(N);

  const size_t stale_budget = max_effort >> 2;
  size_t total_effort = 0;
  size_t effort_since_improvement = 0;

  const double time_limit = mipsolver.options_mip_->time_limit;

  while (total_effort < max_effort) {
    // Pre-epoch checks (sequential, single-thread — safe for
    // terminatorTerminated which is not thread-safe).
    if (mipdata->terminatorTerminated() ||
        mipsolver.timer_.read() >= time_limit) {
      break;
    }
    if (effort_since_improvement > stale_budget) {
      break;
    }

    // Check if all workers are finished.
    bool all_finished = true;
    for (int w = 0; w < N; ++w) {
      if (!workers[w]->finished()) {
        all_finished = false;
        break;
      }
    }
    if (all_finished) break;

    // Epoch: all workers run a pump chain slice in parallel.
    highs::parallel::for_each(
        0, static_cast<HighsInt>(N),
        [&](HighsInt lo, HighsInt hi) {
          for (HighsInt w = lo; w < hi; ++w) {
            epoch_results[w] = workers[w]->run_epoch(epoch_budget);
          }
        },
        1);

    // Post-epoch: merge results in deterministic worker order.
    bool any_improved = false;
    size_t epoch_effort = 0;
    for (int w = 0; w < N; ++w) {
      epoch_effort += epoch_results[w].effort;
      if (epoch_results[w].found_improvement) any_improved = true;
    }
    total_effort += epoch_effort;

    if (any_improved) {
      effort_since_improvement = 0;
      for (int w = 0; w < N; ++w) {
        workers[w]->reset_staleness();
      }
    } else {
      effort_since_improvement += epoch_effort;
    }
  }

  mipdata->heuristic_effort_used += total_effort;

  for (auto &entry : pool.sorted_entries()) {
    mipdata->trySolution(entry.solution, kSolutionSourceScylla);
  }
}

} // namespace scylla
