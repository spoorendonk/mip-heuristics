#include "scylla.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "Highs.h"
#include "fpr_core.h"
#include "heuristic_common.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
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

} // namespace

void run(HighsMipSolver &mipsolver, size_t max_effort) {
  const auto *model = mipsolver.model_;
  auto *mipdata = mipsolver.mipdata_.get();
  const HighsInt ncol = model->num_col_;
  const HighsInt nrow = model->num_row_;
  if (ncol == 0 || nrow == 0) {
    return;
  }

  const bool minimize = (model->sense_ == ObjSense::kMinimize);
  const auto &integrality = model->integrality_;
  const auto &orig_cost = model->col_cost_;

  // Pre-compute cost scaling: √|I| / ‖c‖₂
  HighsInt num_integers = 0;
  double norm_c_sq = 0.0;
  for (HighsInt j = 0; j < ncol; ++j) {
    if (is_integer(integrality, j)) ++num_integers;
    norm_c_sq += orig_cost[j] * orig_cost[j];
  }
  if (num_integers == 0) return;

  double norm_c = std::sqrt(norm_c_sq);
  // When costs are all zero (feasibility problem), scaling is irrelevant;
  // the proximity term dominates the pump objective.
  double cost_scale = (norm_c > 1e-15) ? std::sqrt(num_integers) / norm_c : 1.0;

  // Build LP relaxation and configure PDLP solver
  auto lp = build_lp_relaxation(*model, *mipdata);
  Highs highs;
  highs.setOptionValue("solver", "pdlp");
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("pdlp_scaling", true);
  highs.setOptionValue("pdlp_e_restart_method", 2); // CPU restart
  // Cap each PDLP solve so a single solve can't dominate the budget.
  // With effort = iters * nnz, limit each solve to max_effort/4 of effort.
  size_t nnz_lp = mipdata->ARindex_.size();
  HighsInt pdlp_iter_cap =
      (nnz_lp > 0) ? static_cast<HighsInt>((max_effort >> 2) / nnz_lp) : 10000;
  if (pdlp_iter_cap < 100) pdlp_iter_cap = 100;
  highs.setOptionValue("pdlp_iteration_limit", pdlp_iter_cap);
  highs.passModel(std::move(lp));

  // Build CSC for fpr_attempt
  auto csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_,
                       mipdata->ARvalue_);

  // Algorithm state
  double epsilon = kEpsilonInit;
  double alpha_K = 1.0; // updated after each solve; first solve uses unmodified cost
  const double time_limit = mipsolver.options_mip_->time_limit;
  size_t total_effort = 0;
  size_t effort_since_improvement = 0;
  const size_t stale_budget = max_effort >> 2;

  SolutionPool pool(kPoolCapacity, minimize);
  seed_pool(pool, mipsolver);

  // Cycling history (circular buffer of last kCycleWindow rounded solutions)
  std::vector<std::vector<double>> cycle_history;
  cycle_history.reserve(kCycleWindow);

  std::vector<double> scores(ncol);
  std::vector<double> modified_cost(ncol);
  std::mt19937 rng(42);

  // Outer feasibility pump loop (Algorithm 1.1)
  int pdlp_stall_count = 0;
  for (int K = 1; total_effort < max_effort; ++K) {
    if (mipsolver.timer_.read() >= time_limit) break;
    if (mipdata->terminatorTerminated()) break;
    if (effort_since_improvement > stale_budget) break;

    // Configure PDLP tolerances — progressive refinement (§2.2)
    highs.setOptionValue("pdlp_optimality_tolerance", epsilon);
    double remaining = time_limit - mipsolver.timer_.read();
    if (remaining <= 0.0) break;
    highs.setOptionValue("time_limit", remaining);

    // Solve LP approximately via PDLP
    HighsStatus status = highs.run();

    // Track PDLP effort: iterations × nnz (each iteration does two mat-vec products)
    HighsInt pdlp_iters = 0;
    highs.getInfoValue("pdlp_iteration_count", pdlp_iters);
    size_t iter_effort = static_cast<size_t>(pdlp_iters) * nnz_lp;
    total_effort += iter_effort;
    effort_since_improvement += iter_effort;

    if (status == HighsStatus::kError) break;
    if (highs.getModelStatus() == HighsModelStatus::kInfeasible) break;

    // Stall detection: if PDLP returns 0 iterations, the modified
    // objective isn't changing the LP solution.  Break after consecutive
    // stalls to avoid spinning.
    if (pdlp_iters == 0) {
      ++pdlp_stall_count;
      if (pdlp_stall_count >= kMaxPdlpStalls) break;
    } else {
      pdlp_stall_count = 0;
    }
    const auto &sol = highs.getSolution();
    if (sol.col_value.empty()) break;
    const auto &x_bar = sol.col_value;

    // Line 11: check if PDLP solution is already MIP-feasible (fast path)
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
      // PDLP solutions are approximate — verify row feasibility too
      if (mip_feasible) {
        for (HighsInt i = 0; i < nrow; ++i) {
          double lhs = 0.0;
          for (HighsInt k = mipdata->ARstart_[i]; k < mipdata->ARstart_[i + 1];
               ++k) {
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
        continue; // skip rounding — already integer-feasible
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

    // Line 12: fix-and-propagate to round PDLP solution
    // Scylla uses kDiveprop (default): propagation on, repair on, no backtrack.
    // Deliberate — the pump runs many iterations so fast single-pass rounding
    // is appropriate.  Mexi et al.'s "FixAndPropagate" maps to this mode.
    FprConfig cfg{};
    cfg.max_effort = max_effort - std::min(max_effort, total_effort);
    cfg.rng_seed_offset = 42 + K;
    cfg.hint = x_bar.data();
    cfg.scores = scores.data();
    cfg.cont_fallback = x_bar.data();
    cfg.csc = &csc;

    // attempt_idx=0 ensures fpr_attempt uses the PDLP hint as starting point
    auto result = fpr_attempt(mipsolver, cfg, rng, 0, nullptr);
    total_effort += result.effort;
    effort_since_improvement += result.effort;

    // Line 13: collect feasible solutions
    if (result.found_feasible && !result.solution.empty()) {
      pool.try_add(result.objective, result.solution);
      effort_since_improvement = 0;
    }

    // Use the rounded point for objective update (even if infeasible).
    // If fpr_attempt returned no solution, skip objective update but continue
    // the pump — the next PDLP solve may produce a better starting point.
    auto &x_hat = result.solution;
    if (x_hat.empty()) continue;

    // Line 14: cycling detection + perturbation
    if (detect_cycling(cycle_history, x_hat, integrality, ncol)) {
      perturb(x_hat, *model, rng);
    }
    if (static_cast<int>(cycle_history.size()) < kCycleWindow) {
      cycle_history.push_back(x_hat);
    } else {
      cycle_history[(K - 1) % kCycleWindow] = x_hat;
    }

    // Line 15: objective update
    //   ĉ = α^K · (√|I|/‖c‖) · c + (1-α^K) · Δ(·, x̂)
    alpha_K = std::pow(kAlpha, K);
    compute_pump_objective(orig_cost, x_hat, x_bar, integrality,
                           model->col_lower_, model->col_upper_, alpha_K,
                           cost_scale, ncol, modified_cost);
    highs.changeColsCost(0, ncol - 1, modified_cost.data());

    // Progressive tolerance refinement (§2.2)
    epsilon = std::max(kBeta * epsilon, kEpsilonFloor);
  }

  mipdata->heuristic_effort_used += total_effort;

  // Submit best solutions to solver
  for (auto &entry : pool.sorted_entries()) {
    mipdata->trySolution(entry.solution, kSolutionSourceScylla);
  }
}

} // namespace scylla
