#include "scylla_worker.h"

#include "contested_pdlp.h"
#include "fpr_core.h"
#include "fpr_strategies.h"
#include "heuristic_common.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "pump_common.h"
#include "solution_pool.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

// Workers 0..kNumFprConfigs-1 cover every FPR config exactly once
// (deterministic round-robin, preserving strategy diversity when
// N >= kNumFprConfigs).  Additional workers draw a pseudo-random
// config from their own seed, avoiding the "16 workers, 4 copies of
// each config" pathology while keeping assignment deterministic per
// (seed, worker_idx).
int select_fpr_config(int worker_idx, uint32_t seed) {
    if (worker_idx >= 0 && worker_idx < kNumFprConfigs) {
        return worker_idx;
    }
    if (worker_idx < 0) {
        return ((worker_idx % kNumFprConfigs) + kNumFprConfigs) % kNumFprConfigs;
    }
    std::mt19937 cfg_rng(seed);
    return static_cast<int>(cfg_rng() % static_cast<uint32_t>(kNumFprConfigs));
}

}  // namespace

ScyllaWorker::ScyllaWorker(HighsMipSolver &mipsolver, ContestedPdlp &pdlp, const CscMatrix &csc,
                           SolutionPool &pool, size_t total_budget, uint32_t seed, int worker_idx,
                           int num_workers)
    : mipsolver_(mipsolver),
      pdlp_(pdlp),
      csc_(csc),
      pool_(pool),
      total_budget_(total_budget),
      num_workers_(std::max(num_workers, 1)),
      epsilon_(pump::kEpsilonInit),
      rng_(seed),
      fpr_config_index_(select_fpr_config(worker_idx, seed)) {
    if (!pdlp_.initialized()) {
        finished_ = true;
        return;
    }
    const auto *model = mipsolver_.model_;
    ncol_ = model->num_col_;
    nrow_ = model->num_row_;

    const auto &orig_cost = model->col_cost_;
    const auto &integrality = model->integrality_;
    HighsInt num_integers = 0;
    double norm_c_sq = 0.0;
    for (HighsInt j = 0; j < ncol_; ++j) {
        if (is_integer(integrality, j)) {
            ++num_integers;
        }
        norm_c_sq += orig_cost[j] * orig_cost[j];
    }
    if (num_integers == 0 || ncol_ == 0) {
        finished_ = true;
        return;
    }

    double norm_c = std::sqrt(norm_c_sq);
    cost_scale_ = (norm_c > 1e-15) ? std::sqrt(num_integers) / norm_c : 1.0;

    nnz_lp_ = pdlp_.nnz_lp();
    if (nnz_lp_ == 0) {
        finished_ = true;
        return;
    }

    stale_budget_ = total_budget_ >> 2;
    modified_cost_ = orig_cost;
    cycle_history_.reserve(pump::kCycleWindow);

    // Pre-compute variable order for this worker's static strategy.
    std::mt19937 order_rng(kBaseSeedOffset + static_cast<uint32_t>(fpr_config_index_));
    var_order_ = compute_var_order(mipsolver_, kFprConfigs[fpr_config_index_].strat.var_strategy,
                                   order_rng, nullptr);
}

EpochResult ScyllaWorker::run_epoch(size_t epoch_budget) {
    if (finished_) {
        return {};
    }

    const auto *model = mipsolver_.model_;
    auto *mipdata = mipsolver_.mipdata_.get();
    const auto &integrality = model->integrality_;
    const auto &orig_cost = model->col_cost_;
    const double time_limit = mipsolver_.options_mip_->time_limit;

    EpochResult epoch{};

    while (epoch.effort < epoch_budget && total_effort_ < total_budget_) {
        // HiGHS's timer is not guaranteed thread-safe for concurrent
        // readers; races here are benign (stale reads by ~ms, not data
        // corruption) and the cost of a formal gate isn't worth it.
        if (mipsolver_.timer_.read() >= time_limit) {
            finished_ = true;
            break;
        }
        if (effort_since_improvement_ > stale_budget_) {
            finished_ = true;
            break;
        }

        ++K_;

        double remaining = time_limit - mipsolver_.timer_.read();
        if (remaining <= 0.0) {
            finished_ = true;
            break;
        }

        auto solve = pdlp_.solve(modified_cost_, warm_start_col_value_, warm_start_row_dual_,
                                 warm_start_valid_, epsilon_, remaining);

        // Amortize PDLP iteration cost across the N concurrent workers
        // sharing the contested solver: only one PDLP solve runs at a
        // time, so a chain that holds the mutex for K iters contributes
        // K / N iters worth of shared PDLP work to the global effort
        // accounting.  Keeps per-worker staleness from firing after a
        // single large LP-heavy solve.
        size_t iter_effort =
            (static_cast<size_t>(solve.pdlp_iters) * nnz_lp_) / static_cast<size_t>(num_workers_);
        total_effort_ += iter_effort;
        effort_since_improvement_ += iter_effort;
        epoch.effort += iter_effort;

        if (solve.status == HighsStatus::kError) {
            finished_ = true;
            break;
        }
        if (solve.model_status == HighsModelStatus::kInfeasible) {
            finished_ = true;
            break;
        }

        if (solve.pdlp_iters == 0) {
            ++pdlp_stall_count_;
            if (pdlp_stall_count_ >= pump::kMaxPdlpStalls) {
                finished_ = true;
                break;
            }
        } else {
            pdlp_stall_count_ = 0;
        }
        if (solve.col_value.empty()) {
            finished_ = true;
            break;
        }

        warm_start_col_value_ = std::move(solve.col_value);
        warm_start_row_dual_ = std::move(solve.row_dual);
        warm_start_valid_ = solve.value_valid && solve.dual_valid;

        const auto &x_bar = warm_start_col_value_;

        // Fast path: PDLP solution already MIP-feasible.
        {
            bool mip_feasible = true;
            const double feastol = mipsolver_.options_mip_->mip_feasibility_tolerance;
            for (HighsInt j = 0; j < ncol_; ++j) {
                if (!is_integer(integrality, j)) {
                    continue;
                }
                if (std::abs(x_bar[j] - std::round(x_bar[j])) > feastol) {
                    mip_feasible = false;
                    break;
                }
            }
            if (mip_feasible) {
                for (HighsInt i = 0; i < nrow_; ++i) {
                    double lhs = 0.0;
                    for (HighsInt k = mipdata->ARstart_[i]; k < mipdata->ARstart_[i + 1]; ++k) {
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

        size_t remaining_budget = std::min(epoch_budget - std::min(epoch_budget, epoch.effort),
                                           total_budget_ - std::min(total_budget_, total_effort_));
        if (remaining_budget == 0) {
            break;
        }

        const auto &named = kFprConfigs[fpr_config_index_];
        FprConfig cfg{};
        cfg.max_effort = remaining_budget;
        cfg.rng_seed_offset =
            kBaseSeedOffset + static_cast<uint32_t>(fpr_config_index_) + static_cast<uint32_t>(K_);
        cfg.hint = x_bar.data();
        cfg.scores = nullptr;
        cfg.cont_fallback = x_bar.data();
        cfg.csc = &csc_;
        cfg.mode = named.mode;
        cfg.strategy = &named.strat;
        cfg.lp_ref = nullptr;
        cfg.precomputed_var_order = var_order_.data();
        cfg.precomputed_var_order_size = static_cast<HighsInt>(var_order_.size());

        std::vector<double> restart;
        pool_.get_restart(rng_, restart);
        const double *restart_ptr = restart.empty() ? nullptr : restart.data();

        HeuristicResult rounded = fpr_attempt(mipsolver_, cfg, rng_, 0, restart_ptr);

        total_effort_ += rounded.effort;
        effort_since_improvement_ += rounded.effort;
        epoch.effort += rounded.effort;

        if (rounded.found_feasible && !rounded.solution.empty()) {
            pool_.try_add(rounded.objective, rounded.solution);
            effort_since_improvement_ = 0;
            epoch.found_improvement = true;
        }

        if (rounded.solution.empty()) {
            continue;
        }

        auto &x_hat = rounded.solution;

        if (pump::detect_cycling(cycle_history_, x_hat, integrality, ncol_)) {
            pump::perturb(x_hat, *model, rng_);
        }
        if (static_cast<int>(cycle_history_.size()) < pump::kCycleWindow) {
            cycle_history_.push_back(x_hat);
        } else {
            cycle_history_[(K_ - 1) % pump::kCycleWindow] = x_hat;
        }

        alpha_K_ *= pump::kAlpha;
        pump::compute_pump_objective(orig_cost, x_hat, x_bar, integrality, model->col_lower_,
                                     model->col_upper_, alpha_K_, cost_scale_, ncol_,
                                     modified_cost_);

        epsilon_ = std::max(pump::kBeta * epsilon_, pump::kEpsilonFloor);
    }

    return epoch;
}
