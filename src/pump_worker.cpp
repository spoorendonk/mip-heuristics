#include "pump_worker.h"

#include "fpr_core.h"
#include "heuristic_common.h"
#include "Highs.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "pump_common.h"
#include "solution_pool.h"

#include <algorithm>
#include <cmath>

struct PumpWorker::Impl {
    Highs highs;
    HighsSolution warm_start;
};

PumpWorker::PumpWorker(HighsMipSolver &mipsolver, const CscMatrix &csc, SolutionPool &pool,
                       size_t total_budget, uint32_t seed)
    : impl_(std::make_unique<Impl>()),
      mipsolver_(mipsolver),
      csc_(csc),
      pool_(pool),
      total_budget_(total_budget),
      seed_(seed),
      epsilon_(pump::kEpsilonInit),
      rng_(seed) {
    const auto *model = mipsolver_.model_;
    auto *mipdata = mipsolver_.mipdata_.get();
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

    auto lp = pump::build_lp_relaxation(*model, *mipdata);
    impl_->highs.setOptionValue("solver", "pdlp");
    impl_->highs.setOptionValue("output_flag", false);
    impl_->highs.setOptionValue("pdlp_scaling", true);
    impl_->highs.setOptionValue("pdlp_e_restart_method", 2);
    nnz_lp_ = mipdata->ARindex_.size();
    if (nnz_lp_ == 0) {
        finished_ = true;
        return;
    }
    // nnz_lp_ > 0 guaranteed by the early return above.
    auto pdlp_iter_cap = static_cast<HighsInt>((total_budget_ >> 2) / nnz_lp_);
    if (pdlp_iter_cap < 100) {
        pdlp_iter_cap = 100;
    }
    impl_->highs.setOptionValue("pdlp_iteration_limit", pdlp_iter_cap);
    impl_->highs.passModel(std::move(lp));

    stale_budget_ = total_budget_ >> 2;
    scores_.resize(ncol_);
    modified_cost_.resize(ncol_);
    cycle_history_.reserve(pump::kCycleWindow);
}

PumpWorker::~PumpWorker() = default;

EpochResult PumpWorker::run_epoch(size_t epoch_budget) {
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
        if (mipsolver_.timer_.read() >= time_limit) {
            finished_ = true;
            break;
        }
        if (effort_since_improvement_ > stale_budget_) {
            finished_ = true;
            break;
        }

        ++K_;

        impl_->highs.setOptionValue("pdlp_optimality_tolerance", epsilon_);
        double remaining = time_limit - mipsolver_.timer_.read();
        if (remaining <= 0.0) {
            finished_ = true;
            break;
        }
        impl_->highs.setOptionValue("time_limit", remaining);

        if (impl_->warm_start.value_valid && impl_->warm_start.dual_valid) {
            impl_->highs.setSolution(impl_->warm_start);
        }

        HighsStatus status = impl_->highs.run();

        HighsInt pdlp_iters = 0;
        impl_->highs.getInfoValue("pdlp_iteration_count", pdlp_iters);
        size_t iter_effort = static_cast<size_t>(pdlp_iters) * nnz_lp_;
        total_effort_ += iter_effort;
        effort_since_improvement_ += iter_effort;
        epoch.effort += iter_effort;

        if (status == HighsStatus::kError) {
            finished_ = true;
            break;
        }
        if (impl_->highs.getModelStatus() == HighsModelStatus::kInfeasible) {
            finished_ = true;
            break;
        }

        if (pdlp_iters == 0) {
            ++pdlp_stall_count_;
            if (pdlp_stall_count_ >= pump::kMaxPdlpStalls) {
                finished_ = true;
                break;
            }
        } else {
            pdlp_stall_count_ = 0;
        }
        const auto &sol = impl_->highs.getSolution();
        if (sol.col_value.empty()) {
            finished_ = true;
            break;
        }

        impl_->warm_start.col_value = sol.col_value;
        impl_->warm_start.row_dual = sol.row_dual;
        impl_->warm_start.value_valid = sol.value_valid;
        impl_->warm_start.dual_valid = sol.dual_valid;

        const auto &x_bar = sol.col_value;

        // Check if PDLP solution is already MIP-feasible (fast path)
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
        cfg.max_effort = std::min(epoch_budget - std::min(epoch_budget, epoch.effort),
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
        if (x_hat.empty()) {
            continue;
        }

        // Cycling detection + perturbation
        if (pump::detect_cycling(cycle_history_, x_hat, integrality, ncol_)) {
            pump::perturb(x_hat, *model, rng_);
        }
        if (static_cast<int>(cycle_history_.size()) < pump::kCycleWindow) {
            cycle_history_.push_back(x_hat);
        } else {
            cycle_history_[(K_ - 1) % pump::kCycleWindow] = x_hat;
        }

        // Objective update
        alpha_K_ *= pump::kAlpha;
        pump::compute_pump_objective(orig_cost, x_hat, x_bar, integrality, model->col_lower_,
                                     model->col_upper_, alpha_K_, cost_scale_, ncol_,
                                     modified_cost_);
        impl_->highs.changeColsCost(0, ncol_ - 1, modified_cost_.data());

        epsilon_ = std::max(pump::kBeta * epsilon_, pump::kEpsilonFloor);
    }

    return epoch;
}
