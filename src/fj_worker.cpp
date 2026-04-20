#include "fj_worker.h"

#include "epoch_runner.h"
#include "mip/feasibilityjump.hh"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "solution_pool.h"

#include <algorithm>
#include <cmath>
#include <vector>

static_assert(EpochWorker<FjWorker>, "FjWorker must satisfy EpochWorker concept");

using external_feasibilityjump::CallbackControlFlow;
using external_feasibilityjump::FeasibilityJumpSolver;
using external_feasibilityjump::FJStatus;
using external_feasibilityjump::RowType;
using external_feasibilityjump::VarType;

struct FjWorker::Impl {
    FeasibilityJumpSolver solver;
    std::vector<double> col_value;

    Impl(const HighsLogOptions& log_options, int seed, double epsilon, double feastol)
        : solver(log_options, seed, epsilon, feastol) {}
};

FjWorker::FjWorker(HighsMipSolver& mipsolver, SolutionPool& pool, size_t total_budget,
                   uint32_t seed)
    : mipsolver_(mipsolver), pool_(pool), total_budget_(total_budget), seed_(seed) {}

FjWorker::~FjWorker() = default;

void FjWorker::reset_staleness() {
    effort_since_improvement_ = 0;
}

EpochResult FjWorker::run_epoch(size_t epoch_budget) {
    if (finished_) {
        return {};
    }

    const HighsLp* model = mipsolver_.model_;
    auto* mipdata = mipsolver_.mipdata_.get();
    const double feastol = mipsolver_.options_mip_->mip_feasibility_tolerance;
    const double epsilon = mipdata->epsilon;
    const double sense_multiplier = static_cast<double>(model->sense_);

    // First epoch: build the solver and initial assignments.
    if (!initialized_) {
        initialized_ = true;

#ifdef HIGHSINT64
        finished_ = true;
        return {};
#endif

        const HighsLogOptions& log_options = mipsolver_.options_mip_->log_options;
        impl_ = std::make_unique<Impl>(log_options, static_cast<int>(seed_), epsilon, feastol);

        impl_->col_value.resize(model->num_col_, 0.0);

        const auto& inc = mipdata->incumbent;
        const bool use_incumbent = !inc.empty();

        for (HighsInt col = 0; col < model->num_col_; ++col) {
            double lower = model->col_lower_[col];
            double upper = model->col_upper_[col];

            VarType fjVarType;
            if (model->integrality_[col] == HighsVarType::kContinuous) {
                fjVarType = VarType::Continuous;
            } else {
                fjVarType = VarType::Integer;
                lower = std::ceil(lower - feastol);
                upper = std::floor(upper + feastol);
            }

            const bool legal_bounds = lower <= upper && lower < kHighsInf && upper > -kHighsInf &&
                                      !std::isnan(lower) && !std::isnan(upper);
            if (!legal_bounds) {
                finished_ = true;
                return {};
            }
            impl_->solver.addVar(fjVarType, lower, upper, sense_multiplier * model->col_cost_[col]);

            double initial_assignment = 0.0;
            if (use_incumbent && std::isfinite(inc[col])) {
                initial_assignment = std::max(lower, std::min(upper, inc[col]));
            } else {
                if (std::isfinite(lower)) {
                    initial_assignment = lower;
                } else if (std::isfinite(upper)) {
                    initial_assignment = upper;
                }
            }
            impl_->col_value[col] = initial_assignment;
        }

        HighsSparseMatrix a_matrix;
        a_matrix.createRowwise(model->a_matrix_);

        for (HighsInt row = 0; row < model->num_row_; ++row) {
            bool hasFiniteLower = std::isfinite(model->row_lower_[row]);
            bool hasFiniteUpper = std::isfinite(model->row_upper_[row]);
            if (hasFiniteLower || hasFiniteUpper) {
                HighsInt row_num_nz = a_matrix.start_[row + 1] - a_matrix.start_[row];
                auto row_index = a_matrix.index_.data() + a_matrix.start_[row];
                auto row_value = a_matrix.value_.data() + a_matrix.start_[row];
                if (hasFiniteLower) {
                    impl_->solver.addConstraint(RowType::Gte, model->row_lower_[row], row_num_nz,
                                                row_index, row_value, 0);
                }
                if (hasFiniteUpper) {
                    impl_->solver.addConstraint(RowType::Lte, model->row_upper_[row], row_num_nz,
                                                row_index, row_value, 0);
                }
            }
        }

        const HighsInt nnz = a_matrix.numNz();
        stale_budget_ =
            std::min(static_cast<size_t>(nnz) << 8, total_budget_ > 0 ? total_budget_ : SIZE_MAX);
    }

    if (!impl_) {
        finished_ = true;
        return {};
    }

    // Capture state for the callback closure.
    const bool resume = first_solve_done_;
    size_t epoch_effort_consumed = 0;
    bool found_solution = false;
    double best_obj = 0.0;
    std::vector<double> best_sol;

    auto callback = [&](FJStatus status) -> CallbackControlFlow {
        epoch_effort_consumed = status.totalEffort - total_effort_;

        if (status.solution != nullptr) {
            found_solution = true;
            best_sol.assign(status.solution, status.solution + status.numVars);
            best_obj = model->offset_ + sense_multiplier * status.solutionObjectiveValue;
        }

        // Pause at epoch boundary.
        if (epoch_effort_consumed >= epoch_budget) {
            return CallbackControlFlow::Terminate;
        }
        // Total budget exceeded.
        if (status.totalEffort > total_budget_) {
            return CallbackControlFlow::Terminate;
        }
        // Stall detection.
        size_t esi = status.effortSinceLastImprovement;
        if (esi > stale_budget_) {
            return CallbackControlFlow::Terminate;
        }

        return CallbackControlFlow::Continue;
    };

    impl_->solver.solve(resume ? nullptr : impl_->col_value.data(), callback, resume);
    first_solve_done_ = true;

    EpochResult result{};
    result.effort = epoch_effort_consumed;
    total_effort_ += epoch_effort_consumed;

    if (found_solution) {
        pool_.try_add(best_obj, best_sol, kSolutionSourceFJ);
        result.found_improvement = true;
        effort_since_improvement_ = 0;
    } else {
        effort_since_improvement_ += epoch_effort_consumed;
    }

    // Mark finished if stalled or total budget exceeded.
    if (total_effort_ >= total_budget_ || effort_since_improvement_ > stale_budget_) {
        finished_ = true;
    }

    return result;
}
