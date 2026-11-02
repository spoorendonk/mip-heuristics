#include "early_heuristics.h"

#include "effort_ledger.h"
#include "fj.h"
#include "heuristic_common.h"
#include "heuristic_context.h"
#include "highs_context.h"
#include "io/HighsIO.h"
#include "lp_data/HighsLp.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "solution_sink.h"
#include "util/HighsCDouble.h"

#include <atomic>
#include <cstddef>
#include <exception>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

namespace heuristics {

// Everything the worker thread owns.  `storage`, `problem` and `sink` are
// built on that thread and read by `finish()` only after the join.
struct EarlyFj::Run {
    Run(HighsMipSolver& mipsolver, double effort)
        : model(*mipsolver.orig_model_),
          options(*mipsolver.options_mip_),
          effort_option(effort),
          exec{.num_workers = 1,
               .base_seed = heuristic_base_seed(options.random_seed),
               .deadline = deadline_of(mipsolver),
               .log_options = options.log_options,
               // No terminator: `terminatorTerminated` writes, and this
               // thread is not the solve's.  The stop flag and the clock
               // are the write-free halves (`ExecutionContext`).
               .terminator = {},
               .stop = &stop} {}

    // The worker thread's body: build the view, size the budget, search.
    void search() {
        try {
            auto view = make_problem(model, storage, options.mip_feasibility_tolerance,
                                     options.small_matrix_value);
            if (!view) {
                return;
            }
            problem.emplace(*view);
            const size_t total = heuristic_effort_budget(problem->nnz, effort_option);
            HeuristicBudget budget = make_budget(
                total, 1,
                patience_threshold(problem->nnz, options.mip_heuristic_fj_patience, total));
            // The record bounds the option to the enum's range.
            budget.restart = static_cast<RestartPoint>(options.mip_heuristic_fj_restart);
            sink.emplace(*problem->integrality, kSolutionSourceFJ);
            effort = fj::run_until_stopped(*problem, budget, exec, 0, RestartSource{}, *sink);
        } catch (...) {
            error = std::current_exception();
        }
    }

    const HighsLp& model;
    const HighsOptions& options;
    double effort_option;
    std::atomic<bool> stop{false};
    ExecutionContext exec;
    ProblemStorage storage;
    std::optional<ProblemView> problem;
    std::optional<SolutionSink> sink;
    size_t effort = 0;
    std::exception_ptr error;
    double t0_s = 0.0;
    std::thread thread;
};

EarlyFj::EarlyFj(HighsMipSolver& mipsolver) : mipsolver_(mipsolver) {
    const HighsOptions& options = *mipsolver.options_mip_;
    if (mipsolver.submip || options.mip_heuristic_early_fj_effort <= 0.0) {
        return;
    }
    run_ = std::make_unique<Run>(mipsolver, options.mip_heuristic_early_fj_effort);
    run_->t0_s = EffortLedger(mipsolver).now_s();
    Run* run = run_.get();
    run_->thread = std::thread([run] { run->search(); });
}

EarlyFj::~EarlyFj() {
    if (run_ && run_->thread.joinable()) {
        run_->stop.store(true, std::memory_order_relaxed);
        run_->thread.join();
    }
}

namespace {

// Whether `mipsolver`'s current original-space solution is feasible, by the
// same test `cleanupSolve` applies to it.
bool current_solution_feasible(const HighsMipSolver& mipsolver) {
    const double tol = mipsolver.options_mip_->mip_feasibility_tolerance;
    return mipsolver.solution_objective_ != kHighsInf && mipsolver.bound_violation_ <= tol &&
           mipsolver.integrality_violation_ <= tol && mipsolver.row_violation_ <= tol;
}

}  // namespace

void EarlyFj::finish(bool run_to_budget) {
    if (!run_ || !run_->thread.joinable()) {
        return;
    }
    if (!run_to_budget) {
        run_->stop.store(true, std::memory_order_relaxed);
    }
    run_->thread.join();
    if (run_->error) {
        std::rethrow_exception(run_->error);
    }

    std::vector<double> best;
    const bool found = run_->sink.has_value() && run_->sink->copy_best(best);
    EffortLedger ledger(mipsolver_);
    // Read off the model, not the view: the view may never have been built.
    const auto nnz = static_cast<size_t>(mipsolver_.orig_model_->a_matrix_.numNz());
    ledger.charge_early("early_fj", run_->effort, found, nnz, run_->t0_s, ledger.now_s());
    if (!found) {
        return;
    }

    // Presolve's own verdict decides whether there is anything to hand
    // over: an optimal, infeasible or unbounded model, or one presolve ran
    // out of memory on, keeps what presolve concluded.
    const HighsModelStatus status = mipsolver_.modelstatus_;
    if (status != HighsModelStatus::kNotset && status != HighsModelStatus::kTimeLimit) {
        return;
    }

    // Checked on the original model with HiGHS's own test, which is what
    // `cleanupSolve` will judge `solution_` by.
    const HighsLp& model = *mipsolver_.orig_model_;
    double bound_violation = 0.0;
    double row_violation = 0.0;
    double integrality_violation = 0.0;
    HighsCDouble quad_objective = 0.0;
    mipsolver_.solutionFeasible(&model, best, nullptr, bound_violation, row_violation,
                                integrality_violation, quad_objective);
    const double tol = mipsolver_.options_mip_->mip_feasibility_tolerance;
    if (bound_violation > tol || row_violation > tol || integrality_violation > tol) {
        return;
    }
    const auto objective = static_cast<double>(quad_objective);
    const auto sense = static_cast<double>(static_cast<int>(model.sense_));
    if (current_solution_feasible(mipsolver_) &&
        sense * mipsolver_.solution_objective_ <= sense * objective) {
        return;
    }

    mipsolver_.solution_ = std::move(best);
    mipsolver_.solution_objective_ = objective;
    mipsolver_.bound_violation_ = bound_violation;
    mipsolver_.row_violation_ = row_violation;
    mipsolver_.integrality_violation_ = integrality_violation;
    highsLogUser(mipsolver_.options_mip_->log_options, HighsLogType::kInfo,
                 "Early FeasibilityJump on the original model: objective %.12g, passed on as "
                 "the MIP start\n",
                 objective);

    // Presolve stopped on the time limit, so there is no `runSetup` to take
    // the bound from `solution_`.  `upper_bound` is still in the original
    // minimisation space here and `cleanupSolve` adds the presolved
    // model's offset to it, so this is what makes the run report the
    // solution's objective as its primal bound.
    if (status == HighsModelStatus::kTimeLimit) {
        mipsolver_.mipdata_->upper_bound = (sense * objective) - mipsolver_.model_->offset_;
    }
}

}  // namespace heuristics
