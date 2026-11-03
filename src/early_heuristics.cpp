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
#include <utility>
#include <vector>

namespace heuristics {

namespace {

// Logs each improving point when the worker finds it (#240), since presolve
// may hold the handover back for many seconds.  One worker offers into it, so
// `on_accept` never runs concurrently and `best_logged_` needs no lock.
//
// What the worker thread touches of the solve is read-only for its whole
// length: `orig_model_` (presolve works on its own copy), the options, clock 0
// of `timer_`.  `solutionFeasible` is const and keeps its row values in a
// local.  The log line is one `highsLogUser` call, as `[HeurSol]` is one
// `highsLogDev` call from a worker.  The improving-solution callback is not
// called here: it writes `callback_->data_out`, which the solving thread
// uses too, and a user callback does not expect a second thread.
// `runSetup` fires it for the handed-over point.
class ReportingSink final : public SolutionSink {
public:
    ReportingSink(const std::vector<HighsVarType>& integrality, int source,
                  const HighsMipSolver& mipsolver)
        : SolutionSink(integrality, source), mipsolver_(mipsolver) {}

    // Moves the best point that passed into `out`; whether there is one.  Only
    // after the join.
    [[nodiscard]] bool take_best_checked(std::vector<double>& out) {
        if (best_logged_ == kHighsInf) {
            return false;
        }
        out = std::move(best_checked_);
        return true;
    }

protected:
    void on_accept(double objective, const std::vector<double>& solution, int /*source*/,
                   const WorkerTrace& /*trace*/, size_t /*effort_at*/) override {
        if (objective >= best_logged_) {
            return;
        }
        // The test `finish()` will apply, and the last point to pass is the
        // one it hands over, not the pool's best: a better point that fails
        // here must not cost the run a logged one.  `objective` is the view's
        // minimisation form; the line carries the model's own sense.
        double bound_violation = 0.0;
        double row_violation = 0.0;
        double integrality_violation = 0.0;
        HighsCDouble quad_objective = 0.0;
        if (!mipsolver_.solutionFeasible(mipsolver_.orig_model_, solution, nullptr, bound_violation,
                                         row_violation, integrality_violation, quad_objective)) {
            return;
        }
        best_logged_ = objective;
        best_checked_ = solution;
        highsLogUser(mipsolver_.options_mip_->log_options, HighsLogType::kInfo,
                     "Early FeasibilityJump on the original model: objective %.12g found at "
                     "%.3fs\n",
                     static_cast<double>(quad_objective), mipsolver_.timer_.read());
    }

private:
    const HighsMipSolver& mipsolver_;
    double best_logged_ = kHighsInf;
    std::vector<double> best_checked_;
};

}  // namespace

// Everything the worker thread owns.  `storage`, `problem` and `sink` are
// built on that thread and read by `finish()` only after the join.
struct EarlyFj::Run {
    Run(HighsMipSolver& mipsolver, double effort)
        : solver(mipsolver),
          model(*mipsolver.orig_model_),
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
            const size_t total = heuristic_effort_budget(nnz, effort_option);
            HeuristicBudget budget = make_budget(
                total, 1, patience_threshold(nnz, options.mip_heuristic_fj_patience, total));
            // The record bounds the option to the enum's range.
            budget.restart = static_cast<RestartPoint>(options.mip_heuristic_fj_restart);
            // The objective cut (#222), as the presolve chain sets it, against
            // this thread's own sink.
            budget.objective_cut = {
                .on = options.mip_heuristic_objective_cut && has_objective(*problem),
                .rel = options.mip_heuristic_objective_cut_rel,
                .abs = options.mip_heuristic_objective_cut_abs,
            };
            sink.emplace(*problem->integrality, kSolutionSourceFJ, solver);
            effort = fj::run_until_stopped(*problem, budget, exec, 0, RestartSource{}, *sink);
        } catch (...) {
            error = std::current_exception();
        }
    }

    const HighsMipSolver& solver;
    const HighsLp& model;
    const HighsOptions& options;
    double effort_option;
    // The original model's nnz, the unit of the budget and of the `[Heur]`
    // line both: read off the model, since the view may never be built.
    size_t nnz = static_cast<size_t>(model.a_matrix_.numNz());
    std::atomic<bool> stop{false};
    ExecutionContext exec;
    ProblemStorage storage;
    std::optional<ProblemView> problem;
    std::optional<ReportingSink> sink;
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

// Hands `best` to `solution_` as `finish()` describes; whether it did.
bool install(HighsMipSolver& mipsolver, std::vector<double>& best) {
    // Presolve's own verdict decides whether there is anything to hand
    // over: an optimal, infeasible or unbounded model, or one presolve ran
    // out of memory on, keeps what presolve concluded.
    const HighsModelStatus status = mipsolver.modelstatus_;
    if (status != HighsModelStatus::kNotset && status != HighsModelStatus::kTimeLimit) {
        return false;
    }

    // Checked on the original model with HiGHS's own test, which is what
    // `cleanupSolve` will judge `solution_` by.
    const HighsLp& model = *mipsolver.orig_model_;
    double bound_violation = 0.0;
    double row_violation = 0.0;
    double integrality_violation = 0.0;
    HighsCDouble quad_objective = 0.0;
    mipsolver.solutionFeasible(&model, best, nullptr, bound_violation, row_violation,
                               integrality_violation, quad_objective);
    const double tol = mipsolver.options_mip_->mip_feasibility_tolerance;
    if (bound_violation > tol || row_violation > tol || integrality_violation > tol) {
        return false;
    }
    const auto objective = static_cast<double>(quad_objective);
    const auto sense = static_cast<double>(static_cast<int>(model.sense_));
    if (current_solution_feasible(mipsolver) &&
        sense * mipsolver.solution_objective_ <= sense * objective) {
        return false;
    }

    mipsolver.solution_ = std::move(best);
    mipsolver.solution_objective_ = objective;
    mipsolver.bound_violation_ = bound_violation;
    mipsolver.row_violation_ = row_violation;
    mipsolver.integrality_violation_ = integrality_violation;
    // The time is here because HiGHS prints no display row for a MIP start:
    // this is when HiGHS can first use the point, and the worker's `found at`
    // lines say when it existed.
    highsLogUser(mipsolver.options_mip_->log_options, HighsLogType::kInfo,
                 "Early FeasibilityJump on the original model: objective %.12g at %.3fs, passed "
                 "on as the MIP start\n",
                 objective, EffortLedger(mipsolver).now_s());

    // Presolve stopped on the time limit, so there is no `runSetup` to take
    // the bound from `solution_`.  `upper_bound` is still in the original
    // minimisation space here and `cleanupSolve` adds the presolved
    // model's offset to it, so this is what makes the run report the
    // solution's objective as its primal bound.  The improving-solution
    // callback, the saved list and the file are reported as upstream's
    // presolve-optimal exit reports its solution.
    if (status == HighsModelStatus::kTimeLimit) {
        mipsolver.mipdata_->upper_bound = (sense * objective) - mipsolver.model_->offset_;
        mipsolver.mipdata_->saveReportMipSolution();
    }
    return true;
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
    const bool found = run_->sink.has_value() && run_->sink->take_best_checked(best);
    // Booked as found only when it was handed over, so `found=1` means the
    // run's answer can be this worker's.
    const bool installed = found && install(mipsolver_, best);
    EffortLedger ledger(mipsolver_);
    ledger.charge_early("early_fj", run_->effort, installed, run_->nnz, run_->t0_s, ledger.now_s());
}

}  // namespace heuristics
