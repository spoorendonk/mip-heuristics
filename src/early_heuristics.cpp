#include "early_heuristics.h"

#include "effort_ledger.h"
#include "fj.h"
#include "fj_race.h"
#include "heuristic_common.h"
#include "heuristic_context.h"
#include "highs_context.h"
#include "io/HighsIO.h"
#include "lp_data/HighsLp.h"
#include "lp_data/HighsSolution.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "presolve/HighsPostsolveStack.h"
#include "solution_sink.h"
#include "util/HighsCDouble.h"

#include <atomic>
#include <cmath>
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
        MipViolation violation{};
        HighsCDouble quad_objective = 0.0;
        if (!mipsolver_.solutionFeasible(mipsolver_.orig_model_, solution, nullptr, violation,
                                         quad_objective)) {
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
    // The GPU FJ race's CUDA context, created while presolve runs (#289).
    gpu_race_ = !mipsolver.submip && options.mip_heuristic_fj_gpu &&
                options.mip_heuristic_fj_effort > 0.0 && options.mip_heuristic_run_feasibility_jump;
#ifdef MIP_HEURISTICS_HAS_GPU_FJ
    if (gpu_race_) {
        fj::warm_gpu();
    }
#endif
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
#ifdef MIP_HEURISTICS_HAS_GPU_FJ
    // The end of the solve, past every dispatch: the context's creation
    // must finish before `main` returns into the runtime's teardown (#289).
    if (gpu_race_) {
        fj::wait_gpu_warmup();
    }
#endif
}

namespace {

// Whether `mipsolver`'s current original-space solution is feasible, by the
// same test `cleanupSolve` applies to it.
bool current_solution_feasible(const HighsMipSolver& mipsolver) {
    const double tol = mipsolver.options_mip_->mip_feasibility_tolerance;
    return mipsolver.solution_objective_ != kHighsInf && mipsolver.bound_violation_ <= tol &&
           mipsolver.integrality_violation_ <= tol && mipsolver.row_violation_ <= tol;
}

// Whether `a` differs from `b` by more than `checkAddSolution`'s relative
// tolerance, the test by which it reports a start's objective as changed.
bool objectives_differ(const HighsMipSolver& mipsolver, double a, double b) {
    return std::fabs(a - b) / (1.0 + std::fabs(b)) >
           mipsolver.options_mip_->mip_feasibility_tolerance;
}

// `point` crushed into the presolved space as `checkAddSolution` crushes a
// start, into `crushed`: whether it is feasible on the presolved model, by the
// test `point` passed on the original one, with its objective, in the
// original sense as `checkAddSolution` states it, in `objective`.  The crush
// moves only the postsolve stack's read cursor, which every crush sets before
// it reads.
bool crush(HighsMipSolver& mipsolver, const std::vector<double>& point,
           std::vector<double>& crushed, double& objective) {
    const HighsLp& presolved = *mipsolver.model_;
    crushed = mipsolver.mipdata_->postSolveStack.getReducedPrimalSolution(point);
    MipViolation violation{};
    HighsCDouble quad_objective = 0.0;
    if (crushed.size() != static_cast<std::size_t>(presolved.num_col_) ||
        !mipsolver.solutionFeasible(&presolved, crushed, nullptr, violation, quad_objective)) {
        return false;
    }
    // Presolve states its model as a minimisation; before presolve has run
    // (a test's `finish()`) the model is still the original, in its own sense.
    const int sense_change =
        static_cast<int>(mipsolver.orig_model_->sense_) * static_cast<int>(presolved.sense_);
    objective = static_cast<double>(sense_change) * static_cast<double>(quad_objective);
    return true;
}

// The start `checkAddSolution` will make of `point`, whose original-space
// `objective` and `violation` are given: whether there is one, with `point`,
// `objective` and `violation` then describing it, else why not in `reason`.
//
// `checkAddSolution` crushes `point` into the presolved space as the
// incumbent, takes the upper bound and `solution_objective_` from the crushed
// point's objective, and leaves `solution_` as `point`.  The crush only
// applies duplicate-column and linear transforms and drops removed columns,
// so two things can come apart:
// - The crushed point can be infeasible on the presolved model (#306), e.g.
//   when a dual reduction removed a column that another must compensate
//   for: it would set a cutoff no presolved point reaches.  Dropped.
// - It can be feasible with another objective (#343), e.g. when presolve
//   fixed a dominated column at the bound `point` does not take: the run
//   would report the crushed objective beside a vector that does not attain
//   it.  The crushed point is then postsolved, as `transformNewIntegerFeasible
//   Solution` postsolves every presolved incumbent, and its image becomes the
//   start if it passes both tests in its turn: feasible on the original model,
//   and crushed, feasible on the presolved one with the same objective.
//   Otherwise dropped: there is no repair LP here, unlike upstream's.
bool presolved_start(HighsMipSolver& mipsolver, std::vector<double>& point, double& objective,
                     MipViolation& violation, const char*& reason) {
    HighsSolution image;
    double crushed_objective = 0.0;
    if (!crush(mipsolver, point, image.col_value, crushed_objective)) {
        reason = "infeasible in the presolved model";
        return false;
    }
    if (!objectives_differ(mipsolver, crushed_objective, objective)) {
        return true;
    }

    image.value_valid = true;
    mipsolver.mipdata_->postSolveStack.undoPrimal(*mipsolver.options_mip_, image);
    MipViolation image_violation{};
    HighsCDouble image_quad_objective = 0.0;
    std::vector<double> recrushed;
    double recrushed_objective = 0.0;
    const bool attained =
        mipsolver.solutionFeasible(mipsolver.orig_model_, image.col_value, nullptr, image_violation,
                                   image_quad_objective) &&
        crush(mipsolver, image.col_value, recrushed, recrushed_objective) &&
        !objectives_differ(mipsolver, recrushed_objective,
                           static_cast<double>(image_quad_objective));
    if (!attained) {
        reason = "the presolved model gives it another objective and no feasible point attains it";
        return false;
    }
    const auto image_objective = static_cast<double>(image_quad_objective);
    highsLogUser(mipsolver.options_mip_->log_options, HighsLogType::kInfo,
                 "Early FeasibilityJump on the original model: objective %.12g postsolved from "
                 "its presolved image to objective %.12g\n",
                 objective, image_objective);
    point = std::move(image.col_value);
    objective = image_objective;
    violation = image_violation;
    return true;
}

}  // namespace

bool install_early_start(HighsMipSolver& mipsolver, std::vector<double>& point) {
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
    MipViolation violation{};
    HighsCDouble quad_objective = 0.0;
    if (!mipsolver.solutionFeasible(&model, point, nullptr, violation, quad_objective)) {
        return false;
    }
    auto objective = static_cast<double>(quad_objective);
    // `runSetup` follows only when presolve gave no verdict.  After a
    // time-out nothing searches: the bound set below is the point's own
    // original-space objective, which it attains, and nothing prunes on it.
    const char* reason = nullptr;
    if (status == HighsModelStatus::kNotset &&
        !presolved_start(mipsolver, point, objective, violation, reason)) {
        highsLogUser(mipsolver.options_mip_->log_options, HighsLogType::kInfo,
                     "Early FeasibilityJump on the original model: objective %.12g at %.3fs, "
                     "dropped: %s\n",
                     objective, EffortLedger(mipsolver).now_s(), reason);
        return false;
    }
    const auto sense = static_cast<double>(static_cast<int>(model.sense_));
    if (current_solution_feasible(mipsolver) &&
        sense * mipsolver.solution_objective_ <= sense * objective) {
        return false;
    }

    mipsolver.solution_ = std::move(point);
    mipsolver.solution_objective_ = objective;
    violation.copy(mipsolver.bound_violation_, mipsolver.row_violation_,
                   mipsolver.integrality_violation_);
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
    // presolve-optimal exit reports its solution: with `-kHighsInf` as the
    // new upper limit, so the point always counts as improving.
    if (status == HighsModelStatus::kTimeLimit) {
        mipsolver.mipdata_->upper_bound = (sense * objective) - mipsolver.model_->offset_;
        mipsolver.mipdata_->saveReportMipSolution(-kHighsInf);
    }
    return true;
}

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
    const bool installed = found && install_early_start(mipsolver_, best);
    EffortLedger ledger(mipsolver_);
    ledger.charge_early("early_fj", run_->effort, installed, run_->nnz, run_->t0_s, ledger.now_s());
}

}  // namespace heuristics
