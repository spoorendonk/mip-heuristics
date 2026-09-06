#include "fj_worker.h"

#include "heuristic_context.h"
#include "incumbent_sink.h"
#include "mip/feasibilityjump.hh"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

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

FjWorker::FjWorker(HighsMipSolver& mipsolver, const ExecutionContext& exec, IncumbentSink& sink,
                   size_t total_budget, size_t stale_budget, uint32_t seed,
                   std::vector<double> start, WorkerTrace trace)
    : mipsolver_(mipsolver),
      exec_(exec),
      sink_(sink),
      start_(std::move(start)),
      seed_(seed),
      trace_(trace) {
    base_.total_budget = total_budget;
    base_.stale_budget = stale_budget;
}

FjWorker::~FjWorker() = default;

// Cognitive complexity 50 (threshold 25).  Kept whole: one FeasibilityJump attempt: budget slicing,
// staleness detection and solver rebuild are interleaved with upstream FJ's callback contract,
// which resumes the solver mid-run.
// Decomposing it would move work across a worker's inner loop, and the
// closeout takes no unmeasured performance risk; the standards also rank
// fidelity to the reference algorithm above mechanical extraction.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
AttemptResult FjWorker::run_attempt(size_t attempt_budget) {
    if (base_.finished) {
        return {};
    }

    const HighsLp* model = mipsolver_.model_;
    auto* mipdata = mipsolver_.mipdata_.get();
    const double feastol = mipsolver_.options_mip_->mip_feasibility_tolerance;
    const double epsilon = mipdata->epsilon;
    const auto sense_multiplier = static_cast<double>(model->sense_);

    // First attempt: build the solver and initial assignments.
    if (!initialized_) {
        initialized_ = true;

#ifdef HIGHSINT64
        base_.finished = true;
        return {};
#endif

        const HighsLogOptions& log_options = mipsolver_.options_mip_->log_options;
        impl_ = std::make_unique<Impl>(log_options, static_cast<int>(seed_), epsilon, feastol);

        impl_->col_value.resize(model->num_col_, 0.0);

        // The caller's resolved start, not `mipdata->incumbent`: a peer
        // worker's accepted solution rewrites the live vector while this loop
        // indexes it (issue #98).
        const auto& inc = start_;
        const bool use_incumbent = !inc.empty();

        for (HighsInt col = 0; col < model->num_col_; ++col) {
            double lower = model->col_lower_[col];
            double upper = model->col_upper_[col];

            VarType fj_var_type;
            if (model->integrality_[col] == HighsVarType::kContinuous) {
                fj_var_type = VarType::Continuous;
            } else {
                fj_var_type = VarType::Integer;
                lower = std::ceil(lower - feastol);
                upper = std::floor(upper + feastol);
            }

            const bool legal_bounds = lower <= upper && lower < kHighsInf && upper > -kHighsInf &&
                                      !std::isnan(lower) && !std::isnan(upper);
            if (!legal_bounds) {
                base_.finished = true;
                return {};
            }
            impl_->solver.addVar(fj_var_type, lower, upper,
                                 sense_multiplier * model->col_cost_[col]);

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
            bool has_finite_lower = std::isfinite(model->row_lower_[row]);
            bool has_finite_upper = std::isfinite(model->row_upper_[row]);
            if (has_finite_lower || has_finite_upper) {
                HighsInt row_num_nz = a_matrix.start_[row + 1] - a_matrix.start_[row];
                auto* row_index = a_matrix.index_.data() + a_matrix.start_[row];
                auto* row_value = a_matrix.value_.data() + a_matrix.start_[row];
                if (has_finite_lower) {
                    impl_->solver.addConstraint(RowType::Gte, model->row_lower_[row], row_num_nz,
                                                row_index, row_value, 0);
                }
                if (has_finite_upper) {
                    impl_->solver.addConstraint(RowType::Lte, model->row_upper_[row], row_num_nz,
                                                row_index, row_value, 0);
                }
            }
        }
    }

    if (!impl_) {
        base_.finished = true;
        return {};
    }

    // Capture state for the callback closure.
    const bool resume = first_solve_done_;
    size_t attempt_effort_consumed = 0;
    bool improved_incumbent = false;
    std::vector<double> best_sol;

    auto callback = [&](FJStatus status) -> CallbackControlFlow {
        attempt_effort_consumed = status.totalEffort - base_.total_effort;

        // Publish as the solution arrives, not once the attempt ends (#163).
        //
        // Upstream FJ's callback carries a solution exactly when it has
        // strictly improved on its own best, so this offers on improvement
        // and not on the callback's `CALLBACK_EFFORT` cadence.  It used to
        // overwrite a `best_sol` here and offer it once, after
        // `solver.solve` returned — which made FJ the only worker of the
        // four whose publication cadence depends on a *gate* rather than on
        // its own search: LocalMIP offers from inside its step loop and
        // Scylla from inside its pump loop, while FPR's inner attempts end
        // on their own node limit regardless of budget.
        //
        // Two consequences, and the second is the reason this is a fix
        // rather than a tidy-up. It made FJ's own finds invisible to its
        // peers and to `fj::run`'s staleness rebuild (which seeds from the
        // pool) for the length of an attempt. And it made FJ's yield curve
        // unmeasurable whenever an attempt is long: #113's probe disables
        // every effort gate on purpose, so one attempt is the whole 30 s
        // dispatch, all 16 workers published within a millisecond of the
        // cap, and only 1 of 220 dispatches could be classified as having
        // finished improving. `status.totalEffort` is this worker's
        // cumulative charge, which is exactly what the trailing offer used
        // to reconstruct as `base_.total_effort + attempt_effort_consumed`.
        if (status.solution != nullptr) {
            best_sol.assign(status.solution, status.solution + status.numVars);
            const double obj = model->offset_ + (sense_multiplier * status.solutionObjectiveValue);
            if (sink_.offer(obj, best_sol, trace_, trace_.at(status.totalEffort))
                    .improved_incumbent) {
                improved_incumbent = true;
            }
        }

        // The solve's wall-clock deadline (issue #114).  Every other gate
        // here is denominated in effort units, and the runner checks the
        // clock only between attempts — where one attempt is
        // `attempt_cap` = `total / (10N)`, which scales with
        // `mip_heuristic_fj_effort`.  At the pre-#113 default of one vanilla
        // FJ budget per worker that is `nnz * 102` and the between-attempts
        // check is tight; at 80 of them it is `nnz * 8192` and FJ was
        // measured 1.4-2.0x past the limit, twice to an external SIGKILL.
        //
        // This costs one clock read per `CALLBACK_EFFORT` (500000) effort
        // units, which is upstream FJ's own callback cadence — not a
        // per-iteration read.  `past_deadline()` is the write-free half of
        // `ExecutionContext::terminated()` and needs no poller seat.
        //
        // Deliberately does *not* set `base_.finished`: `fj::run` rebuilds
        // a worker that reports `finished()`, and constructing a fresh
        // FeasibilityJumpSolver is the last thing to do at the deadline.
        // Returning is enough — the runner polls the same predicate on
        // every iteration and stops the loop.
        if (exec_.past_deadline()) {
            return CallbackControlFlow::Terminate;
        }
        // Pause at the attempt boundary.
        if (attempt_effort_consumed >= attempt_budget) {
            return CallbackControlFlow::Terminate;
        }
        // Total budget exceeded.
        if (status.totalEffort > base_.total_budget) {
            return CallbackControlFlow::Terminate;
        }
        // Stall detection.
        size_t esi = status.effortSinceLastImprovement;
        if (esi > base_.stale_budget) {
            return CallbackControlFlow::Terminate;
        }

        return CallbackControlFlow::Continue;
    };

    impl_->solver.solve(resume ? nullptr : impl_->col_value.data(), callback, resume);
    first_solve_done_ = true;

    AttemptResult result{};
    result.effort = attempt_effort_consumed;

    // An attempt counts as productive when it moved the incumbent, not
    // when upstream FJ hands back any solution at all, and not when the
    // pool merely kept one (#116) — FJ's own `effortSinceLastImprovement`
    // tracks the first, inside the solver, and that is the counter its
    // callback gate reads.  This one is the dispatch's, and since #163 it
    // is the disjunction over the offers the callback made: an attempt was
    // productive if *any* of its solutions moved the incumbent.  Every
    // solution reaches the sink through the callback — the discarded
    // return of `solver.solve` carries none — so there is nothing left to
    // offer here.
    const bool improved = improved_incumbent;
    if (improved) {
        result.found_improvement = true;
        base_.charge_improvement(attempt_effort_consumed);
    } else {
        base_.charge_no_improvement(attempt_effort_consumed);
    }

    return result;
}
