#include "fj_worker.h"

#include "fj_personas.h"
#include "heuristic_context.h"
#include "io/HighsIO.h"
#include "lp_data/HighsLp.h"
#include "mip/feasibilityjump.hh"
#include "solution_sink.h"
#include "structure.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
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

FjWorker::FjWorker(const ProblemView& problem, const ExecutionContext& exec, SolutionSink& sink,
                   size_t total_budget, size_t stale_budget, uint32_t seed,
                   std::vector<double> start, WorkerTrace trace, double cutoff,
                   std::span<const ExactKRow> exchange_rows)
    : problem_(problem),
      exec_(exec),
      sink_(sink),
      start_(std::move(start)),
      seed_(seed),
      cutoff_(cutoff),
      exchange_rows_(exchange_rows),
      trace_(trace) {
    base_.total_budget = total_budget;
    base_.stale_budget = stale_budget;
    // A model with no columns or no rows has nothing to search, and an
    // attempt on one would charge no effort, so nothing would ever end it
    // but the deadline.  The HiGHS runners never build a worker for one
    // (`ProblemView::degenerate()`); a caller of the core might.
    base_.finished = problem.degenerate();
}

FjWorker::~FjWorker() = default;

void FjWorker::reset(std::span<const double> start, uint32_t seed, size_t total_budget,
                     size_t stale_budget, WorkerTrace trace, double cutoff) {
    // Whether the model has a cutoff row is fixed at construction, so a reset
    // that would add or drop one is refused: the worker finishes.
    const bool cutoff_kind_changed = std::isfinite(cutoff) != std::isfinite(cutoff_);
    assert(!cutoff_kind_changed);
    start_.assign(start.begin(), start.end());
    seed_ = seed;
    if (!cutoff_kind_changed) {
        cutoff_ = cutoff;
    }
    trace_ = trace;
    best_found_.clear();
    best_found_objective_ = std::numeric_limits<double>::infinity();
    base_ = WorkerBudgetState{.total_budget = total_budget, .stale_budget = stale_budget};
    // A model whose build was refused stays refused.
    base_.finished = problem_.degenerate() || (initialized_ && !impl_) || cutoff_kind_changed;
    first_solve_done_ = false;
    if (impl_) {
        impl_->solver.resetWeights();
        impl_->solver.resetSearch(static_cast<int>(seed_));
    }
}

namespace {

// Column `col`'s bounds as FJ takes them: an integer column's rounded
// inward within `feastol`.
std::pair<double, double> fj_bounds(const ProblemView& problem, HighsInt col) {
    double lower = (*problem.col_lower)[col];
    double upper = (*problem.col_upper)[col];
    if ((*problem.integrality)[col] != HighsVarType::kContinuous) {
        lower = std::ceil(lower - problem.feastol);
        upper = std::floor(upper + problem.feastol);
    }
    return {lower, upper};
}

}  // namespace

void FjWorker::build_model() {
    impl_ = std::make_unique<Impl>(exec_.log_options, static_cast<int>(seed_), problem_.epsilon,
                                   problem_.feastol);
    impl_->col_value.resize(problem_.ncol, 0.0);
    if (!add_columns()) {
        impl_.reset();
        return;
    }
    add_rows();
    // The cutoff row goes in before the exchange groups, which size their
    // per-row scratch from the rows present.
    add_cutoff_row();
    add_exchange_groups();
}

bool FjWorker::add_columns() {
    // The view's minimisation form (#170), so FJ's own minimised objective
    // is the view's up to the offset, with no sense to correct for.
    const std::vector<double>& col_cost = *problem_.col_cost;
    const std::vector<HighsVarType>& integrality = *problem_.integrality;
    for (HighsInt col = 0; col < problem_.ncol; ++col) {
        const auto [lower, upper] = fj_bounds(problem_, col);
        const bool legal_bounds = lower <= upper && lower < kHighsInf && upper > -kHighsInf &&
                                  !std::isnan(lower) && !std::isnan(upper);
        if (!legal_bounds) {
            return false;
        }
        const VarType fj_var_type =
            integrality[col] == HighsVarType::kContinuous ? VarType::Continuous : VarType::Integer;
        impl_->solver.addVar(fj_var_type, lower, upper, col_cost[col]);
    }
    return true;
}

void FjWorker::add_rows() {
    // The view's row-wise matrix, not a transpose of its own (#316): both
    // are the same column-order counting sort of the column-wise matrix
    // (`highsSparseTranspose`, `HighsSparseMatrix::createRowwise`), so each
    // row's coefficients reach FJ in the same order (`test_fj_reuse.cpp`).
    const std::vector<double>& row_lower = *problem_.row_lower;
    const std::vector<double>& row_upper = *problem_.row_upper;
    const std::vector<HighsInt>& ar_start = *problem_.ar_start;
    const HighsInt* ar_index = problem_.ar_index->data();
    const double* ar_value = problem_.ar_value->data();
    for (HighsInt row = 0; row < problem_.nrow; ++row) {
        const HighsInt row_num_nz = ar_start[row + 1] - ar_start[row];
        const HighsInt* row_index = ar_index + ar_start[row];
        const double* row_value = ar_value + ar_start[row];
        if (std::isfinite(row_lower[row])) {
            impl_->solver.addConstraint(RowType::Gte, row_lower[row], row_num_nz, row_index,
                                        row_value, 0);
            fj_nonzeros_ += static_cast<size_t>(row_num_nz);
        }
        if (std::isfinite(row_upper[row])) {
            impl_->solver.addConstraint(RowType::Lte, row_upper[row], row_num_nz, row_index,
                                        row_value, 0);
            fj_nonzeros_ += static_cast<size_t>(row_num_nz);
        }
    }
}

void FjWorker::add_cutoff_row() {
    // Its RHS is set before every search that does not resume.
    if (!std::isfinite(cutoff_)) {
        return;
    }
    const std::vector<double>& col_cost = *problem_.col_cost;
    std::vector<HighsInt> cost_index;
    std::vector<double> cost_value;
    for (HighsInt col = 0; col < problem_.ncol; ++col) {
        if (col_cost[col] != 0.0) {
            cost_index.push_back(col);
            cost_value.push_back(col_cost[col]);
        }
    }
    if (!cost_index.empty()) {
        cutoff_row_ = impl_->solver.addConstraint(RowType::Lte, cutoff_ - problem_.offset,
                                                  static_cast<int>(cost_index.size()),
                                                  cost_index.data(), cost_value.data(), 0);
        fj_nonzeros_ += cost_index.size();
    }
}

void FjWorker::add_exchange_groups() {
    // The exchange groups (#218): each exact-k row's columns, all binary.
    // A row with `k = 0` or `k = width` has every column fixed by it, so
    // no pair: it would only take a draw.
    if (exchange_rows_.empty()) {
        return;
    }
    const std::vector<HighsInt>& ar_start = *problem_.ar_start;
    const std::vector<HighsInt>& index = *problem_.ar_index;
    const auto exchangeable = [&](const ExactKRow& r) {
        return r.k >= 1 && r.k < ar_start[r.row + 1] - ar_start[r.row];
    };
    size_t num_rows = 0;
    size_t num_members = 0;
    for (const ExactKRow& r : exchange_rows_) {
        if (exchangeable(r)) {
            ++num_rows;
            num_members += static_cast<size_t>(ar_start[r.row + 1] - ar_start[r.row]);
        }
    }
    std::vector<uint32_t> offsets{0};
    std::vector<uint32_t> members;
    offsets.reserve(num_rows + 1);
    members.reserve(num_members);
    for (const ExactKRow& r : exchange_rows_) {
        if (!exchangeable(r)) {
            continue;
        }
        for (HighsInt p = ar_start[r.row]; p < ar_start[r.row + 1]; ++p) {
            members.push_back(static_cast<uint32_t>(index[p]));
        }
        offsets.push_back(static_cast<uint32_t>(members.size()));
    }
    impl_->solver.setExchangeGroups(std::move(offsets), std::move(members));
}

void FjWorker::fill_start() {
    // The caller's resolved start, not `mipdata->incumbent`: a peer
    // worker's accepted solution rewrites the live vector while this loop
    // indexes it (issue #98).
    const bool use_start = !start_.empty();
    for (HighsInt col = 0; col < problem_.ncol; ++col) {
        const auto [lower, upper] = fj_bounds(problem_, col);
        impl_->col_value[col] = fj_clamped_start(lower, upper, use_start ? start_[col] : kHighsInf);
    }
}

// Cognitive complexity 29 (threshold 25).  Kept whole: one FeasibilityJump
// attempt, whose budget slicing and staleness detection are interleaved with
// upstream FJ's callback contract, which resumes the solver mid-run.  The
// model build is split out (`build_model`); what is left is one callback.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
AttemptResult FjWorker::run_attempt(size_t attempt_budget) {
    if (base_.finished) {
        return {};
    }

    // First attempt: build the solver.
    if (!initialized_) {
        initialized_ = true;
#ifdef HIGHSINT64
        base_.finished = true;
        return {};
#endif
        build_model();
    }

    if (!impl_) {
        base_.finished = true;
        return {};
    }

    // Capture state for the callback closure.
    const bool resume = first_solve_done_;
    if (!resume) {
        fill_start();
        if (cutoff_row_ >= 0) {
            impl_->solver.setConstraintRhs(cutoff_row_, cutoff_ - problem_.offset);
        }
    }
    size_t attempt_effort_consumed = 0;
    bool improved_incumbent = false;
    bool stalled = false;

    // A held worker's limits (`hold_to_budget`, #309): its budgets count
    // search past FJ's setup, which `init()` charges as two passes over the
    // model's nonzeros (with no continuous relaxation, FJ keeps every
    // coefficient it is given).  Unheld, the setup counts against them and
    // FJ keeps upstream's period, as before.
    const auto saturating_add = [](size_t a, size_t b) {
        return a > std::numeric_limits<size_t>::max() - b ? std::numeric_limits<size_t>::max()
                                                          : a + b;
    };
    const size_t setup = hold_to_budget_ ? 2 * fj_nonzeros_ : 0;
    const size_t attempt_limit = saturating_add(attempt_budget, resume ? 0 : setup);
    const size_t total_limit = saturating_add(base_.total_budget, setup);
    const auto hold_period = [&](size_t consumed) {
        if (hold_to_budget_) {
            impl_->solver.setCallbackEffort(std::min(kFjCallbackEffort, attempt_limit - consumed));
        }
    };
    hold_period(0);

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
            best_found_.assign(status.solution, status.solution + status.numVars);
            const double obj = problem_.offset + status.solutionObjectiveValue;
            best_found_objective_ = obj;
            if (sink_.offer(obj, best_found_, trace_, trace_.at(status.totalEffort))
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
        // This costs one clock read per callback period (upstream's 500000
        // unless `hold_to_budget` lowered it), FJ's own cadence — not a
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
        // Stall detection, and FJ's only patience (#191): FJ's own counter,
        // which restarts at a new fewest violated rows as well as at a
        // better feasible objective.  Ahead of the pause: an attempt is
        // usually shorter than `CALLBACK_EFFORT`, so the pause would
        // otherwise end every callback and the stall would never be seen.
        //
        // The counter is read only here, so the patience has a floor of one
        // callback period, 500000 effort units: a `stale_budget` below that
        // (`128 x nnz` under ~3900 nnz) stalls at the first callback at
        // which effort since the last improvement exceeds it, never sooner
        // than the first callback (#194).  Accepted, not fixed: HiGHS's own FJ
        // (`HighsFeasibilityJump.cpp`, `fjControlCallback`) reads its
        // `nnz << 8` patience at the same callback and has the same floor,
        // and for FJ HiGHS is the reference ahead of SCIP, whose feasjump
        // checks every move.
        size_t esi = status.effortSinceLastImprovement;
        if (esi > base_.stale_budget) {
            stalled = true;
            return CallbackControlFlow::Terminate;
        }

        // Pause at the attempt boundary.
        if (attempt_effort_consumed >= attempt_limit) {
            return CallbackControlFlow::Terminate;
        }
        // Total budget exceeded.
        if (status.totalEffort > total_limit) {
            return CallbackControlFlow::Terminate;
        }
        // Upstream restarts its period at every callback, a solution's too,
        // so a held worker's next one is set to what is left of the attempt.
        hold_period(attempt_effort_consumed);
        return CallbackControlFlow::Continue;
    };

    // A return the callback did not ask for is FJ ending its own search:
    // since v1.16.0 `solve()` breaks when `selectVariable()` finds no move
    // (every sampled score `-inf`).  That can be the luck of one draw — the
    // local-minimum branch samples a single violated row — but upstream
    // ends its FJ run there, and so does this worker: it retires like a
    // stall, for `fj::run` to rebuild.  The attempt is charged from the
    // solver's own effort, since no callback may have fired in it.
    bool callback_stopped = false;
    const std::function<CallbackControlFlow(FJStatus)> observed = [&](FJStatus status) {
        const CallbackControlFlow flow = callback(status);
        callback_stopped = flow == CallbackControlFlow::Terminate;
        return flow;
    };
    impl_->solver.solve(resume ? nullptr : impl_->col_value.data(), observed, resume);
    first_solve_done_ = true;
    attempt_effort_consumed = impl_->solver.effort() - base_.total_effort;
    if (!callback_stopped) {
        base_.finished = true;
    }

    AttemptResult result{};
    result.effort = attempt_effort_consumed;

    // An attempt counts as productive when it moved the incumbent, not
    // when upstream FJ hands back any solution at all, and not when the
    // pool merely kept one (#116).  That is the dispatch's signal, for the
    // runner-level gate, and since #163 it is the disjunction over the
    // offers the callback made: an attempt was productive if *any* of its
    // solutions moved the incumbent.  Every solution reaches the sink
    // through the callback — the discarded return of `solver.solve`
    // carries none — so there is nothing left to offer here.
    result.found_improvement = improved_incumbent;

    // The worker itself retires on FJ's own stall, not on effort since the
    // incumbent moved (#191): before a first feasible solution the latter
    // grows however fast the violations fall, and retired FJ at
    // `worker_stale` on every bundled MIP at SCIP's `128 x nnz`.  So only
    // the total half of `WorkerBudgetState` is charged here.
    base_.total_effort += attempt_effort_consumed;
    if (stalled || base_.total_effort >= total_limit) {
        base_.finished = true;
    }

    return result;
}
