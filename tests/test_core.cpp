// The heuristic core on its own (#170): FJ and LocalMIP run on a model the
// caller owns — read with `Highs::readModel`, with no presolve and no
// `HighsMipSolver` anywhere — through `mip_heuristics_core` alone.  This
// file is `mip_heuristics_core_tests`, which links that target and nothing
// else of ours and includes only core headers.  The hand-built models are
// in test_core_standalone.cpp, which links against a libhighs without the
// MIP solver.  The `[until_stopped]` cases drive the same workers through
// `run_until_stopped` (#171), on a thread the test owns.

#include "core_test_support.h"
#include "fj.h"
#include "heuristic_context.h"
#include "Highs.h"
#include "local_mip.h"

#include <atomic>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <thread>
#include <vector>

using namespace core_test;

namespace {

const std::string kInstancesDir = INSTANCES_DIR;

void read(Highs& highs, const char* instance) {
    highs.setOptionValue("output_flag", false);
    REQUIRE(highs.readModel(kInstancesDir + "/" + instance) == HighsStatus::kOk);
}

}  // namespace

TEST_CASE("core: the view over an original model matches the model", "[core]") {
    Highs highs;
    read(highs, "egout.mps");
    const HighsLp& lp = highs.getLp();
    const Setup s(lp, highs.getOptions().log_options);
    REQUIRE(s.problem.model == &lp);
    REQUIRE(s.problem.col_cost == &lp.col_cost_);  // minimisation: no copy
    REQUIRE(s.problem.offset == lp.offset_);
    REQUIRE(s.problem.integrality == &lp.integrality_);
    REQUIRE(s.problem.ncol == lp.num_col_);
    REQUIRE(s.problem.nrow == lp.num_row_);
    REQUIRE(s.problem.nnz == lp.a_matrix_.index_.size());
    REQUIRE(s.problem.ar_start->size() == static_cast<size_t>(lp.num_row_) + 1);
    REQUIRE(s.problem.csc->col_start.size() == static_cast<size_t>(lp.num_col_) + 1);
    REQUIRE(s.problem.uplocks->size() == static_cast<size_t>(lp.num_col_));
    REQUIRE(s.problem.binary.size() == static_cast<size_t>(lp.num_col_));
    REQUIRE(s.problem.incumbent.empty());
    REQUIRE_FALSE(s.problem.degenerate());
}

TEST_CASE("core: FJ and LocalMIP find feasible solutions on an original model", "[core]") {
    Highs highs;
    read(highs, "egout.mps");
    const HighsLp& lp = highs.getLp();
    const Setup s(lp, highs.getOptions().log_options);
    Found found(*s.problem.integrality);
    run_both(s, found);

    for (BestSink* sink : {&found.fj, &found.local_mip}) {
        REQUIRE(sink->found);
        REQUIRE(is_feasible(lp, sink->solution));
        // Minimisation: the reported objective is the model's own.
        REQUIRE(sink->objective == Catch::Approx(original_objective(lp, sink->solution)));
    }
}

TEST_CASE("core: make_problem refuses semi-continuous and semi-integer columns", "[core]") {
    for (const char* instance : {"semi-continuous.mps", "semi-integer.mps"}) {
        INFO(instance);
        Highs highs;
        read(highs, instance);
        ProblemStorage storage;
        const auto made = make_problem(highs.getLp(), storage, kFeasTol, kEpsilon);
        REQUIRE_FALSE(made.has_value());
        REQUIRE(made.error().kind == ProblemError::Kind::kUnsupported);
    }
}

TEST_CASE("core: a model without integrality is all continuous", "[core]") {
    Highs highs;
    read(highs, "afiro.mps");
    const HighsLp& lp = highs.getLp();
    REQUIRE(lp.integrality_.empty());  // the premise: a pure LP carries none
    ProblemStorage storage;
    const auto made = make_problem(lp, storage, kFeasTol, kEpsilon);
    REQUIRE(made.has_value());
    const ProblemView& problem = *made;
    REQUIRE(problem.integrality->size() == static_cast<size_t>(lp.num_col_));
    for (HighsInt j = 0; j < lp.num_col_; ++j) {
        REQUIRE((*problem.integrality)[j] == HighsVarType::kContinuous);
        REQUIRE(problem.binary[j] == 0);
    }
}

namespace {

// A sink that holds its worker at the first solution it accepts until the
// caller has set the stop flag, so the stop lands at a known point of the
// run: the effort the offer was made at.  `phase` tells the caller's
// thread where the worker is — `kHolding` at that solution, `kReturned`
// once the run is over, which also ends the wait if nothing is ever found.
class HoldingSink : public SolutionSink {
public:
    static constexpr int kRunning = 0;
    static constexpr int kHolding = 1;
    static constexpr int kReturned = 2;

    HoldingSink(const std::vector<HighsVarType>& integrality, const std::atomic<bool>& stop)
        : SolutionSink(integrality, 0), stop_(stop) {}

    std::atomic<int> phase{kRunning};
    // Written by the worker before it publishes `kHolding`.
    size_t effort_at_stop = 0;
    std::vector<double> first;

    void returned() {
        phase.store(kReturned);
        phase.notify_all();
    }

private:
    void on_accept(double /*obj*/, const std::vector<double>& x, int /*source*/,
                   const WorkerTrace& /*trace*/, size_t effort_at) override {
        if (!first.empty()) {
            return;
        }
        first = x;
        effort_at_stop = effort_at;
        phase.store(kHolding);
        phase.notify_all();
        stop_.wait(false);
    }

    const std::atomic<bool>& stop_;
};

// `run` on a thread of the test's own: wait for the sink to hold a
// solution, set the stop flag, join.  Returns the run's effort.
template <typename Run>
size_t stop_at_first_solution(HoldingSink& sink, std::atomic<bool>& stop, Run run,
                              int64_t* polls_at_stop = nullptr) {
    size_t effort = 0;
    std::thread worker([&] {
        effort = run();
        sink.returned();
    });
    sink.phase.wait(HoldingSink::kRunning);
    if (polls_at_stop != nullptr) {
        *polls_at_stop = local_mip::deadline_poll_counters().polls;
    }
    stop.store(true);
    stop.notify_all();
    worker.join();
    return effort;
}

// One attempt as long as the whole run, and a patience that never fires:
// nothing inside the run can end the attempt the stop lands in, so only a
// poll of the stop flag inside it can.  The total is a fuse, so a run that
// never finds anything ends and fails rather than hangs.
HeuristicBudget one_long_attempt() {
    HeuristicBudget budget = make_until_stopped_budget(kBudget, kBudget);
    budget.total = kBudget;
    return budget;
}

}  // namespace

TEST_CASE("core: LocalMIP on a caller's thread stops at its next poll after the stop flag",
          "[core][until_stopped]") {
    Highs highs;
    read(highs, "egout.mps");
    const HighsLp& lp = highs.getLp();
    const Setup s(lp, highs.getOptions().log_options);
    std::atomic<bool> stop{false};
    ExecutionContext exec = s.exec;
    exec.stop = &stop;
    HoldingSink sink(*s.problem.integrality, stop);
    const HeuristicBudget budget = one_long_attempt();

    local_mip::reset_deadline_poll_counters();
    int64_t polls_at_stop = 0;
    const size_t effort = stop_at_first_solution(
        sink, stop,
        [&] {
            return local_mip::run_until_stopped(s.problem, budget, exec, /*worker=*/0,
                                                RestartSource{}, sink);
        },
        &polls_at_stop);

    REQUIRE(sink.phase.load() == HoldingSink::kReturned);
    REQUIRE_FALSE(sink.first.empty());
    REQUIRE(is_feasible(lp, sink.first));
    // The mechanism: the worker polls `past_deadline()` at least every
    // `kTermCheckWork` counted units or steps (#162), and the first poll
    // after the stop ends the attempt, so at most one poll follows it — at
    // most one poll interval of work.  The count is live only in an
    // instrumented build, which the test targets are.
    REQUIRE(polls_at_stop > 0);
    REQUIRE(local_mip::deadline_poll_counters().polls - polls_at_stop <= 1);
    // And it was that poll, not the attempt's cap, that ended the run.
    REQUIRE(effort >= sink.effort_at_stop);
    REQUIRE(effort < budget.attempt_cap);
}

TEST_CASE("core: FJ on a caller's thread stops in the callback that saw the stop flag",
          "[core][until_stopped]") {
    Highs highs;
    read(highs, "egout.mps");
    const HighsLp& lp = highs.getLp();
    const Setup s(lp, highs.getOptions().log_options);
    std::atomic<bool> stop{false};
    ExecutionContext exec = s.exec;
    exec.stop = &stop;
    HoldingSink sink(*s.problem.integrality, stop);
    const HeuristicBudget budget = one_long_attempt();

    const size_t effort = stop_at_first_solution(sink, stop, [&] {
        return fj::run_until_stopped(s.problem, budget, exec, /*worker=*/0, RestartSource{}, sink);
    });

    REQUIRE(sink.phase.load() == HoldingSink::kReturned);
    REQUIRE_FALSE(sink.first.empty());
    REQUIRE(is_feasible(lp, sink.first));
    // FJ offers from inside its callback and polls the flag right after
    // the offer returns, so the stop ends the attempt at the very effort
    // the solution arrived at: nothing is charged after it.
    REQUIRE(effort == sink.effort_at_stop);
}

TEST_CASE("core: a stalled LocalMIP worker restarts from the caller's source",
          "[core][until_stopped]") {
    // minimise x0 + x1 + x2  s.t.  x0 + x1 >= 1,  x1 + x2 >= 1,  x in [0, 10].
    // All continuous, so a restart's perturbation, which moves integer
    // columns only, leaves the source's point as it is, and the rebuilt
    // worker's first step offers exactly that point: it is feasible, and a
    // fresh worker has no best of its own to beat.
    HighsLp lp;
    lp.num_col_ = 3;
    lp.num_row_ = 2;
    lp.col_cost_ = {1.0, 1.0, 1.0};
    lp.col_lower_ = {0.0, 0.0, 0.0};
    lp.col_upper_ = {10.0, 10.0, 10.0};
    lp.row_lower_ = {1.0, 1.0};
    lp.row_upper_ = {kHighsInf, kHighsInf};
    lp.a_matrix_.format_ = MatrixFormat::kColwise;
    lp.a_matrix_.num_col_ = lp.num_col_;
    lp.a_matrix_.num_row_ = lp.num_row_;
    lp.a_matrix_.start_ = {0, 1, 3, 4};
    lp.a_matrix_.index_ = {0, 0, 1, 1};
    lp.a_matrix_.value_ = {1.0, 1.0, 1.0, 1.0};
    // Feasible, interior and dyadic: no move LocalMIP makes lands on it, and
    // it compares exactly.
    const std::vector<double> point = {0.375, 0.8125, 0.5};
    REQUIRE(is_feasible(lp, point));

    Highs highs;  // for its log options only
    highs.setOptionValue("output_flag", false);
    const Setup s(lp, highs.getOptions().log_options);
    std::atomic<bool> stop{false};
    ExecutionContext exec = s.exec;
    exec.stop = &stop;

    // The source declines the first start, so the slot's first worker
    // constructs its own, and serves `point` to every rebuild after it.
    int source_calls = 0;
    const RestartSource source = [&](Rng& /*rng*/, std::vector<double>& out) {
        if (++source_calls == 1) {
            return false;
        }
        out = point;
        return true;
    };

    struct RecordingSink : SolutionSink {
        RecordingSink(const std::vector<HighsVarType>& integrality, const int& source_calls,
                      std::atomic<bool>& stop)
            : SolutionSink(integrality, 0), source_calls_(source_calls), stop_(stop) {}
        std::vector<std::vector<double>> before_restart;
        std::vector<double> after_restart;

    private:
        void on_accept(double /*obj*/, const std::vector<double>& x, int /*source*/,
                       const WorkerTrace& /*trace*/, size_t /*effort_at*/) override {
            if (source_calls_ < 2) {
                before_restart.push_back(x);
            } else if (after_restart.empty()) {
                after_restart = x;
                stop_.store(true);
            }
        }
        const int& source_calls_;
        std::atomic<bool>& stop_;
    } sink(*s.problem.integrality, source_calls, stop);

    // A patience of one unit, so the first worker stalls as soon as it
    // stops improving, and no ceiling on a worker: the rebuild can only be
    // the stall's.  The total is a fuse.
    HeuristicBudget budget = make_until_stopped_budget(kAttemptBudget, 1);
    budget.total = kBudget;
    REQUIRE(budget.per_worker == std::numeric_limits<size_t>::max());

    const size_t effort =
        local_mip::run_until_stopped(s.problem, budget, exec, /*worker=*/0, source, sink);

    REQUIRE(effort > 0);
    REQUIRE(source_calls >= 2);
    REQUIRE(sink.after_restart == point);
    for (const std::vector<double>& x : sink.before_restart) {
        REQUIRE(x != point);
    }
}
