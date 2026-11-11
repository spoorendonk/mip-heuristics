// The bound-prop recombiner (#227) in the heuristic core, on a model the
// caller owns: its adaptive limit, the cases with no offspring, and an
// offspring of two FJ points.

#include "core_test_support.h"
#include "deadline.h"
#include "fj_worker.h"
#include "heuristic_common.h"
#include "heuristic_context.h"
#include "Highs.h"
#include "recombine.h"
#include "rng.h"
#include "ruin_repair.h"
#include "solution_sink.h"
#include "test_paths.h"
#include "util/HighsTimer.h"

#include <algorithm>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

using namespace core_test;

namespace {

void read(Highs& highs, const char* instance) {
    highs.setOptionValue("output_flag", false);
    REQUIRE(highs.readModel(instances_dir() + "/" + instance) == HighsStatus::kOk);
}

// FJ's first feasible point on `s` from `seed`.
std::vector<double> fj_point(const Setup& s, uint64_t seed) {
    BestSink sink(*s.problem.integrality, 0);
    FjWorker fj(s.problem, s.exec, sink, kBudget, kBudget, seed, /*start=*/{}, WorkerTrace{0, 0});
    run_until_found(fj, sink);
    REQUIRE(sink.found);
    return sink.solution;
}

}  // namespace

TEST_CASE("recombine: the limit on free columns follows cuOpt's adaptation", "[recombine]") {
    using recombine::next_max_vars_from_other;
    // An offspring equal to a parent grows it by 10%, rounded up as cuOpt
    // rounds it: `200 * 1.1` is a hair above 220 in double.
    CHECK(next_max_vars_from_other(200, true) == 221);
    CHECK(next_max_vars_from_other(21, true) == 24);
    // At most kBpMaxVarsFromOther.
    CHECK(next_max_vars_from_other(9500, true) == 10000);
    CHECK(next_max_vars_from_other(10000, true) == 10000);
    // Any other shrinks it by 1%, rounded down.
    CHECK(next_max_vars_from_other(200, false) == 198);
    CHECK(next_max_vars_from_other(1000, false) == 990);
    // At least kBpMinVarsFromOther.
    CHECK(next_max_vars_from_other(21, false) == 20);
    CHECK(next_max_vars_from_other(20, false) == 20);
}

TEST_CASE("recombine: no offspring with nothing free or nothing fixed", "[recombine]") {
    Highs highs;
    read(highs, "lseu.mps");
    const HighsLp& lp = highs.getLp();
    const Setup s(lp, highs.getOptions().log_options);
    recombine::BoundPropRecombiner recombiner(s.problem);
    const std::vector<double> a = fj_point(s, 0);
    const double obj_a = s.problem.objective(a);
    Rng rng(1);
    std::vector<double> out = {1.0};

    SECTION("identical parents leave no column free") {
        const recombine::Offspring o =
            recombiner.recombine(a, obj_a, a, obj_a, kBudget, Deadline{}, rng, out);
        CHECK_FALSE(o.success);
        CHECK_FALSE(o.feasible);
        CHECK(out.empty());
    }
    SECTION("parents differing on every integer, within the limit, leave none fixed") {
        // lseu's integers are all binaries, fewer than the initial limit.
        size_t num_integer = 0;
        std::vector<double> b = a;
        for (HighsInt j = 0; j < lp.num_col_; ++j) {
            if (is_integer(lp.integrality_, j)) {
                b[j] = 1.0 - std::round(a[j]);
                ++num_integer;
            }
        }
        REQUIRE(num_integer > 0);
        REQUIRE(num_integer <= recombine::kBpInitialVarsFromOther);
        const recombine::Offspring o = recombiner.recombine(a, obj_a, b, s.problem.objective(b),
                                                            kBudget, Deadline{}, rng, out);
        CHECK_FALSE(o.success);
        CHECK(out.empty());
    }
    CHECK(recombiner.max_vars_from_other() == recombine::kBpInitialVarsFromOther);
}

TEST_CASE("recombine: an offspring of two FJ points keeps what they agree on", "[recombine]") {
    Highs highs;
    read(highs, "p0548.mps");
    const HighsLp& lp = highs.getLp();
    const Setup s(lp, highs.getOptions().log_options);
    const std::vector<double> a = fj_point(s, 0);
    const std::vector<double> b = fj_point(s, 1);
    REQUIRE(a != b);
    const double obj_a = s.problem.objective(a);
    const double obj_b = s.problem.objective(b);

    recombine::BoundPropRecombiner recombiner(s.problem);
    size_t num_feasible = 0;
    std::vector<double> out;
    for (uint64_t seed = 0; seed < 8; ++seed) {
        Rng rng(seed);
        const recombine::Offspring o =
            recombiner.recombine(a, obj_a, b, obj_b, kBudget, Deadline{}, rng, out);
        REQUIRE(out.size() == static_cast<size_t>(lp.num_col_));
        CHECK(o.effort > 0);
        for (HighsInt j = 0; j < lp.num_col_; ++j) {
            CHECK(out[j] >= lp.col_lower_[j] - kFeasTol);
            CHECK(out[j] <= lp.col_upper_[j] + kFeasTol);
            if (is_integer(lp.integrality_, j)) {
                CHECK(out[j] == std::round(out[j]));
                if (std::round(a[j]) == std::round(b[j])) {
                    CHECK(out[j] == std::round(a[j]));
                }
            }
        }
        if (o.feasible) {
            ++num_feasible;
            CHECK(is_feasible(lp, out));
            CHECK(std::abs(o.objective - s.problem.objective(out)) <=
                  1e-6 * (1 + std::abs(o.objective)));
        }
    }
    CHECK(num_feasible > 0);
}

TEST_CASE("recombine: the same seed gives the same offspring", "[recombine]") {
    Highs highs;
    read(highs, "p0548.mps");
    const HighsLp& lp = highs.getLp();
    const Setup s(lp, highs.getOptions().log_options);
    const std::vector<double> a = fj_point(s, 0);
    const std::vector<double> b = fj_point(s, 1);
    REQUIRE(a != b);
    const double obj_a = s.problem.objective(a);
    const double obj_b = s.problem.objective(b);

    // A recombiner that has already recombined another pair, with the
    // limit unchanged, gives what a fresh one gives: nothing of that call's
    // bounds or free set carries over.
    const std::vector<double> c = fj_point(s, 2);
    REQUIRE(c != a);
    std::vector<double> first;
    std::vector<double> second;
    {
        recombine::BoundPropRecombiner recombiner(s.problem);
        Rng rng(3);
        static_cast<void>(
            recombiner.recombine(a, obj_a, b, obj_b, kBudget, Deadline{}, rng, first));
    }
    {
        recombine::BoundPropRecombiner recombiner(s.problem);
        Rng other_rng(5);
        static_cast<void>(recombiner.recombine(c, s.problem.objective(c), a, obj_a, kBudget,
                                               Deadline{}, other_rng, second));
        REQUIRE(recombiner.max_vars_from_other() == recombine::kBpInitialVarsFromOther);
        Rng rng(3);
        static_cast<void>(
            recombiner.recombine(a, obj_a, b, obj_b, kBudget, Deadline{}, rng, second));
    }
    REQUIRE_FALSE(first.empty());
    CHECK(first == second);
}

TEST_CASE("recombine: past the limit the rest differ only where left free, and the limit adapts",
          "[recombine]") {
    Highs highs;
    read(highs, "p0548.mps");
    const HighsLp& lp = highs.getLp();
    const Setup s(lp, highs.getOptions().log_options);
    const std::vector<double> a = fj_point(s, 0);
    // Every binary flipped: more differing columns than the limit allows.
    std::vector<double> b = a;
    size_t num_integer = 0;
    for (HighsInt j = 0; j < lp.num_col_; ++j) {
        if (is_integer(lp.integrality_, j)) {
            REQUIRE(lp.col_lower_[j] == 0.0);
            REQUIRE(lp.col_upper_[j] == 1.0);
            b[j] = 1.0 - std::round(a[j]);
            ++num_integer;
        }
    }
    REQUIRE(num_integer > recombine::kBpInitialVarsFromOther);

    const double obj_a = s.problem.objective(a);
    const double obj_b = s.problem.objective(b);
    REQUIRE(obj_b > obj_a);

    // Whichever is passed first guides, whatever the objectives: the worse
    // one too.  The integer columns differ from the guiding parent's only
    // where left free, and the limit moves the way the offspring says.
    const auto check = [&](const std::vector<double>& guiding, double obj_guiding,
                           const std::vector<double>& other, double obj_other) {
        recombine::BoundPropRecombiner recombiner(s.problem);
        Rng rng(0);
        std::vector<double> out;
        static_cast<void>(recombiner.recombine(guiding, obj_guiding, other, obj_other, kBudget,
                                               Deadline{}, rng, out));
        REQUIRE(out.size() == static_cast<size_t>(lp.num_col_));
        size_t moved = 0;
        bool same_as_guiding = true;
        bool same_as_other = true;
        for (HighsInt j = 0; j < lp.num_col_; ++j) {
            if (is_integer(lp.integrality_, j)) {
                moved += out[j] != std::round(guiding[j]) ? 1 : 0;
                same_as_guiding = same_as_guiding && out[j] == std::round(guiding[j]);
                same_as_other = same_as_other && out[j] == std::round(other[j]);
            }
        }
        CHECK(moved <= recombine::kBpInitialVarsFromOther);
        CHECK(recombiner.max_vars_from_other() ==
              recombine::next_max_vars_from_other(recombine::kBpInitialVarsFromOther,
                                                  same_as_guiding || same_as_other));
    };
    check(a, obj_a, b, obj_b);
    check(b, obj_b, a, obj_a);
}

TEST_CASE("recombine: the bandit pulls every arm once, in order, first", "[recombine]") {
    using recombine::RecombinerArm;
    recombine::RecombinerBandit bandit(7);
    // An arm counts as pulled once rewarded, as in cuOpt.
    CHECK(bandit.select() == RecombinerArm::kBoundProp);
    CHECK(bandit.select() == RecombinerArm::kBoundProp);
    bandit.reward(RecombinerArm::kBoundProp, 0.0);
    CHECK(bandit.select() == RecombinerArm::kGuidedCrossover);
    bandit.reward(RecombinerArm::kGuidedCrossover, 0.0);
    CHECK(bandit.select() == RecombinerArm::kNeighborhoodCrossover);
    bandit.reward(RecombinerArm::kNeighborhoodCrossover, 0.0);
    for (const auto arm : {RecombinerArm::kBoundProp, RecombinerArm::kGuidedCrossover,
                           RecombinerArm::kNeighborhoodCrossover}) {
        CHECK(bandit.pulls(arm) == 1);
    }
}

TEST_CASE("recombine: with equal pulls, UCB picks the arm of larger q", "[recombine]") {
    using recombine::RecombinerArm;
    for (const auto winner : {RecombinerArm::kBoundProp, RecombinerArm::kGuidedCrossover,
                              RecombinerArm::kNeighborhoodCrossover}) {
        recombine::RecombinerBandit bandit(11);
        for (int i = 0; i < 3; ++i) {
            const RecombinerArm arm = bandit.select();
            bandit.reward(arm, arm == winner ? recombine::kRewardBeatsBest : 0.0);
        }
        CHECK(bandit.q(winner) == 0.5 + (recombine::kBanditAlpha * (8.0 - 0.5)));
        CHECK(bandit.select() == winner);
    }
}

TEST_CASE("recombine: UCB breaks ties at random, the same way from the same seed", "[recombine]") {
    using recombine::RecombinerArm;
    // Every reward equals the initial q, so q never moves and the UCB bonus
    // alone decides: the first pick of each round of three (after the
    // initial pulls) is a three-way tie, the second a two-way tie, the third
    // forced.
    auto sequence = [](uint64_t seed) {
        recombine::RecombinerBandit bandit(seed);
        std::vector<RecombinerArm> arms;
        for (int i = 0; i < 60; ++i) {
            const RecombinerArm arm = bandit.select();
            arms.push_back(arm);
            bandit.reward(arm, recombine::kBanditInitialQ);
        }
        return arms;
    };
    const std::vector<RecombinerArm> first = sequence(3);
    CHECK(first == sequence(3));
    // The three-way ties do not all go one way, as a constant pick (say,
    // the lowest index) would make them.
    std::vector<size_t> drawn(recombine::kNumRecombinerArms, 0);
    for (size_t i = recombine::kNumRecombinerArms; i < first.size();
         i += recombine::kNumRecombinerArms) {
        ++drawn[static_cast<size_t>(first[i])];
    }
    for (const size_t count : drawn) {
        CHECK(count > 0);
    }
}

TEST_CASE("recombine: the reward is cuOpt's, normalised by effort", "[recombine]") {
    using recombine::recombination_reward;
    // Beats the best known by more than the margin: 8, whatever the work.
    CHECK(recombination_reward(110.0, 100.0, 99.0, 0) == 8.0);
    CHECK(recombination_reward(110.0, 100.0, 99.0, 4'000'000'000) == 8.0);
    // Within the margin (1e-4 * 100 = 0.01) of the best is not better.
    CHECK(recombination_reward(110.0, 100.0, 99.995, 0) == 4.0);
    // Beats the better parent, within 20% of the best: 4 - effort / 4e8.
    CHECK(recombination_reward(110.0, 100.0, 105.0, 0) == 4.0);
    CHECK(recombination_reward(110.0, 100.0, 105.0, 400'000'000) == 3.0);
    // Floored at 0.1.
    CHECK(recombination_reward(110.0, 100.0, 105.0, 4'000'000'000) == 0.1);
    // Beats the better parent, but 20% or more from the best: 0.
    CHECK(recombination_reward(200.0, 100.0, 150.0, 0) == 0.0);
    // Does not beat the better parent: 0.
    CHECK(recombination_reward(105.0, 100.0, 105.0, 0) == 0.0);
    CHECK(recombination_reward(105.0, 100.0, 120.0, 0) == 0.0);
    // A failed recombination: 0.
    CHECK(recombination_reward(105.0, 100.0, kHighsInf, 0) == 0.0);
    // Near zero, the margin is the absolute 1e-6.
    CHECK(recombination_reward(1.0, 0.0, -2e-6, 0) == 8.0);
    CHECK(recombination_reward(1.0, 0.0, -0.5e-6, 0) == 4.0);
}

namespace {

// Every accepted offer, in order.
class RecordingSink : public SolutionSink {
public:
    using SolutionSink::SolutionSink;

    std::mutex mtx;
    std::vector<std::pair<double, std::vector<double>>> accepted_points;
    std::vector<size_t> accepted_efforts;

protected:
    void on_accept(double obj, const std::vector<double>& x, int /*source*/,
                   const WorkerTrace& /*trace*/, size_t effort_at) override {
        std::scoped_lock guard(mtx);
        accepted_points.emplace_back(obj, x);
        accepted_efforts.push_back(effort_at);
    }
};

struct OperatorRun {
    std::vector<recombine::RecombinerArm> arms;
    std::vector<recombine::RecombineStep> steps;
    std::vector<double> accepted;
    std::vector<std::vector<double>> accepted_points;
    size_t seeded = 0;
};

// `steps` operator steps on p0548 from a pool of four FJ points.
OperatorRun run_operator(const Setup& s, uint64_t seed, size_t steps) {
    RecordingSink sink(*s.problem.integrality, 0);
    for (uint64_t k = 0; k < 4; ++k) {
        const std::vector<double> x = fj_point(s, k);
        static_cast<void>(sink.offer(s.problem.objective(x), x, WorkerTrace{0, 0}, 0));
    }
    OperatorRun run;
    run.seeded = sink.accepted_points.size();
    REQUIRE(run.seeded >= 2);
    recombine::RecombineOperator op(s.problem, s.exec, sink, seed, WorkerTrace{0, 0});
    const Deadline deadline;
    for (size_t k = 0; k < steps; ++k) {
        const std::optional<recombine::RecombineStep> step =
            op.step(1'000'000, 1'000'000, deadline);
        if (!step.has_value()) {
            FAIL("no step on a pool of at least two");
            break;
        }
        run.arms.push_back(step->arm);
        run.steps.push_back(*step);
    }
    size_t pulls = 0;
    for (size_t a = 0; a < recombine::kNumRecombinerArms; ++a) {
        pulls += op.bandit().pulls(static_cast<recombine::RecombinerArm>(a));
    }
    CHECK(pulls == steps);
    for (size_t k = run.seeded; k < sink.accepted_points.size(); ++k) {
        const auto& [obj, x] = sink.accepted_points[k];
        CHECK(ruin_repair::is_feasible(s.problem, x));
        CHECK(std::abs(obj - s.problem.objective(x)) <= 1e-6 * (1.0 + std::abs(obj)));
        run.accepted.push_back(obj);
        run.accepted_points.push_back(x);
        // The slot's charged effort only rises, the raw offspring's offers
        // and FJ's alike.
        if (k > run.seeded) {
            CHECK(sink.accepted_efforts[k] >= sink.accepted_efforts[k - 1]);
        }
    }
    return run;
}

}  // namespace

TEST_CASE("recombine: the operator offers feasible points, and replays from its seed",
          "[recombine]") {
    Highs highs;
    read(highs, "p0548.mps");
    const Setup s(highs.getLp(), highs.getOptions().log_options);
    const OperatorRun first = run_operator(s, 5, 30);
    CHECK_FALSE(first.accepted.empty());
    // Every arm gets its initial pull.
    for (const auto arm :
         {recombine::RecombinerArm::kBoundProp, recombine::RecombinerArm::kGuidedCrossover,
          recombine::RecombinerArm::kNeighborhoodCrossover}) {
        CHECK(std::ranges::count(first.arms, arm) > 0);
    }
    const OperatorRun again = run_operator(s, 5, 30);
    CHECK(again.arms == first.arms);
    CHECK(again.accepted == first.accepted);
    CHECK(again.accepted_points == first.accepted_points);
}

TEST_CASE("recombine: past the deadline the operator builds no FJ and rewards 0", "[recombine]") {
    Highs highs;
    read(highs, "p0548.mps");
    const Setup s(highs.getLp(), highs.getOptions().log_options);
    RecordingSink sink(*s.problem.integrality, 0);
    for (uint64_t k = 0; k < 4; ++k) {
        const std::vector<double> x = fj_point(s, k);
        static_cast<void>(sink.offer(s.problem.objective(x), x, WorkerTrace{0, 0}, 0));
    }
    std::atomic<bool> stop{true};
    ExecutionContext exec = s.exec;
    exec.stop = &stop;
    recombine::RecombineOperator op(s.problem, exec, sink, 5, WorkerTrace{0, 0});
    // The initial pulls: bound-prop, then the two crossovers, whose effort is
    // their setup and the feasibility check, with nothing left for an FJ.
    const auto ncol = static_cast<size_t>(s.problem.ncol);
    for (size_t k = 0; k < recombine::kNumRecombinerArms; ++k) {
        const std::optional<recombine::RecombineStep> step = op.step(kBudget, kBudget, Deadline{});
        if (!step.has_value()) {
            FAIL("no step on a pool of at least two");
            break;
        }
        CHECK(step->reward == 0.0);
        if (step->arm != recombine::RecombinerArm::kBoundProp) {
            CHECK(step->effort <= ncol + s.problem.nnz);
        }
    }
}

TEST_CASE("recombine: the operator takes no step on a pool of one", "[recombine]") {
    Highs highs;
    read(highs, "p0548.mps");
    const Setup s(highs.getLp(), highs.getOptions().log_options);
    RecordingSink sink(*s.problem.integrality, 0);
    recombine::RecombineOperator op(s.problem, s.exec, sink, 1, WorkerTrace{0, 0});
    const Deadline deadline;
    CHECK_FALSE(op.step(1000, 1000, deadline).has_value());
    const std::vector<double> x = fj_point(s, 0);
    static_cast<void>(sink.offer(s.problem.objective(x), x, WorkerTrace{0, 0}, 0));
    CHECK_FALSE(op.step(1000, 1000, deadline).has_value());
    CHECK(op.bandit().pulls(recombine::RecombinerArm::kBoundProp) == 0);
}

TEST_CASE("recombine: the local search's budget is cuOpt's seconds at FJ's rate", "[recombine]") {
    CHECK(recombine::local_search_budget(1000, true) == 1000 + 200'000'000);
    CHECK(recombine::local_search_budget(1000, false) == 1000 + 100'000'000);
}

namespace {

// A pool of FJ points on `s`, at least two.
void fill_pool(const Setup& s, SolutionSink& sink) {
    for (uint64_t k = 0; k < 4; ++k) {
        const std::vector<double> x = fj_point(s, k);
        static_cast<void>(sink.offer(s.problem.objective(x), x, WorkerTrace{0, 0}, 0));
    }
}

}  // namespace

TEST_CASE("recombine: a dispatch spends its budget from a pool of points", "[recombine]") {
    Highs highs;
    read(highs, "p0548.mps");
    const Setup s(highs.getLp(), highs.getOptions().log_options);
    RecordingSink sink(*s.problem.integrality, 0);
    fill_pool(s, sink);
    const size_t seeded = sink.accepted_points.size();
    REQUIRE(seeded >= 2);
    ExecutionContext exec = s.exec;
    constexpr size_t kTotal = 5'000'000;
    const DispatchOutcome outcome = recombine::run(
        s.problem, make_budget(kTotal, 1, std::numeric_limits<size_t>::max()), exec, sink);
    CHECK(outcome.effort >= kTotal);
    CHECK(sink.accepted_points.size() > seeded);
    for (size_t k = seeded; k < sink.accepted_points.size(); ++k) {
        CHECK(ruin_repair::is_feasible(s.problem, sink.accepted_points[k].second));
    }
}

// The trigger: with nothing improving, the patience ends the dispatch long
// before the budget would.  One-sided, so no clock in the fixture: FJ's
// callback period is the only slack, and the budget is far beyond it.
TEST_CASE("recombine: the patience ends a dispatch that stops improving", "[recombine]") {
    Highs highs;
    read(highs, "p0548.mps");
    const Setup s(highs.getLp(), highs.getOptions().log_options);
    RecordingSink sink(*s.problem.integrality, 0);
    fill_pool(s, sink);
    ExecutionContext exec = s.exec;
    constexpr size_t kTotal = std::numeric_limits<size_t>::max() / 2;
    constexpr size_t kPatience = 1'000'000;
    HeuristicBudget budget = make_budget(kTotal, 1, kPatience);
    budget.attempt_cap = kPatience;
    const DispatchOutcome outcome = recombine::run(s.problem, budget, exec, sink);
    CHECK(outcome.effort > 0);
    CHECK(outcome.effort < kTotal / 2);
}

TEST_CASE("recombine: a dispatch with fewer than two points, or no budget, does nothing",
          "[recombine]") {
    Highs highs;
    read(highs, "p0548.mps");
    const Setup s(highs.getLp(), highs.getOptions().log_options);
    ExecutionContext exec = s.exec;
    RecordingSink sink(*s.problem.integrality, 0);
    CHECK(recombine::run(s.problem, make_budget(kBudget, 1, kBudget), exec, sink).effort == 0);
    const std::vector<double> x = fj_point(s, 0);
    static_cast<void>(sink.offer(s.problem.objective(x), x, WorkerTrace{0, 0}, 0));
    CHECK(recombine::run(s.problem, make_budget(kBudget, 1, kBudget), exec, sink).effort == 0);
    const std::vector<double> y = fj_point(s, 1);
    static_cast<void>(sink.offer(s.problem.objective(y), y, WorkerTrace{0, 0}, 0));
    CHECK(recombine::run(s.problem, HeuristicBudget{}, exec, sink).effort == 0);
}

TEST_CASE("recombine: a caller's slot on an empty pool waits and returns at the stop",
          "[recombine][until_stopped]") {
    Highs highs;
    read(highs, "p0548.mps");
    const Setup s(highs.getLp(), highs.getOptions().log_options);
    std::atomic<bool> stop{false};
    ExecutionContext exec = s.exec;
    exec.stop = &stop;
    RecordingSink sink(*s.problem.integrality, 0);
    std::thread stopper([&stop] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        stop.store(true);
    });
    const size_t effort =
        recombine::run_until_stopped(s.problem, make_until_stopped_budget(kBudget, kBudget), exec,
                                     /*worker=*/0, RestartSource{}, sink);
    stopper.join();
    CHECK(effort == 0);
    CHECK(sink.accepted_points.empty());
}

// The caller's stop flag ends a run that is finding things, and the source's
// points are what it recombines: the pool starts empty, the source hands out
// two feasible points one after the other, and the operator takes its first
// step only once both are in.
TEST_CASE("recombine: a caller's slot steps from the source's points until the stop",
          "[recombine][until_stopped]") {
    Highs highs;
    read(highs, "p0548.mps");
    const Setup s(highs.getLp(), highs.getOptions().log_options);
    const std::vector<double> a = fj_point(s, 0);
    const std::vector<double> b = fj_point(s, 1);
    REQUIRE(a != b);

    std::atomic<bool> stop{false};
    ExecutionContext exec = s.exec;
    exec.stop = &stop;
    size_t calls = 0;
    const RestartSource source = [&](Rng& /*rng*/, std::vector<double>& out) {
        out = ++calls % 2 == 1 ? a : b;
        return true;
    };
    // Stop from the sink's side, so the run ends by the flag and no clock:
    // once the operator has offered something beyond the source's two.
    class StoppingSink : public RecordingSink {
    public:
        using RecordingSink::RecordingSink;
        std::atomic<bool>* flag = nullptr;
        size_t after = 0;

    private:
        void on_accept(double obj, const std::vector<double>& x, int source, const WorkerTrace& t,
                       size_t effort_at) override {
            RecordingSink::on_accept(obj, x, source, t, effort_at);
            if (accepted_points.size() >= after) {
                flag->store(true);
            }
        }
    };
    StoppingSink stopping(*s.problem.integrality, 0);
    stopping.flag = &stop;
    stopping.after = 3;
    const size_t effort =
        recombine::run_until_stopped(s.problem, make_until_stopped_budget(1'000'000, 0), exec,
                                     /*worker=*/0, source, stopping);
    CHECK(effort > 0);
    REQUIRE(stopping.accepted_points.size() >= 3);
    CHECK(stopping.accepted_points[0].second == a);
    CHECK(stopping.accepted_points[1].second == b);
    CHECK(ruin_repair::is_feasible(s.problem, stopping.accepted_points[2].second));
}

TEST_CASE("recombine: a source's infeasible point stays out of the pool, and is checked once",
          "[recombine][until_stopped]") {
    Highs highs;
    read(highs, "p0548.mps");
    const Setup s(highs.getLp(), highs.getOptions().log_options);
    RecordingSink sink(*s.problem.integrality, 0);
    recombine::RecombineOperator op(s.problem, s.exec, sink, 1, WorkerTrace{0, 0});
    // The upper bound on every column violates some row of p0548.
    std::vector<double> infeasible(*s.problem.col_upper);
    REQUIRE_FALSE(ruin_repair::is_feasible(s.problem, infeasible));
    const RestartSource source = [&](Rng& /*rng*/, std::vector<double>& out) {
        out = infeasible;
        return true;
    };
    CHECK(op.take_external(source) == s.problem.nnz);
    CHECK(op.take_external(source) == 0);
    CHECK(sink.accepted_points.empty());
}
