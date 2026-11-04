// The line-segment search (#213) in the heuristic core: its visiting order,
// the three roundings, and the search in both modes.

#include "core_test_support.h"
#include "fj_worker.h"
#include "heuristic_context.h"
#include "Highs.h"
#include "line_segment.h"
#include "rng.h"
#include "ruin_repair.h"
#include "test_paths.h"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

using namespace core_test;

namespace {

// `ncol` integer columns in [lb, ub] with costs `cost`, and one row over
// the first `in_row` of them with coefficient 1 and bounds [row_lo, row_hi].
HighsLp one_row_model(const std::vector<double>& cost, double lb, double ub, HighsInt in_row,
                      double row_lo, double row_hi) {
    HighsLp lp;
    const auto ncol = static_cast<HighsInt>(cost.size());
    lp.num_col_ = ncol;
    lp.num_row_ = 1;
    lp.col_cost_ = cost;
    lp.col_lower_.assign(ncol, lb);
    lp.col_upper_.assign(ncol, ub);
    lp.row_lower_ = {row_lo};
    lp.row_upper_ = {row_hi};
    lp.integrality_.assign(ncol, HighsVarType::kInteger);
    lp.a_matrix_.format_ = MatrixFormat::kColwise;
    lp.a_matrix_.num_col_ = ncol;
    lp.a_matrix_.num_row_ = 1;
    lp.a_matrix_.start_.assign(1, 0);
    for (HighsInt j = 0; j < ncol; ++j) {
        if (j < in_row) {
            lp.a_matrix_.index_.push_back(0);
            lp.a_matrix_.value_.push_back(1.0);
        }
        lp.a_matrix_.start_.push_back(static_cast<HighsInt>(lp.a_matrix_.index_.size()));
    }
    return lp;
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

TEST_CASE("line segment: points are visited middle first", "[line_segment]") {
    using line_segment::middle_first_order;
    CHECK(middle_first_order(0).empty());
    CHECK(middle_first_order(1) == std::vector<size_t>{0});
    CHECK(middle_first_order(5) == std::vector<size_t>{2, 0, 3, 1, 4});
    CHECK(middle_first_order(7) == std::vector<size_t>{3, 1, 5, 0, 2, 4, 6});
}

TEST_CASE("line segment: brute force rounds a few fractional columns feasibly", "[line_segment]") {
    Highs highs;
    // x0 + x1 + x2 = 1 over binaries, every one at a third.
    const HighsLp lp = one_row_model({1.0, 1.0, 1.0}, 0.0, 1.0, 3, 1.0, 1.0);
    const Setup s(lp, highs.getOptions().log_options);
    std::vector<double> x = {1.0 / 3, 1.0 / 3, 1.0 / 3};
    Rng rng(0);
    (void)line_segment::round_nearest(s.problem, x, rng);
    // The first configuration in mask order that is feasible: x0 up.
    CHECK(x == std::vector<double>{1.0, 0.0, 0.0});
}

TEST_CASE("line segment: lock rounding follows the locks and the cost sign", "[line_segment]") {
    Highs highs;
    // Nine columns in sum <= 100 (an up lock each), so brute force is
    // skipped and they round down; two columns in no row round by cost.
    std::vector<double> cost(11, 0.0);
    cost[9] = -1.0;
    cost[10] = 1.0;
    const HighsLp lp = one_row_model(cost, 0.0, 3.0, 9, -kHighsInf, 100.0);
    const Setup s(lp, highs.getOptions().log_options);
    std::vector<double> x(11, 1.5);
    Rng rng(0);
    (void)line_segment::round_nearest(s.problem, x, rng);
    for (size_t j = 0; j < 9; ++j) {
        CHECK(x[j] == 1.0);
    }
    CHECK(x[9] == 2.0);
    CHECK(x[10] == 1.0);
}

TEST_CASE("line segment: nearest rounding stays integral and in bounds", "[line_segment]") {
    Highs highs;
    // An equality row locks every column both ways, and nine fractional
    // columns (the tenth is clamped to an integer) skip brute force, so the
    // randomized nearest rounding runs.
    const HighsLp lp = one_row_model(std::vector<double>(10, 1.0), 2.0, 3.0, 10, 25.0, 25.0);
    const Setup s(lp, highs.getOptions().log_options);
    Rng rng(7);
    for (int trial = 0; trial < 20; ++trial) {
        std::vector<double> x(10, 2.5);
        x[0] = 9.0;  // above its bound: clamped first
        (void)line_segment::round_nearest(s.problem, x, rng);
        for (const double v : x) {
            CHECK(v == std::round(v));
            CHECK(v >= 2.0);
            CHECK(v <= 3.0);
        }
        CHECK(x[0] == 3.0);
    }
}

TEST_CASE("line segment: the search reaches a feasible point from an infeasible start",
          "[line_segment]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    REQUIRE(highs.readModel(instances_dir() + "/lseu.mps") == HighsStatus::kOk);
    const Setup s(highs.getLp(), highs.getOptions().log_options);
    const std::vector<double> end = fj_point(s, 0);
    // Every column at its upper bound violates lseu's knapsack rows.
    const std::vector<double> start = *s.problem.col_upper;
    size_t effort = 0;
    REQUIRE_FALSE(line_segment::evaluate(s.problem, start, effort).feasible);

    constexpr size_t kPoints = 5;
    std::vector<double> delta(start.size());
    for (size_t j = 0; j < delta.size(); ++j) {
        delta[j] = (end[j] - start[j]) / (kPoints + 1);
    }
    BestSink sink(*s.problem.integrality, 0);
    WorkerTrace trace{0, 0};
    Rng rng(3);
    const line_segment::SearchParams params{.n_points = kPoints,
                                            .fj_budget_per_point = kBudget,
                                            .total_budget = kBudget,
                                            .mode = line_segment::Mode::kLocalSearch,
                                            .parents = {}};
    const line_segment::SearchResult r =
        line_segment::search(s.problem, s.exec, sink, trace, rng, start, delta, params);
    REQUIRE(r.quality.feasible);
    CHECK(ruin_repair::is_feasible(s.problem, r.point));
    CHECK(std::abs(r.quality.value - s.problem.objective(r.point)) < 1e-6);
    CHECK(r.effort > 0);
    CHECK(trace.effort_base >= r.effort);
}

TEST_CASE("line segment: recombiner mode counts no point equal to a parent", "[line_segment]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    REQUIRE(highs.readModel(instances_dir() + "/lseu.mps") == HighsStatus::kOk);
    const Setup s(highs.getLp(), highs.getOptions().log_options);
    const std::vector<double> a = fj_point(s, 0);
    size_t effort = 0;
    const line_segment::Quality qa = line_segment::evaluate(s.problem, a, effort);
    REQUIRE(qa.feasible);

    // A zero direction: every point is `a`, which beats nothing and differs
    // from no parent, and no FJ runs.
    const std::vector<double> delta(a.size(), 0.0);
    BestSink sink(*s.problem.integrality, 0);
    WorkerTrace trace{0, 0};
    Rng rng(5);
    line_segment::SearchParams params{
        .n_points = 20,
        .fj_budget_per_point = kBudget,
        .total_budget = kBudget,
        .mode = line_segment::Mode::kRecombiner,
        .parents = {.a = &a, .b = &a, .best = qa, .both_infeasible = false}};
    const line_segment::SearchResult r =
        line_segment::search(s.problem, s.exec, sink, trace, rng, a, delta, params);
    CHECK(r.point.empty());
    CHECK_FALSE(sink.found);

    SECTION("but counts it when both parents are infeasible") {
        params.parents.best = {.feasible = false, .value = 1.0};
        params.parents.both_infeasible = true;
        const line_segment::SearchResult again =
            line_segment::search(s.problem, s.exec, sink, trace, rng, a, delta, params);
        CHECK(again.point == a);
        CHECK(again.quality.feasible);
    }
}

TEST_CASE("line segment: a fractional point counts its integrality violation", "[line_segment]") {
    Highs highs;
    const HighsLp lp = one_row_model({1.0, 1.0}, 0.0, 1.0, 2, 0.0, 2.0);
    const Setup s(lp, highs.getOptions().log_options);
    size_t effort = 0;
    const line_segment::Quality q = line_segment::evaluate(s.problem, {0.5, 0.25}, effort);
    CHECK_FALSE(q.feasible);
    CHECK(q.value == 0.75);
    CHECK(effort > 0);
}
