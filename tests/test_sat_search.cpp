#include "heuristic_context.h"
#include "Highs.h"
#include "rng.h"
#include "sat_search.h"
#include "sat_solver.h"
#include "test_common.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kFeastol = 1e-6;
constexpr double kEpsilon = 1e-9;

using sat::make_lit;
using sat::SatResult;
using sat::SatSearchStatus;
using sat::SatSolver;

struct Row {
    std::vector<double> coefficients;
    double lower;
    double upper;
};

// A binary model over the given dense rows, column-wise as `make_problem` wants.
HighsLp binary_model(HighsInt ncol, const std::vector<Row>& rows) {
    HighsLp lp;
    lp.num_col_ = ncol;
    lp.num_row_ = static_cast<HighsInt>(rows.size());
    lp.col_cost_.assign(ncol, 0.0);
    lp.col_lower_.assign(ncol, 0.0);
    lp.col_upper_.assign(ncol, 1.0);
    lp.integrality_.assign(ncol, HighsVarType::kInteger);
    lp.a_matrix_.format_ = MatrixFormat::kColwise;
    lp.a_matrix_.num_col_ = lp.num_col_;
    lp.a_matrix_.num_row_ = lp.num_row_;
    lp.a_matrix_.start_.assign(1, 0);
    for (HighsInt j = 0; j < ncol; ++j) {
        for (size_t i = 0; i < rows.size(); ++i) {
            const double a = rows[i].coefficients[j];
            if (a != 0.0) {
                lp.a_matrix_.index_.push_back(static_cast<HighsInt>(i));
                lp.a_matrix_.value_.push_back(a);
            }
        }
        lp.a_matrix_.start_.push_back(static_cast<HighsInt>(lp.a_matrix_.index_.size()));
    }
    for (const Row& row : rows) {
        lp.row_lower_.push_back(row.lower);
        lp.row_upper_.push_back(row.upper);
    }
    return lp;
}

bool satisfies(const std::vector<Row>& rows, const std::vector<double>& x) {
    for (const Row& row : rows) {
        double lhs = 0.0;
        for (size_t j = 0; j < x.size(); ++j) {
            lhs += row.coefficients[j] * x[j];
        }
        if (lhs < row.lower - kFeastol || lhs > row.upper + kFeastol) {
            return false;
        }
    }
    return true;
}

sat::SatSearchResult search(const HighsLp& lp, std::vector<double>& point,
                            const sat::SatSearchConfig& config = {}) {
    ProblemStorage storage;
    const auto view = make_problem(lp, storage, kFeastol, kEpsilon);
    REQUIRE(view.has_value());
    return sat::sat_search(*view, config, {}, point);
}

// Checks the encoding against every assignment: with all columns free the search
// must agree with brute force on whether a point exists, and with each assignment
// pinned in turn it must accept exactly the feasible ones.  Pinned by one-column
// rows, not by bounds: a fixed column moves to the right-hand side, so bounds
// would never reach the clauses this is checking.
void check_against_brute_force(HighsInt ncol, const std::vector<Row>& rows) {
    bool any = false;
    std::vector<double> x(static_cast<size_t>(ncol));
    for (std::uint32_t mask = 0; mask < (1U << ncol); ++mask) {
        std::vector<Row> pinned = rows;
        for (HighsInt j = 0; j < ncol; ++j) {
            x[j] = ((mask >> j) & 1U) != 0 ? 1.0 : 0.0;
            std::vector<double> unit(static_cast<size_t>(ncol), 0.0);
            unit[j] = 1.0;
            pinned.push_back({unit, x[j], x[j]});
        }
        const bool feasible = satisfies(rows, x);
        any = any || feasible;
        std::vector<double> point;
        const auto result = search(binary_model(ncol, pinned), point);
        CHECK(result.status ==
              (feasible ? SatSearchStatus::kFeasible : SatSearchStatus::kInfeasible));
    }
    std::vector<double> point;
    const auto result = search(binary_model(ncol, rows), point);
    REQUIRE(result.status == (any ? SatSearchStatus::kFeasible : SatSearchStatus::kInfeasible));
    if (any) {
        CHECK(satisfies(rows, point));
    }
}

// The pigeonhole principle: `holes + 1` pigeons do not fit, and proving it needs
// real conflict analysis.
void add_pigeonhole(SatSolver& solver, std::int32_t holes) {
    const std::int32_t pigeons = holes + 1;
    const auto var = [holes](std::int32_t p, std::int32_t h) { return (p * holes) + h; };
    for (std::int32_t p = 0; p < pigeons; ++p) {
        std::vector<sat::Lit> clause;
        clause.reserve(static_cast<size_t>(holes));
        for (std::int32_t h = 0; h < holes; ++h) {
            clause.push_back(make_lit(var(p, h)));
        }
        solver.add_clause(clause);
    }
    for (std::int32_t h = 0; h < holes; ++h) {
        for (std::int32_t p = 0; p < pigeons; ++p) {
            for (std::int32_t q = p + 1; q < pigeons; ++q) {
                const std::array<sat::Lit, 2> clause{make_lit(var(p, h), true),
                                                     make_lit(var(q, h), true)};
                solver.add_clause(clause);
            }
        }
    }
}

}  // namespace

TEST_CASE("SAT solver answers hand-built formulas", "[sat]") {
    SECTION("an empty formula is satisfiable") {
        SatSolver solver(2, 1);
        CHECK(solver.solve({}, {}) == SatResult::kSatisfiable);
    }
    SECTION("opposed units are unsatisfiable") {
        SatSolver solver(1, 1);
        const std::array<sat::Lit, 1> a{make_lit(0)};
        const std::array<sat::Lit, 1> b{make_lit(0, true)};
        CHECK(solver.add_clause(a));
        CHECK_FALSE(solver.add_clause(b));
        CHECK(solver.solve({}, {}) == SatResult::kUnsatisfiable);
    }
    SECTION("a tautology adds nothing, and a duplicate literal counts once") {
        SatSolver solver(2, 1);
        const std::array<sat::Lit, 2> tautology{make_lit(0), make_lit(0, true)};
        CHECK(solver.add_clause(tautology));
        CHECK(solver.num_clauses() == 0);
        const std::array<sat::Lit, 2> duplicate{make_lit(1, true), make_lit(1, true)};
        CHECK(solver.add_clause(duplicate));
        REQUIRE(solver.solve({}, {}) == SatResult::kSatisfiable);
        CHECK_FALSE(solver.model_value(1));
    }
    SECTION("pigeonhole formulas are refuted") {
        SatSolver solver(7 * 6, 3);
        add_pigeonhole(solver, 6);
        CHECK(solver.solve({}, {}) == SatResult::kUnsatisfiable);
        CHECK(solver.conflicts() > 0);
    }
    SECTION("a refutation stays sound across learnt clause reductions") {
        SatSolver solver(9 * 8, 3);
        add_pigeonhole(solver, 8);
        CHECK(solver.solve({}, {}) == SatResult::kUnsatisfiable);
        CHECK(solver.reductions() > 0);
    }
}

TEST_CASE("SAT solver finds models of random 3-SAT", "[sat]") {
    // Below the phase transition's ratio of 4.27, so nearly always satisfiable;
    // any model returned is checked, and a refutation would be legitimate.
    constexpr std::int32_t kVars = 250;
    constexpr std::int32_t kClauses = 1000;
    Rng rng(7);
    std::uniform_int_distribution<std::int32_t> pick(0, kVars - 1);
    std::bernoulli_distribution sign(0.5);
    std::vector<std::array<sat::Lit, 3>> formula;
    SatSolver solver(kVars, 11);
    for (std::int32_t c = 0; c < kClauses; ++c) {
        formula.push_back({make_lit(pick(rng), sign(rng)), make_lit(pick(rng), sign(rng)),
                           make_lit(pick(rng), sign(rng))});
        solver.add_clause(formula.back());
    }
    const SatResult result = solver.solve({}, {});
    REQUIRE(result != SatResult::kStopped);
    CHECK(solver.conflicts() > 0);
    if (result == SatResult::kSatisfiable) {
        for (const auto& clause : formula) {
            bool satisfied = false;
            for (const sat::Lit l : clause) {
                satisfied = satisfied || solver.model_value(l >> 1) == ((l & 1) == 0);
            }
            CHECK(satisfied);
        }
    }
}

TEST_CASE("SAT solver stops on its budgets and its stop check", "[sat]") {
    SECTION("conflict budget") {
        SatSolver solver(9 * 8, 3);
        add_pigeonhole(solver, 8);
        sat::SatLimits limits;
        limits.max_conflicts = 10;
        CHECK(solver.solve(limits, {}) == SatResult::kStopped);
        CHECK(solver.conflicts() == 10);
    }
    SECTION("effort budget") {
        SatSolver solver(9 * 8, 3);
        add_pigeonhole(solver, 8);
        sat::SatLimits limits;
        limits.max_effort = 100;
        CHECK(solver.solve(limits, {}) == SatResult::kStopped);
    }
    SECTION("stop check, and the solver can resume") {
        SatSolver solver(7 * 6, 3);
        add_pigeonhole(solver, 6);
        CHECK(solver.solve({}, [] { return true; }) == SatResult::kStopped);
        CHECK(solver.solve({}, {}) == SatResult::kUnsatisfiable);
    }
}

TEST_CASE("SAT solver is reproducible for a seed", "[sat]") {
    const auto solve = [](std::uint64_t seed) {
        SatSolver solver(30, seed);
        std::vector<std::int8_t> values;
        values.reserve(30);
        for (std::int32_t v = 0; v + 2 < 30; ++v) {
            const std::array<sat::Lit, 3> clause{make_lit(v), make_lit(v + 1, true),
                                                 make_lit(v + 2)};
            solver.add_clause(clause);
        }
        REQUIRE(solver.solve({}, {}) == SatResult::kSatisfiable);
        for (std::int32_t v = 0; v < 30; ++v) {
            values.push_back(solver.model_value(v) ? 1 : 0);
        }
        return values;
    };
    CHECK(solve(5) == solve(5));
}

TEST_CASE("Row encoding agrees with brute force on every row shape", "[sat]") {
    constexpr HighsInt kN = 6;
    SECTION("clause") {
        check_against_brute_force(kN, {{{1, 1, 1, 1, 1, 1}, 1, kInf}});
    }
    SECTION("clause with a negated column") {
        check_against_brute_force(kN, {{{1, -1, 1, -1, 1, 0}, -1, kInf}});
    }
    SECTION("at most one") {
        check_against_brute_force(kN, {{{1, 1, 1, 1, 1, 1}, -kInf, 1}});
    }
    SECTION("exactly one") {
        check_against_brute_force(kN, {{{1, 1, 1, 1, 1, 1}, 1, 1}});
    }
    SECTION("cardinality") {
        check_against_brute_force(kN, {{{1, 1, 1, 1, 1, 1}, 2, 4}});
    }
    SECTION("weighted") {
        check_against_brute_force(kN, {{{5, 4, 3, 3, 2, 1}, -kInf, 9}});
    }
    SECTION("weighted equality with mixed signs") {
        check_against_brute_force(kN, {{{5, -4, 3, -3, 2, 1}, 2, 2}});
    }
    SECTION("fractional coefficients") {
        check_against_brute_force(kN, {{{0.5, 0.25, 1.5, 0.75, 0.1, 2}, 1.1, 2.6}});
    }
    SECTION("a common divisor rounds the bound") {
        check_against_brute_force(kN, {{{2, 2, 4, 4, 6, 6}, -kInf, 7}});
    }
    SECTION("an item heavier than the bound") {
        check_against_brute_force(kN, {{{10, 1, 1, 1, 1, 1}, -kInf, 3}});
    }
    SECTION("an infeasible row") {
        check_against_brute_force(kN, {{{1, 1, 1, 1, 1, 1}, 7, kInf}});
    }
    SECTION("a redundant row") {
        check_against_brute_force(kN, {{{1, 1, 1, 1, 1, 1}, -kInf, 6}});
    }
    SECTION("rows that interact") {
        check_against_brute_force(
            kN,
            {{{1, 1, 1, 0, 0, 0}, 1, 1}, {{0, 0, 1, 1, 1, 0}, 1, 1}, {{3, 0, 2, 0, 2, 3}, 4, 5}});
    }
}

TEST_CASE("Row encoding agrees with brute force on random rows", "[sat]") {
    constexpr HighsInt kN = 7;
    Rng rng(13);
    std::uniform_int_distribution<int> coefficient(-6, 6);
    std::uniform_int_distribution<int> bound(-8, 12);
    for (int trial = 0; trial < 20; ++trial) {
        std::vector<Row> rows;
        for (int r = 0; r < 2; ++r) {
            Row row{std::vector<double>(kN), -kInf, kInf};
            for (double& a : row.coefficients) {
                a = coefficient(rng);
            }
            const int first = bound(rng);
            row.upper = first;
            if ((trial + r) % 3 == 0) {
                row.lower = first;
            }
            rows.push_back(row);
        }
        check_against_brute_force(kN, rows);
    }
}

TEST_CASE("A long row encodes without recursing", "[sat]") {
    // Long enough to overflow a thread's stack if the diagram were built by
    // recursion, with a capacity the single-clause shortcut cannot take.
    constexpr HighsInt kN = 200000;
    Row row{std::vector<double>(kN, 1.0), 1, 1};
    const HighsLp lp = binary_model(kN, {row});
    std::vector<double> point;
    const auto result = search(lp, point);
    REQUIRE(result.status == SatSearchStatus::kFeasible);
    double sum = 0.0;
    for (const double v : point) {
        sum += v;
    }
    CHECK(sum == 1.0);
}

TEST_CASE("sat_search refuses what it cannot encode", "[sat]") {
    std::vector<double> point{42.0};
    const std::vector<double> untouched = point;
    SECTION("a general integer column") {
        HighsLp lp = binary_model(2, {{{1, 1}, 1, kInf}});
        lp.col_upper_[0] = 2.0;
        const auto result = search(lp, point);
        CHECK(result.status == SatSearchStatus::kRefused);
        CHECK(result.refusal == sat::SatRefusal::kNotPureBinary);
    }
    SECTION("a continuous column") {
        HighsLp lp = binary_model(2, {{{1, 1}, 1, kInf}});
        lp.integrality_[1] = HighsVarType::kContinuous;
        const auto result = search(lp, point);
        CHECK(result.status == SatSearchStatus::kRefused);
        CHECK(result.refusal == sat::SatRefusal::kNotPureBinary);
    }
    SECTION("a coefficient no scale makes integral") {
        const HighsLp lp = binary_model(2, {{{1.0 / 3.0, 1}, -kInf, 1}});
        const auto result = search(lp, point);
        CHECK(result.status == SatSearchStatus::kRefused);
        CHECK(result.refusal == sat::SatRefusal::kNotIntegral);
    }
    SECTION("weights too large to add up exactly") {
        const HighsLp lp = binary_model(2, {{{9e14 + 0.5, 1}, -kInf, 1}});
        const auto result = search(lp, point);
        CHECK(result.status == SatSearchStatus::kRefused);
        CHECK(result.refusal == sat::SatRefusal::kNotIntegral);
    }
    SECTION("bounds that are not a binary's, which only a view can carry") {
        // `make_problem` rounds these inward; the adapter's view need not.
        const HighsLp lp = binary_model(2, {{{1, 1}, 1, kInf}});
        ProblemStorage storage;
        const auto view = make_problem(lp, storage, kFeastol, kEpsilon);
        REQUIRE(view.has_value());
        SECTION("a fractional upper bound") {
            storage.col_upper[0] = 0.5;
        }
        SECTION("a fractional lower bound") {
            storage.col_lower[0] = 0.5;
        }
        SECTION("an integer column fixed at a fractional value") {
            storage.col_lower[0] = 0.5;
            storage.col_upper[0] = 0.5;
        }
        const auto result = sat::sat_search(*view, {}, {}, point);
        CHECK(result.status == SatSearchStatus::kRefused);
        CHECK(result.refusal == sat::SatRefusal::kNotPureBinary);
    }
    SECTION("a weight past an int64") {
        const HighsLp lp = binary_model(2, {{{1e13, 1e-6}, -kInf, 1}});
        const auto result = search(lp, point);
        CHECK(result.status == SatSearchStatus::kRefused);
        CHECK(result.refusal == sat::SatRefusal::kNotIntegral);
    }
    SECTION("a model over the clause cap") {
        const HighsLp lp = binary_model(6, {{{1, 1, 1, 1, 1, 1}, 2, 4}});
        sat::SatSearchConfig config;
        config.max_clauses = 3;
        const auto result = search(lp, point, config);
        CHECK(result.status == SatSearchStatus::kRefused);
        CHECK(result.refusal == sat::SatRefusal::kTooLarge);
        CHECK(result.clauses <= 3);
    }
    CHECK(point == untouched);
}

TEST_CASE("A fixed column of any type or value moves to the right-hand side", "[sat]") {
    HighsLp lp = binary_model(3, {{{2, 1, 1}, 5, 5}});
    lp.integrality_[0] = HighsVarType::kContinuous;
    lp.col_lower_[0] = 2.0;
    lp.col_upper_[0] = 2.0;
    std::vector<double> point;
    const auto result = search(lp, point);
    REQUIRE(result.status == SatSearchStatus::kFeasible);
    CHECK(point[0] == 2.0);
    CHECK(point[1] + point[2] == 1.0);
}

TEST_CASE("sat_search stops when told, while encoding", "[sat]") {
    const HighsLp lp = binary_model(2, {{{1, 1}, 1, kInf}});
    ProblemStorage storage;
    const auto view = make_problem(lp, storage, kFeastol, kEpsilon);
    REQUIRE(view.has_value());
    std::vector<double> point;
    const auto result = sat::sat_search(*view, {}, [] { return true; }, point);
    CHECK(result.status == SatSearchStatus::kStopped);
    CHECK(point.empty());
}

TEST_CASE("sat_search stops inside one hard weighted row", "[sat]") {
    // One knapsack row with weights this spread out, so capacities rarely share
    // a node and the diagram is large: a poll only between rows would never come.
    constexpr HighsInt kN = 40;
    Rng rng(17);
    std::uniform_int_distribution<int> weight(100000, 1000000);
    Row row{std::vector<double>(kN), -kInf, 0.0};
    double total = 0.0;
    for (double& a : row.coefficients) {
        a = weight(rng);
        total += a;
    }
    row.upper = std::floor(total / 2.0);
    const HighsLp lp = binary_model(kN, {row});
    ProblemStorage storage;
    const auto view = make_problem(lp, storage, kFeastol, kEpsilon);
    REQUIRE(view.has_value());
    std::vector<double> point;

    SECTION("on the stop check") {
        int polls = 0;
        const auto result = sat::sat_search(*view, {}, [&polls] { return ++polls > 2; }, point);
        CHECK(result.status == SatSearchStatus::kStopped);
        CHECK(polls == 3);
        CHECK(result.conflicts == 0);
    }
    SECTION("on the effort budget, which encoding spends from") {
        sat::SatSearchConfig config;
        config.limits.max_effort = 1000;
        const auto result = sat::sat_search(*view, config, {}, point);
        CHECK(result.status == SatSearchStatus::kStopped);
        CHECK(result.effort >= 1000);
        CHECK(result.effort < 2000);
    }
}

TEST_CASE("The solver gets only what encoding left of the effort budget", "[sat]") {
    // Pigeonhole as rows: encoding is cheap and the search is not.
    constexpr HighsInt kHoles = 5;
    constexpr HighsInt kPigeons = kHoles + 1;
    constexpr HighsInt kCols = kPigeons * kHoles;
    std::vector<Row> rows;
    for (HighsInt p = 0; p < kPigeons; ++p) {
        std::vector<double> a(kCols, 0.0);
        for (HighsInt h = 0; h < kHoles; ++h) {
            a[(p * kHoles) + h] = 1.0;
        }
        rows.push_back({a, 1.0, kInf});
    }
    for (HighsInt h = 0; h < kHoles; ++h) {
        std::vector<double> a(kCols, 0.0);
        for (HighsInt p = 0; p < kPigeons; ++p) {
            a[(p * kHoles) + h] = 1.0;
        }
        rows.push_back({a, -kInf, 1.0});
    }
    const HighsLp lp = binary_model(kCols, rows);
    std::vector<double> point;

    // No conflicts allowed stops the solver before it spends anything, which
    // leaves the encoding's own work.
    sat::SatSearchConfig encode_only;
    encode_only.limits.max_conflicts = 0;
    const auto encoding = search(lp, point, encode_only);
    REQUIRE(encoding.status == SatSearchStatus::kStopped);
    REQUIRE(encoding.effort > 0);

    sat::SatSearchConfig exact;
    exact.limits.max_effort = encoding.effort;
    const auto result = search(lp, point, exact);
    CHECK(result.status == SatSearchStatus::kStopped);
    CHECK(result.effort == encoding.effort);
    CHECK(result.conflicts == 0);

    // Without the budget the search does spend more.
    const auto full = search(lp, point);
    CHECK(full.status == SatSearchStatus::kInfeasible);
    CHECK(full.effort > encoding.effort);
}

TEST_CASE("sat_search on a pure-binary model from HiGHS's own instances", "[sat]") {
    // The original model's rows are checked by sat_search itself; p0548 mixes
    // covering, packing and weighted rows.
    Highs highs;
    highs.setOptionValue("output_flag", false);
    REQUIRE(highs.readModel(instances_dir() + "/p0548.mps") == HighsStatus::kOk);
    const HighsLp& lp = highs.getLp();
    ProblemStorage storage;
    const auto view = make_problem(lp, storage, kFeastol, kEpsilon);
    REQUIRE(view.has_value());
    sat::SatSearchConfig config;
    config.limits.max_conflicts = 200000;
    std::vector<double> point;
    const auto result = sat::sat_search(*view, config, {}, point);
    REQUIRE(result.status == SatSearchStatus::kFeasible);
    for (HighsInt j = 0; j < lp.num_col_; ++j) {
        CHECK(point[j] >= lp.col_lower_[j]);
        CHECK(point[j] <= lp.col_upper_[j]);
    }
}
