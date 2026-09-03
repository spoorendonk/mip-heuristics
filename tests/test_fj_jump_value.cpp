// The negative-coefficient jump value in HiGHS's vendored FeasibilityJump
// (issue #139).
//
// Luteberget & Sartor, "Feasibility Jump: an LP-free Lagrangian MIP
// heuristic", MPC 15:365-388, 2023.  Eq. (5)/(6) define a row's critical
// value for a variable with explicit positive- and negative-coefficient
// cases, and Algorithm 1 accumulates the pre-bound slope by that same sign.
//
// `JumpMove::updateValue` builds a row's valid range for a variable by
// dividing both endpoints of the row's bound interval by the coefficient.
// Upstream does not swap the endpoints when the coefficient is negative,
// which reverses the interval, so `validRange.first > validRange.second`
// fires and the row is dropped: an Lte row comes out (+inf, t) and a Gte row
// (t, -inf).  Such a row contributes neither a critical value nor a slope,
// and the jump value is computed as if it were not there.
// `apply_patch.cmake` inserts the swap.
//
// The cases below drive `JumpMove` directly rather than only through
// `FeasibilityJumpSolver::solve`, because that is where the defect lives and
// because the solver-level observable (the first feasible solution reported)
// exists only for models FeasibilityJump can actually satisfy — which rules
// out the two cases that pin the *slope* half of the fix.  The last case runs
// the issue's end-to-end repro through the real solver, so the unit cases
// cannot pass against a `JumpMove` the solver has stopped using.

#include "io/HighsIO.h"
#include "mip/feasibilityjump.hh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <limits>
#include <vector>

namespace {

namespace fj = external_feasibilityjump;

constexpr double kEqualityTolerance = 1e-5;
constexpr double kViolationTolerance = 1e-5;

// One row: sense, right-hand side, and the single variable's coefficient.
// Every model here is one-variable, which is what makes the expected jump
// value hand-computable from eq. (5)/(6) rather than an opaque fixture.
struct Row {
    fj::RowType sense;
    double rhs;
    double coeff;
};

// Builds the one-variable problem and returns the jump value `updateValue`
// computes for it at `incumbent`.
double jump_value(fj::VarType vartype, double lower, double upper, double incumbent,
                  const std::vector<Row>& rows) {
    fj::Problem problem(kEqualityTolerance, kViolationTolerance);
    problem.addVar(vartype, lower, upper, 0.0);

    for (const Row& row : rows) {
        int index = 0;
        double coeff = row.coeff;
        problem.addConstraint(row.sense, row.rhs, 1, &index, &coeff,
                              /*relax_continuous=*/0);
    }

    double initial = incumbent;
    problem.resetIncumbent(&initial);

    fj::JumpMove jump_move(kEqualityTolerance);
    jump_move.init(problem);
    jump_move.updateValue(problem, 0);

    double value = std::numeric_limits<double>::quiet_NaN();
    jump_move.forEachVarMove(0, [&value](fj::Move& move) { value = move.value; });
    return value;
}

// Silences the vendored solver: `highsLogDev` dereferences these pointers
// before it looks at anything else, so they may not be null.
struct SilentLog {
    bool output_flag = false;
    bool log_to_console = false;
    HighsInt log_dev_level = 0;
    HighsLogOptions options;

    SilentLog() {
        options.output_flag = &output_flag;
        options.log_to_console = &log_to_console;
        options.log_dev_level = &log_dev_level;
    }
};

// Runs the whole solver on the one-variable model and returns the first
// feasible value it reports through the callback.
double first_reported_value(fj::VarType vartype, double lower, double upper, double incumbent,
                            const std::vector<Row>& rows) {
    SilentLog log;
    fj::FeasibilityJumpSolver solver(log.options, /*seed=*/0, kEqualityTolerance,
                                     kViolationTolerance);
    solver.addVar(vartype, lower, upper, 0.0);

    for (const Row& row : rows) {
        int index = 0;
        double coeff = row.coeff;
        solver.addConstraint(row.sense, row.rhs, 1, &index, &coeff,
                             /*relax_continuous=*/0);
    }

    double reported = std::numeric_limits<double>::quiet_NaN();
    bool have_solution = false;
    // A bound on a test that would otherwise run to INT_MAX steps if the
    // solver never reached feasibility.
    constexpr size_t kEffortCap = 100000;

    auto callback = [&reported, &have_solution](fj::FJStatus status) {
        if (status.solution != nullptr && !have_solution) {
            have_solution = true;
            reported = status.solution[0];
            return fj::CallbackControlFlow::Terminate;
        }
        if (status.totalEffort > kEffortCap) {
            return fj::CallbackControlFlow::Terminate;
        }
        return fj::CallbackControlFlow::Continue;
    };

    double start = incumbent;
    solver.solve(&start, callback);
    REQUIRE(have_solution);
    return reported;
}

using Catch::Matchers::WithinAbs;

}  // namespace

// The issue's repro, at the level of the defective function: one constraint
// written two algebraically identical ways.  Before the fix the `-1` spelling
// yields 10 (the upper bound, the row having been dropped) while the `+1`
// spelling yields 3.
TEST_CASE("fj: a row's critical value does not depend on its spelling", "[fj][jump-value]") {
    const double positive =
        jump_value(fj::VarType::Integer, 0.0, 10.0, 0.0, {{fj::RowType::Gte, 3.0, 1.0}});
    const double negative =
        jump_value(fj::VarType::Integer, 0.0, 10.0, 0.0, {{fj::RowType::Lte, -3.0, -1.0}});

    REQUIRE_THAT(positive, WithinAbs(3.0, kEqualityTolerance));
    REQUIRE_THAT(negative, WithinAbs(3.0, kEqualityTolerance));
}

// Continuous columns take the same path minus the ceil/floor, so the fix must
// not be integer-only.  A non-integral critical value also rules out a "fix"
// that rounds its way to the right answer.
TEST_CASE("fj: a continuous column gets its negative-coefficient critical value",
          "[fj][jump-value]") {
    const double value =
        jump_value(fj::VarType::Continuous, 0.0, 10.0, 0.0, {{fj::RowType::Lte, -3.5, -1.0}});

    REQUIRE_THAT(value, WithinAbs(3.5, kEqualityTolerance));
}

// Where the swap sits relative to the ceil/floor is observable, and the two
// orders disagree exactly when the critical value is fractional.  `-x <= -3.5`
// admits x >= 3.5, so an integer column's critical value is 4.  Swapping the
// endpoints *after* the rounding rounds the true lower endpoint with `floor`
// and answers 3 — a value that violates the row, which is worse than the
// upper bound the unfixed code returns, because it is wrong rather than
// merely uninformed.
TEST_CASE("fj: the swap precedes the integer rounding", "[fj][jump-value]") {
    const double value =
        jump_value(fj::VarType::Integer, 0.0, 10.0, 0.0, {{fj::RowType::Lte, -3.5, -1.0}});

    REQUIRE_THAT(value, WithinAbs(4.0, kEqualityTolerance));
}

// The critical value alone is not the whole of eq. (5)/(6): Algorithm 1 also
// accumulates each row's weight into the pre-bound slope, and the jump is the
// minimum of the resulting piecewise-linear function rather than the nearest
// critical value.  With `x >= 3` and `x >= 8` both spelled with a negative
// coefficient, the slope stays negative past 3 and the minimum is at 8.
//
// This separates three implementations: upstream drops both rows and answers
// 10, a fix that registered the critical values but not the slope answers 3,
// and the paper answers 8.
TEST_CASE("fj: negative-coefficient rows accumulate the pre-bound slope", "[fj][jump-value]") {
    const double value =
        jump_value(fj::VarType::Integer, 0.0, 10.0, 0.0,
                   {{fj::RowType::Lte, -3.0, -1.0}, {fj::RowType::Lte, -8.0, -1.0}});

    REQUIRE_THAT(value, WithinAbs(8.0, kEqualityTolerance));
}

// The other half of the slope accounting: a row whose valid range ends at or
// below the current level contributes a *positive* weight to the slope
// (`validRange.second <= currentValue`).  For a negative coefficient that is
// the Gte branch, and it is reachable only once the endpoints are the right
// way round.
//
// `-x >= 3` admits x <= -3, below the lower bound, so it adds +1; the two
// positive-coefficient rows subtract 1 each.  The slope therefore reaches
// zero at 4 and the walk stops there.  Upstream drops the first row, leaves
// the slope at -2, and runs on to 8.
//
// The model is infeasible over [0, 10], which is exactly why this case is
// stated against `updateValue` and not against a reported solution: the jump
// value is defined for a violated row whether or not the model has an answer.
TEST_CASE("fj: a negative-coefficient row below the level lifts the slope", "[fj][jump-value]") {
    const double value = jump_value(fj::VarType::Integer, 0.0, 10.0, 0.0,
                                    {{fj::RowType::Gte, 3.0, -1.0},
                                     {fj::RowType::Gte, 4.0, 1.0},
                                     {fj::RowType::Gte, 8.0, 1.0}});

    REQUIRE_THAT(value, WithinAbs(4.0, kEqualityTolerance));
}

// An equality row expands into three bound intervals, all three of which are
// reversed by a negative coefficient.  `-x = -6` fixes x at 6; upstream drops
// all three and jumps to the upper bound.
TEST_CASE("fj: an equality row with a negative coefficient is not dropped", "[fj][jump-value]") {
    const double value =
        jump_value(fj::VarType::Integer, 0.0, 10.0, 0.0, {{fj::RowType::Equal, -6.0, -1.0}});

    REQUIRE_THAT(value, WithinAbs(6.0, kEqualityTolerance));
}

// A positive coefficient needs no swap, and swapping one would empty its
// range and drop the row — so this is what stops the fix from being written
// as an unconditional swap, or as a swap on the wrong side of the test.
// Written as a solver-level run, so it also pins that the fixed `JumpMove` is
// the one `FeasibilityJumpSolver` uses: the unit cases above cannot tell a
// live `updateValue` from a dead one.
TEST_CASE("fj: the solver reaches the critical value from either spelling", "[fj][jump-value]") {
    const double positive =
        first_reported_value(fj::VarType::Integer, 0.0, 10.0, 0.0, {{fj::RowType::Gte, 3.0, 1.0}});
    const double negative = first_reported_value(fj::VarType::Integer, 0.0, 10.0, 0.0,
                                                 {{fj::RowType::Lte, -3.0, -1.0}});

    REQUIRE_THAT(positive, WithinAbs(3.0, kEqualityTolerance));
    REQUIRE_THAT(negative, WithinAbs(3.0, kEqualityTolerance));
}
