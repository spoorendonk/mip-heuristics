// The objective term's sign in HiGHS's vendored FeasibilityJump (issue #139).
//
// Luteberget & Sartor, "Feasibility Jump: an LP-free Lagrangian MIP
// heuristic", MPC 15:365-388, 2023, Sect. 2.6: the score is a *minimized* sum
// of an objective term and the violation terms.
//
// In the vendored code every other part of that sum is improvement-positive —
// a constraint term is `weight * (score(new) - score(old))` with `score`
// returning minus the violation, `selectVariable` takes the maximum, and
// `updateGoodMoves` calls a move good when its score is positive — while the
// objective term is *added*, over coefficients the HiGHS wrapper has already
// multiplied by the model sense.  A move that worsens the objective therefore
// scored positively, and after first feasibility the improving mode steered
// away from better objectives.  `apply_patch.cmake` negates the term, at both
// of the two sites that spell it (`resetMoves` recomputes it, `updateWeights`
// applies it incrementally).
//
// What this case pins and what it does not, from running each mutation rather
// than reasoning about it.  Best reported objective, optimum 0:
//
//   both sites subtracted (shipped)     0    passes
//   both sites restored to upstream   190    fails
//   `resetMoves` restored only        130    fails
//   `updateWeights` restored only       0    PASSES — the gap
//
// So this case *does* pin `resetMoves`, which is the site that decides the
// direction here: the model is feasible from the start and `objectiveWeight`
// begins at 0.0, so the first scores come entirely from `updateWeights`'s
// increment, and `resetMoves` is what re-signs a column after it has moved.
// Restore it alone and every column the solver moves ping-pongs back to the
// bound it came from.
//
// The uncovered site is `updateWeights`, and it is uncovered because the two
// disagree only by drift: it applies incrementally what `resetMoves`
// recomputes, so a wrong sign there is corrected on any column that
// subsequently moves, and on this model that is every column that matters.
// The patch block's post-check catches the half-applied *patch* — it refuses
// to configure unless both sites subtract and no `+=` spelling survives — but
// it is a spelling check and not a substitute for a test: it runs after a
// pre-check that already matched both anchors, and it would pass
// `move.score -= -weightUpdateIncrement * ...`, a flipped
// `objectiveWeight += weightUpdateIncrement`, or a deleted incremental loop.
// Stated rather than left implicit, because the disagreement is real.
//
// The score itself is private to `FeasibilityJumpSolver`, so the case below
// pins the behaviour the sign *decides*: on a model that is feasible from the
// start, where every variable's jump is a bound and half of the jumps improve
// the objective while half worsen it, the improving mode either walks down to
// the optimum or walks away from it.  Which half `selectVariable` prefers is
// the whole content of the sign, and nothing else in the solver chooses
// between them: the jump value never reads the objective.

#include "io/HighsIO.h"
#include "mip/feasibilityjump.hh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>
#include <limits>
#include <vector>

namespace {

namespace fj = external_feasibilityjump;

constexpr double kEqualityTolerance = 1e-5;
constexpr double kViolationTolerance = 1e-5;

// Half the columns start at their upper bound and half at their lower bound.
// A column at a bound with no row pushing it jumps to the opposite bound, so
// with `min sum x_j` exactly half the jumps improve the objective by 10 each
// and half worsen it by 10 each.
constexpr int kNumColumns = 40;
constexpr double kUpper = 10.0;

// Generous: the corrected sign needs about twenty moves, each costing on the
// order of the single row's length.  Large enough that "did not get there" is
// a statement about the search direction and not about the budget.
constexpr size_t kEffortCap = 2000000;

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

// Runs the model described above and returns the best objective the solver
// reported through its improvement callback.  The callback stops the solve as
// soon as the optimum is reached, so a passing run is also a short one.
double best_reported_objective() {
    SilentLog log;
    fj::FeasibilityJumpSolver solver(log.options, /*seed=*/0, kEqualityTolerance,
                                     kViolationTolerance);

    std::vector<int> index;
    std::vector<double> row_coeff;
    std::vector<double> start;
    for (int col = 0; col < kNumColumns; ++col) {
        solver.addVar(fj::VarType::Integer, 0.0, kUpper, /*objCoeff=*/1.0);
        index.push_back(col);
        row_coeff.push_back(1.0);
        start.push_back(col % 2 == 0 ? kUpper : 0.0);
    }

    // One row over every column, satisfied at every point of the box, so the
    // violation half of the score is identically zero and the objective term
    // is the only thing selecting a move.  It is not decoration: `resetMoves`
    // and `updateGoodMoves` reach a column only through a row it appears in,
    // so a model with no rows would leave the good-move set frozen.
    double rhs = 0.0;
    solver.addConstraint(fj::RowType::Gte, rhs, kNumColumns, index.data(), row_coeff.data(),
                         /*relax_continuous=*/0);

    double best = std::numeric_limits<double>::infinity();
    auto callback = [&best](fj::FJStatus status) {
        if (status.solution != nullptr && status.solutionObjectiveValue < best) {
            best = status.solutionObjectiveValue;
            if (best <= kEqualityTolerance) {
                return fj::CallbackControlFlow::Terminate;
            }
        }
        if (status.totalEffort > kEffortCap) {
            return fj::CallbackControlFlow::Terminate;
        }
        return fj::CallbackControlFlow::Continue;
    };

    solver.solve(start.data(), callback);
    return best;
}

}  // namespace

// The starting point is feasible with objective 200 (half the columns at 10),
// and the optimum is 0.  With the term subtracted, the good-move set is
// exactly the columns whose jump lowers the objective, and the solver walks
// down to 0.  With upstream's sign the good-move set is the other half, the
// solver walks up to 400, and 0 is out of reach: a column it drives to the
// upper bound is immediately "good" again in the direction it came from.
TEST_CASE("fj: the improving mode moves toward a better objective", "[fj][objective-sign]") {
    const double start_objective = kNumColumns * kUpper / 2.0;

    const double best = best_reported_objective();

    REQUIRE(best < start_objective);
    REQUIRE_THAT(best, Catch::Matchers::WithinAbs(0.0, kEqualityTolerance));
}
