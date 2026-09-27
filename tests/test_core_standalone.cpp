// The core on a libhighs without the MIP solver (#170).
//
// `mip_heuristics_core_standalone_tests` links `libmip_heuristics_core.a`
// against a copy of `libhighs.a` with every member built from
// `highs/mip/*.cpp` and every adapter object removed.  It links at all only
// if nothing the core needs lives in the MIP solver or the adapter, which is
// the `core_boundary` claim checked by the linker; and these cases then run
// FJ and LocalMIP on it.  Nothing here may use the `Highs` class, whose own
// `run` reaches the MIP solver: the model is built by hand.

#include "core_test_support.h"
#include "heuristic_context.h"
#include "io/HighsIO.h"
#include "util/HighsTimer.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <optional>
#include <vector>

using namespace core_test;

namespace {

// `HighsLogOptions`' own defaults are null pointers the logger dereferences.
struct QuietLog {
    bool output_flag = false;
    bool log_to_console = false;
    HighsInt log_dev_level = 0;
    HighsLogOptions options;
    QuietLog() {
        options.output_flag = &output_flag;
        options.log_to_console = &log_to_console;
        options.log_dev_level = &log_dev_level;
    }
};

}  // namespace

TEST_CASE("core: a maximisation model is searched and reported in minimisation form", "[core]") {
    const QuietLog log;
    const HighsLp lp = maximisation_model();
    const Setup s(lp, log.options);

    // Normalised: negated costs and offset, the model itself untouched.
    REQUIRE(*s.problem.col_cost == std::vector<double>{-3.0, -2.0, -4.0, -1.0});
    REQUIRE(s.problem.offset == -7.0);
    REQUIRE(lp.col_cost_ == std::vector<double>{3.0, 2.0, 4.0, 1.0});

    Found found(*s.problem.integrality);
    run_both(s, found);

    for (BestSink* sink : {&found.fj, &found.local_mip}) {
        REQUIRE(sink->found);
        REQUIRE(is_feasible(lp, sink->solution));
        // The documented convention: the sink sees the minimisation
        // objective, offset included, i.e. minus the model's own.
        REQUIRE(sink->objective == Catch::Approx(-original_objective(lp, sink->solution)));
    }
}

namespace {

// What `make_problem` says about `lp`: nothing (accepted), or the kind of
// refusal.
std::optional<ProblemError::Kind> refusal(const HighsLp& lp) {
    ProblemStorage storage;
    const auto made = make_problem(lp, storage, kFeasTol, kEpsilon);
    if (made.has_value()) {
        return std::nullopt;
    }
    INFO(made.error().message);
    return made.error().kind;
}

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

}  // namespace

TEST_CASE("core: make_problem accepts the well-formed model", "[core][validation]") {
    REQUIRE_FALSE(refusal(maximisation_model()).has_value());
}

TEST_CASE("core: make_problem refuses a malformed model", "[core][validation]") {
    HighsLp lp = maximisation_model();
    SECTION("integrality of the wrong size") {
        lp.integrality_.pop_back();
    }
    SECTION("costs of the wrong size") {
        lp.col_cost_.push_back(0.0);
    }
    SECTION("column lower bounds of the wrong size") {
        lp.col_lower_.pop_back();
    }
    SECTION("column upper bounds of the wrong size") {
        lp.col_upper_.pop_back();
    }
    SECTION("row lower bounds of the wrong size") {
        lp.row_lower_.pop_back();
    }
    SECTION("row upper bounds of the wrong size") {
        lp.row_upper_.push_back(1.0);
    }
    SECTION("a row-wise matrix") {
        lp.a_matrix_.format_ = MatrixFormat::kRowwise;
    }
    SECTION("matrix dimensions that disagree") {
        lp.a_matrix_.num_row_ = 2;
    }
    SECTION("column starts of the wrong size") {
        lp.a_matrix_.start_.pop_back();
    }
    SECTION("column starts that are not monotone") {
        lp.a_matrix_.start_ = {0, 2, 1, 6, 8};
    }
    SECTION("starts that do not cover the entries") {
        lp.a_matrix_.index_.push_back(0);
    }
    SECTION("a row index out of range") {
        lp.a_matrix_.index_[3] = 3;
    }
    SECTION("a negative row index") {
        // Not in column 0: without the lower check, `last_col[-1]` there
        // reads memory that can equal 0 and trip the repeat test instead.
        lp.a_matrix_.index_[3] = -1;
    }
    SECTION("a NaN coefficient") {
        lp.a_matrix_.value_[2] = kNaN;
    }
    SECTION("a NaN cost") {
        lp.col_cost_[1] = kNaN;
    }
    SECTION("a NaN column bound") {
        lp.col_upper_[3] = kNaN;
    }
    SECTION("a NaN row bound") {
        lp.row_lower_[2] = kNaN;
    }
    SECTION("a NaN offset") {
        lp.offset_ = kNaN;
    }
    SECTION("column starts that do not begin at 0") {
        // Consistent otherwise: the first column just starts one entry in.
        lp.a_matrix_.start_[0] = 1;
    }
    SECTION("a row repeated within a column") {
        lp.a_matrix_.index_[1] = 0;
    }
    SECTION("an infinite coefficient") {
        lp.a_matrix_.value_[0] = kHighsInf;
    }
    SECTION("a coefficient at large_matrix_value") {
        lp.a_matrix_.value_[0] = -1e15;
    }
    SECTION("an infinite cost") {
        lp.col_cost_[0] = -kHighsInf;
    }
    SECTION("a cost at infinite_cost") {
        lp.col_cost_[0] = 1e20;
    }
    SECTION("an infinite offset") {
        lp.offset_ = kHighsInf;
    }
    // An enum value outside its enumerators, written as a corrupt input
    // would carry it: the raw bytes, not a cast of a literal.
    SECTION("an integrality outside HighsVarType") {
        const std::uint8_t raw = 7;
        std::memcpy(&lp.integrality_[1], &raw, sizeof raw);
    }
    SECTION("a sense outside ObjSense") {
        const int raw = 0;
        std::memcpy(&lp.sense_, &raw, sizeof raw);
    }
    SECTION("a column whose lower bound is +infinity") {
        lp.col_lower_[3] = kHighsInf;
        lp.col_upper_[3] = kHighsInf;
    }
    SECTION("a column whose upper bound is -infinity") {
        lp.col_lower_[3] = -kHighsInf;
        lp.col_upper_[3] = -kHighsInf;
    }
    SECTION("a row whose lower bound is +infinity") {
        lp.row_lower_[2] = kHighsInf;
        lp.row_upper_[2] = kHighsInf;
    }
    SECTION("a row whose upper bound is -infinity") {
        lp.row_lower_[0] = -kHighsInf;
        lp.row_upper_[0] = -kHighsInf;
    }
    SECTION("a row lower bound at +infinite_bound") {
        lp.row_lower_[2] = 1e20;
    }
    REQUIRE(refusal(lp) == ProblemError::Kind::kMalformed);
}

TEST_CASE("core: make_problem refuses a model infeasible on its face", "[core][validation]") {
    HighsLp lp = maximisation_model();
    SECTION("crossed continuous column bounds") {
        lp.col_lower_[3] = 2.0;
        lp.col_upper_[3] = 1.0;
    }
    SECTION("crossed row bounds") {
        lp.row_lower_[0] = 2.0;
        lp.row_upper_[0] = 1.0;
    }
    SECTION("integer bounds with no integer between them") {
        lp.col_lower_[0] = 0.5;
        lp.col_upper_[0] = 0.7;
    }
    SECTION("an empty row whose bounds exclude 0") {
        // A fourth row with no entries, bounds [1, 2].
        lp.num_row_ = 4;
        lp.a_matrix_.num_row_ = 4;
        lp.row_lower_.push_back(1.0);
        lp.row_upper_.push_back(2.0);
    }
    REQUIRE(refusal(lp) == ProblemError::Kind::kInfeasible);
}

TEST_CASE("core: make_problem accepts an empty row whose bounds hold 0", "[core][validation]") {
    HighsLp lp = maximisation_model();
    lp.num_row_ = 4;
    lp.a_matrix_.num_row_ = 4;
    lp.row_lower_.push_back(-1.0);
    lp.row_upper_.push_back(kHighsInf);
    REQUIRE_FALSE(refusal(lp).has_value());
}

TEST_CASE("core: make_problem rounds integer bounds inward", "[core][validation]") {
    HighsLp lp = maximisation_model();
    lp.col_lower_[0] = -1e-9;       // within feastol of 0: rounds to 0, not -0
    lp.col_upper_[0] = 1.0 + 1e-9;  // within feastol of 1: rounds to 1
    lp.col_lower_[1] = 0.3;         // rounds up to 1
    lp.col_upper_[1] = 2.7;         // rounds down to 2
    lp.col_upper_[3] = 2.5;         // continuous: untouched
    ProblemStorage storage;
    const auto made = make_problem(lp, storage, kFeasTol, kEpsilon);
    REQUIRE(made.has_value());
    const ProblemView& p = *made;
    REQUIRE((*p.col_lower)[0] == 0.0);
    REQUIRE_FALSE(std::signbit((*p.col_lower)[0]));
    REQUIRE((*p.col_upper)[0] == 1.0);
    REQUIRE(p.binary[0] == 1);  // binary once rounded
    REQUIRE((*p.col_lower)[1] == 1.0);
    REQUIRE((*p.col_upper)[1] == 2.0);
    REQUIRE(p.binary[1] == 0);
    REQUIRE((*p.col_upper)[3] == 2.5);
    REQUIRE(lp.col_lower_[1] == 0.3);  // the model itself is untouched
}

TEST_CASE("core: make_problem builds the same view in reused storage", "[core][validation]") {
    // A caller running several models keeps one `ProblemStorage`: each view
    // built in it must equal one built in fresh storage, whatever the
    // storage held before, a starts array left non-zero at [0] included
    // (the transpose `resize`s its starts rather than assigning them).

    // minimise x0 - x1, x0 + 2 x1 <= 3, x0 integer
    const HighsLp other = [] {
        HighsLp lp;
        lp.num_col_ = 2;
        lp.num_row_ = 1;
        lp.col_cost_ = {1.0, -1.0};
        lp.col_lower_ = {0.0, -1.0};
        lp.col_upper_ = {4.0, 1.0};
        lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kContinuous};
        lp.row_lower_ = {-kHighsInf};
        lp.row_upper_ = {3.0};
        lp.a_matrix_.format_ = MatrixFormat::kColwise;
        lp.a_matrix_.num_col_ = 2;
        lp.a_matrix_.num_row_ = 1;
        lp.a_matrix_.start_ = {0, 1, 2};
        lp.a_matrix_.index_ = {0, 0};
        lp.a_matrix_.value_ = {1.0, 2.0};
        return lp;
    }();
    const HighsLp maximise = maximisation_model();

    ProblemStorage reused;
    SECTION("storage a previous make_problem filled") {}
    SECTION("storage holding anything") {
        reused.ar_start.assign(8, 5);
        reused.uplocks.assign(8, 5);
        reused.col_cost.assign(8, 5.0);
    }
    for (const HighsLp* lp : {&maximise, &other, &maximise}) {
        const auto view = make_problem(*lp, reused, kFeasTol, kEpsilon);
        ProblemStorage fresh;
        const auto want = make_problem(*lp, fresh, kFeasTol, kEpsilon);
        REQUIRE(view.has_value());
        REQUIRE(want.has_value());
        REQUIRE(*view->ar_start == *want->ar_start);
        REQUIRE(*view->ar_index == *want->ar_index);
        REQUIRE(*view->ar_value == *want->ar_value);
        REQUIRE(view->csc->col_start == want->csc->col_start);
        REQUIRE(view->csc->col_row == want->csc->col_row);
        REQUIRE(view->csc->col_val == want->csc->col_val);
        REQUIRE(*view->uplocks == *want->uplocks);
        REQUIRE(*view->downlocks == *want->downlocks);
        REQUIRE(*view->col_cost == *want->col_cost);
        REQUIRE(view->offset == want->offset);
        REQUIRE(*view->integrality == *want->integrality);
        REQUIRE(*view->col_lower == *want->col_lower);
        REQUIRE(*view->col_upper == *want->col_upper);
        REQUIRE(*view->row_lower == *want->row_lower);
        REQUIRE(*view->row_upper == *want->row_upper);
        REQUIRE(view->binary == want->binary);
    }
}

TEST_CASE("core: make_problem maps bounds of infinite_bound or more to infinity",
          "[core][validation]") {
    HighsLp lp = maximisation_model();
    lp.col_lower_[1] = -1e30;  // integer column: mapped, then rounding keeps it
    lp.col_upper_[1] = 1e30;
    lp.col_upper_[3] = 1e20;  // exactly `infinite_bound`
    lp.row_upper_[2] = 1e25;
    lp.row_lower_[0] = -1e21;
    ProblemStorage storage;
    const auto made = make_problem(lp, storage, kFeasTol, kEpsilon);
    REQUIRE(made.has_value());
    const ProblemView& p = *made;
    REQUIRE((*p.col_lower)[1] == -kHighsInf);
    REQUIRE((*p.col_upper)[1] == kHighsInf);
    REQUIRE((*p.col_upper)[3] == kHighsInf);
    REQUIRE((*p.row_upper)[2] == kHighsInf);
    REQUIRE((*p.row_lower)[0] == -kHighsInf);
    REQUIRE(lp.col_upper_[1] == 1e30);  // the model itself is untouched
}

TEST_CASE("core: workers stay feasible with huge integer bounds", "[core][validation]") {
    // Before the mapping, FJ took [-1e30, 1e30] literally and reported a
    // point violating a row.
    const QuietLog log;
    HighsLp lp = maximisation_model();
    lp.col_lower_[1] = -1e30;
    lp.col_upper_[1] = 1e30;
    lp.row_upper_[2] = 5.0;  // x1 + x3 <= 5 keeps the maximisation bounded
    const Setup s(lp, log.options);
    Found found(*s.problem.integrality);
    run_both(s, found);
    for (BestSink* sink : {&found.fj, &found.local_mip}) {
        REQUIRE(sink->found);
        REQUIRE(is_feasible(lp, sink->solution));
    }
}

TEST_CASE("core: workers return at once on a degenerate view", "[core][validation]") {
    // Two models make_problem accepts and no worker can search: nothing at
    // all, and one fixed integer column with no rows.  An attempt on either
    // charges no effort, so before the workers checked for it nothing but
    // the clock ended LocalMIP's, and with no deadline nothing did.
    const QuietLog log;
    HighsLp empty;
    empty.a_matrix_.format_ = MatrixFormat::kColwise;
    empty.a_matrix_.start_ = {0};
    HighsLp fixed_column = empty;
    fixed_column.num_col_ = 1;
    fixed_column.col_cost_ = {1.0};
    fixed_column.col_lower_ = {2.0};
    fixed_column.col_upper_ = {2.0};
    fixed_column.integrality_ = {HighsVarType::kInteger};
    fixed_column.a_matrix_.num_col_ = 1;
    fixed_column.a_matrix_.start_ = {0, 0};
    for (const HighsLp* lp : {&empty, &fixed_column}) {
        const Setup s(*lp, log.options);  // no deadline, no terminator
        REQUIRE(s.problem.degenerate());
        SolutionSink sink(*s.problem.integrality, 0);
        FjWorker fj(s.problem, s.exec, sink, kBudget, kBudget, 0, {}, WorkerTrace{0, 0});
        local_mip_detail::LocalMipWorker local_mip(s.problem, s.exec, sink, kBudget, kBudget, 0,
                                                   nullptr, WorkerTrace{0, 0});
        REQUIRE(fj.finished());
        REQUIRE(local_mip.finished());
        REQUIRE(fj.run_attempt(kAttemptBudget).effort == 0);
        REQUIRE(local_mip.run_attempt(kAttemptBudget).effort == 0);
    }
}

TEST_CASE("core: LocalMIP sees its deadline when its steps charge nothing", "[core][validation]") {
    // One integer column fixed at 2 in a row that needs 5: accepted (only
    // an empty row is checked on its face), not degenerate, and LocalMIP has
    // no move that could repair the row, so its steps charge no effort and
    // the work-based poll never comes round again.  Only the
    // step-count poll sees the deadline; without it the attempt never
    // returns, which the watchdog turns into a failure.
    const QuietLog log;
    HighsLp lp;
    lp.num_col_ = 1;
    lp.num_row_ = 1;
    lp.col_cost_ = {1.0};
    lp.col_lower_ = {2.0};
    lp.col_upper_ = {2.0};
    lp.integrality_ = {HighsVarType::kInteger};
    lp.row_lower_ = {5.0};
    lp.row_upper_ = {kHighsInf};
    lp.a_matrix_.format_ = MatrixFormat::kColwise;
    lp.a_matrix_.num_col_ = 1;
    lp.a_matrix_.num_row_ = 1;
    lp.a_matrix_.start_ = {0, 1};
    lp.a_matrix_.index_ = {0};
    lp.a_matrix_.value_ = {1.0};
    Setup s(lp, log.options);
    REQUIRE_FALSE(s.problem.degenerate());
    HighsTimer timer;
    timer.start();
    s.exec.deadline = Deadline{.timer = &timer, .limit = timer.read() + 0.05};
    SolutionSink sink(*s.problem.integrality, 0);
    local_mip_detail::LocalMipWorker local_mip(s.problem, s.exec, sink, kBudget, kBudget, 0,
                                               nullptr, WorkerTrace{0, 0});
    auto attempt =
        std::async(std::launch::async, [&] { return local_mip.run_attempt(kBudget).effort; });
    if (attempt.wait_for(std::chrono::seconds(30)) != std::future_status::ready) {
        std::fputs("LocalMIP never saw its deadline\n", stderr);
        std::_Exit(1);  // the attempt still runs on the stack it reads
    }
    REQUIRE(attempt.get() < kBudget);
    REQUIRE(s.exec.past_deadline());
}

TEST_CASE("core: workers stay inside integer bounds that need rounding", "[core][validation]") {
    // x1's bounds [0.5, 1.7] hold only the integer 1, and its cost pulls
    // it up.  Unrounded, LocalMIP's `clamp_and_round` rounds first and
    // clamps second, so it lands on 1.7 and reports a non-integral point.
    const QuietLog log;
    HighsLp lp = maximisation_model();
    lp.col_lower_[1] = 0.5;
    lp.col_upper_[1] = 1.7;
    lp.row_upper_[0] = 4.0;  // room for x1 to be worth pushing up
    const Setup s(lp, log.options);
    Found found(*s.problem.integrality);
    run_both(s, found);
    for (BestSink* sink : {&found.fj, &found.local_mip}) {
        REQUIRE(sink->found);
        REQUIRE(is_feasible(lp, sink->solution));
    }
}
