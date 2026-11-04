// cuOpt's cardinality exchange in HiGHS's vendored FeasibilityJump (#218):
// one binary of an exact-k row from 1 to 0 and another from 0 to 1 in one
// step, `FeasibilityJumpSolver::setExchangeGroups` as `apply_patch.cmake`
// inserts it.
//
// The models are hand-built so that the first step is a local minimum for
// scalar moves: taking the forced column off the exact-k row fixes one row
// and breaks another, a zero score at equal weights.  The exchange keeps the
// exact-k row whole, which only shows if a row both columns touch is scored
// on their summed delta; scored per column it would read as two violations.
// So a feasible point reached through exactly one exchange pins that
// combination as well as the move.

#include "io/HighsIO.h"
#include "mip/feasibilityjump.hh"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <vector>

namespace {

namespace fj = external_feasibilityjump;

constexpr double kEqualityTolerance = 1e-5;
constexpr double kViolationTolerance = 1e-5;
// A bound on a run that never reaches feasibility.
constexpr size_t kEffortCap = 1000000;

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

struct Column {
    double lower = 0.0;
    double upper = 1.0;
    double cost = 0.0;
    double start = 0.0;
};

struct Row {
    fj::RowType sense = fj::RowType::Equal;
    double rhs = 0.0;
    std::vector<int> index;
    std::vector<double> value;
};

struct Model {
    std::vector<Column> columns;
    std::vector<Row> rows;
};

enum class Groups : std::uint8_t { kNotSet, kEmpty, kRowZero };

struct Outcome {
    bool feasible = false;
    std::vector<double> solution;
    size_t effort = 0;
    size_t exchanges = 0;
};

// Runs FJ to its first feasible point; with `kRowZero`, row 0's columns are
// the one exchange group.
Outcome run(const Model& model, Groups groups, uint32_t seed = 0) {
    SilentLog log;
    fj::FeasibilityJumpSolver solver(log.options, static_cast<int>(seed), kEqualityTolerance,
                                     kViolationTolerance);
    std::vector<double> start;
    for (const Column& c : model.columns) {
        solver.addVar(fj::VarType::Integer, c.lower, c.upper, c.cost);
        start.push_back(c.start);
    }
    for (const Row& r : model.rows) {
        std::vector<int> index = r.index;
        std::vector<double> value = r.value;
        solver.addConstraint(r.sense, r.rhs, static_cast<int>(index.size()), index.data(),
                             value.data(), /*relax_continuous=*/0);
    }
    if (groups == Groups::kEmpty) {
        solver.setExchangeGroups({0}, {});
    } else if (groups == Groups::kRowZero) {
        const std::vector<int>& members = model.rows.front().index;
        solver.setExchangeGroups({0, static_cast<uint32_t>(members.size())},
                                 std::vector<uint32_t>(members.begin(), members.end()));
    }

    Outcome out;
    auto callback = [&out](fj::FJStatus status) {
        if (status.solution != nullptr) {
            out.feasible = true;
            out.solution.assign(status.solution, status.solution + status.numVars);
            out.effort = status.totalEffort;
            return fj::CallbackControlFlow::Terminate;
        }
        return status.totalEffort > kEffortCap ? fj::CallbackControlFlow::Terminate
                                               : fj::CallbackControlFlow::Continue;
    };
    solver.solve(start.data(), callback);
    out.exchanges = solver.numExchangeMoves();
    return out;
}

// x0 + ... + x(n-1) == 1 with x0 = 1 at the start, and x0 <= 0.
Model forced_off(int n) {
    Model m;
    m.columns.resize(static_cast<size_t>(n));
    m.columns[0].start = 1.0;
    for (int j = 1; j < n; ++j)
        m.columns[static_cast<size_t>(j)].cost = j;
    Row exact{fj::RowType::Equal, 1.0, {}, {}};
    for (int j = 0; j < n; ++j) {
        exact.index.push_back(j);
        exact.value.push_back(1.0);
    }
    m.rows.push_back(exact);
    m.rows.push_back({fj::RowType::Lte, 0.0, {0}, {1.0}});
    return m;
}

double row_activity(const Row& r, const std::vector<double>& x) {
    double lhs = 0.0;
    for (size_t p = 0; p < r.index.size(); ++p)
        lhs += r.value[p] * x[static_cast<size_t>(r.index[p])];
    return lhs;
}

}  // namespace

TEST_CASE("FJ exchange off: no groups and empty groups search alike", "[fj][exchange]") {
    const Model m = forced_off(12);
    for (uint32_t seed : {0u, 1u, 7u}) {
        const Outcome none = run(m, Groups::kNotSet, seed);
        const Outcome empty = run(m, Groups::kEmpty, seed);
        REQUIRE(none.feasible);
        CHECK(empty.feasible);
        CHECK(empty.solution == none.solution);
        CHECK(empty.effort == none.effort);
        CHECK(none.exchanges == 0);
        CHECK(empty.exchanges == 0);
    }
}

TEST_CASE("FJ exchange: one exchange keeps the exact-k row whole", "[fj][exchange]") {
    const Model m = forced_off(12);
    for (uint32_t seed : {0u, 1u, 7u}) {
        const Outcome out = run(m, Groups::kRowZero, seed);
        REQUIRE(out.feasible);
        CHECK(out.exchanges == 1);
        CHECK(out.solution[0] == 0.0);
        CHECK(row_activity(m.rows[0], out.solution) == 1.0);
    }
}

TEST_CASE("FJ exchange: a fixed column is never moved", "[fj][exchange]") {
    // x0 + x1 + x2 + x3 == 2 from (1, 1, 0, 0), with x1 <= 0.  x0 is fixed
    // at 1 and x2 at 0, so the only exchange within bounds is x1 off, x3 on.
    Model m;
    m.columns = {
        {1.0, 1.0, 0.0, 1.0}, {0.0, 1.0, 0.0, 1.0}, {0.0, 0.0, -5.0, 0.0}, {0.0, 1.0, 5.0, 0.0}};
    m.rows.push_back({fj::RowType::Equal, 2.0, {0, 1, 2, 3}, {1.0, 1.0, 1.0, 1.0}});
    m.rows.push_back({fj::RowType::Lte, 0.0, {1}, {1.0}});
    for (uint32_t seed : {0u, 1u, 7u, 11u}) {
        const Outcome out = run(m, Groups::kRowZero, seed);
        REQUIRE(out.feasible);
        CHECK(out.exchanges >= 1);
        CHECK(out.solution == std::vector<double>{1.0, 0.0, 0.0, 1.0});
    }
}
