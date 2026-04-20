#include "heuristic_common.h"
#include "Highs.h"
#include "prop_engine.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <vector>

// ===================================================================
// PropEngine unit tests
// ===================================================================

// Helper: build a small test model for PropEngine tests.
// 3 variables: x0 (binary), x1 (integer [0,5]), x2 (continuous [0,10])
// 2 constraints:
//   row 0: x0 + x1 >= 2   (row_lo=2, row_hi=inf)
//   row 1: x1 + x2 <= 8   (row_lo=-inf, row_hi=8)
namespace {
struct SmallModel {
    static constexpr HighsInt ncol = 3;
    static constexpr HighsInt nrow = 2;
    std::vector<HighsInt> ar_start = {0, 2, 4};
    std::vector<HighsInt> ar_index = {0, 1, 1, 2};
    std::vector<double> ar_value = {1.0, 1.0, 1.0, 1.0};
    std::vector<double> col_lb = {0.0, 0.0, 0.0};
    std::vector<double> col_ub = {1.0, 5.0, 10.0};
    double row_lo[2] = {2.0, -kHighsInf};
    double row_hi[2] = {kHighsInf, 8.0};
    std::vector<HighsVarType> integrality = {HighsVarType::kInteger, HighsVarType::kInteger,
                                             HighsVarType::kContinuous};
    CscMatrix csc;

    SmallModel() { csc = build_csc(ncol, nrow, ar_start, ar_index, ar_value); }

    PropEngine make_engine(double feastol = 1e-6) {
        return PropEngine(ncol, nrow, ar_start.data(), ar_index.data(), ar_value.data(), csc,
                          col_lb.data(), col_ub.data(), row_lo, row_hi, integrality.data(),
                          feastol);
    }
};
}  // namespace

TEST_CASE("PropEngine: fix and propagate", "[prop-engine]") {
    SmallModel m;
    auto eng = m.make_engine();

    // Initial state: all variables unfixed with global bounds
    REQUIRE_FALSE(eng.var(0).fixed);
    REQUIRE(eng.var(1).lb == Catch::Approx(0.0));
    REQUIRE(eng.var(1).ub == Catch::Approx(5.0));

    // Fix x0 = 0, propagate: row 0 (x0+x1 >= 2) forces x1 >= 2
    REQUIRE(eng.fix(0, 0.0));
    REQUIRE(eng.propagate(0));
    REQUIRE(eng.var(0).fixed);
    REQUIRE(eng.var(0).val == Catch::Approx(0.0));
    REQUIRE(eng.var(1).lb >= 2.0 - 1e-6);

    // Fix x1 = 5, propagate: row 1 (x1+x2 <= 8) forces x2 <= 3
    REQUIRE(eng.fix(1, 5.0));
    REQUIRE(eng.propagate(1));
    REQUIRE(eng.var(2).ub <= 3.0 + 1e-6);
}

TEST_CASE("PropEngine: backtrack restores state", "[prop-engine]") {
    SmallModel m;
    auto eng = m.make_engine();

    HighsInt vs_m = eng.vs_mark();
    HighsInt sol_m = eng.sol_mark();

    REQUIRE(eng.fix(0, 1.0));
    REQUIRE(eng.propagate(0));
    REQUIRE(eng.var(0).fixed);

    eng.backtrack_to(vs_m, sol_m);
    REQUIRE_FALSE(eng.var(0).fixed);
    REQUIRE(eng.var(0).lb == Catch::Approx(0.0));
    REQUIRE(eng.var(0).ub == Catch::Approx(1.0));
    REQUIRE(eng.var(1).lb == Catch::Approx(0.0));
}

TEST_CASE("PropEngine: tighten bounds and auto-fix", "[prop-engine]") {
    SmallModel m;
    auto eng = m.make_engine();

    REQUIRE(eng.tighten_lb(1, 3.0));
    REQUIRE(eng.var(1).lb >= 3.0 - 1e-6);
    REQUIRE(eng.var(1).ub == Catch::Approx(5.0));

    // Tighten ub to match lb — should auto-fix
    REQUIRE(eng.tighten_ub(1, 3.0));
    REQUIRE(eng.var(1).fixed);
    REQUIRE(eng.var(1).val == Catch::Approx(3.0));
}

TEST_CASE("PropEngine: infeasible propagation", "[prop-engine]") {
    SmallModel m;
    auto eng = m.make_engine();

    // Fix x0=0, tighten x1 ub to 1. Propagation from row 0 (x0+x1 >= 2)
    // tries to tighten x1 lb to 2, but ub is 1 → lb > ub → infeasible.
    REQUIRE(eng.fix(0, 0.0));
    REQUIRE(eng.tighten_ub(1, 1.0));
    REQUIRE_FALSE(eng.propagate(0));
}

TEST_CASE("PropEngine: reset clears state", "[prop-engine]") {
    SmallModel m;
    auto eng = m.make_engine();

    eng.fix(0, 1.0);
    eng.propagate(0);
    REQUIRE(eng.var(0).fixed);

    eng.reset();
    REQUIRE_FALSE(eng.var(0).fixed);
    REQUIRE(eng.var(0).lb == Catch::Approx(0.0));
    REQUIRE(eng.var(0).ub == Catch::Approx(1.0));
    REQUIRE(eng.var(1).lb == Catch::Approx(0.0));
    REQUIRE(eng.var(1).ub == Catch::Approx(5.0));
}

TEST_CASE("PropEngine: effort tracking", "[prop-engine]") {
    SmallModel m;
    auto eng = m.make_engine();

    size_t before = eng.effort();
    eng.fix(0, 1.0);
    eng.propagate(0);
    REQUIRE(eng.effort() > before);

    eng.add_effort(100);
    REQUIRE(eng.effort() >= before + 100);
}
