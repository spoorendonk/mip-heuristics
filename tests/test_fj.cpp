#include "Highs.h"
#include "test_common.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

// ── FJ standalone: HiGHS dispatches FJ via fj::run ──

TEST_CASE("FJ standalone: flugpl", "[heuristic][fj]") {
    REQUIRE(solve_suite("flugpl.mps", "fj") == Catch::Approx(1201500.0).epsilon(1e-6));
}
