#include "Highs.h"
#include "test_common.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

// ── FJ standalone: HiGHS dispatches FJ via fj::run ──

TEST_CASE("FJ standalone: flugpl", "[heuristic][fj]") {
    REQUIRE(solve_suite("flugpl.mps", "fj") == Catch::Approx(1201500.0).epsilon(1e-6));
}

// ── FJ personas (#206): behind `mip_heuristic_fj_personas`, default off ──

TEST_CASE("FJ personas: p0548 keeps its optimum and logs its row structure",
          "[heuristic][fj][fj_personas]") {
    double objective = 0.0;
    const auto lines = solve_capturing_log(
        "p0548.mps",
        [](Highs& h) {
            select_heuristics(h, "fj");
            require_option(h, "mip_heuristic_fj_personas", true);
            require_option(h, "log_dev_level", 3);
        },
        [&](Highs& h) { h.getInfoValue("objective_function_value", objective); });
    CHECK(objective == Catch::Approx(8691.0).epsilon(1e-6));
    CHECK(log_contains(lines, "[FjPersonas] partitioning="));
}

TEST_CASE("FJ personas: off by default, with no detection pass", "[heuristic][fj][fj_personas]") {
    const auto lines = solve_capturing_log("p0548.mps", [](Highs& h) {
        select_heuristics(h, "fj");
        require_option(h, "log_dev_level", 3);
    });
    CHECK(heuristic_reported_effort(lines, "fj"));
    CHECK_FALSE(log_contains(lines, "[FjPersonas]"));
}
