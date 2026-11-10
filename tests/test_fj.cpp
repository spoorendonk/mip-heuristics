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
    const ScopedThreadPin pin;
    double objective = 0.0;
    const auto lines = solve_capturing_log(
        "p0548.mps",
        [](Highs& h) {
            select_heuristics(h, "fj");
            require_option(h, "mip_heuristic_fj_personas", true);
            require_option(h, "log_dev_level", 3);
            // Slot 0 is always seeded: a persona needs a second worker.
            require_option(h, "threads", 4);
        },
        [&](Highs& h) { h.getInfoValue("objective_function_value", objective); });
    CHECK(objective == Catch::Approx(8691.0).epsilon(1e-6));
    CHECK(log_contains(lines, "[FjPersonas] partitioning="));
}

TEST_CASE("FJ personas: a partitioning model gets a covering and an exact-k slot",
          "[heuristic][fj][fj_personas]") {
    // issue-2095 keeps 23 `= 1` binary rows through presolve: partitioning,
    // so exact-k, rows that bind from below.
    const ScopedThreadPin pin;
    const auto lines = solve_capturing_log("issue-2095.mps", [](Highs& h) {
        select_heuristics(h, "fj");
        require_option(h, "mip_heuristic_fj_personas", true);
        require_option(h, "log_dev_level", 3);
        require_option(h, "threads", 4);
        require_option(h, "mip_max_nodes", 1);
    });
    CHECK(log_contains(lines, "slots_covering=1 slots_exact_k=1"));
}

TEST_CASE("FJ exchange: a partitioning model gets an exchange slot without personas",
          "[heuristic][fj][fj_exchange]") {
    // The exchange (#218) needs the detection pass on its own: slot 1 of
    // four is cuOpt's first exchange lane, and the exact-k slot stays unset.
    const ScopedThreadPin pin;
    const auto lines = solve_capturing_log("issue-2095.mps", [](Highs& h) {
        select_heuristics(h, "fj");
        require_option(h, "mip_heuristic_fj_exchange", true);
        require_option(h, "log_dev_level", 3);
        require_option(h, "threads", 4);
        require_option(h, "mip_max_nodes", 1);
    });
    CHECK(log_contains(lines, "slots_covering=0 slots_exact_k=0 slots_exchange=1"));
}

TEST_CASE("FJ personas: off by default, with no detection pass", "[heuristic][fj][fj_personas]") {
    const auto lines = solve_capturing_log("p0548.mps", [](Highs& h) {
        select_heuristics(h, "fj");
        require_option(h, "log_dev_level", 3);
    });
    CHECK(heuristic_reported_effort(lines, "fj"));
    CHECK_FALSE(log_contains(lines, "[FjPersonas]"));
}

// ── FJ's CDCL lane (#237): behind `mip_heuristic_fj_cdcl`, default off ──

TEST_CASE("FJ CDCL lane: p0548 keeps its optimum with the lane on",
          "[heuristic][fj][fj_cdcl][mode-matrix]") {
    // Whether the lane's search runs or stands down depends on what the
    // other slots and presolve found first, so the slot is checked, not the
    // search's status.
    const ScopedThreadPin pin;
    double objective = 0.0;
    const auto lines = solve_capturing_log(
        "p0548.mps",
        [](Highs& h) {
            select_heuristics(h, "fj");
            require_option(h, "mip_heuristic_fj_cdcl", true);
            require_option(h, "log_dev_level", 3);
            // Slot 0 is vanilla FJ: the lane needs a second worker.
            require_option(h, "threads", 4);
        },
        [&](Highs& h) { h.getInfoValue("objective_function_value", objective); });
    CHECK(objective == Catch::Approx(8691.0).epsilon(1e-6));
    CHECK(log_contains(lines, "[FjCdcl] slot=3 workers=4"));
}

TEST_CASE("FJ CDCL lane: off by default", "[heuristic][fj][fj_cdcl][mode-matrix]") {
    const ScopedThreadPin pin;
    bool on = true;
    Highs probe;
    REQUIRE(probe.getOptionValue("mip_heuristic_fj_cdcl", on) == HighsStatus::kOk);
    CHECK_FALSE(on);
    const auto lines = solve_capturing_log("p0548.mps", [](Highs& h) {
        select_heuristics(h, "fj");
        require_option(h, "log_dev_level", 3);
        require_option(h, "threads", 4);
    });
    CHECK(heuristic_reported_effort(lines, "fj"));
    CHECK_FALSE(log_contains(lines, "[FjCdcl]"));
}
