#include "heuristic_common.h"
#include "Highs.h"
#include "test_common.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Smoke test: solve small MIP", "[basic]") {
    // min x + y
    // s.t. x + y >= 1
    //      x, y in {0, 1}
    Highs highs;
    highs.setOptionValue("output_flag", false);

    highs.addVar(0.0, 1.0);
    highs.addVar(0.0, 1.0);
    highs.changeColCost(0, 1.0);
    highs.changeColCost(1, 1.0);
    highs.changeColIntegrality(0, HighsVarType::kInteger);
    highs.changeColIntegrality(1, HighsVarType::kInteger);

    HighsInt idx[] = {0, 1};
    double val[] = {1.0, 1.0};
    highs.addRow(1.0, kHighsInf, 2, idx, val);

    HighsStatus status = highs.run();
    REQUIRE(status == HighsStatus::kOk);

    HighsInt sol_status;
    highs.getInfoValue("primal_solution_status", sol_status);
    REQUIRE(sol_status == kSolutionStatusFeasible);

    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == 1.0);
}

TEST_CASE("Options: disable custom heuristics", "[options]") {
    REQUIRE(solve_suite("flugpl.mps", "off") == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("Options: effort split defaults", "[options]") {
    // The effort-option split contract: mip_heuristic_effort keeps vanilla
    // HiGHS semantics and default (0.05, the B&B LP-iteration fraction that
    // gates RENS/RINS and sizes fpr_lp), while the presolve heuristics draw
    // their budget from mip_heuristic_presolve_effort (default 0.30, the
    // pre-split patched default).  A patched binary at default options must
    // match vanilla's B&B heuristic budget exactly.
    Highs highs;
    highs.setOptionValue("output_flag", false);
    double effort = -1.0;
    REQUIRE(highs.getOptionValue("mip_heuristic_effort", effort) == HighsStatus::kOk);
    REQUIRE(effort == 0.05);
    double presolve_effort = -1.0;
    REQUIRE(highs.getOptionValue("mip_heuristic_presolve_effort", presolve_effort) ==
            HighsStatus::kOk);
    REQUIRE(presolve_effort == 0.30);
    // Settable across the documented [0, 1] range.
    REQUIRE(highs.setOptionValue("mip_heuristic_presolve_effort", 0.0) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_heuristic_presolve_effort", 1.0) == HighsStatus::kOk);
}

// #92 and #93 deleted their options from the patch rather than leaving
// them silently ignored knobs.  Epic #88's coupling B is that every
// pre-existing HiGHS build tree still registers the old option set; the
// PATCH_VERSION and retired-identifier guards in apply_patch.cmake are the
// primary defence and this is the runtime backstop for a stale tree or a
// patch-script regression that re-adds one.
TEST_CASE("Options: retired options are gone", "[options]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    REQUIRE(highs.setOptionValue("mip_heuristic_opportunistic", true) != HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_heuristic_preset", std::string("off")) != HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_heuristic_run_fpr", true) != HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_heuristic_run_local_mip", true) != HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_heuristic_run_scylla", true) != HighsStatus::kOk);
}

TEST_CASE("Options: suite defaults to all and accepts every value", "[options][suite]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    std::string suite;
    REQUIRE(highs.getOptionValue("mip_heuristic_suite", suite) == HighsStatus::kOk);
    REQUIRE(suite == "all");
    // HiGHS does not validate string option *values*, so every one of these
    // returns kOk — including the bogus one below.  What this asserts is that
    // the option exists under this exact name; the dispatcher is what
    // distinguishes a known value from an unknown one.
    for (const char* value : {"off", "fj", "fpr", "local_mip", "scylla", "all"}) {
        REQUIRE(highs.setOptionValue("mip_heuristic_suite", std::string(value)) ==
                HighsStatus::kOk);
    }
}

TEST_CASE("Options: unknown suite value warns and runs everything", "[options][suite]") {
    const auto lines = solve_capturing_log("flugpl.mps", [](Highs& h) {
        require_option(h, "log_dev_level", 3);
        set_suite(h, "bogus");
    });
    bool warned = false;
    for (const auto& line : lines) {
        if (line.find("Unknown mip_heuristic_suite value \"bogus\"") != std::string::npos) {
            warned = true;
        }
    }
    REQUIRE(warned);
    // Fail-open: all four heuristics are dispatched, exactly as at `all`.
    // Asserted on the presence of the trace line rather than on non-zero
    // effort — Scylla can legitimately report zero on an instance this
    // small, and what is under test here is the flag set, not the search.
    for (const char* heur : {"fj", "fpr", "local_mip", "scylla"}) {
        const std::string tag = std::string("[Sequential] heur=") + heur + " ";
        bool dispatched = false;
        for (const auto& line : lines) {
            dispatched = dispatched || line.find(tag) != std::string::npos;
        }
        INFO("heuristic " << heur);
        REQUIRE(dispatched);
    }
}
