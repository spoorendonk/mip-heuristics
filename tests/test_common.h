#pragma once

// Shared helpers for the Catch2 test suite.
//
// This header gathers the small number of helpers that more than one
// per-topic test translation unit needs.  It intentionally keeps the
// surface area small — anything used by a single file stays in that
// file as a file-local helper.

#include "Highs.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <string>

// Path to the HiGHS-provided `check/instances/` directory, injected by
// CMake via `INSTANCES_DIR`.  Defined inline so every translation unit
// that includes this header gets its own const reference.
inline const std::string kInstancesDir = INSTANCES_DIR;

// Solve `inst` with the requested (portfolio × opportunistic) cell of
// the execution matrix and return the final objective.  Used by the
// mode-matrix cross-heuristic parity tests.
inline double solve_mode(const char* inst, bool portfolio, bool opp) {
    Highs h;
    h.setOptionValue("output_flag", false);
    h.setOptionValue("mip_heuristic_portfolio", portfolio);
    h.setOptionValue("mip_heuristic_opportunistic", opp);
    REQUIRE(h.readModel(std::string(INSTANCES_DIR) + "/" + inst) == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    double obj;
    h.getInfoValue("objective_function_value", obj);
    return obj;
}

// Solve flugpl with every custom heuristic disabled in the requested
// (portfolio × opportunistic) cell — verifies none of the mode paths
// blocks HiGHS's built-in B&B fallback.
inline double solve_mode_no_heuristics(bool portfolio, bool opp) {
    Highs h;
    h.setOptionValue("output_flag", false);
    h.setOptionValue("mip_heuristic_portfolio", portfolio);
    h.setOptionValue("mip_heuristic_opportunistic", opp);
    h.setOptionValue("mip_heuristic_run_fpr", false);
    h.setOptionValue("mip_heuristic_run_local_mip", false);
    h.setOptionValue("mip_heuristic_run_feasibility_jump", false);
    h.setOptionValue("mip_heuristic_run_scylla", false);
    REQUIRE(h.readModel(std::string(INSTANCES_DIR) + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    double obj;
    h.getInfoValue("objective_function_value", obj);
    return obj;
}
