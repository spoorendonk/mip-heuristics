#include "heuristic_common.h"
#include "Highs.h"
#include "test_common.h"

#include <array>
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

    const auto idx = std::to_array<HighsInt>({0, 1});
    const auto val = std::to_array<double>({1.0, 1.0});
    highs.addRow(1.0, kHighsInf, 2, idx.data(), val.data());

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
    // gates RENS/RINS and sizes fpr_lp), while each presolve heuristic draws
    // its budget from its own mip_heuristic_<name>_effort option (#110).  A
    // patched binary at default options must match vanilla's B&B heuristic
    // budget exactly.
    //
    // The four defaults are pinned here because they are registered in
    // third_party/highs_patch/apply_patch.cmake, which nothing else compiles
    // or checks — a typo there is otherwise silent.  They are derived from
    // what the retired shared envelope handed each heuristic (FJ's
    // `nnz << 10` per worker; 0.30 x w/sum(w) for the retired weights
    // 2.99 / 6.16 / 1.00), but they only *approximate* it — see that file
    // for how far off, and in which direction, at each worker count and
    // suite value.
    Highs highs;
    highs.setOptionValue("output_flag", false);
    double effort = -1.0;
    REQUIRE(highs.getOptionValue("mip_heuristic_effort", effort) == HighsStatus::kOk);
    REQUIRE(effort == 0.05);
    struct EffortDefault {
        const char* name;
        double value;
    };
    const auto presolve_efforts = std::to_array<EffortDefault>({
        {"mip_heuristic_fj_effort", 2.84},
        {"mip_heuristic_fpr_effort", 7.672},
        {"mip_heuristic_local_mip_effort", 29.232},
        {"mip_heuristic_scylla_effort", 1.136},
    });
    for (const auto& [name, expected] : presolve_efforts) {
        double value = -1.0;
        REQUIRE(highs.getOptionValue(name, value) == HighsStatus::kOk);
        REQUIRE(value == expected);
        // Settable across the documented range.  `1.0` is the top of what
        // ships and tunes; the ceiling is `1e6`, which exists so #113's
        // calibration probe can hand a heuristic a budget that cannot bind
        // — with the patience gate off as well, the wall clock is then the
        // single stopping rule and the trace measures the heuristic rather
        // than the setting being derived from it.
        REQUIRE(highs.setOptionValue(name, 0.0) == HighsStatus::kOk);
        REQUIRE(highs.setOptionValue(name, 1.0) == HighsStatus::kOk);
        REQUIRE(highs.setOptionValue(name, 1e6) == HighsStatus::kOk);
        // Still bounded: an out-of-range write is rejected, not clamped,
        // and a rejected write is silent on every Highs instance we build
        // (they all set output_flag=false).
        REQUIRE(highs.setOptionValue(name, 1e6 * 10) != HighsStatus::kOk);
        REQUIRE(highs.setOptionValue(name, -1.0) != HighsStatus::kOk);
        REQUIRE(highs.setOptionValue(name, expected) == HighsStatus::kOk);
    }
}

TEST_CASE("Options: patience defaults", "[options][patience]") {
    // The four patience values were `constexpr` values in the heuristics'
    // own headers until #106 made them options, because the patience gate
    // — not the effort budget — is what actually limits a presolve
    // dispatch, and a constant cannot be swept without a rebuild per point.
    // They were spelled `mip_heuristic_<name>_stall` until #116, which
    // renamed them with **no alias**: the parameter is a floor on spend,
    // not a description of a state the search is in, and this project does
    // not carry option aliases.
    //
    // Pinned here for the same reason the effort defaults above are: they
    // are registered in third_party/highs_patch/apply_patch.cmake, which
    // nothing else compiles or checks.  They are **not** comparable across
    // heuristics (FJ counts step units, FPR and LocalMIP coefficient
    // accesses, Scylla PDLP iters x nnz), so read these as four unrelated
    // numbers that happen to share a suffix.
    Highs highs;
    highs.setOptionValue("output_flag", false);
    struct PatienceDefault {
        const char* name;
        double value;
        double effort;
    };
    // Same unit as the effort option beside it since #116 -- a multiple of
    // `nnz << 10` -- so the pair is directly comparable and every one of
    // these is a quarter of its own ceiling, which is what the clamp in the
    // #113 derivation produced and what `patience_threshold` now enforces
    // for any value.
    const auto patiences = std::to_array<PatienceDefault>({
        {"mip_heuristic_fj_patience", 0.71, 2.84},
        {"mip_heuristic_fpr_patience", 1.918, 7.672},
        {"mip_heuristic_local_mip_patience", 7.308, 29.232},
        {"mip_heuristic_scylla_patience", 0.284, 1.136},
    });
    for (const auto& [name, expected, effort] : patiences) {
        double value = -1.0;
        REQUIRE(highs.getOptionValue(name, value) == HighsStatus::kOk);
        REQUIRE(value == expected);
        // Below its own ceiling, or the gate can never fire.  Exactly a
        // quarter of it, in fact, which is where `kPatienceCeilingDivisor`
        // would clamp anything larger.
        REQUIRE(value < effort);
        REQUIRE(value == effort / 4.0);
        // 0 is legal and means "no patience gate at all" — load-bearing
        // for the patience-axis search, which needs a point where the gate
        // provably never fires.  If the registered lower bound ever moved
        // above zero that semantic would become inexpressible from the
        // option, silently.
        REQUIRE(highs.setOptionValue(name, 0.0) == HighsStatus::kOk);
        REQUIRE(highs.setOptionValue(name, 1e6) == HighsStatus::kOk);
        REQUIRE(highs.setOptionValue(name, expected) == HighsStatus::kOk);
    }
}

TEST_CASE("Options: the retired stall spelling is gone, with no alias", "[options][patience]") {
    // #116 renamed `mip_heuristic_<name>_stall` to `_patience` and this
    // project does not carry option aliases — `apply_patch.cmake` refuses
    // any HiGHS tree whose patch-version marker is not the current one and
    // never upgrades an older option layout in place, so a build that
    // still answered to the old name would mean the patch had been applied
    // twice or the rename was incomplete.  HiGHS's own `getOptionValue`
    // reports an unknown name only through its return status, and every
    // instance we build sets `output_flag=false`, so nothing would say so.
    Highs highs;
    highs.setOptionValue("output_flag", false);
    for (const char* name : {"mip_heuristic_fj_stall", "mip_heuristic_fpr_stall",
                             "mip_heuristic_local_mip_stall", "mip_heuristic_scylla_stall"}) {
        double value = -1.0;
        INFO(name);
        REQUIRE(highs.getOptionValue(name, value) != HighsStatus::kOk);
        REQUIRE(highs.setOptionValue(name, 1.0) != HighsStatus::kOk);
    }
}

TEST_CASE("Options: presolve-only defaults to false", "[options][presolve-only]") {
    // Registered in apply_patch.cmake like the rest, so nothing but this
    // checks the default.  It has to be false: true makes every solve stop
    // before the root LP, which is a measurement mode, not a solver.
    Highs highs;
    highs.setOptionValue("output_flag", false);
    bool presolve_only = true;
    REQUIRE(highs.getOptionValue("mip_heuristic_presolve_only", presolve_only) == HighsStatus::kOk);
    REQUIRE_FALSE(presolve_only);
    REQUIRE(highs.setOptionValue("mip_heuristic_presolve_only", true) == HighsStatus::kOk);
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
    // distinguishes a known value from an unknown one, and what the
    // comma-separated list form means (see test_suite_option.cpp).
    for (const char* value :
         {"off", "fj", "fpr", "local_mip", "scylla", "all", "fj,fpr", "fj,fpr,local_mip"}) {
        REQUIRE(highs.setOptionValue("mip_heuristic_suite", std::string(value)) ==
                HighsStatus::kOk);
    }
}

// The exact substrings `bench/run_benchmark.py` greps a solve's log for, as
// `CONFIG_IGNORED_WARNINGS`.  Both mark a run that solved cleanly — exit 0,
// complete log — while ignoring the configuration it was given, so the
// harness discards it instead of recording a results directory named for one
// configuration and holding runs of another.
//
// Keeping them here, spelled out, is what makes the coupling visible from
// both ends: the emitter is `run_presolve` in `src/mode_dispatch.cpp` (which
// carries the matching note), the consumer is the bench harness, and this
// test is the thing that fails if either side moves without the other.  A
// substring rather than the whole line, because that is precisely what the
// harness matches — pinning more would make this test stricter than the
// contract it exists to protect.
namespace {
constexpr const char* kBenchWarningUnknownSuite = "Unknown mip_heuristic_suite value";
constexpr const char* kBenchWarningNoHeuristic = "no heuristic will run";
}  // namespace

TEST_CASE("Options: the warnings the bench harness greps for are emitted verbatim",
          "[options][suite][bench-contract]") {
    SECTION("unknown suite value") {
        const auto lines =
            solve_capturing_log("flugpl.mps", [](Highs& h) { set_suite(h, "bogus"); });
        REQUIRE(log_contains(lines, kBenchWarningUnknownSuite));
    }

    SECTION("suite=fj with FeasibilityJump switched off") {
        // Asks for FJ and then takes it away: heuristic-free without being
        // `off`, so it also loses the native FJ call site.  A benchmark row
        // labelled "FJ isolated" would silently measure vanilla-minus-FJ.
        const auto lines = solve_capturing_log("flugpl.mps", [](Highs& h) {
            set_suite(h, "fj");
            require_option(h, "mip_heuristic_run_feasibility_jump", false);
        });
        REQUIRE(log_contains(lines, kBenchWarningNoHeuristic));
    }

    SECTION("an ordinary run trips neither") {
        // The other half of the contract: these must not fire on a good run,
        // or the harness would discard every result it collected.
        const auto lines = solve_capturing_log("flugpl.mps", [](Highs& h) { set_suite(h, "all"); });
        REQUIRE_FALSE(log_contains(lines, kBenchWarningUnknownSuite));
        REQUIRE_FALSE(log_contains(lines, kBenchWarningNoHeuristic));
    }
}

TEST_CASE("Options: unknown suite value warns and runs everything", "[options][suite]") {
    const auto lines = solve_capturing_log("flugpl.mps", [](Highs& h) {
        require_option(h, "log_dev_level", 3);
        set_suite(h, "bogus");
    });
    bool warned = false;
    for (const auto& line : lines) {
        if (line.contains("Unknown mip_heuristic_suite value \"bogus\"")) {
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
            dispatched = dispatched || line.contains(tag);
        }
        INFO("heuristic " << heur);
        REQUIRE(dispatched);
    }
}
