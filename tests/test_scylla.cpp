#include "Highs.h"
#include "test_common.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <mutex>
#include <regex>
#include <string>
#include <vector>

// ── Scylla as the only enabled heuristic ──
// `suite=scylla` clears FJ too, so a solution here can only have come from
// the pump chains.  There used to be three sections here — "standalone",
// "parallel" and "only" — distinguished by `mip_heuristic_opportunistic`
// (deleted in #92) and then by the per-heuristic bool flags (#93).  With
// neither left they are the same configuration, so the duplicates were
// removed rather than kept as six extra full HiGHS solves per suite run
// under names claiming distinct coverage.

TEST_CASE("Scylla standalone: flugpl general integers", "[heuristic][scylla]") {
    REQUIRE(solve_suite("flugpl.mps", "scylla") == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("Scylla standalone: gt2 pure binary instance", "[heuristic][scylla]") {
    REQUIRE(solve_suite("gt2.mps", "scylla") == Catch::Approx(21166.0).epsilon(1e-3));
}

TEST_CASE("Scylla standalone: egout mixed integers", "[heuristic][scylla]") {
    REQUIRE(solve_suite("egout.mps", "scylla") == Catch::Approx(568.1007).epsilon(1e-4));
}

// ── Sequential orchestrator: weighted effort allocation ──

TEST_CASE("Sequential orchestrator: flugpl weighted effort", "[heuristic][sequential]") {
    REQUIRE(solve_suite("flugpl.mps", "all") == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("Sequential orchestrator: egout all arms", "[heuristic][sequential]") {
    REQUIRE(solve_suite("egout.mps", "all") == Catch::Approx(568.1007).epsilon(1e-4));
}

// ── Scylla stale-snapshot overlap (issue #76) ──
//
// Regression guard for the new `[ScyllaOverlap] fresh=<F> stale=<S>
// ratio=<R>` trace line emitted at the end of Scylla's parallel
// runners.  The line surfaces the #76 acceptance criterion — operators
// running with `log_dev_level=3` can read the overlap ratio from the
// log.  We assert the line is emitted at all and that `fresh >= 1`
// (Scylla ran at least one real solve).  Stale rounds are environment-
// dependent (contention between N workers fighting the PDLP mutex);
// on small instances the PDLP solve is fast enough that a single
// worker can finish before peers retry, so we don't require
// `stale > 0` as a hard assertion.  Coverage of the full stale
// branches is via the `ContestedPdlp` unit tests in
// `tests/test_contested_pdlp.cpp` plus MIPLIB bench runs.
TEST_CASE("Scylla overlap trace line: fresh count emitted (#76)", "[heuristic][scylla][overlap]") {
    const std::vector<std::string> lines = solve_capturing_log("flugpl.mps", [](Highs& h) {
        h.setOptionValue("log_dev_level", 3);
        set_suite(h, "scylla");
    });

    // Parse out the fresh / stale counts from the [ScyllaOverlap] line
    // so we assert the plumbing, not just the presence of a substring.
    const std::regex re(R"(\[ScyllaOverlap\] fresh=(\d+) stale=(\d+) ratio=([0-9.]+))");
    std::uint64_t fresh = 0;
    std::uint64_t stale = 0;
    bool seen = false;
    for (const auto& line : lines) {
        std::smatch match;
        if (std::regex_search(line, match, re)) {
            fresh = std::stoull(match[1].str());
            stale = std::stoull(match[2].str());
            seen = true;
            break;
        }
    }
    REQUIRE(seen);        // Line was emitted — closes #76's "new trace lines" ask.
    REQUIRE(fresh >= 1);  // Scylla actually ran at least one solve.
    (void)stale;          // Best-effort — see comment above.
}
