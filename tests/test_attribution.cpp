#include "Highs.h"
#include "test_common.h"

#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>

// ===================================================================
// Per-heuristic solution attribution sentinels (#90)
//
// Every other correctness test asserts only the final objective, so a
// heuristic can be deleted or silently filtered out and the suite stays
// green — FeasibilityJump alone solves the bundled instances.  These
// cases assert *which* heuristic produced a solution, by reading the
// solution-source code HiGHS prints on its MIP display lines:
//
//   A  FPR        M  LocalMIP        G  Scylla        J  FeasibilityJump
//
// They are the regression signal carried through the closeout deletions
// (epic #88), so they deliberately depend on nothing but the option
// surface and the source tags on the shared solution pool.
//
// Two properties of these codes are worth knowing before relying on them:
//
//   * `A`, `M` and `G` are patch-only.  `J` is not: upstream's own
//     `kSolutionSourceFeasibilityJump` also renders as `J`.  That is
//     unambiguous today only because the patch disables upstream's FJ
//     call site.  Epic #88 restores native FJ under `suite=off`, and at
//     that point `J` stops identifying *which* FJ ran.
//   * Scylla's `G` is worker-count dependent — see the Scylla case.
//
// gt2 is the instance for all cases: with one heuristic enabled at a time
// it yields that heuristic's code and never another heuristic's (verified
// across seeds and across thread counts 1-12).
// ===================================================================

namespace {

// Solve gt2 with exactly one custom presolve heuristic enabled and
// return the captured log.  `which` is the `mip_heuristic_run_<which>`
// option suffix.
std::vector<std::string> gt2_log_for(const char* which) {
    return solve_capturing_log("gt2.mps", [&](Highs& h) {
        // Every option is set through `require_option`, which fails the
        // test if the name does not exist.  HiGHS only returns `kError`
        // for an unknown option, so when the option surface collapses to
        // `mip_heuristic_suite` (#93) an unchecked call would leave the
        // solve at its defaults — all four heuristics on — and three of
        // the four positive cases below would keep passing while
        // asserting nothing.  That migration must fail loudly.
        require_option(h, "log_dev_level", 3);
        require_option(h, "mip_heuristic_portfolio", false);
        require_option(h, "mip_heuristic_opportunistic", false);
        require_option(h, "mip_heuristic_run_fpr", false);
        require_option(h, "mip_heuristic_run_local_mip", false);
        require_option(h, "mip_heuristic_run_scylla", false);
        require_option(h, "mip_heuristic_run_feasibility_jump", false);
        require_option(h, std::string("mip_heuristic_run_") + which, true);
    });
}

std::string gt2_codes_for(const char* which) { return source_codes(gt2_log_for(which)); }

}  // namespace

TEST_CASE("attribution: FPR-only run is credited with A", "[attribution]") {
    REQUIRE(gt2_codes_for("fpr").find('A') != std::string::npos);
}

TEST_CASE("attribution: LocalMIP-only run is credited with M", "[attribution]") {
    REQUIRE(gt2_codes_for("local_mip").find('M') != std::string::npos);
}

// Scylla is the one heuristic whose code is not machine-independent.
// Each worker takes a fixed FPR rounding strategy (`kFprConfigs[w % N]`),
// and on gt2 the strategy that lands a solution is only instantiated from
// the third worker on: a seq/det Scylla-only run emits `G` at >= 3 workers
// and none at all below that.  HiGHS derives its default worker count from
// the machine (`(hardware_concurrency() + 1) / 2`), so asserting `G`
// unconditionally would pass on a developer box and fail on a 2-vCPU CI
// runner.  Pinning `threads` is not an option either — the task executor
// is a process-global singleton, so a pinned value makes `run()` fail
// whenever an earlier case in the same process initialised a different one.
//
// So this takes the fallback issue #90 sanctioned: `G`, or else evidence
// that Scylla ran and consumed budget.  That is deliberately *not* "the
// solve succeeded" — it still fails if Scylla is deleted, disabled or
// filtered out of the chain.
TEST_CASE("attribution: Scylla-only run is credited with G", "[attribution]") {
    const std::vector<std::string> lines = gt2_log_for("scylla");
    REQUIRE((source_codes(lines).find('G') != std::string::npos ||
             heuristic_reported_effort(lines, "scylla")));
}

TEST_CASE("attribution: FJ-only run is credited with J", "[attribution]") {
    REQUIRE(gt2_codes_for("feasibility_jump").find('J') != std::string::npos);
}

// The negative direction: enabling one heuristic must not let another one
// run.  This is what catches a suite-filter leak when the option surface
// is collapsed to `mip_heuristic_suite` (#93) — a filter that silently
// admits every arm still passes all four cases above.  `D` is checked
// alongside the presolve codes because `fpr_lp` is gated on the FPR bit
// (epic coupling E), so a leak that re-admits FPR re-admits the dive-time
// heuristic too.
TEST_CASE("attribution: FJ-only run emits no other custom-heuristic solution", "[attribution]") {
    const std::string codes = gt2_codes_for("feasibility_jump");
    REQUIRE(codes.find('J') != std::string::npos);
    REQUIRE(codes.find('A') == std::string::npos);  // FPR
    REQUIRE(codes.find('D') == std::string::npos);  // fpr_lp
    REQUIRE(codes.find('M') == std::string::npos);  // LocalMIP
    REQUIRE(codes.find('G') == std::string::npos);  // Scylla
}
