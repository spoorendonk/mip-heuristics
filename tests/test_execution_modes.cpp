#include "Highs.h"
#include "test_common.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <string>

// ===================================================================
// Execution-mode correctness tests
//
// Since #92 there is one parallel runner: continuous workers, no epoch
// barriers, no bit-identical guarantee.  Reproducible runs come from
// `threads=1` plus a fixed `random_seed` — that is the configuration the
// determinism cases below pin, and the only one that guarantees a
// repeatable node count.
//
// Cases that only assert an objective deliberately run at the default
// thread count so the multi-worker path stays covered.
// ===================================================================

// ── 3 tests: objective on the characterized instances ──

TEST_CASE("execution-mode: flugpl objective", "[mode-matrix]") {
    REQUIRE(solve_default("flugpl.mps") == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("execution-mode: egout objective", "[mode-matrix]") {
    REQUIRE(solve_default("egout.mps") == Catch::Approx(568.1007).epsilon(1e-4));
}

// p0548 is the largest instance the suite characterizes.  Its only
// coverage lived in a test file deleted by #91; carried over here so the
// closeout does not silently drop an instance size class.
TEST_CASE("execution-mode: p0548 objective", "[mode-matrix]") {
    REQUIRE(solve_default("p0548.mps") == Catch::Approx(8691.0).epsilon(1e-3));
}

// ── 1 test: infeasibility detection ──

TEST_CASE("execution-mode: infeasible detected", "[mode-matrix]") {
    Highs h;
    h.setOptionValue("output_flag", false);
    REQUIRE(h.readModel(std::string(INSTANCES_DIR) + "/infeasible-mip0.mps") == HighsStatus::kOk);
    h.run();
    REQUIRE(h.getModelStatus() == HighsModelStatus::kInfeasible);
}

// ── 1 test: all custom heuristics disabled ──
// At `suite=off` the custom dispatcher is a no-op and HiGHS's own
// pipeline — native FeasibilityJump included — must still solve flugpl.

TEST_CASE("execution-mode: all heuristics disabled still solves", "[mode-matrix]") {
    REQUIRE(solve_no_heuristics() == Catch::Approx(1201500.0).epsilon(1e-6));
}

// ── 1 test: single-heuristic (FJ-only) ──
// Only feasibility_jump enabled: exercises the runner's
// single-worker-type path, which is easy to break with worker-count logic.

TEST_CASE("execution-mode: FJ-only flugpl", "[mode-matrix]") {
    REQUIRE(solve_suite("flugpl.mps", "fj") == Catch::Approx(1201500.0).epsilon(1e-6));
}

// ── 3 tests: the reproducible single-worker configuration ──
//
// `threads=1` + a fixed `random_seed` is the project's reproducibility
// contract after #92.  The default multi-worker path makes no such
// promise: `HighsTaskExecutor` is a lazily-initialised global singleton
// whose work-stealing order depends on prior runs in the same process,
// so a node count taken there is not repeatable even within one binary.

namespace {
struct SeededRun {
    double obj = 0.0;
    // `int64_t`, not `HighsInt`: `mip_node_count` is registered as an
    // `InfoRecordInt64`, and in a default (32-bit `HighsInt`) build the
    // `HighsInt&` overload of `getInfoValue` rejects the type, returns
    // `kError` and leaves the value *untouched* — every node-count
    // assertion below would compare 0 against 0 forever.
    int64_t nodes = 0;
    // Concatenated `heur=<name> effort=<N>` fields of the `[Sequential]`
    // traces, which fingerprint how much work each heuristic did.  A
    // finer signal than the node count: the presolve heuristics can
    // diverge without moving the B&B tree.  The `wall_ms` /
    // `effort_per_ms` fields of those lines are deliberately excluded —
    // they are wall-clock measurements and differ between two runs of
    // identical work.
    std::string effort_trace;
};

// Solve flugpl in the reproducible configuration (`threads=1` plus a
// fixed `random_seed`) and fingerprint the run.
SeededRun run_seeded(int seed) {
    SeededRun res;
    const ScopedThreadPin pin;
    const auto lines = solve_capturing_log(
        "flugpl.mps",
        [&](Highs& h) {
            require_option(h, "threads", 1);
            require_option(h, "random_seed", seed);
            require_option(h, "log_dev_level", 3);
            // Same rationale as `solve_default`: the objective assertion
            // below is tighter than HiGHS's default `mip_rel_gap` (1e-4)
            // can guarantee, so require a proven-optimal solve.
            require_option(h, "mip_rel_gap", 0.0);
        },
        [&](Highs& h) {
            // Status-checked: a silently-failing `getInfoValue` leaves the
            // field at its initialiser and makes the equality assertions
            // in the callers vacuous.
            REQUIRE(h.getInfoValue("objective_function_value", res.obj) == HighsStatus::kOk);
            REQUIRE(h.getInfoValue("mip_node_count", res.nodes) == HighsStatus::kOk);
        });
    for (const auto& line : lines) {
        const auto heur = line.find("heur=");
        if (line.find("[Sequential] ") == std::string::npos || heur == std::string::npos) {
            continue;
        }
        // Keep `heur=<name> effort=<N>`, drop the wall-clock tail.
        const auto wall = line.find(" wall_ms=", heur);
        res.effort_trace +=
            line.substr(heur, wall == std::string::npos ? std::string::npos : wall - heur) + ";";
    }
    return res;
}
}  // namespace

TEST_CASE("execution-mode: threads=1 same seed reproduces the run", "[mode-matrix]") {
    auto first = run_seeded(7);
    auto second = run_seeded(7);
    REQUIRE(first.obj == Catch::Approx(second.obj).epsilon(1e-12));
    REQUIRE(first.nodes == second.nodes);
    REQUIRE(first.effort_trace == second.effort_trace);
    // Guard against the assertions above passing vacuously if the trace
    // tag is ever renamed without updating this test.
    REQUIRE_FALSE(first.effort_trace.empty());
}

TEST_CASE("execution-mode: threads=1 different seeds take different search paths",
          "[mode-matrix]") {
    // Proves the seed reaches *our* workers rather than being silently
    // ignored.  Asserted on the effort trace specifically, not on the
    // node count: HiGHS consumes `random_seed` in its own branching, so
    // a differing node count would be satisfied with zero contribution
    // from the heuristics and would prove nothing about them.  The
    // trace is bit-stable per seed and moves in 3 of the 4 heuristics
    // between seeds 7 and 8.
    auto a = run_seeded(7);
    auto b = run_seeded(8);
    REQUIRE(a.obj == Catch::Approx(b.obj).epsilon(1e-6));
    REQUIRE(a.effort_trace != b.effort_trace);
}

TEST_CASE("execution-mode: threads=1 still finds the optimum", "[mode-matrix]") {
    // The reproducible configuration must not be a degenerate one: a
    // single worker still has to solve the instance.
    REQUIRE(run_seeded(7).obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

namespace {

// Helper used by the shared-pool test.  Runs a Highs solve with only FJ
// enabled, captures the MIP display lines, and returns whether a `J`
// source code was emitted among them.  `J` appearing for lseu proves
// that FJ's pool entry round-tripped through the shared flush in
// mode_dispatch::run_sequential with kSolutionSourceFJ preserved.
bool lseu_emits_fj_tag() {
    const std::string codes =
        solve_capturing_source_codes("lseu.mps", [](Highs& h) { set_suite(h, "fj"); });
    return codes.find('J') != std::string::npos;
}
}  // namespace

// ── 1 test: shared pool round-trip (#72) ──
// Verifies that FJ's pool entries survive the end-of-chain flush in
// mode_dispatch::run_sequential and reach HiGHS tagged as
// kSolutionSourceFJ (`J`).
//
// Pre-#72, each heuristic (FJ/FPR/LocalMIP/Scylla) owned a private
// SolutionPool and emitted its own trySolution loop inside
// <heuristic>::run_parallel.  The tags on that path were correct, but
// FPR/LocalMIP/Scylla could not see FJ's entries as pool-restart seeds:
// each pool was destroyed at the end of its heuristic.
//
// Post-#72, mode_dispatch::run_sequential owns one shared SolutionPool,
// seeds it from the incumbent once, and hands it to every heuristic's
// run_parallel as an `&` parameter.  Each solution accepted by the pool
// is immediately forwarded to HiGHS via the on_accept callback (so
// timestamps reflect find time, not flush time).  The per-entry source
// tag (#73) is preserved and forwarded by the callback so HiGHS logs
// `J`/`A`/`M`/`G` per heuristic.  This test proves the callback path
// round-trips FJ's tag; the pool-restart semantic for downstream
// heuristics is exercised transitively (FPR's get_restart reads from
// the same pool that FJ wrote to).

TEST_CASE("execution-mode: FJ entries survive shared pool flush", "[mode-matrix]") {
    REQUIRE(lseu_emits_fj_tag());
}
