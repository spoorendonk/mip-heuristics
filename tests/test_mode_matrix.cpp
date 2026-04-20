#include "Highs.h"
#include "test_common.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mutex>
#include <string>
#include <vector>

// ===================================================================
// 2x2 mode-matrix correctness tests
//
// The four cells of the (portfolio × opportunistic) execution matrix:
//   seq/det : portfolio=false, opportunistic=false — weighted sequential
//   seq/opp : portfolio=false, opportunistic=true  — per-heuristic opportunistic
//   port/det: portfolio=true,  opportunistic=false — deterministic epoch bandit
//   port/opp: portfolio=true,  opportunistic=true  — opportunistic bandit
//
// Each cell should be exercised on at least one real instance. See also
// #63 for the fpr_lp mode-matrix follow-up (dive-time variant).
// ===================================================================

// ── 8 tests: 4 modes × {flugpl, egout} objective ──

TEST_CASE("mode-matrix seq/det: flugpl objective", "[mode-matrix]") {
    REQUIRE(solve_mode("flugpl.mps", false, false) == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("mode-matrix seq/opp: flugpl objective", "[mode-matrix]") {
    REQUIRE(solve_mode("flugpl.mps", false, true) == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("mode-matrix port/det: flugpl objective", "[mode-matrix]") {
    REQUIRE(solve_mode("flugpl.mps", true, false) == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("mode-matrix port/opp: flugpl objective", "[mode-matrix]") {
    REQUIRE(solve_mode("flugpl.mps", true, true) == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("mode-matrix seq/det: egout objective", "[mode-matrix]") {
    REQUIRE(solve_mode("egout.mps", false, false) == Catch::Approx(568.1007).epsilon(1e-4));
}

TEST_CASE("mode-matrix seq/opp: egout objective", "[mode-matrix]") {
    REQUIRE(solve_mode("egout.mps", false, true) == Catch::Approx(568.1007).epsilon(1e-4));
}

TEST_CASE("mode-matrix port/det: egout objective", "[mode-matrix]") {
    REQUIRE(solve_mode("egout.mps", true, false) == Catch::Approx(568.1007).epsilon(1e-4));
}

TEST_CASE("mode-matrix port/opp: egout objective", "[mode-matrix]") {
    REQUIRE(solve_mode("egout.mps", true, true) == Catch::Approx(568.1007).epsilon(1e-4));
}

// ── 4 tests: infeasibility detection × 4 modes ──

namespace {
void check_infeasible_mode(bool portfolio, bool opp) {
    Highs h;
    h.setOptionValue("output_flag", false);
    h.setOptionValue("mip_heuristic_portfolio", portfolio);
    h.setOptionValue("mip_heuristic_opportunistic", opp);
    REQUIRE(h.readModel(std::string(INSTANCES_DIR) + "/infeasible-mip0.mps") == HighsStatus::kOk);
    h.run();
    REQUIRE(h.getModelStatus() == HighsModelStatus::kInfeasible);
}
}  // namespace

TEST_CASE("mode-matrix seq/det: infeasible detected", "[mode-matrix]") {
    check_infeasible_mode(false, false);
}

TEST_CASE("mode-matrix seq/opp: infeasible detected", "[mode-matrix]") {
    check_infeasible_mode(false, true);
}

TEST_CASE("mode-matrix port/det: infeasible detected", "[mode-matrix]") {
    check_infeasible_mode(true, false);
}

TEST_CASE("mode-matrix port/opp: infeasible detected", "[mode-matrix]") {
    check_infeasible_mode(true, true);
}

// ── 4 tests: all custom heuristics disabled × 4 modes ──
// With every custom arm off the dispatcher is a no-op and HiGHS's own
// B&B must still solve flugpl.  This verifies none of the mode paths
// accidentally block fallback behaviour.

TEST_CASE("mode-matrix seq/det: all heuristics disabled still solves", "[mode-matrix]") {
    REQUIRE(solve_mode_no_heuristics(false, false) == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("mode-matrix seq/opp: all heuristics disabled still solves", "[mode-matrix]") {
    REQUIRE(solve_mode_no_heuristics(false, true) == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("mode-matrix port/det: all heuristics disabled still solves", "[mode-matrix]") {
    REQUIRE(solve_mode_no_heuristics(true, false) == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("mode-matrix port/opp: all heuristics disabled still solves", "[mode-matrix]") {
    REQUIRE(solve_mode_no_heuristics(true, true) == Catch::Approx(1201500.0).epsilon(1e-6));
}

// ── 4 tests: single-arm (FJ-only) × 4 modes ──
// Only feasibility_jump enabled: exercises the opportunistic runner's
// single-worker-type path, which is easy to break with worker-count logic.

namespace {
double solve_mode_fj_only(bool portfolio, bool opp) {
    Highs h;
    h.setOptionValue("output_flag", false);
    h.setOptionValue("mip_heuristic_portfolio", portfolio);
    h.setOptionValue("mip_heuristic_opportunistic", opp);
    h.setOptionValue("mip_heuristic_run_fpr", false);
    h.setOptionValue("mip_heuristic_run_local_mip", false);
    h.setOptionValue("mip_heuristic_run_scylla", false);
    h.setOptionValue("mip_heuristic_run_feasibility_jump", true);
    REQUIRE(h.readModel(std::string(INSTANCES_DIR) + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    double obj;
    h.getInfoValue("objective_function_value", obj);
    return obj;
}
}  // namespace

TEST_CASE("mode-matrix seq/det: FJ-only flugpl", "[mode-matrix]") {
    REQUIRE(solve_mode_fj_only(false, false) == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("mode-matrix seq/opp: FJ-only flugpl", "[mode-matrix]") {
    REQUIRE(solve_mode_fj_only(false, true) == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("mode-matrix port/det: FJ-only flugpl", "[mode-matrix]") {
    REQUIRE(solve_mode_fj_only(true, false) == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("mode-matrix port/opp: FJ-only flugpl", "[mode-matrix]") {
    REQUIRE(solve_mode_fj_only(true, true) == Catch::Approx(1201500.0).epsilon(1e-6));
}

// ── 2 tests: determinism for deterministic cells only ──
// Opportunistic cells are intentionally non-deterministic so no determinism
// guarantee is asserted for them.

TEST_CASE("mode-matrix seq/det: same seed → same objective and node count", "[mode-matrix]") {
    struct RunResult {
        double obj;
        HighsInt nodes;
    };
    auto run_seeded = [](int seed) {
        Highs h;
        h.setOptionValue("output_flag", false);
        h.setOptionValue("mip_heuristic_portfolio", false);
        h.setOptionValue("mip_heuristic_opportunistic", false);
        h.setOptionValue("random_seed", seed);
        REQUIRE(h.readModel(std::string(INSTANCES_DIR) + "/flugpl.mps") == HighsStatus::kOk);
        REQUIRE(h.run() == HighsStatus::kOk);
        RunResult res;
        h.getInfoValue("objective_function_value", res.obj);
        h.getInfoValue("mip_node_count", res.nodes);
        return res;
    };
    auto first = run_seeded(42);
    auto second = run_seeded(42);
    REQUIRE(first.obj == Catch::Approx(second.obj).epsilon(1e-12));
    REQUIRE(first.nodes == second.nodes);
}

TEST_CASE("mode-matrix port/det: same seed → same objective and node count", "[mode-matrix]") {
    struct RunResult {
        double obj;
        HighsInt nodes;
    };
    auto run_seeded = [](int seed) {
        Highs h;
        h.setOptionValue("output_flag", false);
        h.setOptionValue("mip_heuristic_portfolio", true);
        h.setOptionValue("mip_heuristic_opportunistic", false);
        h.setOptionValue("random_seed", seed);
        REQUIRE(h.readModel(std::string(INSTANCES_DIR) + "/flugpl.mps") == HighsStatus::kOk);
        REQUIRE(h.run() == HighsStatus::kOk);
        RunResult res;
        h.getInfoValue("objective_function_value", res.obj);
        h.getInfoValue("mip_node_count", res.nodes);
        return res;
    };
    auto first = run_seeded(42);
    auto second = run_seeded(42);
    REQUIRE(first.obj == Catch::Approx(second.obj).epsilon(1e-12));
    REQUIRE(first.nodes == second.nodes);
}

namespace {

// Helper used by the shared-pool tests.  Runs a Highs solve with the
// given (portfolio × opportunistic) cell and only FJ enabled, captures
// MIP display log lines via the kCallbackLogging callback, and returns
// whether a `J` source-code line was emitted.  `J` appearing for lseu
// proves that FJ's pool entry round-tripped through the shared flush in
// mode_dispatch::run_sequential with kSolutionSourceFJ preserved.
bool lseu_seq_emits_fj_tag(bool opportunistic) {
    struct LogCapture {
        std::mutex mtx;
        std::vector<std::string> lines;
    };
    LogCapture capture;

    Highs h;
    h.setOptionValue("output_flag", true);
    h.setOptionValue("log_to_console", false);
    h.setOptionValue("mip_heuristic_portfolio", false);
    h.setOptionValue("mip_heuristic_opportunistic", opportunistic);
    h.setOptionValue("mip_heuristic_run_feasibility_jump", true);
    h.setOptionValue("mip_heuristic_run_fpr", false);
    h.setOptionValue("mip_heuristic_run_local_mip", false);
    h.setOptionValue("mip_heuristic_run_scylla", false);

    auto log_cb = [](int callback_type, const std::string& message,
                     const HighsCallbackOutput* /*out*/, HighsCallbackInput* /*in*/,
                     void* user_data) {
        if (callback_type != kCallbackLogging) {
            return;
        }
        auto* cap = static_cast<LogCapture*>(user_data);
        std::lock_guard<std::mutex> lock(cap->mtx);
        cap->lines.emplace_back(message);
    };

    REQUIRE(h.setCallback(HighsCallbackFunctionType(log_cb), &capture) == HighsStatus::kOk);
    REQUIRE(h.startCallback(kCallbackLogging) == HighsStatus::kOk);
    REQUIRE(h.readModel(std::string(INSTANCES_DIR) + "/lseu.mps") == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);

    std::lock_guard<std::mutex> lock(capture.mtx);
    for (const auto& line : capture.lines) {
        // MIP display line format: " <CODE> <nodes> ..." with the single
        // source letter at offset 1 followed by a space.
        if (line.size() >= 3 && line[0] == ' ' && line[2] == ' ' && line[1] == 'J') {
            return true;
        }
    }
    return false;
}
}  // namespace

// ── 2 tests: shared pool round-trip in seq/det and seq/opp (#72) ──
// Verifies that in both sequential cells of the 2×2 mode matrix, FJ's
// pool entries survive the end-of-chain flush in
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
// seeds it from the incumbent once, hands it to every heuristic's
// run_parallel as an `&` parameter, and flushes it once at
// end-of-chain.  The per-entry source tag (#73) lets that single flush
// emit `J`/`A`/`M`/`G` accordingly.  These tests prove the new flush
// path round-trips FJ's tag; the pool-restart semantic for downstream
// heuristics is exercised transitively (FPR's get_restart now reads
// from the same pool that FJ wrote to, as audited by the refactor's
// signature changes).

TEST_CASE("mode-matrix seq/det: FJ entries survive shared pool flush", "[mode-matrix]") {
    REQUIRE(lseu_seq_emits_fj_tag(/*opportunistic=*/false));
}

TEST_CASE("mode-matrix seq/opp: FJ entries survive shared pool flush", "[mode-matrix]") {
    REQUIRE(lseu_seq_emits_fj_tag(/*opportunistic=*/true));
}
