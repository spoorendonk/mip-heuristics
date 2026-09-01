#include "fpr_core.h"
#include "heuristic_context.h"
#include "Highs.h"
#include "incumbent_sink.h"
#include "parallel/HighsParallel.h"
#include "pump_common.h"
#include "scylla_worker.h"
#include "test_common.h"
#include "worker_base.h"

#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <limits>
#include <mutex>
#include <regex>
#include <string>
#include <utility>
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

// ── Cycle-history ring buffer (#126) ──
// `pump::record_cycle_entry` is the slot rule of Algorithm 1.1 line 13
// (Mexi, Besancon, Pokutta et al., *Scylla*).  It used to write
// `(K - 1) % kCycleWindow`, which overwrote the newest entry at the first
// wrap: the window at K4 was {K0, K1, K3} and at K5 {K1, K3, K4} instead
// of the last three iterates.  This drives six fresh rounds in the real
// order ScyllaWorker uses — record with the pre-increment counter, then
// increment — and pins the window contents after every one.  The old rule
// gets rounds 4 and 5 wrong; round 4 is where an execution dies, since
// `REQUIRE` aborts the case at the first failure.

TEST_CASE("Scylla cycle history keeps the last kCycleWindow iterates", "[heuristic][scylla]") {
    constexpr int kRounds = 6;
    std::vector<std::vector<double>> history;
    history.reserve(pump::kCycleWindow);
    int completed_rounds = 0;

    for (int round = 1; round <= kRounds; ++round) {
        // Distinguishable iterate: round 1 is {1.0}, round 2 is {2.0}, ...
        const std::vector<double> x{static_cast<double>(round)};
        pump::record_cycle_entry(history, completed_rounds, x);
        ++completed_rounds;

        const int expected_size = std::min(round, pump::kCycleWindow);
        REQUIRE(std::cmp_equal(history.size(), expected_size));

        // The window is the last `expected_size` iterates, as a set.
        std::vector<double> got;
        got.reserve(history.size());
        for (const auto& entry : history) {
            REQUIRE(entry.size() == 1);
            got.push_back(entry[0]);
        }
        std::ranges::sort(got);

        std::vector<double> want;
        want.reserve(static_cast<size_t>(expected_size));
        for (int r = round - expected_size + 1; r <= round; ++r) {
            want.push_back(static_cast<double>(r));
        }
        // `want` ends at `round`, so this pins the newest iterate's presence
        // too — the entry the old rule dropped at the wrap.
        REQUIRE(got == want);
    }
}

// ===================================================================
// Issue #121: the PDLP iterate reaches the rounding (Algorithm 1.1 line 12)
// ===================================================================
//
// `ScyllaWorker::run_attempt` builds its `FprConfig` with `cfg.lp_ref =
// x_bar.data()` and a `kFprConfigs` entry whose value strategy is
// `ValStrategy::kLp`. This drives that exact call shape (strategy +
// lp_ref + cont_fallback, all from `kFprConfigs`) at the `fpr_attempt`
// level directly, with two very different `x_bar` inputs, and — like the
// #120 test in test_fpr.cpp — via the begin/step lifecycle rather than
// the one-shot wrapper, since a single DFS attempt on these small bundled
// instances routinely does not complete within the `ncol+1` node budget.
namespace {

struct ScyllaProbe {
    std::vector<uint8_t> fixed;
    std::vector<double> sol;
};

ScyllaProbe round_once(HighsMipSolver& mipsolver, const CscMatrix& csc, const ProblemView& problem,
                       const NamedConfig& named, const double* x_bar, HighsInt ncol) {
    FprScratch scratch;
    FprConfig cfg{};
    cfg.max_effort = std::numeric_limits<size_t>::max() / 2;
    cfg.csc = &csc;
    cfg.mode = named.mode;
    cfg.strategy = &named.strat;
    cfg.lp_ref = x_bar;
    cfg.cont_fallback = x_bar;
    cfg.binary_mask = problem.binary.data();
    cfg.scratch = &scratch;
    Rng rng(2024);
    FprAttemptState state;
    fpr_attempt_begin(state, mipsolver, cfg, rng, /*attempt_idx=*/0);
    while (state.phase == FprAttemptState::Phase::kDfs) {
        fpr_attempt_step(state, mipsolver, cfg, rng, cfg.max_effort);
    }
    ScyllaProbe p;
    for (HighsInt j = 0; j < ncol; ++j) {
        p.fixed.push_back(scratch.prop_engine->var(j).fixed ? 1 : 0);
        p.sol.push_back(scratch.prop_engine->sol_data()[j]);
    }
    return p;
}

}  // namespace

TEST_CASE("Scylla: two different x_bar inputs round to different x_hat (#121)",
          "[heuristic][scylla][lp_ref]") {
    highs::parallel::initialize_scheduler();
    Highs highs;
    highs.setOptionValue("output_flag", false);
    HighsCallback cb(&highs);
    auto mipsolver = build_bare_mipsolver(highs, cb, "flugpl.mps");

    CscMatrix csc;
    const ProblemView problem = make_problem(*mipsolver, csc);
    const HighsInt ncol = mipsolver->model_->num_col_;

    std::vector<double> x_bar_lo(mipsolver->model_->col_lower_);
    std::vector<double> x_bar_hi(mipsolver->model_->col_upper_);
    for (HighsInt j = 0; j < ncol; ++j) {
        if (x_bar_lo[j] <= -1e30) {
            x_bar_lo[j] = -1e5;
        }
        if (x_bar_hi[j] >= 1e30) {
            x_bar_hi[j] = 1e5;
        }
    }

    // Non-empty solution pool: seeded here for defense-in-depth documented
    // in the design, not because this call path can reach it. `fpr_attempt`
    // takes no `IncumbentSink&` — the pool restart that used to feed
    // `initial_solution` was removed from `ScyllaWorker::run_attempt`
    // itself (issue #121), so no candidate in the pool can be seen by this
    // call regardless of what is in it. Populating it here rules out (for
    // a reader re-deriving the old code path) the theory the design flags
    // as the risk this removal defends against — a pool restart silently
    // overriding `cfg.lp_ref`'s starting point — which issue #122's own
    // investigation found was never actually possible: the seeded value is
    // provably unread on every path (overwritten by `fix()`/auto-fix for
    // fixed columns, by the Phase 2.5 fill loop for the rest).
    IncumbentSink sink(*mipsolver, kSolutionSourceHeuristic);
    std::vector<double> adversarial(x_bar_lo.begin(), x_bar_lo.end());
    static_cast<void>(sink.offer(0.0, adversarial, WorkerTrace{0, 0}, 0));

    for (int i = 0; i < kNumFprConfigs; ++i) {
        INFO("kFprConfigs[" << i << "]");
        const ScyllaProbe with_lo =
            round_once(*mipsolver, csc, problem, kFprConfigs[i], x_bar_lo.data(), ncol);
        const ScyllaProbe with_hi =
            round_once(*mipsolver, csc, problem, kFprConfigs[i], x_bar_hi.data(), ncol);

        bool any_column_differs = false;
        for (HighsInt j = 0; j < ncol; ++j) {
            const auto idx = static_cast<size_t>(j);
            if (with_lo.fixed[idx] != with_hi.fixed[idx] ||
                (with_lo.fixed[idx] != 0 && with_lo.sol[idx] != with_hi.sol[idx])) {
                any_column_differs = true;
                break;
            }
        }
        REQUIRE(any_column_differs);
    }
}
