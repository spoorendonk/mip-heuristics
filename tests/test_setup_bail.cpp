#include "effort_ledger.h"
#include "fpr.h"
#include "heuristic_common.h"
#include "heuristic_context.h"
#include "Highs.h"
#include "incumbent_sink.h"
#include "parallel/HighsParallel.h"
#include "scylla.h"
#include "test_common.h"

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// ===================================================================
// A dispatch that abandoned setup is not one that found nothing (#119)
//
// #117 made the presolve dispatch's *sequential* setup give up when the
// wall-clock deadline has already passed: `fpr::precompute_var_orders` and
// `scylla::precompute_config_var_orders` check the clock between variable
// orders and return false, and the entry point returns without searching.
// That fixed the 5.5x time-limit overruns and introduced a reporting hole
// — the dispatch booked `[Heur] ... effort=0 ... found=0`, byte-identical
// to a dispatch that ran normally and produced nothing.
//
// The #113 calibration probe reads exactly those lines and bins both as
// "barren".  A setup bail is not barren: the heuristic never ran, so its
// spend is a cost of setup rather than evidence about how long this
// heuristic goes without improving.  And a bail is only possible on the
// large, hard instances — the same population #117 was already biasing —
// so the mis-binning lands where the estimate is most sensitive.
//
// The channel is the runner contract's return type: `DispatchOutcome`
// instead of a bare `size_t`, carrying the effort and the bail flag
// together from the site that knows to `run_sequential`, which is the only
// caller and the only booker.  The cases below pin the three things that
// makes true, in the order a reader needs them:
//
//   1. the bail sites set the flag, and nothing else does;
//   2. the ledger puts it on the line, on both the presolve and the dive
//      path, so there is one line format and one parser path;
//   3. a dispatch that really did search and find nothing still reports
//      `abandoned_setup=0`, which is what keeps the distinction a
//      distinction.
//
// What these cases deliberately do NOT do is reach the bail through a
// whole solve.  `run_sequential` checks `exec.terminated()` at the top of
// each chain iteration, so a solve whose limit has already expired emits
// no `[Heur]` line at all rather than a bailed one; the bail's real window
// is the deadline passing *inside* a setup that the loop-top check just
// waved through, which on the bundled instances is microseconds wide.
// Pinning it end-to-end would be a race.  So the flag is pinned at its
// source (case 1) and its sink (case 2) separately, which is also what
// makes case 2 able to assert the `=1` form at all.
// ===================================================================

namespace {

// A bare `HighsMipSolver` on `instance` whose solve clock has already
// expired, so the first `Deadline::expired()` any setup asks is true.
//
// `1e-6` rather than 0: the limit is applied after the model is read and
// `HighsMipSolver`'s clock starts from zero there, so a microsecond limit
// is expired after the first microsecond of `runSetup`.  By construction,
// not by a race — the same argument `test_deadline.cpp` makes for the
// single-attempt cases.
constexpr double kExpired = 1e-6;

// The dispatch arguments every heuristic entry point takes, built over one
// solver so a case can call `fpr::run` / `scylla::run` directly.
//
// Declaration order is load-bearing: `make_problem` builds the transpose
// into `csc` and returns a view over it, so `csc` must be the first member
// initialised and must outlive `problem`.
struct Dispatch {
    CscMatrix csc;
    ProblemView problem;
    ExecutionContext exec;
    IncumbentSink sink;

    explicit Dispatch(HighsMipSolver& mipsolver)
        : problem(make_problem(mipsolver, csc)),
          exec(make_exec(mipsolver)),
          sink(mipsolver, kSolutionSourceHeuristic) {}
};

// A budget large enough that no gate derived from it can be what stops the
// dispatch.  The deadline has to be the only stopping rule for these cases
// to say anything: a dispatch that returned because its budget was
// exhausted would report `effort=0 abandoned_setup=0` for a completely
// different reason.
constexpr size_t kAmpleBudget = size_t{1} << 24;

}  // namespace

// ── 1. The bail sites, and only the bail sites, set the flag ──
//
// Both FPR and Scylla, because the two setups are different code with the
// same contract: FPR precomputes eight variable orders, Scylla builds a
// shared `ContestedPdlp` first and then five.  Each is checked against
// itself on a live clock rather than against a constant, so the case is a
// statement about the deadline and not about what these heuristics happen
// to do on `flugpl`.
TEST_CASE("setup-bail: an expired deadline is reported as an abandoned setup", "[setup-bail]") {
    // `HighsMipSolverData::init` reads `parallel::num_threads()`; see the
    // note at the other `build_bare_mipsolver` call sites.
    highs::parallel::initialize_scheduler();

    auto outcome_at = [](double time_limit, bool scylla) {
        Highs highs;
        highs.setOptionValue("output_flag", false);
        HighsCallback cb(&highs);
        auto mipsolver = build_bare_mipsolver(highs, cb, "flugpl.mps", time_limit);

        Dispatch d(*mipsolver);
        const HeuristicBudget budget =
            make_budget(kAmpleBudget, d.exec.num_workers, kAmpleBudget >> 2);
        return scylla ? scylla::run(d.problem, budget, d.exec, d.sink)
                      : fpr::run(d.problem, budget, d.exec, d.sink);
    };

    SECTION("fpr") {
        const DispatchOutcome bailed = outcome_at(kExpired, /*scylla=*/false);
        CHECK(bailed.abandoned_setup);
        // The flag implies the zero; the converse is what it exists to deny.
        CHECK(bailed.effort == 0);

        // The same call with a clock that has not passed reaches the
        // search, so the flag is discriminating rather than always set.
        const DispatchOutcome ran = outcome_at(kHighsInf, /*scylla=*/false);
        CHECK_FALSE(ran.abandoned_setup);
        CHECK(ran.effort > 0);
    }

    SECTION("scylla") {
        const DispatchOutcome bailed = outcome_at(kExpired, /*scylla=*/true);
        CHECK(bailed.abandoned_setup);
        CHECK(bailed.effort == 0);

        const DispatchOutcome ran = outcome_at(kHighsInf, /*scylla=*/true);
        CHECK_FALSE(ran.abandoned_setup);
        CHECK(ran.effort > 0);
    }
}

// A heuristic that declines for a reason other than the clock reports a
// plain zero, not a bail.  Without this, "abandoned setup" would drift into
// meaning "returned without searching", which is the coarser distinction
// the issue is trying to get away from — `mip_heuristic_<name>_effort=0` is
// how #107 spells "this heuristic is excluded", and an excluded heuristic
// is not one the clock cut short.
TEST_CASE("setup-bail: a disabled heuristic is not an abandoned setup", "[setup-bail]") {
    highs::parallel::initialize_scheduler();

    Highs highs;
    highs.setOptionValue("output_flag", false);
    HighsCallback cb(&highs);
    auto mipsolver = build_bare_mipsolver(highs, cb);

    Dispatch d(*mipsolver);
    const HeuristicBudget none = make_budget(0, d.exec.num_workers, 0);
    REQUIRE(none.disabled());

    const DispatchOutcome fpr_out = fpr::run(d.problem, none, d.exec, d.sink);
    CHECK_FALSE(fpr_out.abandoned_setup);
    CHECK(fpr_out.effort == 0);

    const DispatchOutcome scylla_out = scylla::run(d.problem, none, d.exec, d.sink);
    CHECK_FALSE(scylla_out.abandoned_setup);
    CHECK(scylla_out.effort == 0);
}

// ── 2. The ledger puts it on the line, on both paths ──
//
// Driven through `EffortLedger` directly rather than through a solve: the
// `=1` form needs a bail, and a bail cannot be arranged end-to-end without
// racing the clock (see the header comment).  What is being pinned here is
// the emission, which is the ledger's job and nobody else's — it is the one
// place in `src/` that writes these lines.
//
// Both `charge_presolve` and `charge_dive`, because #118 is adding a
// dive-time bail in `fpr_lp` and the field has to already mean the same
// thing there.  One line format, one parser path.
namespace {

// Every log line `emit` produced, captured off the Highs logging callback.
// `log_dev_level=3` because the `[Heur]` line is `kVerbose`.
template <typename Emit>
std::vector<std::string> ledger_lines(Emit&& emit) {
    struct LogCapture {
        std::mutex mtx;
        std::vector<std::string> lines;
    };
    LogCapture capture;

    Highs highs;
    highs.setOptionValue("output_flag", true);
    highs.setOptionValue("log_to_console", false);
    require_option(highs, "log_dev_level", 3);

    auto log_cb = [](int callback_type, const std::string& message,
                     const HighsCallbackOutput* /*out*/, HighsCallbackInput* /*in*/,
                     void* user_data) {
        if (callback_type != kCallbackLogging) {
            return;
        }
        auto* cap = static_cast<LogCapture*>(user_data);
        std::scoped_lock lock(cap->mtx);
        cap->lines.emplace_back(message);
    };
    REQUIRE(highs.setCallback(HighsCallbackFunctionType(log_cb), &capture) == HighsStatus::kOk);
    REQUIRE(highs.startCallback(kCallbackLogging) == HighsStatus::kOk);

    highs::parallel::initialize_scheduler();
    HighsCallback cb(&highs);
    auto mipsolver = build_bare_mipsolver(highs, cb);
    EffortLedger ledger(*mipsolver);
    emit(ledger);

    std::scoped_lock lock(capture.mtx);
    return capture.lines;
}

// The one `[Heur]` line in `lines`.
std::string sole_heur_line(const std::vector<std::string>& lines) {
    std::string found;
    for (const std::string& line : lines) {
        if (line.contains("[Heur] name=")) {
            REQUIRE(found.empty());
            found = line;
        }
    }
    REQUIRE_FALSE(found.empty());
    return found;
}

}  // namespace

TEST_CASE("setup-bail: [Heur] carries abandoned_setup on both phases", "[setup-bail]") {
    SECTION("presolve, abandoned") {
        const std::string line = sole_heur_line(ledger_lines([](EffortLedger& ledger) {
            ledger.charge_presolve("fpr", 0, /*found=*/false, 0.0, 0.5,
                                   /*abandoned_setup=*/true);
        }));
        INFO(line);
        CHECK(line.contains("abandoned_setup=1"));
        CHECK(line.contains("phase=presolve"));
    }

    SECTION("presolve, ran") {
        const std::string line = sole_heur_line(ledger_lines([](EffortLedger& ledger) {
            ledger.charge_presolve("fpr", 0, /*found=*/false, 0.0, 0.5,
                                   /*abandoned_setup=*/false);
        }));
        INFO(line);
        // The discriminating half: same effort, same `found`, different
        // field.  This is the pair the issue says a consumer could not tell
        // apart.
        CHECK(line.contains("abandoned_setup=0"));
    }

    SECTION("dive, abandoned") {
        // The path #118 will use.  `nnz` must be non-zero (`charge_dive`
        // divides by it); the LP-envelope arithmetic is not what this
        // asserts.
        const std::string line = sole_heur_line(ledger_lines([](EffortLedger& ledger) {
            ledger.charge_dive("fpr_lp", 0, /*found=*/false, /*setup_lp_iters=*/0,
                               /*nnz=*/1, 0.0, 0.5, /*abandoned_setup=*/true);
        }));
        INFO(line);
        CHECK(line.contains("abandoned_setup=1"));
        CHECK(line.contains("phase=dive"));
    }

    SECTION("dive, ran") {
        const std::string line = sole_heur_line(ledger_lines([](EffortLedger& ledger) {
            ledger.charge_dive("fpr_lp", 100, /*found=*/false, 0, 1, 0.0, 0.5);
        }));
        INFO(line);
        CHECK(line.contains("abandoned_setup=0"));
    }
}

// ── 3. A real solve reports the negative ──
//
// The end-to-end half of the pair: every dispatch of an ordinary solve ran,
// so every `[Heur]` line it emits must say so.  Without this the field
// could be emitted only on the bail path and a consumer reading a current
// log would still be inferring from absence — the exact failure mode the
// issue describes, one level down.
TEST_CASE("setup-bail: an ordinary solve reports abandoned_setup=0 throughout", "[setup-bail]") {
    const auto lines = solve_capturing_log("flugpl.mps", [](Highs& h) {
        require_option(h, "log_dev_level", 3);
        set_suite(h, "all");
    });

    int heur_lines = 0;
    for (const std::string& line : lines) {
        if (!line.contains("[Heur] name=")) {
            continue;
        }
        ++heur_lines;
        INFO(line);
        CHECK(line.contains("abandoned_setup=0"));
    }
    // The four presolve entries at minimum; `fpr_lp` adds dive-time ones.
    CHECK(heur_lines >= 4);
}
