#pragma once

// Shared helpers for the Catch2 test suite.
//
// This header gathers the small number of helpers that more than one
// per-topic test translation unit needs.  It intentionally keeps the
// surface area small — anything used by a single file stays in that
// file as a file-local helper.

#include "Highs.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// Path to the HiGHS-provided `check/instances/` directory, injected by
// CMake via `INSTANCES_DIR`.  Defined inline so every translation unit
// that includes this header gets its own const reference.
inline const std::string kInstancesDir = INSTANCES_DIR;

// `setOptionValue` that fails the test when the option does not exist.
// HiGHS returns `kError` for an unknown name and otherwise does nothing,
// so a test that sets a renamed option silently keeps the solve at its
// defaults instead of the configuration it asked for.  That matters most
// where an option is *disabling* something: the test then measures a
// completely different solve and can still pass.
template <typename T>
inline void require_option(Highs& h, const std::string& name, const T& value) {
    REQUIRE(h.setOptionValue(name, value) == HighsStatus::kOk);
}

// Makes a `threads=N` pin work regardless of what else ran first in this
// process.
//
// The HiGHS task executor is a process-global singleton, initialised on
// the first `Highs::run` of the process.  A later solve that asks for a
// different count does not silently get the old one — it fails outright:
// `Highs::initializeMultiThreading` returns `kError`, which surfaces as
// an opaque `run() != kOk` with nothing pointing at the thread count.
// Under ctest that never bites, because `catch_discover_tests` forks one
// process per case, but `./mip_heuristics_tests "[tag]"` (documented in
// CLAUDE.md) runs many cases in one process and every pinned case fails.
//
// Tearing the scheduler down on both ends fixes it in both worlds: on
// entry so this solve gets the count it asked for, on exit so the next
// case re-initialises at the default rather than silently inheriting our
// pin — which would quietly strip multi-worker coverage from everything
// downstream while still passing.  RAII rather than two bare calls
// precisely so the exit half cannot be forgotten.
class ScopedThreadPin {
public:
    ScopedThreadPin() { Highs::resetGlobalScheduler(/*blocking=*/true); }
    ~ScopedThreadPin() { Highs::resetGlobalScheduler(/*blocking=*/true); }
    ScopedThreadPin(const ScopedThreadPin&) = delete;
    ScopedThreadPin& operator=(const ScopedThreadPin&) = delete;
};

// Solve `inst` at default options and return the final objective.  Used
// by the execution-mode cross-heuristic parity tests.
inline double solve_default(const char* inst) {
    Highs h;
    h.setOptionValue("output_flag", false);
    // Callers assert the known optimum at tolerances tighter than
    // HiGHS's default `mip_rel_gap` (1e-4) allows it to guarantee, so
    // require a proven-optimal solve.
    //
    // Why only some tests carry this guard: a solve permitted to stop at
    // relative 1e-4 may return an incumbent short of the optimum, which
    // makes any `Approx(optimum).epsilon(<1e-4)` assertion unsound in
    // principle.  In practice only `bell5` ever exercises that freedom —
    // 15 default-option runs of each bundled instance produced 3 distinct
    // primal bounds for bell5 and exactly 1 for flugpl, egout, gt2,
    // p0548 and lseu.  So the guard goes on the helpers here and on every
    // bell5 assertion (`test_fpr.cpp`, `test_fpr_lp.cpp`); the remaining
    // exact-optimum assertions are left alone deliberately, not by
    // oversight.  If a new instance is added, re-run that check before
    // asserting its optimum tightly.
    require_option(h, "mip_rel_gap", 0.0);
    REQUIRE(h.readModel(std::string(INSTANCES_DIR) + "/" + inst) == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    double obj;
    h.getInfoValue("objective_function_value", obj);
    return obj;
}

// Solve `inst` and return every log message HiGHS emitted, in order.
// `configure` receives the `Highs` object after logging is wired up and
// before the model is read, so it can set any option the caller needs.
//
// A callback is the only way to observe HiGHS's MIP display and its
// `log_dev_level=3` traces — neither exists in any info field — so this is
// the capture primitive the whole suite shares.  Declaration order below is
// load-bearing: `capture` outlives `h`, so the callback can never fire
// against a destroyed buffer.
// `inspect` runs on the same `Highs` object after a successful `run()`,
// which is the only point where both the captured log and the solve's
// info values are reachable — `Highs` is destroyed on return.
template <typename Configure, typename Inspect>
inline std::vector<std::string> solve_capturing_log(const char* inst, Configure&& configure,
                                                    Inspect&& inspect) {
    struct LogCapture {
        std::mutex mtx;
        std::vector<std::string> lines;
    };
    LogCapture capture;

    Highs h;
    h.setOptionValue("output_flag", true);
    h.setOptionValue("log_to_console", false);

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

    REQUIRE(h.setCallback(HighsCallbackFunctionType(log_cb), &capture) == HighsStatus::kOk);
    REQUIRE(h.startCallback(kCallbackLogging) == HighsStatus::kOk);
    std::forward<Configure>(configure)(h);
    REQUIRE(h.readModel(kInstancesDir + "/" + inst) == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    std::forward<Inspect>(inspect)(h);

    std::scoped_lock lock(capture.mtx);
    return capture.lines;
}

// Log-only overload for the majority of callers, which assert on the
// captured lines alone.
template <typename Configure>
inline std::vector<std::string> solve_capturing_log(const char* inst, Configure&& configure) {
    return solve_capturing_log(inst, std::forward<Configure>(configure), [](Highs&) {});
}

// Every solution-source code HiGHS printed on a MIP display line of
// `lines`, in emission order.
//
// The source code is the single character HiGHS's `printSolutionSourceKey`
// maps a `kSolutionSource*` value to: `A` FPR, `D` fpr_lp, `M` LocalMIP,
// `G` Scylla, `J` FeasibilityJump, plus upstream's own codes.  Asserting on
// these is the only way a test can tell *which* heuristic found a solution.
//
// Display line format is `" %s %7s ..."` with the one-character code at
// offset 1 (`HighsMipSolverData::printDisplayLine`).  `kSolutionSourceNone`
// and `kSolutionSourceCleanup` both render as a space and are skipped —
// they carry no attribution.
inline std::string source_codes(const std::vector<std::string>& lines) {
    std::string codes;
    for (const auto& line : lines) {
        if (line.size() >= 3 && line[0] == ' ' && line[2] == ' ' && line[1] != ' ') {
            codes.push_back(line[1]);
        }
    }
    return codes;
}

// Convenience composition of the two helpers above.
template <typename Configure>
inline std::string solve_capturing_source_codes(const char* inst, Configure&& configure) {
    return source_codes(solve_capturing_log(inst, std::forward<Configure>(configure)));
}

// Whether `lines` carries a `[Sequential] heur=<heur> effort=<N>` trace
// with a non-zero effort, i.e. whether that heuristic actually ran and
// consumed budget.  Requires `log_dev_level=3` on the solve.
//
// This is a weaker signal than a solution-source code — it proves the
// heuristic ran, not that it found anything — so prefer `source_codes`
// where the code is reliably emitted.
inline bool heuristic_reported_effort(const std::vector<std::string>& lines,
                                      const std::string& heur) {
    const std::string tag = "[Sequential] heur=" + heur + " effort=";
    return std::ranges::any_of(lines, [&](const std::string& line) {
        const auto pos = line.find(tag);
        return pos != std::string::npos &&
               std::strtoull(line.c_str() + pos + tag.size(), nullptr, 10) > 0;
    });
}

// Whether any captured line carries `tag`, e.g. "[Heur] " or
// "[Heur] name=fpr ".  Requires `log_dev_level=3` on the solve.
//
// `find` rather than a prefix match: HiGHS routes some log lines through
// a formatter that prepends nothing today, but the assertions this backs
// are about a line being emitted at all, not about its column 0.
inline bool log_contains(const std::vector<std::string>& lines, const std::string& tag) {
    return std::ranges::any_of(lines, [&](const std::string& line) { return line.contains(tag); });
}

// Restrict the solve to a subset of the heuristics.  `suite` is a
// `mip_heuristic_suite` value: the alias `off` or `all`, or a
// comma-separated list of `fj`, `fpr`, `local_mip`, `scylla`.
//
// `require_option` rather than a bare set: a typo'd suite value would
// otherwise leave the solve at the `all` default and the test would
// measure every heuristic while claiming to isolate one.  HiGHS does not
// validate string option *values*, so this catches a renamed option, not a
// misspelt value — `mip_heuristic_suite` itself warns on those at solve
// time (see the unknown-value case in test_smoke.cpp).
inline void set_suite(Highs& h, const char* suite) {
    require_option(h, "mip_heuristic_suite", std::string(suite));
}

// Solve `inst` with `suite` selected and return the final objective.
inline double solve_suite(const char* inst, const char* suite) {
    Highs h;
    h.setOptionValue("output_flag", false);
    set_suite(h, suite);
    REQUIRE(h.readModel(kInstancesDir + "/" + inst) == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    double obj;
    h.getInfoValue("objective_function_value", obj);
    return obj;
}

// Solve flugpl with every custom heuristic disabled — verifies the
// dispatch path does not block HiGHS's built-in B&B fallback.
inline double solve_no_heuristics() {
    return solve_suite("flugpl.mps", "off");
}

// Stand up a real `HighsMipSolver` (with `mipdata_`) on `instance`
// without going through `Highs::run`'s heuristics, so a test can call a
// heuristic's `run` — or `make_problem`, or one FPR attempt — itself.
// Mirrors the minimal init sequence from `HighsMipSolver::run` (init →
// runMipPresolve → runSetup); the heuristics and B&B that follow are
// skipped.
//
// Options are read from `highs` as they stand at the call, so a caller
// configures the solve before calling — except `time_limit`, which is a
// parameter here because it cannot be set before the read: `Highs::readModel`
// returns `kError` outright once the instance's own clock is past the limit,
// so a limit small enough to be expired by the time a heuristic runs fails
// the read instead.  It is applied after the model is in and before the
// solver — and therefore its own clock — exists.
//
// Callers must have started the HiGHS task scheduler first (see the
// `initialize_scheduler()` note at each call site).
inline std::unique_ptr<HighsMipSolver> build_bare_mipsolver(Highs& highs, HighsCallback& cb,
                                                            const char* instance = "flugpl.mps",
                                                            double time_limit = kHighsInf) {
    // Disable HiGHS presolve so `runMipPresolve` is a near-no-op
    // that leaves `mipsolver.model_` pointing at the original LP.
    // The heuristics' `run` only needs the LP shape and
    // the `mipdata_` row-major buffers (`ARstart_/ARindex_/ARvalue_`)
    // that `runSetup` populates; the heavier LP-relaxation
    // machinery that comes later in `Highs::run` is not needed and
    // skipping presolve keeps this minimal.
    highs.setOptionValue("presolve", "off");
    REQUIRE(highs.readModel(kInstancesDir + "/" + instance) == HighsStatus::kOk);
    require_option(highs, "time_limit", time_limit);
    auto mipsolver = std::make_unique<HighsMipSolver>(cb, highs.getOptions(), highs.getLp(),
                                                      highs.getSolution());
    mipsolver->timer_.start();
    // `HighsMipSolver::run` initialises this before anything can find a
    // solution; constructing the solver directly leaves it holding
    // garbage, and `addIncumbent` -> `saveReportMipSolution` writes to
    // it unconditionally when non-null.
    mipsolver->improving_solution_file_ = nullptr;
    mipsolver->mipdata_ = std::make_unique<HighsMipSolverData>(*mipsolver);
    mipsolver->mipdata_->init();
    mipsolver->mipdata_->runMipPresolve(mipsolver->options_mip_->presolve_reduction_limit);
    mipsolver->mipdata_->runSetup();
    // `HighsMipSolver::run` creates the master worker right after
    // runSetup and before it dispatches the presolve heuristics.  Do
    // the same: `addIncumbent` — reached through the sink's accept
    // callback as soon as a heuristic finds something — reads
    // `mipdata_->workers[0]`, so a harness that skips this crashes on
    // the first solution rather than on any assertion.
    mipsolver->mipdata_->workers.emplace_back(
        *mipsolver, &mipsolver->mipdata_->getLp(), &mipsolver->mipdata_->getDomain(),
        &mipsolver->mipdata_->getCutPool(), &mipsolver->mipdata_->getConflictPool(),
        &mipsolver->mipdata_->getPseudoCost());
    return mipsolver;
}
