#include "Highs.h"
#include "test_common.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// ===================================================================
// The solve's wall-clock deadline binds every presolve heuristic (#114)
//
// Every gate that stops a presolve heuristic is denominated in *effort*
// units.  The deadline used to be checked in one place — the runner, in
// `run_opportunistic_loop`, between attempts — and one attempt is
// `HeuristicBudget::attempt_cap` = `total / (10N)`, which scales with
// that heuristic's effort option.  So the documented "~2 attempts of
// overshoot" was a bound proportional to the knob: tight at the shipped
// defaults, and worthless at `effort=1.0`, where FJ was measured running
// 1.4-2.0x past a 60 s limit and twice to an external SIGKILL at 3.5x.
//
// A bound that stops binding as a knob is turned is not a bound, so
// `ExecutionContext::past_deadline()` — the write-free half of
// `terminated()`, and therefore callable from a worker thread without the
// poller seat — is now polled inside all four heuristics on a cadence of
// their own, plus unconditionally by the runner on every iteration.
//
// What these cases deliberately do NOT assert is that the whole chain
// runs.  The chain is FJ -> FPR -> LocalMIP -> Scylla and `run_sequential`
// skips a heuristic once `exec.terminated()`, so a heuristic that
// legitimately consumes the whole limit still truncates its successors.
// Bounding an overrun does not create time that is not there.
// ===================================================================

namespace {

// The instance these cases run on.  gesa2 is the largest bundled MIP that
// still reads and presolves quickly.  At `effort=1.0` and `threads=1` its
// budget is `nnz * 81920` = ~4.07e8 effort units, which every one of the
// four heuristics needs seconds to spend (measured 8.5 s for FJ, 4.1 s for
// LocalMIP, 3.3 s for FPR, 1.5 s for Scylla).  That headroom over the
// limit below is the whole reason the instance is this one.
constexpr const char* kInstance = "gesa2.mps";

// Far below the time any of the four needs to exhaust its budget, and —
// the property that gives these cases teeth — far below the time any of
// them needs to finish a single *attempt*.  See `kAttemptCapUnits`.
constexpr double kLimit = 0.1;

// Allowance above the limit for the polling cadence plus teardown.
// Measured overshoot on the development machine is 0-6 ms for FJ, FPR and
// LocalMIP, and 26 ms for Scylla, whose floor is a whole PDLP solve; a
// slow shared runner is given far more rather than a tight bound that
// turns scheduler jitter into a red suite.  Without a sub-attempt
// deadline check these runs end at 0.66 s (FJ, measured), so the
// assertion keeps its teeth at this slack.
constexpr double kSlack = 0.20;

// Scylla's own allowance.  Its deadline guard sits between pump
// iterations and one iteration charges a whole PDLP solve, which
// `attempt_cap` does not govern once started — the granularity floor
// `docs/PARAMETERS.md` documents and no constant can cross.  That floor is
// deterministic in *effort* (identical idle and under a full parallel
// ctest run) but not in wall time: measured 27 ms overshoot on an idle
// machine and 267 ms with the rest of the suite running, the difference
// being ContestedPdlp construction and a contended PDLP solve.  Sized
// against the loaded measurement, because that is the condition the suite
// actually runs in.
constexpr double kPdlpSlack = 0.60;

// `HeuristicBudget::attempt_cap` for this configuration, in effort units.
//
// `make_budget` computes `total / (num_workers * 10)`, and at
// `threads = 1` all four heuristics land on the same number: FJ's option
// is per-worker so its `total` is `nnz * 81920 * N`, the other three size
// a whole dispatch at `nnz * 81920`, and `N` is 1.
//
// This is the discriminator.  A build whose deadline is checked only
// between attempts cannot report *less* than one `attempt_cap`, because
// nothing can stop the attempt it is inside — measured at exactly this
// value with the FJ callback's check removed, identically at a 0.25 s and
// a 0.5 s limit.  Spending materially less than one attempt is therefore
// positive evidence that the heuristic stopped mid-attempt, which is what
// the deadline check does and what nothing else in the run would do.
//
// The margin is ~4x on this machine (9.6-11.0e6 spent against 4.07e7) and
// it is the one direction that is machine-dependent: a slower runner
// spends less and passes more easily, while a runner ~4x faster than the
// development machine would begin to erode it.
constexpr size_t kAttemptCapUnits = size_t{4968} * 8192;  // gesa2 nnz * 8192

// The whole allowance, `heuristic_effort_budget(nnz, 1.0)` — what a
// heuristic spends when nothing stops it before the budget does.  Scylla
// is measured against this rather than against `kAttemptCapUnits`, for the
// reason `kPdlpSlack` gives: one Scylla attempt legitimately charges more
// than one attempt cap, so the cap says nothing about it, while the full
// budget still separates "stopped by the clock" (3.5e7 measured, and the
// same number idle or loaded) from "ran to the budget" (4.07e8, measured
// at a 60 s limit).
constexpr size_t kBudgetUnits = size_t{4968} * 81920;  // gesa2 nnz * 81920

// Value of `key=` in `line`, as text.
std::string field_of(const std::string& line, const std::string& key) {
    const std::string tag = key + "=";
    const auto pos = line.find(tag);
    REQUIRE(pos != std::string::npos);
    const auto start = pos + tag.size();
    const auto end = line.find_first_of(" \n", start);
    return line.substr(start, end == std::string::npos ? end : end - start);
}

// Second at which a `[Heur]` dispatch ended, on the solver's own clock —
// the same origin `time_limit` is measured against, which is why
// `ExecutionContext::past_deadline()` reads `HighsTimer` rather than a
// `steady_clock` of its own.
double end_s_of(const std::string& line) {
    return std::strtod(field_of(line, "end_s").c_str(), nullptr);
}

size_t effort_of(const std::string& line) {
    return std::strtoull(field_of(line, "effort").c_str(), nullptr, 10);
}

// Every presolve-phase `[Heur]` line of one solve.
//
// The suite's shared `solve_capturing_log` cannot be used: it `REQUIRE`s
// `run() == kOk`, and both a presolve-only solve and a time-limited one
// return `kWarning` (every limit status maps to it in
// `highsStatusFromHighsModelStatus`).  Same workaround
// `test_presolve_only.cpp` documents, with the log callback kept.
template <typename Configure>
std::vector<std::string> presolve_heur_lines(Configure&& configure) {
    struct LogCapture {
        std::mutex mtx;
        std::vector<std::string> lines;
    };
    LogCapture capture;

    const ScopedThreadPin pin;
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

    // `threads=1` and a fixed seed are the project's reproducible
    // configuration, and `kAttemptCapUnits` above is derived at that worker
    // count.  `log_dev_level=3` is what emits `[Heur]` at all, and
    // `presolve_only` keeps B&B from adding time this test would then have
    // to model.
    require_option(h, "threads", 1);
    require_option(h, "random_seed", 0);
    require_option(h, "log_dev_level", 3);
    require_option(h, "mip_heuristic_presolve_only", true);
    require_option(h, "time_limit", kLimit);
    std::forward<Configure>(configure)(h);

    REQUIRE(h.readModel(kInstancesDir + "/" + kInstance) == HighsStatus::kOk);
    static_cast<void>(h.run());

    std::scoped_lock lock(capture.mtx);
    std::vector<std::string> out;
    for (const auto& line : capture.lines) {
        if (line.contains("[Heur] name=") && line.contains("phase=presolve")) {
            out.push_back(line);
        }
    }
    return out;
}

// One heuristic, alone, with its effort option at the maximum and its
// stall gate disabled (`0` means no gate at all, not "give up
// immediately"), so the budget is the only thing competing with the
// deadline.
std::string alone_at_limit(const std::string& heuristic) {
    const auto lines = presolve_heur_lines([&](Highs& h) {
        require_option(h, "mip_heuristic_suite", heuristic);
        require_option(h, "mip_heuristic_" + heuristic + "_effort", 1.0);
        require_option(h, "mip_heuristic_" + heuristic + "_stall", 0);
    });
    REQUIRE(lines.size() == 1);
    return lines.front();
}

// The property, for one heuristic that charges effort in units an attempt
// cap can be compared against.
void require_stopped_mid_attempt(const char* heuristic) {
    const std::string line = alone_at_limit(heuristic);
    INFO(line);
    CHECK(end_s_of(line) <= kLimit + kSlack);
    CHECK(effort_of(line) < kAttemptCapUnits);
}

}  // namespace

// FJ is the heuristic #114 was reported against: its callback had gates
// for the attempt budget, the total budget and staleness, and no clock.
TEST_CASE("deadline: FJ stops at the time limit, not at its effort budget", "[deadline]") {
    require_stopped_mid_attempt("fj");
}

// LocalMIP had the identical hole and was never reported, because nothing
// had swept its effort option to where the hole opens.
// `kTermCheckInterval` had documented a termination-check cadence for this
// loop since it was introduced while being referenced nowhere in the tree;
// it is now what paces the check.
TEST_CASE("deadline: LocalMIP stops at the time limit, not at its effort budget", "[deadline]") {
    require_stopped_mid_attempt("local_mip");
}

// FPR already polled the deadline per inner attempt — via `terminated()`,
// which writes `mipsolver.termination_status_` from a worker thread when a
// terminator is attached.  It polls `past_deadline()` now; this pins that
// the behaviour it had is the behaviour it kept.
TEST_CASE("deadline: FPR stops at the time limit, not at its effort budget", "[deadline]") {
    require_stopped_mid_attempt("fpr");
}

// Scylla is measured against the full budget rather than the attempt cap,
// and at its own wall-clock slack.  Its guard is `past_deadline()` inlined
// so the same clock read also yields PDLP's input time limit, and it is
// correct — but it can only act between pump iterations, and one iteration
// charges a whole PDLP solve.  So the question this case can answer is not
// "did it stop mid-attempt" but "did it stop at all before its budget",
// which is the one a removed guard would fail.
TEST_CASE("deadline: Scylla stops at the time limit", "[deadline]") {
    const std::string line = alone_at_limit("scylla");
    INFO(line);
    CHECK(effort_of(line) < kBudgetUnits / 4);
    CHECK(end_s_of(line) <= kLimit + kPdlpSlack);
}

// The dispatch-level statement, which is the one the benchmark harness
// depends on: whatever subset of the chain gets to run, none of it is
// still running past the limit.  Note the *subset* — at `effort=1.0` the
// first heuristic legitimately consumes the whole limit and its
// successors are skipped by `run_sequential`'s own `terminated()` guard.
//
// At `kPdlpSlack`, because which subset runs is a timing fact and Scylla
// may be in it.
TEST_CASE("deadline: no presolve heuristic outlives the limit", "[deadline]") {
    const auto lines = presolve_heur_lines([](Highs& h) {
        require_option(h, "mip_heuristic_suite", std::string("all"));
        require_option(h, "mip_heuristic_fj_effort", 1.0);
        require_option(h, "mip_heuristic_fpr_effort", 1.0);
        require_option(h, "mip_heuristic_local_mip_effort", 1.0);
        require_option(h, "mip_heuristic_scylla_effort", 1.0);
    });
    REQUIRE(!lines.empty());
    for (const auto& line : lines) {
        INFO(line);
        CHECK(end_s_of(line) <= kLimit + kPdlpSlack);
    }
}
