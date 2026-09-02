#include "fpr_core.h"
#include "fpr_lp.h"
#include "fpr_strategies.h"
#include "heuristic_common.h"
#include "heuristic_context.h"
#include "Highs.h"
#include "parallel/HighsParallel.h"
#include "rng.h"
#include "test_common.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <limits>
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

// Every case built on `presolve_heur_lines` below is tagged `[serial]`,
// which `CMakeLists.txt` registers with ctest's `RUN_SERIAL` (issue #146).
// These are wall-clock bounds and they are protected by serialisation, not
// by their constants — say so plainly rather than pretending otherwise.
//
// Note the scope: `[serial]` follows the *fixture*, not the assertion.  The
// Scylla case asserts on effort alone and is tagged anyway, because the
// 0.1 s limit is still running while the chain is dispatched, so its
// `REQUIRE(lines.size() == 1)` is exposed to the clock whatever the `CHECK`
// below it looks at.  The attempt-level and dive-setup cases at the bottom
// of the file are *not* tagged: they use an already-expired clock by
// construction rather than by a race.
//
// The constants above were measured on an idle machine, and `kLimit` is
// 0.1 s.  Under a full `ctest -j$(nproc)` on a saturated host that budget
// is gone before the code under test is reached.  Two distinct failures
// were observed on this tree, unmodified, with 32 CPU-saturating spinners
// beside `ctest -j32`, and **neither is a near-miss on `kSlack`**:
//
//   * `REQUIRE(h.readModel(...) == kOk)` returning -1 — the MPS parser's
//     own budget, described at the `time_limit` write in
//     `presolve_heur_lines`.  This one has a real fix and now has it: the
//     limit is set after the read, so parsing is no longer inside it.
//     `[serial]` is not what addresses it, and must not be read as if it
//     were — the tag would only have made it rarer.
//   * `REQUIRE(lines.size() == 1)` returning 0 — the solve reached
//     `run_sequential` with the clock already past, so the heuristic was
//     skipped by its own `terminated()` guard and emitted no `[Heur]`
//     line.  This one is genuine in-run wall clock: the limit is 0.1 s of
//     solver time and a descheduled thread spends it doing nothing.
//     Nothing in the code can bound it, so it is what the tag is for.
//
// Widening `kSlack` addresses neither, and would be wrong even if it did:
// it would blunt the assertion that still works.  Raising `kLimit` is no
// better — the effort bound `effort < kAttemptCapUnits` (9.6-11.0e6 spent
// against 4.07e7) erodes as the limit grows, trading one flake for
// another.  What the second mode needs is an unloaded machine, which is
// what `RUN_SERIAL` gives it and what `RESOURCE_LOCK` would not.
//
// Note what serialisation does *not* cover: `RUN_SERIAL` excludes other
// ctest tests, not other processes.  A CI runner sharing a box, or another
// build on the same machine, is load this tag cannot see — which is
// exactly why the parser mode was worth fixing at the source rather than
// tagging around.
//
// The effort half of each assertion needs none of this — a starved runner
// spends *less* effort, so `effort_of(line) < kAttemptCapUnits` only gets
// easier under load.  It is the fixture and the `end_s` bound that need an
// unloaded machine, and that is the whole of what the tag buys.

// Scylla is asserted on effort alone, and this is the reasoning, because
// it is the one place a slack constant would have been a fudge.
//
// Its deadline guard sits between pump iterations, and one iteration
// charges a whole PDLP solve that `attempt_cap` does not govern once
// started — the granularity floor `docs/PARAMETERS.md` documents and that
// no constant can cross.  The floor is exactly reproducible in *effort*,
// and its *wall time* is not: 127 ms, 284 ms and 367 ms were measured for
// the same work, and during a clean-rebuild push gate it exceeded 700 ms.
//
// So a wall-clock *upper* bound on Scylla measures how loaded the machine
// is, and the only way to make one pass everywhere is to widen it until it
// can no longer fail — at which point it asserts nothing.  That is why the
// cases below bound Scylla on effort.  It does not apply to the #152 case,
// which bounds `end_s` from *below*: load moves a lower bound the safe way.
// The three other heuristics keep their wall-clock upper bound: their
// floors are milliseconds, so for them it is a real one.
//
// The effort figures quoted through this file were re-measured after #152,
// which roughly doubled what a clock-bound Scylla dispatch charges by
// giving it its whole deadline instead of half — the earlier 35409804 was
// taken before both #152 and #140, and #140 moved the axis again by
// cutting charged effort per pump round.  Current, on this model at
// `threads=1`, seed 0, presolve-only, patience off, and reproducible to
// the digit across repeats:
//
//   scylla, effort=1.0,           any limit 0.1-5 s : 5152601  (end_s 0.099)
//   scylla, effort=kUnbindable..., limit 0.1 s      : 9779068
//   scylla, effort=kUnbindable..., limit 0.5 s      : 42921044-42970944
//
// Read the first row carefully, because it is not what its case's name
// says: `heuristic_effort_budget(nnz, 1.0)` is `nnz << 10` = 5087232, so
// that dispatch spends 101.3% of its budget and ends at the same effort
// whether the limit is 0.1 s or 5 s.  At `effort=1.0` on this model Scylla
// is *budget*-bound, not clock-bound, and so are FJ (5089737, end_s 0.025)
// and FPR (5088164, end_s 0.058); only LocalMIP (2156255, end_s 0.105)
// still runs into the clock.  See the note on `kAttemptCapUnits` below.

// `HeuristicBudget::attempt_cap` for this configuration, in effort units
// **as the budget was denominated before #116**, which is 80x the unit in
// force today.  Read the warning below before using either constant.
//
// `make_budget` computes `total / (num_workers * 10)`, and at
// `threads = 1` all four heuristics land on the same number: FJ's option
// is per-worker so its `total` is `nnz * <base> * N`, the other three size
// a whole dispatch at `nnz * <base>`, and `N` is 1.
//
// The discriminator these were chosen to be: a build whose deadline is
// checked only between attempts cannot report *less* than one
// `attempt_cap`, because nothing can stop the attempt it is inside —
// measured at exactly this value with the FJ callback's check removed,
// identically at a 0.25 s and a 0.5 s limit.  Spending materially less
// than one attempt is positive evidence that the heuristic stopped
// mid-attempt.
//
// **They are stale by 80x and the bounds keyed to them are correspondingly
// loose.**  `<base>` here is 81920, which is the pre-#116 `nnz << 12 *
// (value / 0.05)`; #116 re-denominated the option to `nnz << 10` and this
// file was not updated, so `heuristic_effort_budget(nnz, 1.0)` is 5087232,
// not the 406978560 `kBudgetUnits` claims, and one `attempt_cap` is 508723,
// not 40697856.  Verified against the built binary: at `effort` 1.0 / 2.0 /
// 4.0 a Scylla dispatch charges 5.15e6 / 1.08e7 / 2.27e7 against budgets of
// 5.09e6 / 1.02e7 / 2.03e7, i.e. it tracks `nnz << 10` exactly, plus its
// documented one-attempt overshoot.  The consequence is that three of the
// four `require_stopped_mid_attempt` cases and the two Scylla cases here
// are now measuring *budget*-bound dispatches while asserting a bound 80x
// above the budget, so they pass without discriminating.  Fixing that means
// re-choosing `kLimit` and the per-case effort so the clock actually binds,
// which is its own measurement exercise and is filed as #154; #152
// deliberately did not attempt it, and only re-sized the two bounds it had
// re-measured.  Do not read a pass from a case keyed to these constants as
// evidence that a deadline check works.
// gesa2's post-presolve nonzero count, which the two sizes below are
// derived from.  Read back off the `[Heur] nnz=` field of every line these
// cases parse rather than trusted: a HiGHS tag bump that changes presolve
// would otherwise erode the margins silently instead of failing here.
constexpr size_t kNnz = 4968;

constexpr size_t kAttemptCapUnits = kNnz * 8192;

// What the whole allowance used to be, on the same pre-#116 unit as
// `kAttemptCapUnits` — see the warning above, which applies to this
// constant equally.  Scylla is measured against a budget rather than
// against an attempt cap for the reason the "asserted on effort alone"
// note at the top of this file gives: one Scylla attempt legitimately
// charges more than one attempt cap, so the cap says nothing about it.
constexpr size_t kBudgetUnits = kNnz * 81920;

// What a clock-bound Scylla dispatch actually charges at each of the two
// configurations the cases below use, measured on the fixed (#152) binary
// and reproducible to the digit — see the table in the note at the top of
// this file.  The bounds are these values with room for a slower or faster
// host; effort is the load-safe direction for an upper bound, since a
// starved runner charges less.
constexpr size_t kScyllaEffortAtDefaultEffort = 5152601;  // effort=1.0, any limit
constexpr size_t kScyllaEffortAtLimit = 9779068;          // effort=1e6, 0.1 s
constexpr size_t kScyllaEffortAtRatioLimit = 42970944;    // effort=1e6, 0.5 s

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

// Fail loudly if the model these sizes were derived from has moved.
void require_expected_nnz(const std::string& line) {
    INFO(line);
    REQUIRE(std::strtoull(field_of(line, "nnz").c_str(), nullptr, 10) == kNnz);
}

// Every presolve-phase `[Heur]` line of one solve.
//
// The suite's shared `solve_capturing_log` cannot be used: it `REQUIRE`s
// `run() == kOk`, and both a presolve-only solve and a time-limited one
// return `kWarning` (every limit status maps to it in
// `highsStatusFromHighsModelStatus`).  Same workaround
// `test_presolve_only.cpp` documents, with the log callback kept.
template <typename Configure>
std::vector<std::string> presolve_heur_lines(Configure&& configure, double limit = kLimit) {
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
    std::forward<Configure>(configure)(h);

    REQUIRE(h.readModel(kInstancesDir + "/" + kInstance) == HighsStatus::kOk);
    // `time_limit` goes in *after* the read, and this is load-bearing.
    //
    // The free-format MPS reader takes `options.time_limit` as a budget of
    // its own (`io/FilereaderMps.cpp`) against a clock it starts at parse
    // start (`HMpsFF::start_time`, checked by `HMpsFF::timeout`), so a limit
    // set beforehand is a limit on *parsing* as well as on solving.  Parsing
    // `gesa2` costs 4-5 ms idle — measured by bisection on the built binary,
    // which reports "Free format reader reached time_limit while parsing" at
    // `--time_limit 0.004` and reads cleanly at `0.005` — so against the
    // 0.1 s default `kLimit` the margin is only 20-25x, and `ctest -j$(nproc)`
    // beside a saturating load closes it.  `readModel` then returns `kError` and the
    // `REQUIRE` above fails on a build with nothing wrong with it.
    //
    // Nothing is lost by moving it: `Highs::run` starts the run clock
    // (`timer_.start()`) as one of its first acts, so read time never
    // entered `end_s` and every assertion in this file is unchanged.  Only
    // `run()` has to see the option.  `build_bare_mipsolver` in
    // `test_common.h` already orders it this way, for the same reason —
    // `presolve_heur_lines` was the outlier.
    require_option(h, "time_limit", limit);
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

// One heuristic, alone, at `effort` with its patience gate disabled (`0`
// means no gate at all, not "give up immediately"), so the budget is the
// only thing competing with the deadline.  The default is one whole
// vanilla FJ budget, which on this instance is already more than the limit
// allows; `kUnbindableEffort` is the setting at which the budget stops
// competing at all.
std::string alone_at_limit(const std::string& heuristic, double effort = 1.0) {
    const auto lines = presolve_heur_lines([&](Highs& h) {
        require_option(h, "mip_heuristic_suite", heuristic);
        require_option(h, "mip_heuristic_" + heuristic + "_effort", effort);
        require_option(h, "mip_heuristic_" + heuristic + "_patience", 0);
    });
    REQUIRE(lines.size() == 1);
    require_expected_nnz(lines.front());
    return lines.front();
}

// The `mip_heuristic_<name>_effort` ceiling (#116).  At this setting the
// budget cannot bind on any model — that headroom exists for #113's
// calibration probe, and it is also what makes `attempt_cap`, and
// therefore every unit the deadline is polled between, as large as it can
// get (issue #117).
constexpr double kUnbindableEffort = 1e6;

// The property, for one heuristic that charges effort in units an attempt
// cap can be compared against.
void require_stopped_mid_attempt(const char* heuristic) {
    const std::string line = alone_at_limit(heuristic);
    INFO(line);
    CHECK(end_s_of(line) <= kLimit + kSlack);
    CHECK(effort_of(line) < kAttemptCapUnits);
}

// The limit the #152 ratio case below runs at, longer than `kLimit`
// deliberately — see the note above that case.
constexpr double kRatioLimit = 0.5;

// The fraction of its limit a clock-bound Scylla dispatch must spend.
// Below the after-fix ratio (1.00-1.03 measured) by a wide margin and far
// above the before-fix one (0.52).  Not tighter than that: the point is to
// separate "spends its limit" from "spends half of it", and a bound close
// to 1.0 would be measuring the last PDLP solve's length instead.
constexpr double kMinRatio = 0.85;

}  // namespace

// FJ is the heuristic #114 was reported against: its callback had gates
// for the attempt budget, the total budget and staleness, and no clock.
TEST_CASE("deadline: FJ stops at the time limit, not at its effort budget", "[deadline][serial]") {
    require_stopped_mid_attempt("fj");
}

// LocalMIP had the identical hole and was never reported, because nothing
// had swept its effort option to where the hole opens.
// `kTermCheckInterval` had documented a termination-check cadence for this
// loop since it was introduced while being referenced nowhere in the tree;
// it is now what paces the check.
TEST_CASE("deadline: LocalMIP stops at the time limit, not at its effort budget",
          "[deadline][serial]") {
    require_stopped_mid_attempt("local_mip");
}

// FPR already polled the deadline per inner attempt — via `terminated()`,
// which writes `mipsolver.termination_status_` from a worker thread when a
// terminator is attached.  It polls `past_deadline()` now; this pins that
// the behaviour it had is the behaviour it kept.
TEST_CASE("deadline: FPR stops at the time limit, not at its effort budget", "[deadline][serial]") {
    require_stopped_mid_attempt("fpr");
}

// Scylla is measured against the full budget rather than the attempt cap,
// and on effort rather than wall time — see `kBudgetUnits` and the note
// above it.  Its guard is `past_deadline()` inlined so the same clock read
// also yields PDLP's input time limit, and it is correct; but it can only
// act between pump iterations, so the question this case can answer is not
// "did it stop mid-attempt" but "did it stop at all before its budget",
// which is the one a removed guard would fail.
//
// **What this case establishes today is weaker than its name.**  Its
// `effort=1.0` dispatch charges 5152601 against a real budget of
// `nnz << 10` = 5087232, and it reports that same effort at a 0.1 s limit
// and at a 5 s one — so it is *budget*-bound, and it stops just past its
// budget rather than before it.  The old `kBudgetUnits / 4` bound did not
// notice because `kBudgetUnits` is 80x the current budget (see the warning
// on `kAttemptCapUnits`).  The bound below is the measured charge with 2x
// of room, which at least fails on a regression; making the case mean what
// its name says needs a limit and an effort at which the clock actually
// binds, which is #154.
TEST_CASE("deadline: Scylla stops before spending its budget", "[deadline][serial]") {
    const std::string line = alone_at_limit("scylla");
    INFO(line);
    require_expected_nnz(line);
    CHECK(effort_of(line) < 2 * kScyllaEffortAtDefaultEffort);
}

// Scylla's *lower* bound, and the one #152 was filed for: a clock-bound
// dispatch has to spend substantially all of its limit, not half of it.
//
// The measured defect, on this instance and on `bell5`, was end_s/limit =
// 0.515-0.525 at every limit from 0.5 s to 16 s (0.580 at 0.25 s on this
// model, where the dispatch is only a handful of pump rounds long) — near
// enough a constant, because the trip point is `T/(1+alpha)` for `alpha`
// the fraction of dispatch wall time spent inside the wrapped instance's
// `run()`, and `alpha` does not depend on `T`.  Scylla was handing back
// half of every deadline silently: the dispatch closes with an ordinary
// `effort`/`found` line and nothing in it says the chain stopped early.
//
// Cause, and why this case cannot be satisfied by loosening anything:
// `ContestedPdlp` reuses one `Highs` instance for every pump solve, so
// `runPresolve`'s `left = options_.time_limit - timer_.read()` compared a
// *remaining* time against an *accumulated* one.  The long note at
// `zeroAllClocks()` in `contested_pdlp.cpp` carries the derivation.
// Handing the sub-solver a bigger number would move this ratio too, which
// is exactly why it is not the whole test — "ContestedPdlp: the wrapped
// instance's clock does not accumulate across solves" in
// `test_contested_pdlp.cpp` pins the cause, and the two are only sound
// together.
//
// `kRatioLimit` is longer than `kLimit` deliberately: the defect needs
// enough pump rounds to accumulate `alpha * T` of run time, and 0.1 s on
// this model is only a handful of PDLP solves.  At 0.5 s the ratio is
// fully developed — 0.518 measured before the fix, 1.002 after.
//
// Load-safety, since this is a wall-clock assertion: it is a *lower*
// bound, so every way a starved runner can behave makes it easier.  A
// descheduled worker notices the deadline later, not sooner, and the
// overshoot is bounded by one PDLP solve — the same granularity floor that
// makes an *upper* bound on Scylla a fudge (see the "asserted on effort
// alone" note at the top of this file) works in this bound's favour.  The budget cannot end the
// dispatch first
// (`kUnbindableEffort`) and neither can patience (`0` is no gate at all).
// What is exposed to the clock is the fixture, not the assertion:
// `REQUIRE(lines.size() == 1)` needs the solve to reach `run_sequential`
// before the limit expires.  That is the `[serial]` rule this file already
// states — the tag follows the fixture — and it applies here with 5x the
// headroom the 0.1 s cases have.
TEST_CASE("deadline: a clock-bound Scylla dispatch spends its whole limit",
          "[deadline][scylla][serial]") {
    const auto lines = presolve_heur_lines(
        [](Highs& h) {
            require_option(h, "mip_heuristic_suite", std::string("scylla"));
            require_option(h, "mip_heuristic_scylla_effort", kUnbindableEffort);
            require_option(h, "mip_heuristic_scylla_patience", 0);
        },
        kRatioLimit);
    REQUIRE(lines.size() == 1);
    const std::string& line = lines.front();
    INFO(line);
    require_expected_nnz(line);
    // The upper half: it stopped for the clock and did not run wild.  The
    // budget genuinely cannot bind here — at `kUnbindableEffort` it is
    // `nnz << 10 * 1e6` = 5.09e12, five orders above anything a 0.5 s
    // dispatch reaches — so this is not a budget comparison and must not be
    // written as one.  It is the measured charge with 2x of room, which is
    // the load-safe direction for an upper bound on effort.
    CHECK(effort_of(line) < 2 * kScyllaEffortAtRatioLimit);
    // The lower half, and the point of the case.
    CHECK(end_s_of(line) >= kMinRatio * kRatioLimit);
}

// The dispatch-level statement, which is the one the benchmark harness
// depends on: whatever subset of the chain gets to run, none of it is
// still running past the limit.  Note the *subset* — at `effort=1.0` the
// first heuristic legitimately consumes the whole limit and its
// successors are skipped by `run_sequential`'s own `terminated()` guard.
//
// Which subset runs is itself a timing fact, so Scylla may be in it and is
// held to the same effort bound it gets on its own rather than to a wall
// clock it cannot honour.
TEST_CASE("deadline: no presolve heuristic outlives the limit", "[deadline][serial]") {
    const auto lines = presolve_heur_lines([](Highs& h) {
        require_option(h, "mip_heuristic_suite", std::string("all"));
        require_option(h, "mip_heuristic_fj_effort", 1.0);
        require_option(h, "mip_heuristic_fpr_effort", 1.0);
        require_option(h, "mip_heuristic_local_mip_effort", 1.0);
        require_option(h, "mip_heuristic_scylla_effort", 1.0);
    });
    REQUIRE(!lines.empty());
    for (const auto& line : lines) {
        require_expected_nnz(line);
        INFO(line);
        if (line.contains("name=scylla ")) {
            CHECK(effort_of(line) < kBudgetUnits / 4);
        } else {
            CHECK(end_s_of(line) <= kLimit + kSlack);
        }
    }
}

// ===================================================================
// ...and it binds the work units *below* a heuristic's runner (#117)
//
// #114 put a deadline poll in each heuristic's own loop, which bounds the
// overrun at one of that loop's iterations.  It does not bound the
// iteration.  FPR's is a whole DFS attempt, sized from the effort option
// (`HeuristicBudget::attempt_cap`), and Scylla's is a whole pump round,
// which contains a PDLP solve and an FPR rounding sized the same way; and
// before either loop starts, both heuristics run a sequential setup that
// nothing was watching the clock during at all.  Measured on `rail02`
// (542k nonzeros, 16 workers, presolve-only, budget deliberately
// unreachable): a 20 s limit produced a 38.1 s FPR dispatch and a 28.7 s
// Scylla one, both of which had spent every second of the overrun inside
// setup — 34.5 s of `compute_var_order` calls for FPR, 20.7 s of shared-LP
// construction plus var orders for Scylla — and the probe behind #113 had
// 23 such runs SIGKILLed at 5.5x their limit.
//
// **The cases below do not reproduce that.**  On the bundled instances a
// whole FPR attempt and a whole Scylla pump round are milliseconds and the
// setup is smaller still, so an end-to-end run here already ends within
// its limit on the *unfixed* build — measured 0.10-0.19 s against a 0.1 s
// limit for both heuristics at the maximum effort option.  A wall-clock
// assertion on this hardware would therefore pass either way, which is
// exactly the vacuous bound the "asserted on effort alone" note at the
// top of this file refuses.
//
// So the sub-attempt gate is pinned where it is decidable: one FPR attempt
// against a clock that has already passed, compared with the same attempt
// against a clock that has not.  Same instance, same seed, same config,
// same unbindable effort budget — the deadline is the only difference, and
// a build without the poll inside the DFS cannot tell them apart.  The
// PDLP half of the same statement lives in `test_contested_pdlp.cpp`,
// where the sub-solver's limit is observable.

namespace {

// Effort charged by one one-shot FPR attempt on `instance` when the solve
// carries `time_limit`, with an effort budget that cannot bind.
//
// The mode is `kRepairSearch`, which is what the shipped rotation gives
// worker 6 (`kInitialFprConfigs`), so the *unbounded* arm runs Phase 3 as
// well as the Phase 2 DFS and the two arms are a like-for-like config.
//
// What this discriminates is the Phase 2 DFS poll, and only that.  Phase 3
// is entry-gated on `!deadline.expired()`, so `repair_search` cannot be
// entered with a deadline that has already passed: its per-node poll gets
// no coverage here or anywhere else, and reaching it would need a deadline
// that expires *during* the repair — a timing-dependent thing to arrange.
// Phase 3's entry gate is covered only when the DFS reaches a leaf inside
// `kDeadlinePollNodes` nodes, which is instance-dependent and not asserted.
size_t one_attempt_effort(const char* instance, double time_limit) {
    // `HighsMipSolverData::init` reads `parallel::num_threads()`; see the
    // note at the other `build_bare_mipsolver` call site.
    highs::parallel::initialize_scheduler();
    Highs highs;
    highs.setOptionValue("output_flag", false);
    HighsCallback cb(&highs);
    auto mipsolver = build_bare_mipsolver(highs, cb, instance, time_limit);

    CscMatrix csc;
    const ProblemView problem = make_problem(*mipsolver, csc);

    FprConfig cfg{};
    // "Cannot bind": the effort gates in `fpr_attempt_step` and
    // `fpr_attempt_finish` are subtractions against this, so it stands in
    // for the `mip_heuristic_fpr_effort=1e6` an unreachable budget is
    // spelled as end-to-end.  Half of SIZE_MAX rather than all of it so
    // the `max_effort - already_used` arithmetic has room on both sides.
    cfg.max_effort = std::numeric_limits<size_t>::max() / 2;
    cfg.csc = &csc;
    cfg.mode = FrameworkMode::kRepairSearch;
    cfg.strategy = &kStratLocks;
    cfg.binary_mask = problem.binary.data();

    Rng rng(0);
    return fpr_attempt(*mipsolver, cfg, rng, 0).effort;
}

}  // namespace

TEST_CASE("deadline: one FPR attempt stops on the clock, not on its effort budget", "[deadline]") {
    // `p0548` is a bundled MIP whose DFS runs long enough for the two
    // measurements to separate by orders of magnitude; `kInstance` above
    // is not reused because these cases measure a single attempt rather
    // than a dispatch, and want the attempt to be as long as possible.
    constexpr const char* kAttemptInstance = "p0548.mps";

    // The clock has passed before the attempt begins: the limit is applied
    // after the model is read and `HighsMipSolver`'s own clock starts at
    // zero from there, so a microsecond limit is expired after the first
    // microsecond of `runSetup` — by construction, not by a race.
    const size_t stopped = one_attempt_effort(kAttemptInstance, 1e-6);
    // No limit at all, which is HiGHS's own default.
    const size_t unbounded = one_attempt_effort(kAttemptInstance, kHighsInf);

    INFO("effort with an expired deadline: " << stopped << ", without one: " << unbounded);
    // The attempt that was allowed to run is the control: if it did not do
    // real work, the comparison below proves nothing.
    REQUIRE(unbounded > 0);
    CHECK(stopped * 10 < unbounded);
}

// The end-to-end statement at the extreme setting, and an honest note
// about what it can and cannot catch here.
//
// At `effort=1e6` the budget cannot bind, so `attempt_cap` — the size of
// the unit the runner polls the deadline between — is as large as the
// option surface allows.  On `rail02` that is the configuration that
// produced a 38.1 s dispatch under a 20 s limit.  On `gesa2` it produces
// nothing of the kind on *either* build: one attempt and one pump round
// are milliseconds here, so the unfixed binary was measured ending at
// 0.102 s (FPR) and 0.185 s (Scylla) against the same 0.1 s limit.
//
// This case is therefore a guard rather than a discriminator — it fails if
// a future change lets the effort option buy time again — and the
// discriminating measurement is the attempt-level case below it.
TEST_CASE("deadline: an unbindable budget does not loosen the deadline", "[deadline][serial]") {
    const std::string fpr_line = alone_at_limit("fpr", kUnbindableEffort);
    INFO(fpr_line);
    require_expected_nnz(fpr_line);
    CHECK(end_s_of(fpr_line) <= kLimit + kSlack);

    // Scylla on effort, for the reason the "asserted on effort alone" note
    // at the top of this file gives: its floor is one whole PDLP solve,
    // whose wall time varies 127-700 ms with machine load while its charge
    // does not.  Measured charge with 2x of room, rather than the stale
    // `kBudgetUnits / 4` (which is 10x above it and bounded nothing).
    const std::string scylla_line = alone_at_limit("scylla", kUnbindableEffort);
    INFO(scylla_line);
    require_expected_nnz(scylla_line);
    CHECK(effort_of(scylla_line) < 2 * kScyllaEffortAtLimit);
}

// ===================================================================
// ...and it binds the *dive-time* setup too (#118)
//
// `fpr_lp`'s dispatch setup has the shape #117 fixed in the two presolve
// heuristics — ten `compute_var_order` calls plus two reference LP solves,
// all sequential on the dispatching thread before a worker exists, none of
// it sized by any option — and #117 left it ungated, its whole evidence
// base being presolve-only runs.
//
// It also had a way of hiding the problem that the presolve path does not:
// the reference solves return an empty vector once the clock has passed,
// which the `ac_ptr`/`zv_ptr` fallbacks read as "LP failed, use the
// full-obj solution", and the setup went on to build all ten orders.  So
// the bounded half of the setup masked the unbounded half.
//
// And it differs in the half that is not a copy of #117 at all: `fpr_lp`
// draws from upstream's dive-time LP-iteration envelope and charges its
// work back, so a setup abandoned *after* a reference solve still owes
// that envelope.  `SetupResult` carries the count out of the bail for
// `run` to charge; the presolve fix could report zero and be done.
//
// **What is decidable here is the entry gate**, for the reason the #117
// note above gives: on the bundled instances the whole setup is
// microseconds, so a wall-clock assertion around it would pass on the
// unfixed build too.  A deadline that has *already* passed is not a timing
// fact, though — the limit is applied after the model is read, so it is
// expired by construction — and the two cases below separate "declined the
// clock" from "skipped for want of an optimal LP", which is the
// distinction the bail exists to make and the one `run` books on.
//
// Not covered, and deliberately: the per-arm poll and the polls around the
// reference solves. Reaching those needs a setup that gets past the
// LP-status check, i.e. a scaled-optimal LP relaxation, which a bare
// `HighsMipSolver` does not have — `HighsLpRelaxation::resolveLp` on one
// segfaults inside the simplex on solver state that only
// `HighsMipSolver::run` installs. Their coverage is the same gap #117
// documents for its own setup bail-outs, on the same instances, for the
// same reason.

namespace {

// The dive-time setup's verdict on `instance` under a solve carrying
// `time_limit`, without running the dispatch it would have set up.
fpr_lp::SetupProbe dive_setup_at(const char* instance, double time_limit) {
    // `HighsMipSolverData::init` reads `parallel::num_threads()`; see the
    // note at the other `build_bare_mipsolver` call site.
    highs::parallel::initialize_scheduler();
    Highs highs;
    highs.setOptionValue("output_flag", false);
    HighsCallback cb(&highs);
    auto mipsolver = build_bare_mipsolver(highs, cb, instance, time_limit);
    // The per-call budget `run` would have derived from the LP-iteration
    // envelope.  It is recorded in the setup and steers none of what is
    // asserted below — any in-range value does; this is the shape of the
    // real one.
    const size_t max_effort = mipsolver->mipdata_->ARindex_.size() << 12;
    return fpr_lp::probe_setup(*mipsolver, max_effort);
}

// The instance the attempt-level case above uses, for the same reason: a
// bundled MIP with enough structure that the setup would do real work.
constexpr const char* kDiveInstance = "p0548.mps";

}  // namespace

TEST_CASE("deadline: the fpr_lp dive setup declines an expired clock", "[deadline]") {
    // The limit is applied after the model is read and `HighsMipSolver`'s
    // own clock starts at zero from there, so a microsecond limit has
    // passed before the setup is entered — by construction, not by a race.
    const auto expired = dive_setup_at(kDiveInstance, 1e-6);
    INFO("expired limit: built=" << expired.built << " bail=" << expired.deadline_bail
                                 << " lp_iterations=" << expired.lp_iterations);
    CHECK(expired.deadline_bail);
    CHECK(!expired.built);
    // The accounting half of the fix. This bail is ahead of both reference
    // LP solves — that ordering is what the entry check buys — so there is
    // nothing for `run` to charge the shared RENS/RINS envelope. A bail
    // after one carries its iterations out in `SetupResult` instead, and
    // `run` charges them on a path disjoint from the normal one, so
    // neither path can charge the other's work.
    CHECK(expired.lp_iterations == 0);
}

// A deadline bail is not a skip. An empty model or an unsolved LP
// relaxation consumes nothing and is not a clock event, so `run` books it
// differently — silently, where a bail leaves a `[Heur]` line. This case
// is also what keeps the one above honest: a `deadline_bail` hard-wired to
// `true` would pass it and fail this.
TEST_CASE("deadline: an fpr_lp setup skipped for want of an LP is not a deadline bail",
          "[deadline]") {
    const auto skipped = dive_setup_at(kDiveInstance, kHighsInf);
    INFO("no limit, no root LP: built=" << skipped.built << " bail=" << skipped.deadline_bail
                                        << " lp_iterations=" << skipped.lp_iterations);
    CHECK(!skipped.built);
    CHECK(!skipped.deadline_bail);
    CHECK(skipped.lp_iterations == 0);
}

// ===================================================================
// ...and the engine is armed where production builds it (#151)
//
// #151 put the poll inside `PropEngine::propagate`, and an engine only
// polls a clock somebody handed it.  Three lines do that handing: the
// `set_deadline` in `fpr_core.cpp`'s `acquire_engine`, the two in
// `repair_search`, and — for the poll to be worth anything to the DFS —
// the `kDeadlineExpired` break in `fpr_attempt_step`.  A cold review of
// the first version of this fix found all three uncovered: deleting the
// `acquire_engine` arming, which is the exact bug its own comment warns
// about, left the whole suite green, and so did neutralising the DFS
// break together with `repair_search`'s arming.  The unit cases in
// `tests/test_prop_engine.cpp` arm an engine by hand, so they pin the
// poll and nothing about who arms it — the same shape as the
// `select_ref` finding #128 records, and there is no `switch` on
// `PropResult` anywhere, so `-Werror=switch` supplies no checklist here
// either.
//
// This drives the *lifecycle* API rather than the one-shot
// `fpr_attempt`, because the hazard is the reuse path: the engine is
// cached in `FprScratch` across attempts, so an arming that runs only
// where the engine is constructed leaves every later attempt polling a
// stale limit.  The limit is therefore moved *between* the two
// attempts, and attempt 2's assertion is what a construction-only
// arming fails.
TEST_CASE("deadline: the cached FPR engine is armed on every attempt, not just the first",
          "[deadline]") {
    // A limit no attempt here can reach, and one already behind the
    // solver's clock — which started when the model was read, so a
    // microsecond is past by construction rather than by a race.  Not
    // `kHighsInf` for the live one: `make_deadline` collapses an infinite
    // limit to a null timer, which is the un-armed case and would make
    // the assertions below say nothing.
    constexpr double kLiveLimit = 3600.0;
    constexpr double kExpiredLimit = 1e-6;

    highs::parallel::initialize_scheduler();
    Highs highs;
    highs.setOptionValue("output_flag", false);
    HighsCallback cb(&highs);
    // `p0548` for the reason `one_attempt_effort` above uses it: a DFS
    // whose per-node propagation cascade is big enough to reach the poll.
    auto mipsolver = build_bare_mipsolver(highs, cb, "p0548.mps", kLiveLimit);

    CscMatrix csc;
    const ProblemView problem = make_problem(*mipsolver, csc);

    FprScratch scratch;
    FprConfig cfg{};
    cfg.max_effort = std::numeric_limits<size_t>::max() / 2;
    cfg.csc = &csc;
    cfg.mode = FrameworkMode::kDfs;
    cfg.strategy = &kStratLocks;
    cfg.binary_mask = problem.binary.data();
    cfg.scratch = &scratch;

    Rng rng(0);

    // Attempt 1 — the construction path.
    FprAttemptState first;
    fpr_attempt_begin(first, *mipsolver, cfg, rng, 0);
    REQUIRE(scratch.prop_engine.has_value());
    // Without this the node assertion at the end is vacuous: a
    // non-propagating mode never calls `propagate` from the DFS at all.
    REQUIRE(first.do_propagate);
    CHECK(scratch.prop_engine->deadline().timer == &mipsolver->timer_);
    CHECK(scratch.prop_engine->deadline().limit == kLiveLimit);

    // Run it out, so the next `begin` sees a genuinely reused engine
    // rather than one abandoned mid-DFS.
    while (fpr_attempt_step(first, *mipsolver, cfg, rng, cfg.max_effort) ==
           FprStepResult::kBudgetGate) {
    }
    static_cast<void>(fpr_attempt_finish(first, *mipsolver, cfg, rng));

    // Attempt 2 — the reuse path, with the limit moved behind the clock.
    // `deadline_of` reads `options_mip_`, which aliases the live `Highs`
    // options, so this is the limit the next `begin` is obliged to pick
    // up.  An arming confined to the construction path leaves the engine
    // on `kLiveLimit` and fails here.
    require_option(highs, "time_limit", kExpiredLimit);
    FprAttemptState second;
    fpr_attempt_begin(second, *mipsolver, cfg, rng, 1);
    REQUIRE(scratch.prop_engine.has_value());
    CHECK(scratch.prop_engine->deadline().timer == &mipsolver->timer_);
    CHECK(scratch.prop_engine->deadline().limit == kExpiredLimit);

    // And the expiry reaches the DFS: the first propagating node's
    // fixpoint returns `kDeadlineExpired` and `step` breaks out at once,
    // instead of running the whole `kDeadlinePollNodes` batch that its
    // own pre-node poll would allow.  Both halves are load-bearing — an
    // un-armed engine never returns the state, and a `step` that ignores
    // it visits the full batch.
    static_cast<void>(fpr_attempt_step(second, *mipsolver, cfg, rng, cfg.max_effort));
    INFO("nodes visited under an already-expired deadline: " << second.nodes_visited);
    // The DFS stopped rather than finished -- otherwise the node count
    // below would be small for a reason that has nothing to do with the
    // clock, and both mutations would pass it.
    CHECK_FALSE(second.found_complete);
    // Fewer than 15, which is what a `step` that never sees a
    // `kDeadlineExpired` visits: `kDeadlinePollNodes` is 16 and the
    // pre-node poll fires at the *top* of the sixteenth iteration, before
    // that node is counted.  Measured 10 with the wiring in place and 15
    // with either half of it removed (the `acquire_engine` arming, or the
    // break itself).  The exact 10 is not asserted -- it is a function of
    // where `kPropagateDeadlinePollWork` falls in each node's cascade --
    // but the gap between the two is the mechanism, and it is what this
    // discriminates.
    CHECK(second.nodes_visited < 15);
    static_cast<void>(fpr_attempt_finish(second, *mipsolver, cfg, rng));
}
