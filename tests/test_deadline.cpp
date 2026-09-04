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
// still reads and presolves quickly — reading it fast is what the instance
// is chosen for.  It is *not* what buys headroom over the limit: at
// `threads=1` the budget is `heuristic_effort_budget(nnz, effort)` =
// `nnz << 10 * effort` (#116), which at `effort=1.0` is 5087232 units, and
// three of the four heuristics spend the whole of that in tens of
// milliseconds on this model (measured: FJ 26 ms, FPR 46 ms, Scylla 90 ms).
// So `effort=1.0` here is a *budget*-bound configuration and not a
// clock-bound one — that was #154 — and the headroom comes from the effort
// option instead.  See `kClockBindingEffort`.
constexpr const char* kInstance = "gesa2.mps";

// Far below the time any of the four needs to exhaust its budget at
// `kClockBindingEffort`, and — the property that gives these cases teeth —
// far below the time any of them needs to finish a single *attempt* at that
// setting.  See `kAttemptCapUnits`.
constexpr double kLimit = 0.1;

// The second limit, 5x `kLimit`.  Two things use it: `require_clock_bound`
// below, which certifies that a `kLimit` dispatch was stopped by the clock
// by showing the same configuration charges strictly more when given more
// time; and the #152 ratio case at the foot of the presolve group — see the
// note above that case for why *that* one needs the longer limit
// specifically.
constexpr double kRatioLimit = 0.5;

// Allowance above the limit for the polling cadence plus teardown.
// Measured overshoot on the development machine is 0-6 ms for FJ, FPR and
// LocalMIP, and 26 ms for Scylla, whose floor is a whole PDLP solve; a
// slow shared runner is given far more rather than a tight bound that
// turns scheduler jitter into a red suite.  Without a sub-attempt
// deadline check these runs end at 2.75 s (FJ at `kClockBindingEffort`,
// measured with its callback's clock check neutralised), so the assertion
// keeps its teeth at this slack.
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
// better — the effort bound `effort < kAttemptCapUnits` (2.2-17.6e6 spent
// against 5.09e8) erodes as the limit grows, trading one flake for
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
// The measured table this file's constants and margins are chosen from
// follows.  All rows on this model at `threads=1`, seed 0, presolve-only, patience 0
// (no gate at all), three repeats each, taken after #152 — which roughly
// doubled what a clock-bound Scylla dispatch charges by giving it its whole
// deadline instead of half — after #140, which moved the same axis again by
// cutting charged effort per pump round, and after #153, which moved the
// Scylla rows alone: `presolve=off` on the shared PDLP instance means the
// pump iterates over the unreduced LP, and charged effort per round is
// `pdlp_iters * nnz`.  Measured on this model, that roughly doubled every
// clock-bound Scylla row again (0.5 s: 41.4-42.9e6 -> 86.0-88.0e6) and left
// the budget-bound one where it was, since a budget is a budget.
//
// A clock-bound row is a *rate*, so it is a range and not a figure: it is
// what the host got through in the time, and it moves a few percent between
// repeats and a few percent again between the standalone CLI and this
// fixture, which pins the thread and installs a log callback.  The ranges
// below span both, so a reader re-measuring either way lands inside one.
// Only the `1.0` rows and Scylla's 0.1 s rows are exact, and they are exact
// for the reason the whole issue was about: those dispatches are stopped by
// something other than the clock.
//
//   heuristic  effort               0.1 s               0.5 s
//   ---------  -------------------  ------------------  ------------------
//   fj         kClockBindingEffort  17.1-18.2e6 (0.102)  83.6-88.2e6
//   fpr        kClockBindingEffort  9.9-10.3e6           54.6-55.3e6
//   local_mip  kClockBindingEffort  2156255     (0.104)  9.9-10.1e6
//   scylla     kClockBindingEffort  15.7-18.9e6 (0.100)  84.8-87.8e6
//   scylla     kUnbindableEffort    15.7-18.9e6 (0.103)  84.8-88.0e6
//   fj         1.0                  5110832     (0.036)  5110832
//   fpr        1.0                  5087251     (0.055)  5087251
//   scylla     1.0                  5092240     (0.046)  5092240
//
// The last three rows are what #154 was about, and they are the reason the
// cases below do not run at `effort=1.0`: those dispatches report the *same*
// effort at both limits, which is what a budget-bound dispatch does and what
// a clock-bound one cannot do.  The first four rows increase 4.2-5.6x
// between the two limits, which is the certification `require_clock_bound`
// makes of every case that claims the clock stopped it.
//
// Scylla's two `kClockBindingEffort` and `kUnbindableEffort` rows used to be
// identical to the digit, because at `threads=1` the clock-bound trajectory
// does not depend on the effort option once the option stops binding.  Since
// #153 they agree only to about 20% at 0.1 s: a clock-bound Scylla row is
// quantised in whole pump rounds — nothing can stop the round it is inside —
// and #153 made a round expensive enough on this model that whether the last
// one fits inside 0.1 s is no longer settled.  Both rows are still the same
// trajectory; only the point it is cut at moves.  Hence the range, and hence
// the constants below taking the top of it.

// gesa2's post-presolve nonzero count, which the two sizes below are
// derived from.  Read back off the `[Heur] nnz=` field of every line these
// cases parse rather than trusted: a HiGHS tag bump that changes presolve
// would otherwise erode the margins silently instead of failing here.
constexpr size_t kNnz = 4968;

// The `mip_heuristic_<name>_effort` every case that claims the clock
// stopped a dispatch runs at.  One variable, feeding both the option those
// cases write and the two bounds below, so the pair cannot drift from the
// configuration it describes — which is exactly how #154 happened: the
// bounds were literals in a unit the option had stopped being denominated
// in, and nothing recomputed them.
//
// Why this value.  It is two-sided, and both sides are measured:
//
//   * From below, it has to make `effort < kAttemptCapUnits` a real
//     discriminator.  A build whose deadline is checked only *between*
//     attempts cannot report less than one `attempt_cap`, since nothing can
//     stop the attempt it is inside; so the bound only says anything if a
//     clock-bound dispatch charges materially less than one cap.  Measured
//     margins against the 508723200 below: 28.9x (fj), 50.0x (fpr), 236x
//     (local_mip), and 520x for Scylla against its budget.  At `effort=100`
//     FJ's margin is 2.9x, which is too thin to survive a faster host.
//
//   * From above, one `attempt_cap` has to stay small enough that a
//     *regressed* build — one that runs its whole attempt out — finishes
//     inside ctest's timeout, so a regression is a red assertion and not a
//     hang.  At the option's 1e6 ceiling one cap is 5.09e11 units, which is
//     roughly 45 minutes of FJ; here it is 508723200 units, 2.7 s of FJ and
//     23 s of LocalMIP.
//
// It is the configuration under test, not a fuse: raising it to make an
// assertion pass would be re-introducing #154 from the other end.
constexpr double kClockBindingEffort = 1000.0;

// `heuristic_effort_budget(nnz, kClockBindingEffort)` and
// `HeuristicBudget::attempt_cap` for this configuration — *derived*, not
// transcribed, because a literal is what #116's re-denomination silently
// invalidated.  Both helpers are `inline` rather than `constexpr`, so these
// are runtime `const`s; nothing here needs them at compile time.
//
// `make_budget` computes `total / (num_workers * 10)`, and at `threads = 1`
// all four heuristics land on the same number: FJ's option is per-worker,
// so `run_sequential` scales its total by the pool
// (`saturating_mul(sized, N)` in `mode_dispatch.cpp`), and at `N = 1` that
// is the identity; the other three size a whole dispatch and are not
// scaled.  The fixture pins `threads=1`, which is what makes one derivation
// cover all four.
//
// The third argument is the staleness ceiling, which does not enter
// `attempt_cap` at all; it is spelled the way `run_sequential` spells it,
// at the patience option these cases set (`0`, i.e. no gate — which
// `patience_threshold` returns as SIZE_MAX).
//
// Note what the name does *not* say: this is the budget at
// `kClockBindingEffort`, not "the budget".  The two `kUnbindableEffort`
// cases below run at a different effort and their budget is a thousand
// times larger, so `kBudgetUnits` is not a bound they may be given — a
// `CHECK(effort < kBudgetUnits)` there would be wrong and green.
const size_t kBudgetUnits = heuristic_effort_budget(kNnz, kClockBindingEffort);
const size_t kAttemptCapUnits =
    make_budget(kBudgetUnits, /*num_workers=*/1, patience_threshold(kNnz, 0.0, kBudgetUnits))
        .attempt_cap;

// What a clock-bound Scylla dispatch charges at the two `kUnbindableEffort`
// configurations the last two cases in the presolve group use, measured on
// the fixed (#152) binary and reproducible to the digit — see the table in
// the note at the top of this file.  The bounds are these values with room
// for a slower or faster host; effort is the load-safe direction for an
// upper bound, since a starved runner charges less.  They are transcribed
// measurements rather than derived sizes, so they have to be re-measured if
// `kLimit` or `kRatioLimit` moves — and, as #153 showed, whenever a change
// moves what a pump round *charges*.  Each is the largest of five repeats,
// which is the conservative end for an upper bound: a starved runner charges
// less, so the risk is a bound that never fires, not one that fires
// spuriously.  Values before #153 took LP presolve off the shared PDLP
// instance: 9779068 and 42970944, i.e. this roughly doubled.
//
// What the `2 x` in the two bounds below is buying, since #154's whole
// point is that a transcribed constant has to say: the observed spread
// across five repeats is 86.0-88.0e6 at `kRatioLimit`, about 2%, so the
// factor is roughly 100x that spread and is host headroom, not slack in
// the assertion.  What it still catches is a *mechanism* failure -- a
// dispatch that ignores the deadline and runs to its unbindable budget
// charges orders of magnitude more, not 2x.  And note how the old value
// behaved when the axis moved under it: it did not quietly keep passing,
// it went red at 0.4% over, which is why these were re-measured rather
// than widened.
constexpr size_t kScyllaEffortAtLimit = 18888466;       // effort=1e6, 0.1 s
constexpr size_t kScyllaEffortAtRatioLimit = 87975670;  // effort=1e6, 0.5 s

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

// One heuristic, alone, under `limit`, at `effort` with its patience gate
// disabled (`0` means no gate at all, not "give up immediately"), so the
// wall clock and the effort budget are the only two things that can stop
// it — which is what makes "which of the two stopped it" answerable.
std::string alone_at(const std::string& heuristic, double effort, double limit) {
    const auto lines = presolve_heur_lines(
        [&](Highs& h) {
            require_option(h, "mip_heuristic_suite", heuristic);
            require_option(h, "mip_heuristic_" + heuristic + "_effort", effort);
            require_option(h, "mip_heuristic_" + heuristic + "_patience", 0);
        },
        limit);
    REQUIRE(lines.size() == 1);
    require_expected_nnz(lines.front());
    return lines.front();
}

// The same at `kLimit`.  The default effort is the clock-binding one, so a
// case that does not name an effort is asking for the configuration the
// margins above were measured at; `kUnbindableEffort` is the setting at
// which the budget stops competing at all.
std::string alone_at_limit(const std::string& heuristic, double effort = kClockBindingEffort) {
    return alone_at(heuristic, effort, kLimit);
}

// The `mip_heuristic_<name>_effort` ceiling (#116).  At this setting the
// budget cannot bind on any model — that headroom exists for #113's
// calibration probe, and it is also what makes `attempt_cap`, and
// therefore every unit the deadline is polled between, as large as it can
// get (issue #117).
constexpr double kUnbindableEffort = 1e6;

// The certification, and the check whose absence #154 was: the identical
// configuration, given 5x the time, charges strictly more effort.
//
// At `threads=1` with a fixed seed a dispatch stopped by its budget or by
// its attempt cap reports the *same* effort at every limit — measured to
// the digit, and it is how #154 was found.  A strict increase therefore
// cannot come from anywhere but the clock, so this certifies that the
// `kLimit` run passed in here was clock-bound.  That run and no other: the
// longer run may itself be budget-bound and nothing here bounds its effort,
// which is deliberate — it is a witness, not a second assertion.
//
// Strict, not `!=`: a starved runner charges *less*, so `!=` would accept a
// longer run that was descheduled into charging less than the short one and
// call that a certification.  `[serial]` is why the increase is reliable at
// all; it cannot be arranged by taking the clock out of the fixture, since
// the clock is the thing under test.
//
// `effort` is the effort the caller took `at_limit` at, and is not defaulted:
// the two runs have to be the *identical* configuration for the comparison to
// mean anything, and a default would let a caller pair a witness taken at one
// effort with a longer run taken at another and still pass.
void require_clock_bound(const std::string& heuristic, double effort, const std::string& at_limit) {
    const std::string longer = alone_at(heuristic, effort, kRatioLimit);
    INFO("at " << kLimit << " s: " << at_limit);
    INFO("at " << kRatioLimit << " s: " << longer);
    CHECK(effort_of(at_limit) < effort_of(longer));
}

// The property, for one heuristic that charges effort in units an attempt
// cap can be compared against.
void require_stopped_mid_attempt(const char* heuristic) {
    const std::string line = alone_at_limit(heuristic);
    INFO(line);
    CHECK(end_s_of(line) <= kLimit + kSlack);
    CHECK(effort_of(line) < kAttemptCapUnits);
    require_clock_bound(heuristic, kClockBindingEffort, line);
}

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
// So the assertion is the budget comparison the name promises, against the
// budget this dispatch was actually given — 9779068 charged against
// 5087232000, a 520x margin — and `require_clock_bound` is what says the
// number on the left is a clock-bound one rather than a heuristic that
// happened to stop early for some other reason.  Both halves are needed:
// at `effort=1.0` this dispatch charges 5152601 against a budget of
// 5087232, i.e. it *exceeds* its budget and reports the same effort at
// 0.1 s and at 5 s, so a budget comparison alone was satisfiable only
// because the constant it used was 80x the budget (#154).
TEST_CASE("deadline: Scylla stops before spending its budget", "[deadline][serial]") {
    const std::string line = alone_at_limit("scylla");
    INFO(line);
    CHECK(effort_of(line) < kBudgetUnits);
    require_clock_bound("scylla", kClockBindingEffort, line);
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
// still running past the limit.  Note the *subset* — at
// `kClockBindingEffort` the first heuristic legitimately consumes the whole
// limit and its successors are skipped by `run_sequential`'s own
// `terminated()` guard.  Measured here: FJ alone, ending at 0.100-0.107 s.
//
// The whole configuration is the point, and it is the same one every other
// case in this group runs — `kClockBindingEffort` and patience `0` on all
// four.  Before #154 this case ran at `effort=1.0` with the patience
// options left at their shipped defaults, which made every heuristic in the
// chain *patience*-bound (measured 3/3: fj ended at 0.021 s, fpr 0.032,
// local_mip 0.071, scylla 0.092) — so the `end_s <= kLimit + kSlack` bound
// was being met by a gate that has nothing to do with the deadline, and
// Scylla was bounded by a quarter of a `kBudgetUnits` that was itself 80x
// its budget.  Neither half said anything about the clock.
//
// Which subset runs is itself a timing fact, so Scylla may be in it and is
// held to the same budget bound it gets on its own rather than to a wall
// clock it cannot honour — its floor is one whole PDLP solve; see the
// "asserted on effort alone" note at the top of this file.
TEST_CASE("deadline: no presolve heuristic outlives the limit", "[deadline][serial]") {
    const auto lines = presolve_heur_lines([](Highs& h) {
        require_option(h, "mip_heuristic_suite", std::string("all"));
        for (const std::string name : {"fj", "fpr", "local_mip", "scylla"}) {
            require_option(h, "mip_heuristic_" + name + "_effort", kClockBindingEffort);
            require_option(h, "mip_heuristic_" + name + "_patience", 0);
        }
    });
    REQUIRE(!lines.empty());
    for (const auto& line : lines) {
        require_expected_nnz(line);
        INFO(line);
        if (line.contains("name=scylla ")) {
            CHECK(effort_of(line) < kBudgetUnits);
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
    // does not.  Measured charge with 2x of room; a quarter of the budget,
    // which is what this bound used to be, is 1.27e12 at `kUnbindableEffort`
    // and bounds nothing.
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
