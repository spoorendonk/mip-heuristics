#include "continuous_loop.h"
#include "heuristic_common.h"
#include "Highs.h"
#include "mip/HighsMipSolverData.h"  // for kSolutionSource* constants
#include "solution_pool.h"
#include "test_common.h"
#include "worker_base.h"

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

// ===================================================================
// Patience: the absolute threshold (#111), the signal it reads (#116)
// and the ceiling it is clamped below (#116)
//
// A patience gate has two operands and both were broken.
//
// The *threshold* was `total_budget >> 2` in three of the four presolve
// heuristics.  A quarter of the budget cannot bound over-budgeting:
// doubling the budget doubles the tolerance, so the gate never fires
// relatively sooner.  It is now an absolute, instance-scaled
// `mip_heuristic_<name>_patience x nnz << 10` — a `constexpr` per
// heuristic at first, an option since #106, in the effort option's own
// unit since #116, where `0` still means no gate at all.
//
// The *signal* was each worker's own "I beat my own best", which restarts
// at infinity on every rebuild, so the pool's verdict was computed,
// discarded, and replaced by one that a rebuilt worker could satisfy by
// rediscovering a solution the pool already held.  #111 pointed the gates
// at `IncumbentSink::offer`'s acceptance verdict, which fixed that and
// left a subtler version of the same defect: the pool keeps a top-K, so a
// heuristic beating its own *worst* entry resets staleness forever
// without the solve's best objective moving.  #113 measured how far apart
// the two are over 233 instances, presolve-only, 30 s, 16 workers:
//
//                pool acceptances   incumbent improvements
//   fpr                    ~3.3 M                      590
//   local_mip              ~3.3 M                   24,598
//   scylla                367,801                      374
//   fj                      1,557                      297
//
// Five orders of magnitude for FPR.  A patience calibrated on
// improvements — which is the only thing it can honestly be calibrated on
// — cannot be spent against a gate that resets on acceptances, so since
// #116 both gate levels read `OfferResult::improved_incumbent` while
// `[Heur] found` and `[HeurSol] accepted` keep reporting the acceptance.
//
// Fixing either operand alone leaves the gate toothless on some
// instances, which is why all three landed.  Ratios below are charged
// effort at the top of a 20x budget sweep over its bottom, at
// `threads=1`, seed 0, `[Heur] ... phase=presolve effort=` summed over
// the solve:
//
//                     pre-#111   threshold only   + #111 signal   + #116
//   fpr / flugpl        19.98x        19.98x           1.03x       1.80x
//   fpr / p0548         20.00x         0.89x           0.89x       2.26x
//   fpr / gt2           20.00x        20.00x         1.5-4.4x       1.18x
//   fpr / egout         19.98x        19.98x          19.98x       1.82x
//   fj / p0548          15.67x         2.00x           2.00x       1.50x
//   scylla / flugpl     17.60x         1.00x           1.00x       1.00x
//
// Two of those *rose* between the last two columns while the gate got
// tighter, which is worth understanding before reading the column as a
// score: a stricter improvement signal retires a worker sooner at **both**
// ends of the sweep, and the low end has further to fall because it was
// budget-bound.  fpr/p0548 spends 4,231,951 -> 1,624,678 at the low end
// and 3,770,052 -> 3,675,057 at the high end, so the ratio goes 0.89x ->
// 2.26x on strictly less work at both points.  The ratio measures whether
// the gate binds; it does not measure how much it saves.  Medians over
// several seeds also differ from seed 0, so do not mix the two.
//
// egout is the row #111 recorded as a known miss and #116 exists to
// close: FPR earns 40+ acceptances there against four incumbent
// improvements, so an acceptance-driven gate never fired.  19.98x ->
// 1.82x, and it is asserted on below.  The last column is measured with
// `improved_best` taken against a monotone watermark; against the pool's
// front entry it is 2.45x, because the diversity path evicts that entry
// on egout and the degraded value is then beatable for free.
// ===================================================================

namespace {

// Total effort the `[Heur]` instrumentation attributes to `heur`'s
// presolve dispatches, summed over the solve.  Requires
// `log_dev_level=3`; the line is
//
//   [Heur] name=<n> phase=presolve start_s=.. end_s=.. effort=<N> ...
//
// which is the ground truth for how much budget a heuristic actually
// spent — `[Sequential]` carries the same number but not the phase, and
// `fpr_lp` reports under `phase=dive` against a different envelope.
size_t presolve_effort(const std::vector<std::string>& lines, const std::string& heur) {
    const std::string head = "[Heur] name=" + heur + " phase=presolve ";
    const std::string key = " effort=";
    size_t total = 0;
    for (const auto& line : lines) {
        const auto at = line.find(head);
        if (at == std::string::npos) {
            continue;
        }
        const auto eff = line.find(key, at + head.size());
        if (eff == std::string::npos) {
            continue;
        }
        total += std::strtoull(line.c_str() + eff + key.size(), nullptr, 10);
    }
    return total;
}

// Solve `inst` with exactly `heur` enabled and its effort option set to
// `effort`, and report what the heuristic spent.
//
// `threads` and `random_seed` are both pinned, for reproducibility
// rather than for coverage: the ratio turns out to be flat in the worker
// count (fpr/flugpl is ~19-20x before the fix and ~1.0-1.5x after, at
// N = 1, 2, 4 and 12 alike), because each worker retires at `stale / N`
// of its own effort so all N cross at `stale` of aggregate effort and N
// cancels.  Only the N = 1 figures are exact: above one worker the
// schedule is nondeterministic and repeated runs of the same case move
// within that band, which is why the multi-worker case below asserts a
// bound rather than a value.  What an unpinned test would
// buy is a number that differs between a 12-core laptop and a 2-vCPU CI
// runner, on an issue labelled portable.  `ScopedThreadPin` is what makes
// the pin survive whatever initialised the global task executor first —
// see its comment.
// `patience` is that heuristic's `mip_heuristic_<name>_patience` option,
// or a negative value to leave it at its shipped default.  Since #106 the
// threshold is an option rather than a constant, which is what lets the
// gate be switched off from a test instead of from a rebuild.
size_t effort_at(const char* inst, const char* heur, const char* option, double effort, int threads,
                 double patience = -1.0) {
    const auto lines = solve_capturing_log(inst, [&](Highs& h) {
        require_option(h, "log_dev_level", 3);
        require_option(h, "threads", threads);
        require_option(h, "random_seed", 0);
        require_option(h, option, effort);
        if (patience >= 0.0) {
            require_option(h, std::string("mip_heuristic_") + heur + "_patience", patience);
        }
        set_suite(h, heur);
    });
    return presolve_effort(lines, heur);
}

// The sweep is 20x wide (0.05 -> 1.00, the option's whole documented
// range).  A gate that binds holds the growth to a small constant; the
// pre-#111 numbers in the header comment are 16x-20x at this thread pin.
// 4x is deliberately loose — the worst post-fix ratio measured over four
// seeds at `threads=1` was 2.00x (fj/p0548, which sits on its structural
// ceiling of one gate plus one call of overshoot) and 4.43x at the worst
// seed of fpr/gt2, which is why gt2 is not asserted on.  The bound only
// has to separate "bounded by an absolute threshold" from "bounded by
// the budget".
constexpr double kMaxGrowth = 4.0;
// Both options are multiples of `nnz << 10` since #116, so the sweep that
// used to run 0.05 -> 1.00 now runs 4 -> 80.  Same 20x width, same budgets.
constexpr double kLowEffort = 4.0;
constexpr double kHighEffort = 80.0;

void check_gate_binds(const char* inst, const char* heur, const char* option, int threads = 1) {
    // Held across both solves so they share one pinned worker count.
    const ScopedThreadPin pin;
    INFO("instance=" << inst << " heuristic=" << heur << " threads=" << threads);
    const size_t low = effort_at(inst, heur, option, kLowEffort, threads);
    const size_t high = effort_at(inst, heur, option, kHighEffort, threads);
    // A zero here means the heuristic never ran, which would make the
    // ratio below vacuously true.
    REQUIRE(low > 0);
    REQUIRE(high > 0);
    INFO("effort at " << kLowEffort << " = " << low << ", at " << kHighEffort << " = " << high);
    REQUIRE(static_cast<double>(high) < kMaxGrowth * static_cast<double>(low));
}

}  // namespace

// ── The headline property: a 20x budget does not buy 20x the effort ──

TEST_CASE("patience gate: FPR exits on staleness rather than spending 20x", "[patience]") {
    check_gate_binds("p0548.mps", "fpr", "mip_heuristic_fpr_effort");
}

TEST_CASE("patience gate: FeasibilityJump exits on staleness rather than spending 20x",
          "[patience]") {
    check_gate_binds("p0548.mps", "fj", "mip_heuristic_fj_effort");
}

TEST_CASE("patience gate: Scylla exits on staleness rather than spending 20x", "[patience]") {
    check_gate_binds("flugpl.mps", "scylla", "mip_heuristic_scylla_effort");
}

// The case that needs *both* halves of the fix, and the reason the
// improvement signal came into scope.  With the threshold absolute but
// the signal still worker-local, flugpl spent 2,785,359 effort against a
// 69,632 ceiling — forty ceilings' worth — while the pool accepted
// exactly one solution all dispatch.  Every one of those resets came
// from a feasible point FPR had already found and the pool refused.
TEST_CASE("patience gate: FPR on flugpl needs a shared verdict, not its own", "[patience]") {
    check_gate_binds("flugpl.mps", "fpr", "mip_heuristic_fpr_effort");
}

// One multi-worker case.  The ratio is flat in N (see `effort_at`), so
// this is not extra coverage of a different regime so much as a guard
// against a future change that makes the gate depend on the pool size —
// the runner-level gate is aggregated over workers and the per-worker
// share is `stale / N`, which is exactly the kind of relation a refactor
// drops.  Four is safe on any host: if the executor caps the pool lower,
// the flatness means the assertion still holds.
TEST_CASE("patience gate: FPR on flugpl binds at four workers too", "[patience]") {
    check_gate_binds("flugpl.mps", "fpr", "mip_heuristic_fpr_effort", /*threads=*/4);
}

// ── The option, end to end: 0 really does remove the gate ──

TEST_CASE("patience gate: mip_heuristic_fpr_patience=0 restores budget-tracking", "[patience]") {
    // The unit assertions below pin `patience_threshold(nnz, 0, budget)
    // == SIZE_MAX`; this pins that the option reaches it, through the
    // record registration, `kChain`, `run_sequential` and `make_budget`.  #113's
    // probe needs a run where the gate provably never fires, and a
    // semantic that is only true in a header is no use to it.
    //
    // flugpl/FPR is the case where the gate is measurably doing the work:
    // 19.98x charged-effort growth over a 20x budget sweep before #111,
    // 1.03x after.  Switching the gate off at the top of that sweep has to
    // put the spend back where the budget puts it — so the ratio here is
    // between two solves at the *same* effort, differing only in the
    // patience option, and the bound is the same 4x that separates
    // "bounded by an absolute threshold" from "bounded by the budget".
    const ScopedThreadPin pin;
    const size_t gated = effort_at("flugpl.mps", "fpr", "mip_heuristic_fpr_effort", kHighEffort, 1);
    const size_t ungated =
        effort_at("flugpl.mps", "fpr", "mip_heuristic_fpr_effort", kHighEffort, 1, /*patience=*/0);
    REQUIRE(gated > 0);
    INFO("gated=" << gated << " ungated=" << ungated);
    REQUIRE(static_cast<double>(ungated) > kMaxGrowth * static_cast<double>(gated));
}

// The instance #111 recorded as its known miss, now asserted on.
//
// `fpr/egout` was 19.98x before #111 and 19.98x after it: FPR genuinely
// earns forty-odd pool acceptances there against only four incumbent
// improvements, because `SolutionPool` keeps a top-`kPoolCapacity` and
// FPR keeps beating its worst entry.  #111 called those acceptances the
// project's definition of production and ruled the fix out of scope;
// #113's probe overruled that, because the pool's admission policy is not
// a measure of a heuristic's productivity and a threshold cannot be
// calibrated against it.  The gate reads incumbent improvements now, so
// egout is a ratio case like the rest.
TEST_CASE("patience gate: FPR on egout binds once the gate counts improvements", "[patience]") {
    check_gate_binds("egout.mps", "fpr", "mip_heuristic_fpr_effort");
}

// LocalMIP is the one heuristic still outside `kMaxGrowth`, and it is a
// seed-dependent spread rather than a single number.  Post-#111 it was
// 6.00x / 19.99x / 19.99x / 19.98x / 9.93x on p0548 over seeds 0-4 and
// 11.22 / 7.25 / 6.50 / 13.36 / 5.73 on gt2; post-#116 seed 0 of p0548 is
// 7.81x on 3,255,835 -> 25,417,115, i.e. **less** spend at both ends of
// the sweep than the 4.6M -> 27.8M it was, with a higher ratio — the same
// arithmetic as the fpr/p0548 row above.  Asserting on a quantity that
// swings 3x with the seed would be a flake, so LocalMIP's property is
// pinned as a unit test on the signal instead (below), and the remaining
// spread is the honest state of it: LocalMIP genuinely keeps improving
// the incumbent on some seeds, which is the case the gate is *supposed*
// to let run.
//
// The issue's own motivating evidence is `fiball`, a MIPLIB instance out
// of reach of this suite; read it carefully, because it is weaker than it
// looks.  It reports `found=1` at every budget level while effort scaled
// 20x, and `found` is not a count: `EffortLedger` sets it from
// `sink.accepted() > accepted_before` (see effort_ledger.cpp and
// `IncumbentSink::accepted`'s own comment), so it says only that *at
// least one* offer was accepted during that dispatch.  It is consistent
// with one solution and with fifty.  Establishing that a dispatch stopped
// producing needs the accepted objectives themselves, not this field.

// ── Why the threshold is absolute ──

TEST_CASE("patience gate: the threshold does not move with the budget", "[patience][unit]") {
    constexpr size_t kNnz = 1000;
    // A quarter of the base, i.e. the same 256 effort units per nonzero the
    // threshold has always been -- `0.25 * (1 << 10)`.
    constexpr double kPerBase = 0.25;
    constexpr size_t kPerNnz = 256;

    // The defining property, and the one `total >> 2` did not have: the
    // same instance yields the same threshold no matter how large an
    // allowance the heuristic was handed.
    const size_t small = patience_threshold(kNnz, kPerBase, 100'000'000);
    const size_t large = patience_threshold(kNnz, kPerBase, 100'000'000'000);
    REQUIRE(small == kNnz * kPerNnz);
    REQUIRE(large == small);

    // It does scale with the instance — that is what makes it usable as a
    // single constant across MIPLIB.
    REQUIRE(patience_threshold(2 * kNnz, kPerBase, 100'000'000) == 2 * small);

    // A threshold at or above the allowance fires exactly at exhaustion,
    // which is indistinguishable from no gate at all, so it is clamped to
    // a quarter of the allowance instead (#116) — see the dedicated case
    // below.
    REQUIRE(patience_threshold(kNnz, kPerBase, 1000) == 250);

    // Degenerate inputs: never zero, or the gate would trip before any
    // work happened.
    REQUIRE(patience_threshold(0, kPerBase, 1000) == 1);
    REQUIRE(patience_threshold(kNnz, kPerBase, 0) == kNnz * kPerNnz);
}

// ── Why the threshold is clamped strictly below the ceiling ──

TEST_CASE("patience gate: patience is clamped strictly below the ceiling", "[patience][unit]") {
    // A patience at or above the ceiling fires exactly at budget
    // exhaustion.  That is not a gate — it is the budget, reported under
    // another name, and nothing in the log tells the two apart.  #113
    // measured a p95 inter-improvement gap above the ceiling on three of
    // the four heuristics (FJ's by 4,400x), so this is the common case for
    // an honestly derived value, not a corner.
    constexpr size_t kNnz = 1000;
    constexpr size_t kBudget = 1'000'000;

    // Any multiplier large enough to reach the ceiling lands on the same
    // clamped value, strictly below the budget.
    for (const double per_base : {1.0, 10.0, 1e6}) {
        const size_t threshold = patience_threshold(kNnz, per_base, kBudget);
        INFO("per_base=" << per_base << " threshold=" << threshold);
        REQUIRE(threshold == kBudget / kPatienceCeilingDivisor);
        REQUIRE(threshold < kBudget);
    }

    // Below the ceiling the option is used as it stands: the clamp bounds
    // the parameter, it does not replace it.
    REQUIRE(patience_threshold(kNnz, 0.1, kBudget) == 102'400);

    // The shipped vector sits exactly on the clamp — every default is
    // `0.25 x` its effort option — so applying the clamp moved no shipped
    // behaviour.  Checked at FPR's pair; the other three have the same
    // ratio.
    constexpr double kFprEffort = 7.672;
    constexpr double kFprPatience = 1.918;
    const size_t ceiling = heuristic_effort_budget(kNnz, kFprEffort);
    REQUIRE(patience_threshold(kNnz, kFprPatience, ceiling) ==
            heuristic_effort_budget(kNnz, kFprPatience));

    // A ceiling smaller than the divisor still yields a threshold of at
    // least 1, for the same reason a zero-nnz model does: a gate that
    // trips before any work happens is worse than one that never trips.
    //
    // Which is also where "strictly below the ceiling" stops holding, and
    // deliberately so.  `stale()` is `> stale_budget` while `exhausted()`
    // is `>= total_budget`, so a threshold `S` fires at effort `S + 1` and
    // precedes exhaustion only while `S <= budget - 2`.  The floor pins
    // `S` at 1 here, so at `budget == 2` the gate coincides with
    // exhaustion and at `budget == 1` it never fires; from `budget == 3`
    // up it precedes exhaustion again.  The floor outranks strictness on
    // purpose — the alternative failure is a gate that retires every
    // worker before it does anything.  Unreachable in practice anyway:
    // `budget` is `effort x (nnz << 10)` (times the worker count for FJ),
    // so two effort units needs about a one-nonzero model at the bottom of
    // the option's range, and `budget == 0` never arrives here because
    // `make_budget` treats it as "this heuristic does not run".
    REQUIRE(patience_threshold(kNnz, 1.0, 2) == 1);
}

TEST_CASE("patience gate: a zero multiplier disables the gate outright", "[patience][unit]") {
    // `mip_heuristic_<name>_patience = 0` means **no gate at all** (#106),
    // not "give up immediately".  The patience axis cannot be searched
    // without a point where the gate provably never fires —
    // otherwise "how much does this gate cost?" has no zero to measure
    // against — and that point has to be reachable from the option.
    constexpr size_t kNnz = 1000;

    // Unbounded, and unbounded *before* the clamp.  A clamped value fires
    // strictly before exhaustion (#116) and an unclamped one at it; "no
    // gate" is neither, and has to stay reachable for a probe that needs a
    // run with nothing stopping the heuristic but the clock.
    REQUIRE(patience_threshold(kNnz, 0.0, 1000) == SIZE_MAX);
    REQUIRE(patience_threshold(kNnz, 0.0, 0) == SIZE_MAX);
    REQUIRE(patience_threshold(0, 0.0, 1000) == SIZE_MAX);

    // And a worker handed that threshold never retires on staleness.
    WorkerBudgetState worker;
    worker.total_budget = SIZE_MAX;
    worker.stale_budget = patience_threshold(kNnz, 0, 1000);
    for (int i = 0; i < 1000; ++i) {
        worker.charge_no_improvement(1'000'000);
    }
    REQUIRE_FALSE(worker.stale());
    REQUIRE_FALSE(worker.finished);
}

TEST_CASE("patience gate: the threshold saturates instead of wrapping", "[patience][unit]") {
    // Both factors are user-supplied — the multiplier is an option with a
    // very wide upper bound, and `nnz` is whatever model was loaded — so
    // the product overflows at the top of the range on a large
    // instance.  A wrapped product is the worst possible answer: it yields
    // a *small* threshold, so the gate fires almost immediately and the
    // heuristic silently does nothing, which reads as "this parameter
    // value is terrible" to whatever is searching the space.
    REQUIRE(saturating_mul(SIZE_MAX, 2) == SIZE_MAX);
    REQUIRE(saturating_mul(2, SIZE_MAX) == SIZE_MAX);
    REQUIRE(saturating_mul(SIZE_MAX, 0) == 0);
    REQUIRE(saturating_mul(3, 5) == 15);

    // Monotone, which is the property the search actually depends on: a
    // larger multiplier never produces a tighter gate.
    REQUIRE(patience_threshold(SIZE_MAX, 2, 0) == SIZE_MAX);
    REQUIRE(patience_threshold(SIZE_MAX / 2, 4, 0) == SIZE_MAX);
    // Still clamped to a quarter of a finite allowance.
    REQUIRE(patience_threshold(SIZE_MAX, 2, 1000) == 250);
}

// ── Why FPR's pause/resume cannot falsely trip its new gate ──

TEST_CASE("patience gate: staleness is invariant to how an attempt is sliced", "[patience][unit]") {
    // `FprWorker` pauses an in-flight DFS at the per-call budget gate and
    // resumes it on the next call.  Its patience gate must not read that
    // interruption as a stall — a worker that was interrupted has not
    // stopped producing, it was asked to yield.
    //
    // What makes that true is that the gate counts *effort* since the last
    // improvement, never attempts and never calls: an attempt spanning K
    // calls is charged exactly the sum of what those K calls spent, which
    // is what the same attempt run in one call would have charged.  This
    // pins that invariance on the shared bookkeeping both paths use.  An
    // implementation that counted "calls that returned without a solution"
    // instead would fail here, and would retire a paused worker after
    // `stale_budget` calls regardless of how little work they did.
    constexpr size_t kThreshold = 1000;
    constexpr size_t kAttemptEffort = 900;  // below the threshold on its own

    auto fresh = [] {
        WorkerBudgetState s;
        s.total_budget = SIZE_MAX;
        s.stale_budget = kThreshold;
        return s;
    };

    // One uninterrupted attempt.
    WorkerBudgetState whole = fresh();
    whole.charge_no_improvement(kAttemptEffort);

    // The same attempt, paused and resumed nine times.
    WorkerBudgetState sliced = fresh();
    for (int i = 0; i < 9; ++i) {
        sliced.charge_no_improvement(kAttemptEffort / 9);
    }

    REQUIRE(sliced.effort_since_improvement == whole.effort_since_improvement);
    REQUIRE(sliced.total_effort == whole.total_effort);
    REQUIRE_FALSE(sliced.stale());
    REQUIRE_FALSE(sliced.finished);

    // And an attempt that finally verdicts feasible clears the whole
    // accumulated slice, so the pauses that preceded the find cost the
    // worker nothing.
    sliced.charge_improvement(50);
    REQUIRE(sliced.effort_since_improvement == 0);
    REQUIRE_FALSE(sliced.finished);

    // The gate still fires on effort, though — being sliced buys no extra
    // allowance either.
    WorkerBudgetState over = fresh();
    for (int i = 0; i < 11; ++i) {
        over.charge_no_improvement(kThreshold / 10);
    }
    REQUIRE(over.stale());
    REQUIRE(over.finished);
}

// ── Why the gate reads an incumbent improvement and nothing weaker ──

TEST_CASE("patience gate: an offer the pool refuses does not reset staleness", "[patience][unit]") {
    // Read what this does and does not guard, because it is easy to
    // over-read.
    //
    // It is **not** a regression test for #111's signal fix: it drives
    // `SolutionPool` and `WorkerBudgetState` directly rather than any of
    // the four workers, so it passes against the pre-fix tree as happily
    // as against this one.  The behavioural guard is
    // `patience gate: FPR on flugpl needs a shared verdict, not its own`
    // above, which does fail pre-fix (19.96x against a 4x bound).
    //
    // The *wiring* — that no worker drops the verdict again — is guarded
    // by the compiler: `IncumbentSink::offer` is `[[nodiscard]]` and
    // `CMakeLists.txt` puts `-Werror=unused-result` on our two targets,
    // so a re-drop is a hard build failure rather than a warning that
    // scrolls past.  (Without that flag it was only a warning: nothing
    // here passes `-Werror`, and the clang-tidy gate cannot see
    // `clang-diagnostic-*` because `.clang-tidy` opens with `-*`.)
    //
    // What is left for a test is the *policy*: that an offer the pool
    // threw away is not productivity.  The pool is the predicate `offer`
    // wraps — `offer` is `pool_.try_add(...)` plus an accept counter — so
    // drive `WorkerBudgetState` from a real `SolutionPool` the way the
    // four workers now do, and again the way they used to, and watch the
    // two diverge.
    constexpr size_t kThreshold = 1000;
    constexpr size_t kEffortPerAttempt = 300;
    constexpr int kAttempts = 20;

    SolutionPool pool(/*capacity=*/2, /*minimize=*/true);
    REQUIRE(pool.try_add(1.0, {1.0, 0.0}, kSolutionSourceLocalMIP).accepted);
    REQUIRE(pool.try_add(2.0, {0.0, 1.0}, kSolutionSourceLocalMIP).accepted);

    // A full pool refuses a solution worse than everything in it.  This
    // is the rediscovery case: a rebuilt worker finds a feasible point it
    // has no memory of, calls it an improvement on its own (infinite)
    // baseline, and offers something the dispatch already knows about.
    REQUIRE_FALSE(pool.try_add(9.0, {1.0, 1.0}, kSolutionSourceLocalMIP).accepted);

    auto fresh = [] {
        WorkerBudgetState s;
        s.total_budget = SIZE_MAX;
        s.stale_budget = kThreshold;
        return s;
    };

    // Post-#111: the worker resets only on a verdict of true.
    WorkerBudgetState now = fresh();
    // Pre-#111: the worker reset on its own notion, which rediscovery
    // satisfied every time.
    WorkerBudgetState before = fresh();

    for (int i = 0; i < kAttempts; ++i) {
        // The full pool refuses every one of these, which is what makes
        // the two counters below diverge; a matching `charge_improvement`
        // arm would be dead code here, so the accept path is exercised by
        // `charge_improvement` in the sliced-attempt case above instead.
        REQUIRE_FALSE(pool.try_add(9.0, {1.0, 1.0}, kSolutionSourceLocalMIP).accepted);

        // Post-#111: no acceptance, no reset.
        now.charge_no_improvement(kEffortPerAttempt);

        // The shape the workers had: "I produced a feasible solution"
        // stood in for "the dispatch produced something".
        before.charge_improvement(kEffortPerAttempt);
    }

    // 20 x 300 = 6000 units of effort bought the dispatch nothing at all.
    REQUIRE(now.stale());
    REQUIRE(now.finished);
    REQUIRE(now.effort_since_improvement > kThreshold);

    // The old signal cleared the counter on every one of them, so the
    // gate could not fire however low the threshold was set — which is
    // why an absolute threshold alone left FPR at 19.98x on flugpl.
    REQUIRE_FALSE(before.stale());
    REQUIRE_FALSE(before.finished);
    REQUIRE(before.effort_since_improvement == 0);
}

TEST_CASE("patience gate: the pool separates acceptance from improvement", "[patience][unit]") {
    // The two facts `SolutionPool::try_add` reports, and the three ways
    // they can come apart.  This is the predicate `IncumbentSink::offer`
    // forwards as `OfferResult`, so it is where the #116 policy lives.
    SolutionPool pool(/*capacity=*/3, /*minimize=*/true);

    // First solution of the solve: nothing to improve on, so it improves
    // by definition.  The pool is seeded from the incumbent at sink
    // construction, so an empty pool really does mean "no feasible
    // solution known".
    const auto first = pool.try_add(10.0, {1.0}, kSolutionSourceFPR);
    REQUIRE(first.accepted);
    REQUIRE(first.improved_best);

    // (1) Filling.  The pool admits its first `capacity` offers
    // unconditionally, so these are accepted while the best objective
    // does not move.  This alone is enough to keep an acceptance-driven
    // gate from ever firing early in a dispatch.
    const auto filling = pool.try_add(20.0, {2.0}, kSolutionSourceFPR);
    REQUIRE(filling.accepted);
    REQUIRE_FALSE(filling.improved_best);

    // (2) Beating the worst.  With the pool full, a solution better than
    // the worst entry replaces it — accepted, and still no movement in
    // the best.  This is the top-K mechanism that earns FPR ~3.3 M
    // acceptances against 590 improvements.
    REQUIRE(pool.try_add(30.0, {3.0}, kSolutionSourceFPR).accepted);
    const auto beats_worst = pool.try_add(25.0, {4.0}, kSolutionSourceFPR);
    REQUIRE(beats_worst.accepted);
    REQUIRE_FALSE(beats_worst.improved_best);

    // (3) An actual improvement: both flags.
    const auto better = pool.try_add(5.0, {5.0}, kSolutionSourceFPR);
    REQUIRE(better.accepted);
    REQUIRE(better.improved_best);

    // Ties do not count.  The margin is relative with an absolute floor,
    // matching what `bench/analyze_presolve_probe.py` reconstructs the
    // calibration's improvement trajectory with, so re-offering the best
    // — or a value a rounding error away from it — is not production.
    const auto tie = pool.try_add(5.0, {6.0}, kSolutionSourceFPR);
    REQUIRE_FALSE(tie.improved_best);
    const auto epsilon = pool.try_add(5.0 - 1e-12, {7.0}, kSolutionSourceFPR);
    REQUIRE_FALSE(epsilon.improved_best);

    // And the same three cases for a maximization pool, since the
    // comparison is the one thing that flips with the sense.
    SolutionPool maxpool(/*capacity=*/2, /*minimize=*/false);
    REQUIRE(maxpool.try_add(10.0, {1.0}, kSolutionSourceFPR).improved_best);
    REQUIRE_FALSE(maxpool.try_add(5.0, {2.0}, kSolutionSourceFPR).improved_best);
    REQUIRE(maxpool.try_add(20.0, {3.0}, kSolutionSourceFPR).improved_best);
}

TEST_CASE("patience gate: an accepted non-improvement resets neither gate level",
          "[patience][unit]") {
    // The acceptance criterion of #116, at both levels of the gate.  A
    // pool acceptance that does not move the best objective must advance
    // the worker's `effort_since_improvement` *and* the runner's, or a
    // heuristic that keeps beating its own worst entry never runs out of
    // patience — which is what the shipped defaults were measured against
    // and could not previously be spent on.
    constexpr size_t kThreshold = 1000;
    constexpr size_t kEffortPerAttempt = 300;

    SolutionPool pool(/*capacity=*/3, /*minimize=*/true);
    REQUIRE(pool.try_add(10.0, {1.0}, kSolutionSourceLocalMIP).improved_best);

    WorkerBudgetState worker;
    worker.total_budget = SIZE_MAX;
    worker.stale_budget = kThreshold;

    ContinuousLoopState loop;

    // Four accepted offers, none of them an improvement: two while the
    // pool fills, then two that beat the worst entry.
    for (const double obj : {20.0, 30.0, 25.0, 22.0}) {
        const auto added = pool.try_add(obj, {obj}, kSolutionSourceLocalMIP);
        INFO("obj=" << obj);
        REQUIRE(added.accepted);
        REQUIRE_FALSE(added.improved_best);

        // Exactly what a worker and the runner do with that verdict.
        if (added.improved_best) {
            worker.charge_improvement(kEffortPerAttempt);
        } else {
            worker.charge_no_improvement(kEffortPerAttempt);
        }
        loop.note_staleness(kEffortPerAttempt, added.improved_best, kThreshold);
    }

    // 4 x 300 = 1200 units, past the threshold at both levels, despite
    // four acceptances.  Under #111's signal both counters would read
    // zero here and neither gate could ever fire.
    REQUIRE(worker.effort_since_improvement == 4 * kEffortPerAttempt);
    REQUIRE(worker.stale());
    REQUIRE(worker.finished);
    REQUIRE(loop.effort_since_improvement.load() == 4 * kEffortPerAttempt);
    REQUIRE(loop.stopped());

    // And a real improvement does still clear both, so the gate is a
    // patience and not a second budget.
    const auto improvement = pool.try_add(1.0, {1.0}, kSolutionSourceLocalMIP);
    REQUIRE(improvement.improved_best);
    worker.charge_improvement(kEffortPerAttempt);
    ContinuousLoopState fresh_loop;
    fresh_loop.note_staleness(kEffortPerAttempt, /*improved=*/false, kThreshold);
    fresh_loop.note_staleness(kEffortPerAttempt, improvement.improved_best, kThreshold);
    REQUIRE(worker.effort_since_improvement == 0);
    REQUIRE(fresh_loop.effort_since_improvement.load() == 0);
    REQUIRE_FALSE(fresh_loop.stopped());
}

TEST_CASE("patience gate: evicting the pool's best does not manufacture an improvement",
          "[patience][unit]") {
    // `improved_best` must mean "moved the best objective the solve knows",
    // and the solve's best objective never goes backwards: `addIncumbent`
    // keeps whatever was submitted.  The *pool's* front entry does go
    // backwards — the diversity path replaces the entry most similar to the
    // offer, and that entry can be the best one — so a "best before this
    // offer" read off `entries_.front()` degrades, and the next offer to
    // clear the degraded value looks like an improvement while HiGHS still
    // holds something better.  That is a free staleness reset, which is the
    // exact defect #116 exists to remove.
    constexpr int kNumIntVars = 20;
    SolutionPool pool(/*capacity=*/2, /*minimize=*/true);
    pool.set_integer_mask(std::vector<bool>(kNumIntVars, true));

    std::vector<double> best(kNumIntVars, 0.0);
    std::vector<double> other(kNumIntVars, 1.0);
    // Hamming 1 from `best` (5% of 20, exactly `kDiversityMinHammingFrac`)
    // and 19 from `other`, so `best` is the most similar entry and the one
    // the diversity path erases.
    std::vector<double> diverse(kNumIntVars, 0.0);
    diverse[0] = 1.0;

    REQUIRE(pool.try_add(100.0, best, kSolutionSourceFPR).improved_best);
    REQUIRE(pool.try_add(101.0, other, kSolutionSourceFPR).accepted);
    REQUIRE(pool.snapshot().best_objective == 100.0);

    // Dominated (ties the worst) but within `kDiversityObjTolerance` of the
    // best and diverse enough, so it is admitted on the diversity path.
    const auto evicting = pool.try_add(101.0, diverse, kSolutionSourceFPR);
    REQUIRE(evicting.accepted);
    REQUIRE_FALSE(evicting.improved_best);

    // HiGHS was told about the 100.0 solution when the pool accepted it, so
    // that is still the incumbent whatever the pool now holds.  An offer of
    // 100.5 is therefore not an improvement, and must not reset a gate.
    const auto not_an_improvement = pool.try_add(100.5, best, kSolutionSourceFPR);
    REQUIRE_FALSE(not_an_improvement.improved_best);
}
