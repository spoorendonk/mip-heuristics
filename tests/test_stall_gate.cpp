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
// Absolute stall thresholds and the improvement signal they read (#111)
//
// A stall gate has two operands and both were broken.
//
// The *threshold* was `total_budget >> 2` in three of the four presolve
// heuristics.  A quarter of the budget cannot bound over-budgeting:
// doubling the budget doubles the tolerance, so the gate never fires
// relatively sooner.  It is now an absolute, instance-scaled
// `mip_heuristic_<name>_stall x nnz` — a `constexpr` per heuristic at
// first, an option since #106, where `0` means no gate at all.
//
// The *signal* was each worker's own "I beat my own best", which restarts
// at infinity on every rebuild, so the pool's verdict was computed,
// discarded, and replaced by one that a rebuilt worker could satisfy by
// rediscovering a solution the pool already held.  Workers now read what
// `IncumbentSink::offer` returns, which is the project's single
// definition of production and the same predicate `[Heur] found` reports.
//
// Fixing either alone leaves the gate toothless on some instances, which
// is why both are here.  Ratios below are charged effort at effort option
// 1.00 over charged effort at 0.05 — a 20x sweep — at `threads=1`:
//
//                     pre-#111   threshold only   + signal
//   fpr / flugpl        19.98x        19.98x        1.03x
//   fpr / p0548         20.00x         0.89x        0.89x
//   fpr / gt2           20.00x        20.00x        1.5-4.4x
//   fj / p0548          15.67x         2.00x        2.00x
//   scylla / flugpl     17.60x         1.00x        1.00x
//
// Read those as this file's own harness reports them: one seed (0), one
// worker, `[Heur] ... phase=presolve effort=` summed over the solve.
// fpr/p0548's 0.89x is not a typo and not drift — both binaries report
// low = 4,231,951 and high = 3,770,052 — the gate simply lands below the
// low end's budget-clamped spend.  Medians over several seeds differ
// (that row is 1.42x over seeds 0-2), so do not mix the two.
//
// The gate ends up bounded everywhere but tight only in places, and this
// file says where it is not — see the notes on `fpr/egout` and on
// LocalMIP below.  Calibrating the constants is #106's job.
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
// `stall` is that heuristic's `mip_heuristic_<name>_stall` option, or a
// negative value to leave it at its shipped default.  Since #106 the
// threshold is an option rather than a constant, which is what lets the
// gate be switched off from a test instead of from a rebuild.
size_t effort_at(const char* inst, const char* heur, const char* option, double effort, int threads,
                 double stall = -1.0) {
    const auto lines = solve_capturing_log(inst, [&](Highs& h) {
        require_option(h, "log_dev_level", 3);
        require_option(h, "threads", threads);
        require_option(h, "random_seed", 0);
        require_option(h, option, effort);
        if (stall >= 0.0) {
            require_option(h, std::string("mip_heuristic_") + heur + "_stall", stall);
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

TEST_CASE("stall gate: FPR exits on staleness rather than spending 20x", "[stall]") {
    check_gate_binds("p0548.mps", "fpr", "mip_heuristic_fpr_effort");
}

TEST_CASE("stall gate: FeasibilityJump exits on staleness rather than spending 20x", "[stall]") {
    check_gate_binds("p0548.mps", "fj", "mip_heuristic_fj_effort");
}

TEST_CASE("stall gate: Scylla exits on staleness rather than spending 20x", "[stall]") {
    check_gate_binds("flugpl.mps", "scylla", "mip_heuristic_scylla_effort");
}

// The case that needs *both* halves of the fix, and the reason the
// improvement signal came into scope.  With the threshold absolute but
// the signal still worker-local, flugpl spent 2,785,359 effort against a
// 69,632 ceiling — forty ceilings' worth — while the pool accepted
// exactly one solution all dispatch.  Every one of those resets came
// from a feasible point FPR had already found and the pool refused.
TEST_CASE("stall gate: FPR on flugpl needs the pool's verdict, not its own", "[stall]") {
    check_gate_binds("flugpl.mps", "fpr", "mip_heuristic_fpr_effort");
}

// One multi-worker case.  The ratio is flat in N (see `effort_at`), so
// this is not extra coverage of a different regime so much as a guard
// against a future change that makes the gate depend on the pool size —
// the runner-level gate is aggregated over workers and the per-worker
// share is `stale / N`, which is exactly the kind of relation a refactor
// drops.  Four is safe on any host: if the executor caps the pool lower,
// the flatness means the assertion still holds.
TEST_CASE("stall gate: FPR on flugpl binds at four workers too", "[stall]") {
    check_gate_binds("flugpl.mps", "fpr", "mip_heuristic_fpr_effort", /*threads=*/4);
}

// ── The option, end to end: 0 really does remove the gate ──

TEST_CASE("stall gate: mip_heuristic_fpr_stall=0 restores budget-tracking", "[stall]") {
    // The unit assertions below pin `stall_threshold(nnz, 0, budget) ==
    // SIZE_MAX`; this pins that the option reaches it, through the record
    // registration, `kChain`, `run_sequential` and `make_budget`.  #113's
    // probe needs a run where the gate provably never fires, and a
    // semantic that is only true in a header is no use to it.
    //
    // flugpl/FPR is the case where the gate is measurably doing the work:
    // 19.98x charged-effort growth over a 20x budget sweep before #111,
    // 1.03x after.  Switching the gate off at the top of that sweep has to
    // put the spend back where the budget puts it — so the ratio here is
    // between two solves at the *same* effort, differing only in the
    // stall option, and the bound is the same 4x that separates "bounded
    // by an absolute threshold" from "bounded by the budget".
    const ScopedThreadPin pin;
    const size_t gated = effort_at("flugpl.mps", "fpr", "mip_heuristic_fpr_effort", kHighEffort, 1);
    const size_t ungated =
        effort_at("flugpl.mps", "fpr", "mip_heuristic_fpr_effort", kHighEffort, 1, /*stall=*/0);
    REQUIRE(gated > 0);
    INFO("gated=" << gated << " ungated=" << ungated);
    REQUIRE(static_cast<double>(ungated) > kMaxGrowth * static_cast<double>(gated));
}

// Two instances are deliberately absent from the ratio cases above, and
// both absences are findings rather than omissions.
//
// `fpr/egout` is not fixed: 19.98x before, 19.98x after.  FPR genuinely
// earns forty-odd pool acceptances there against only four incumbent
// improvements, because `SolutionPool` keeps a top-`kPoolCapacity` and
// FPR keeps beating its worst entry.  Those acceptances are the project's
// stated notion of production — the pool admits the first
// `kPoolCapacity` offers unconditionally while it fills, and admits
// structurally diverse near-best solutions afterwards — so closing this
// would mean redefining improvement as "accepted *and* beat the pool's
// best", which is the change #111 rules out.
//
// LocalMIP is bounded but not tight, and unevenly: on p0548 at
// `threads=1` the fix takes effort at option 1.00 from 92,736,483 to
// 27,824,095 for the identical fifteen incumbents at seed 0, but leaves
// it unchanged at seeds 1-3 and halves it at seed 4 — a ratio of 6.00x,
// 19.99x, 19.99x, 19.98x, 9.93x across five seeds.  gt2 is 11.22, 7.25,
// 6.50, 13.36, 5.73 over the same five.  Both exceed `kMaxGrowth`, and
// asserting on a quantity that swings 3x with the seed would be a flake,
// so LocalMIP's property is pinned as a unit test on the signal instead
// (below).  Neither spread is a bimodal flip — each solve is
// bit-reproducible at `threads=1`, and p0548 over seeds 0-9 gives a
// continuous 20.7M-92.7M with three seeds at the ceiling.  LocalMIP is
// legitimately earning acceptances on the seeds where nothing improves.
// The residual is pool-fill and diversity accepts, same mechanism as
// egout.
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

TEST_CASE("stall gate: the threshold does not move with the budget", "[stall][unit]") {
    constexpr size_t kNnz = 1000;
    // A quarter of the base, i.e. the same 256 effort units per nonzero the
    // threshold has always been -- `0.25 * (1 << 10)`.
    constexpr double kPerBase = 0.25;
    constexpr size_t kPerNnz = 256;

    // The defining property, and the one `total >> 2` did not have: the
    // same instance yields the same threshold no matter how large an
    // allowance the heuristic was handed.
    const size_t small = stall_threshold(kNnz, kPerBase, 100'000'000);
    const size_t large = stall_threshold(kNnz, kPerBase, 100'000'000'000);
    REQUIRE(small == kNnz * kPerNnz);
    REQUIRE(large == small);

    // It does scale with the instance — that is what makes it usable as a
    // single constant across MIPLIB.
    REQUIRE(stall_threshold(2 * kNnz, kPerBase, 100'000'000) == 2 * small);

    // A threshold above the allowance can never fire, so it reports the
    // allowance instead.
    REQUIRE(stall_threshold(kNnz, kPerBase, 1000) == 1000);

    // Degenerate inputs: never zero, or the gate would trip before any
    // work happened.
    REQUIRE(stall_threshold(0, kPerBase, 1000) == 1);
    REQUIRE(stall_threshold(kNnz, kPerBase, 0) == kNnz * kPerNnz);
}

TEST_CASE("stall gate: a zero multiplier disables the gate outright", "[stall][unit]") {
    // `mip_heuristic_<name>_stall = 0` means **no staleness gate at all**
    // (#106), not "give up immediately".  The stall axis cannot be
    // searched without a point where the gate provably never fires —
    // otherwise "how much does this gate cost?" has no zero to measure
    // against — and that point has to be reachable from the option.
    constexpr size_t kNnz = 1000;

    // Unbounded, and unbounded *before* the clamp.  Clamping to the budget
    // would make the gate fire exactly at budget exhaustion, which looks
    // the same on most runs and is not the same thing.
    REQUIRE(stall_threshold(kNnz, 0.0, 1000) == SIZE_MAX);
    REQUIRE(stall_threshold(kNnz, 0.0, 0) == SIZE_MAX);
    REQUIRE(stall_threshold(0, 0.0, 1000) == SIZE_MAX);

    // And a worker handed that threshold never retires on staleness.
    WorkerBudgetState worker;
    worker.total_budget = SIZE_MAX;
    worker.stale_budget = stall_threshold(kNnz, 0, 1000);
    for (int i = 0; i < 1000; ++i) {
        worker.charge_no_improvement(1'000'000);
    }
    REQUIRE_FALSE(worker.stale());
    REQUIRE_FALSE(worker.finished);
}

TEST_CASE("stall gate: the threshold saturates instead of wrapping", "[stall][unit]") {
    // Both factors are now user-supplied — the multiplier is an option
    // with an upper bound of `kHighsIInf`, and `nnz` is whatever model was
    // loaded — so the product overflows at the top of the range on a large
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
    REQUIRE(stall_threshold(SIZE_MAX, 2, 0) == SIZE_MAX);
    REQUIRE(stall_threshold(SIZE_MAX / 2, 4, 0) == SIZE_MAX);
    // Still clamped to a finite allowance.
    REQUIRE(stall_threshold(SIZE_MAX, 2, 1000) == 1000);
}

// ── Why FPR's pause/resume cannot falsely trip its new gate ──

TEST_CASE("stall gate: staleness is invariant to how an attempt is sliced", "[stall][unit]") {
    // `FprWorker` pauses an in-flight DFS at the per-call budget gate and
    // resumes it on the next call.  Its stall gate must not read that
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

// ── Why the gate reads the pool's verdict and not the worker's ──

TEST_CASE("stall gate: a refused offer does not reset staleness", "[stall][unit]") {
    // Read what this does and does not guard, because it is easy to
    // over-read.
    //
    // It is **not** a regression test for #111's signal fix: it drives
    // `SolutionPool` and `WorkerBudgetState` directly rather than any of
    // the four workers, so it passes against the pre-fix tree as happily
    // as against this one.  The behavioural guard is
    // `stall gate: FPR on flugpl needs the pool's verdict, not its own`
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
    // What is left for a test is the *policy*: that a refused offer is
    // not productivity.  The pool is the predicate `offer` wraps — `offer`
    // is `pool_.try_add(...)` plus an accept counter — so drive
    // `WorkerBudgetState` from a real `SolutionPool` the way the four
    // workers now do, and again the way they used to, and watch the two
    // diverge.
    constexpr size_t kThreshold = 1000;
    constexpr size_t kEffortPerAttempt = 300;
    constexpr int kAttempts = 20;

    SolutionPool pool(/*capacity=*/2, /*minimize=*/true);
    REQUIRE(pool.try_add(1.0, {1.0, 0.0}, kSolutionSourceLocalMIP));
    REQUIRE(pool.try_add(2.0, {0.0, 1.0}, kSolutionSourceLocalMIP));

    // A full pool refuses a solution worse than everything in it.  This
    // is the rediscovery case: a rebuilt worker finds a feasible point it
    // has no memory of, calls it an improvement on its own (infinite)
    // baseline, and offers something the dispatch already knows about.
    REQUIRE_FALSE(pool.try_add(9.0, {1.0, 1.0}, kSolutionSourceLocalMIP));

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
        REQUIRE_FALSE(pool.try_add(9.0, {1.0, 1.0}, kSolutionSourceLocalMIP));

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
