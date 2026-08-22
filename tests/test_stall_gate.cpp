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
// `kStallPerNnz<H> x nnz`.
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
//   fpr / p0548         20.00x         1.42x        1.42x
//   fpr / gt2           20.00x        20.00x        1.5-4.4x
//   fj / p0548          15.67x         2.00x        2.00x
//   scylla / flugpl     17.60x         1.00x        1.00x
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
// count (fpr/flugpl is 19.98 / 19.18 / 18.96 / 19.53 before the fix at
// N = 1 / 2 / 4 / 12, and 1.03 / 1.46 / 1.30 / 1.25 after), because each
// worker retires at `stale / N` of its own effort so all N cross at
// `stale` of aggregate effort and N cancels.  What an unpinned test would
// buy is a number that differs between a 12-core laptop and a 2-vCPU CI
// runner, on an issue labelled portable.  `ScopedThreadPin` is what makes
// the pin survive whatever initialised the global task executor first —
// see its comment.
size_t effort_at(const char* inst, const char* heur, const char* option, double effort,
                 int threads) {
    const auto lines = solve_capturing_log(inst, [&](Highs& h) {
        require_option(h, "log_dev_level", 3);
        require_option(h, "threads", threads);
        require_option(h, "random_seed", 0);
        require_option(h, option, effort);
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
constexpr double kLowEffort = 0.05;
constexpr double kHighEffort = 1.00;

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

// Two instances are deliberately absent from the ratio cases above, and
// both absences are findings rather than omissions.
//
// `fpr/egout` is not fixed: 19.98x before, 19.98x after.  FPR genuinely
// earns forty-odd pool acceptances there against only three incumbent
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
// 19.99x, 19.99x, 19.98x, 9.93x across five seeds.  gt2 lands at 7.25x.
// Both exceed `kMaxGrowth`, and asserting on a quantity that swings 3x
// with the seed would be a flake, so LocalMIP's property is pinned as a
// unit test on the signal instead (below).  The residual is pool-fill and
// diversity accepts, same mechanism as egout.
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
    constexpr size_t kPerNnz = 256;

    // The defining property, and the one `total >> 2` did not have: the
    // same instance yields the same threshold no matter how large an
    // allowance the heuristic was handed.
    const size_t small = stall_threshold(kNnz, kPerNnz, 100'000'000);
    const size_t large = stall_threshold(kNnz, kPerNnz, 100'000'000'000);
    REQUIRE(small == kNnz * kPerNnz);
    REQUIRE(large == small);

    // It does scale with the instance — that is what makes it usable as a
    // single constant across MIPLIB.
    REQUIRE(stall_threshold(2 * kNnz, kPerNnz, 100'000'000) == 2 * small);

    // A threshold above the allowance can never fire, so it reports the
    // allowance instead.
    REQUIRE(stall_threshold(kNnz, kPerNnz, 1000) == 1000);

    // Degenerate inputs: never zero, or the gate would trip before any
    // work happened.
    REQUIRE(stall_threshold(0, kPerNnz, 1000) == 1);
    REQUIRE(stall_threshold(kNnz, kPerNnz, 0) == kNnz * kPerNnz);
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
    // The policy half of the improvement-signal fix.  The *wiring* half
    // is guarded at compile time — `IncumbentSink::offer` is
    // `[[nodiscard]]`, so a worker that drops the verdict again will not
    // build — but nothing stops a future edit from reading the verdict
    // and then resetting anyway.  This pins what the verdict is for.
    //
    // The pool is the predicate `offer` wraps: `offer` is
    // `pool_.try_add(...)` plus an accept counter.  So drive
    // `WorkerBudgetState` from a real `SolutionPool` exactly as the four
    // workers now do, and again as they used to, and watch the two
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
        const bool accepted = pool.try_add(9.0, {1.0, 1.0}, kSolutionSourceLocalMIP);
        REQUIRE_FALSE(accepted);

        if (accepted) {
            now.charge_improvement(kEffortPerAttempt);
        } else {
            now.charge_no_improvement(kEffortPerAttempt);
        }

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
