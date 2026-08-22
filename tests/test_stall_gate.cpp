#include "heuristic_common.h"
#include "Highs.h"
#include "test_common.h"
#include "worker_base.h"

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

// ===================================================================
// Absolute stall thresholds (#111)
//
// Three of the four presolve heuristics used to derive their stall
// threshold as `total_budget >> 2`.  A quarter of the budget cannot bound
// over-budgeting: doubling the budget doubles the tolerance, so the gate
// never fires relatively sooner and charged effort tracks the effort
// option one-for-one.  Measured on the pre-#111 binary across a 20x
// sweep of each heuristic's own option, at 12 workers:
//
//   fj / p0548        9.50M ->  144.0M   15.3x
//   fpr / p0548       4.91M ->  100.3M   20.4x
//   fpr / dcmulti     1.83M ->   28.5M   16.2x
//   scylla / flugpl   0.05M ->    0.75M  13.7x
//
// The cases below are the acceptance signal for that issue: with the
// thresholds absolute, the same sweep must not buy anything like 20x the
// effort, because the heuristic exits on staleness before it exhausts an
// allowance it was never going to use.  At the `threads=1` pin these
// cases run under, the pre-#111 numbers are fpr/p0548 28.3x, fj/p0548
// 16.0x, scylla/flugpl 17.6x; after, 1.58x / 2.00x / 1.00x.
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
// `threads` and `random_seed` are both pinned.  Unpinned, the worker
// count is the *run machine's* core count, and effort depends on it: the
// runner-level gate is aggregated over the pool, and the overshoot past
// it is one per-worker call from each of N workers.  A ratio measured on
// a 12-core laptop is then not the ratio a 2-vCPU CI runner measures, on
// an issue labelled portable.  One worker also makes the comparison a
// clean before/after of the same search rather than of two different
// schedules.  `ScopedThreadPin` is what makes the pin survive whatever
// initialised the global task executor first — see its comment.
size_t effort_at(const char* inst, const char* heur, const char* option, double effort) {
    const auto lines = solve_capturing_log(inst, [&](Highs& h) {
        require_option(h, "log_dev_level", 3);
        require_option(h, "threads", 1);
        require_option(h, "random_seed", 0);
        require_option(h, option, effort);
        set_suite(h, heur);
    });
    return presolve_effort(lines, heur);
}

// The sweep is 20x wide (0.05 -> 1.00, the option's whole documented
// range).  A gate that binds holds the growth to a small constant; the
// pre-#111 numbers in the header comment are 16x-28x at this thread pin.
// 4x is deliberately loose — the worst post-fix ratio measured over four
// seeds at `threads=1` was 2.00x (fj/p0548, which sits on its structural
// ceiling of one gate plus one call of overshoot), and the bound only has
// to separate "bounded by an absolute threshold" from "bounded by the
// budget".
constexpr double kMaxGrowth = 4.0;
constexpr double kLowEffort = 0.05;
constexpr double kHighEffort = 1.00;

void check_gate_binds(const char* inst, const char* heur, const char* option) {
    // Held across both solves so they share one pinned worker count.
    const ScopedThreadPin pin;
    INFO("instance=" << inst << " heuristic=" << heur);
    const size_t low = effort_at(inst, heur, option, kLowEffort);
    const size_t high = effort_at(inst, heur, option, kHighEffort);
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

// LocalMIP is deliberately absent from the list above, and that is a
// finding rather than an omission.  On every instance bundled with HiGHS
// it keeps *improving* — 7 to 43 accepted incumbents on p0548, dcmulti,
// egout, gt2, rgn — so its staleness counter keeps resetting and the gate
// correctly declines to fire; effort still tracks the option, for the
// right reason.  `flugpl` is the one bundled instance where it stalls
// (63x before, 1.7x after) but only on some seeds, which is too thin to
// assert on.  The issue's own evidence is `fiball`, a MIPLIB instance out
// of reach of this suite; read that evidence carefully, because it is
// weaker than it looks.  It reports `found=1` at every budget level while
// effort scaled 20x, and `found` is not a count: `EffortLedger` sets it
// from `sink.accepted() > accepted_before` (see effort_ledger.cpp and
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
