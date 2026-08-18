#pragma once

#include <cstddef>
#include <cstdint>

class HighsMipSolver;

// The one place heuristic effort is booked and reported (issue #94).
//
// There used to be two accounting paths, and the split was deliberate but
// documented only in a code comment:
//
//   * presolve — `mode_dispatch.cpp` centrally incremented
//     `heuristic_effort_used` and emitted the `[Sequential]` line;
//   * dive — `fpr_lp.cpp` self-booked `heuristic_lp_iterations`,
//     `total_lp_iterations` *and* `heuristic_effort_used`, and emitted
//     nothing, so its work was invisible to `bench/parse_highs_log.py`.
//
// Both now funnel through one private `book()`, so the counter increment
// and the log emission each happen in exactly one place.  What remains
// different between them is not duplication to remove: `charge_dive` also
// depletes the RENS/RINS LP-iteration envelope, which is `fpr_lp`'s
// genuine extra obligation for competing with those heuristics for the
// same budget.
//
// State-mutation invariant: this class is the *only* thing in `src/`
// allowed to write upstream `HighsMipSolverData` counters, and only in
// `charge_dive`, where depleting the envelope is the deliberate contract.
// Neither `charge_presolve` nor `note_presolve_span` touches anything
// upstream — `heuristic_effort_used`, `presolve_heuristic_time` and
// `fpr_lp_lp_iterations` are all patch-added fields with no upstream
// reader.  A heuristic that is disabled must reach none of the three.
//
// Threading invariant: every counter update here is a plain non-atomic
// `+=`.  Both methods must be called from the dispatching thread with every
// parallel region already joined — `run_sequential` books between
// heuristics, and `fpr_lp::run` books after its worker loop returns (and
// only when `parallelLockActive()` is false).  Do not call either from a
// worker without making the counters atomic first.
class EffortLedger {
public:
    explicit EffortLedger(HighsMipSolver& mipsolver) : mipsolver_(mipsolver) {}

    // Elapsed solve seconds, for the `t0_s` / `t1_s` arguments below.
    //
    // Not monotonic: `HighsTimer` bottoms out in
    // `std::chrono::high_resolution_clock`, which libstdc++ aliases to
    // `system_clock`, so a wall-clock step can make `t1_s - t0_s`
    // negative.  Shared origin with the solver's own timestamps is worth
    // more than monotonicity here — a negative `wall_ms` is a visible
    // artefact in one sample, which the bench parser accepts and reports
    // rather than silently dropping.
    // Exposed here so every caller times against one clock — and it is
    // deliberately the *solver's* clock (`HighsMipSolver::timer_`) rather
    // than a raw `steady_clock`, so the `start_s` / `end_s` fields of the
    // `[Heur]` line share an origin with the `[Root] lp_time_s` timestamp
    // and with HiGHS's own display-line time column.  Comparing a
    // heuristic's window against when the root LP started is the whole
    // point of the cannibalization instrumentation (issue #95).
    [[nodiscard]] double now_s() const;

    // A presolve-chain heuristic (FJ / FPR / LocalMIP / Scylla) consumed
    // `effort` units between `t0_s` and `t1_s`.  `found` is whether the
    // shared `IncumbentSink` accepted at least one of its solutions.
    void charge_presolve(const char* name, size_t effort, bool found, double t0_s, double t1_s);

    // The whole presolve chain occupied the solver from `t0_s` to `t1_s`.
    //
    // Deliberately *not* the sum of the `charge_presolve` windows: those
    // are scoped to what `kWeight*` calibrates and exclude the shared
    // setup `run_sequential` hoisted out of all four heuristics
    // (`make_problem` / `build_csc` / `seed_pool`) — sub-millisecond on
    // the bundled test instances, but O(nnz) and attributed to nobody at
    // any size.  `[Root] presolve_heur_s`
    // asks "how much wall time did the chain cost the solver before the
    // root node", so it takes the full span.  Keeping the two quantities
    // separate is what lets the calibration basis stay untouched.
    // NOLINTNEXTLINE(readability-make-member-function-const): the ledger
    // holds `HighsMipSolver&`, so every method here technically leaves the
    // ledger object untouched — but this is the one place in src/ that
    // writes the solver's own counters.  `const` would advertise the
    // opposite and invite a `const EffortLedger&` caller to assume the
    // call is free of side effects.
    void note_presolve_span(double t0_s, double t1_s);

    // A B&B-dive heuristic (fpr_lp) did the same, and additionally owes
    // the shared heuristic LP-iteration envelope: its `setup_lp_iters`
    // reference-LP solves plus `effort` converted at `nnz` effort-units
    // per LP iteration are charged to `heuristic_lp_iterations` and
    // `total_lp_iterations`, mirroring how RENS/RINS flush their sub-MIP
    // LP iterations.  `nnz` must be non-zero.
    void charge_dive(const char* name, size_t effort, bool found, int64_t setup_lp_iters,
                     size_t nnz, double t0_s, double t1_s);

private:
    // NOLINTNEXTLINE(readability-make-member-function-const): the ledger
    // holds `HighsMipSolver&`, so every method here technically leaves the
    // ledger object untouched — but this is the one place in src/ that
    // writes the solver's own counters.  `const` would advertise the
    // opposite and invite a `const EffortLedger&` caller to assume the
    // call is free of side effects.
    void book(const char* name, const char* phase, size_t effort, bool found, double t0_s,
              double t1_s);

    HighsMipSolver& mipsolver_;
};
