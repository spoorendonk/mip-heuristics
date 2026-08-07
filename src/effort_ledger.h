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
// `charge_presolve` touches nothing upstream — `heuristic_effort_used` is
// a patch-added field with no upstream reader.  A heuristic that is
// disabled must reach neither method.
class EffortLedger {
public:
    explicit EffortLedger(HighsMipSolver &mipsolver) : mipsolver_(mipsolver) {}

    // Seconds on a monotonic clock, for the `t0_s` / `t1_s` arguments
    // below.  Exposed here so every caller times against one clock.
    static double now_s();

    // A presolve-chain heuristic (FJ / FPR / LocalMIP / Scylla) consumed
    // `effort` units between `t0_s` and `t1_s`.
    void charge_presolve(const char *name, size_t effort, double t0_s, double t1_s);

    // A B&B-dive heuristic (fpr_lp) did the same, and additionally owes
    // the shared heuristic LP-iteration envelope: its `setup_lp_iters`
    // reference-LP solves plus `effort` converted at `nnz` effort-units
    // per LP iteration are charged to `heuristic_lp_iterations` and
    // `total_lp_iterations`, mirroring how RENS/RINS flush their sub-MIP
    // LP iterations.  `nnz` must be non-zero.
    void charge_dive(const char *name, size_t effort, int64_t setup_lp_iters, size_t nnz,
                     double t0_s, double t1_s);

private:
    void book(const char *name, size_t effort, double t0_s, double t1_s);

    HighsMipSolver &mipsolver_;
};
