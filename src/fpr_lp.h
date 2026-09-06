#pragma once
#include <cstddef>
#include <cstdint>
class HighsMipSolver;
namespace fpr_lp {
// Run LP-dependent FPR configs (paper Classes 2-3) using the root LP solution.
// Requires an optimal LP relaxation. Called during B&B dive (after RINS/RENS).
//
// Gating and budget are derived internally so fpr_lp participates in the
// same B&B heuristic budget as RENS/RINS (issue: pre-split it drew an
// unaccounted nnz-based budget per call):
//  - enabled iff heuristics::effective_flags(options).fpr_lp — i.e. only at
//    a mip_heuristic_suite value naming fpr_lp, so suite=off really
//    disables it (and so does every subset that omits fpr_lp, deliberately:
//    per-heuristic attribution has to cover the dive-time heuristic too).
//    Its own token since #164: it followed presolve FPR's bit before, so
//    "presolve FPR without fpr_lp" had no spelling at all;
//  - and iff mip_heuristic_fpr_lp_effort > 0, which is the same "0 means
//    the heuristic does not run" the four presolve effort options carry.
//    Both gates return from the same place, above every counter read;
//  - per-call effort budget = mip_heuristic_fpr_lp_effort times the lesser
//    of the remaining LP-iteration headroom of the moreHeuristicsAllowed()
//    envelope (total_lp_iterations * mip_heuristic_effort + 10000 -
//    heuristic_lp_iterations), converted at nnz effort-units per LP
//    iteration, and heuristic_effort_budget(nnz, mip_heuristic_effort).
//    The option is a *share* of that slice rather than an absolute
//    multiplier, because the quantity is zero-sum against RENS/RINS.  See
//    `dive_budget` below;
//  - all consumed work — the reference-LP solves in setup plus worker
//    effort / nnz — is charged back to heuristic_lp_iterations and
//    total_lp_iterations, mirroring how RENS/RINS book their sub-MIP LP
//    iterations, so the shared envelope depletes;
//  - skipped entirely while parallelLockActive() (multi-worker B&B
//    search under parallel=on): the counters above are shared and fpr_lp
//    has no worker-local flush infrastructure, so running there would
//    race.  Never fires on the default single-search-worker runs.
void run(HighsMipSolver& mipsolver);

// Test hook: counter incremented once per fpr_lp dispatch (one bump per
// `run_workers` call, not per worker).  fpr_lp is a
// single heuristic family, so it always runs arm-aligned parallel
// workers.  Process-global; reset before each test that inspects it.
struct DispatchCounts {
    size_t dispatches = 0;
};
DispatchCounts dispatch_counts();
void reset_dispatch_counts();

// Test hook: the outcome of one dispatch *setup* — the sequential work
// `run` does before any worker exists — without running the dispatch
// (issue #118).
//
// The setup is where a dive-time overrun comes from: ten
// `compute_var_order` calls and two reference LP solves, none of which any
// budget option touches, and until #118 none of which looked at the clock.
// It polls the solve's wall-clock deadline now, and this reports what the
// poll decided.  The three fields are the whole observable contract; the
// setup itself stays private to `fpr_lp.cpp`.
struct SetupProbe {
    // A complete setup came back and the dispatch would have run.
    bool built = false;
    // The wall-clock deadline stopped the setup.  False for the
    // model-shape / LP-status skips, which consume nothing — telling the
    // two apart is the point of this hook.
    bool deadline_bail = false;
    // Reference-LP iterations consumed before returning, which `run`
    // charges to the shared RENS/RINS envelope on either path.
    int64_t lp_iterations = 0;
};

// `max_effort` is the per-call effort budget `run` would have derived; it
// is recorded in the setup and does not steer any of the decisions above.
//
// **Test-only, and — unlike `dispatch_counts` above, which is one atomic
// load — this one does real work and charges nobody for it.**  It calls
// the actual `build_setup`, so on a solver whose LP relaxation *is*
// scaled-optimal it runs both reference LP solves and all ten
// `compute_var_order` calls and then destroys the result: LP iterations
// drawn and booked to nothing, which is precisely what `fpr_lp`'s
// charge-back exists to prevent, and `compute_var_order` mutates
// `HighsCliqueTable` state, so reaching it off the dispatching thread is a
// data race (#99).  Neither bites at today's call sites — the tests probe
// a bare solver, on one thread, whose LP relaxation stops the setup before
// either — but do not call this from a solve, and never from a worker.
SetupProbe probe_setup(HighsMipSolver& mipsolver, size_t max_effort);

// The per-call effort budget `run` derives, as a pure function of the four
// numbers it derives it from: the remaining envelope headroom in LP
// iterations, the model's nnz, upstream's mip_heuristic_effort, and this
// heuristic's own mip_heuristic_fpr_lp_effort share.
//
//   share * min(headroom_iters * nnz, vanilla_effort_budget(nnz, effort))
//
// saturating at SIZE_MAX, and 0 when the share, the headroom or nnz is
// non-positive.
//
// **The share multiplies the whole `min`.**  At `share == 1.0` that is
// `min(headroom, cap)` — what the call took before the option existed, so
// the shipped default is unmoved.  Above 1.0 it grows without bound, which
// is what lets a calibration hand fpr_lp a budget that cannot bind (the
// reason the option's ceiling is 1e6, as for the four presolve ones);
// below it, both terms throttle together.
//
// Declared here rather than left inline in `run` so a test can pin the
// arithmetic: no assertion on a whole solve separates "the share scaled the
// budget" from "the budget happened to land there", and the first version
// of #164 shipped with that gap — deleting the share from the expression
// left the entire suite green.  This is not a test hook; `run` is its
// production caller, and `tests/test_fpr_lp.cpp` asserts on both halves
// (this function's arithmetic, and that a share still moves what a real
// dispatch does) because a pure function nothing calls would pass either
// way.
size_t dive_budget(double headroom_iters, size_t nnz, double heuristic_effort, double share);
}  // namespace fpr_lp
