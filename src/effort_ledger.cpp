#include "effort_ledger.h"

#include "io/HighsIO.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

#include <cassert>
#include <chrono>

double EffortLedger::now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void EffortLedger::charge_presolve(const char *name, size_t effort, double t0_s, double t1_s) {
    book(name, effort, t0_s, t1_s);
}

void EffortLedger::charge_dive(const char *name, size_t effort, int64_t setup_lp_iters, size_t nnz,
                               double t0_s, double t1_s) {
    assert(nnz > 0);
    auto *mipdata = mipsolver_.mipdata_.get();
    // Reference-LP iterations directly, worker effort converted at nnz
    // units per LP iteration.  This is what makes the dive heuristic
    // compete with RENS/RINS for the vanilla `mip_heuristic_effort`
    // envelope instead of drawing unaccounted work.
    const int64_t charged = setup_lp_iters + static_cast<int64_t>(effort / nnz);
    mipdata->heuristic_lp_iterations += charged;
    mipdata->total_lp_iterations += charged;
    book(name, effort, t0_s, t1_s);
}

void EffortLedger::book(const char *name, size_t effort, double t0_s, double t1_s) {
    mipsolver_.mipdata_->heuristic_effort_used += effort;

    // Emit `[Sequential] heur=<name> effort=<N> wall_ms=<X.X> effort_per_ms=<R>`.
    // Parsed by `bench/parse_highs_log.py` and used by
    // `bench/check_effort_drift.py` to calibrate the kWeight* constants in
    // `mode_dispatch.cpp`.  Zero-effort observations are emitted too
    // (local_mip often skips with non-zero setup wall_ms when the incumbent
    // is empty; a deadline can fire before setup).  `check_effort_drift.py`
    // filters `effort_per_ms <= 0` before aggregation, so these lines inform
    // a human reader without poisoning the geomean.  The `%.3f` format
    // preserves precision for slow heuristics whose rate would otherwise
    // round to 0.
    const double wall_ms = (t1_s - t0_s) * 1000.0;
    const double effort_per_ms =
        (effort > 0 && wall_ms > 0.0) ? static_cast<double>(effort) / wall_ms : 0.0;
    highsLogDev(mipsolver_.options_mip_->log_options, HighsLogType::kVerbose,
                "[Sequential] heur=%s effort=%zu wall_ms=%.1f effort_per_ms=%.3f\n", name, effort,
                wall_ms, effort_per_ms);
}
