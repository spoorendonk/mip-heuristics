#include "effort_ledger.h"

#include "io/HighsIO.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

#include <cassert>

double EffortLedger::now_s() const {
    return mipsolver_.timer_.read();
}

void EffortLedger::charge_presolve(const char* name, size_t effort, bool found, double t0_s,
                                   double t1_s) {
    book(name, "presolve", effort, found, t0_s, t1_s);
}

void EffortLedger::charge_dive(const char* name, size_t effort, bool found, int64_t setup_lp_iters,
                               size_t nnz, double t0_s, double t1_s) {
    assert(nnz > 0);
    auto* mipdata = mipsolver_.mipdata_.get();
    // Reference-LP iterations directly, worker effort converted at nnz
    // units per LP iteration.  This is what makes the dive heuristic
    // compete with RENS/RINS for the vanilla `mip_heuristic_effort`
    // envelope instead of drawing unaccounted work.
    // `assert` alone is not enough: the project builds Release, where
    // NDEBUG removes it and this would be a SIGFPE rather than a wrong
    // number.  The one current caller guards `nnz == 0` far upstream.
    const int64_t charged = setup_lp_iters + static_cast<int64_t>(nnz == 0 ? 0 : effort / nnz);
    mipdata->heuristic_lp_iterations += charged;
    mipdata->total_lp_iterations += charged;
    book(name, "dive", effort, found, t0_s, t1_s);
}

// NOLINTNEXTLINE(readability-make-member-function-const): see the declaration in effort_ledger.h.
void EffortLedger::book(const char* name, const char* phase, size_t effort, bool found, double t0_s,
                        double t1_s) {
    mipsolver_.mipdata_->heuristic_effort_used += effort;

    // Two lines per observation, deliberately:
    //
    //   * `[Sequential] heur=<name> effort=<N> wall_ms=<X.X> effort_per_ms=<R>`
    //     is the legacy calibration line, kept byte-identical because
    //     external tooling parses it.
    //   * `[Heur] ... phase=<presolve|dive> start_s=<S> end_s=<E> ... found=<0|1>`
    //     carries what `[Sequential]` cannot: *when* in the solve the
    //     heuristic ran, on the solver's own clock; which side of the
    //     patch boundary it ran on; and whether it produced anything.
    //     This is the line the per-heuristic budget calibration reads.
    //
    // Zero-effort observations are emitted too (local_mip often skips with
    // non-zero setup wall_ms when the incumbent is empty; a deadline can
    // fire before setup).  A rate of 0 is a real observation rather than a
    // parse failure, so anything aggregating these lines should filter
    // `effort_per_ms <= 0` rather than average it in.  The `%.3f` format
    // preserves precision for slow heuristics whose rate would otherwise
    // round to 0.
    const double wall_ms = (t1_s - t0_s) * 1000.0;
    const double effort_per_ms =
        (effort > 0 && wall_ms > 0.0) ? static_cast<double>(effort) / wall_ms : 0.0;
    const HighsLogOptions& log_options = mipsolver_.options_mip_->log_options;
    highsLogDev(log_options, HighsLogType::kVerbose,
                "[Sequential] heur=%s effort=%zu wall_ms=%.1f effort_per_ms=%.3f\n", name, effort,
                wall_ms, effort_per_ms);
    highsLogDev(log_options, HighsLogType::kVerbose,
                "[Heur] name=%s phase=%s start_s=%.3f end_s=%.3f effort=%zu wall_ms=%.1f "
                "effort_per_ms=%.3f found=%d\n",
                name, phase, t0_s, t1_s, effort, wall_ms, effort_per_ms, found ? 1 : 0);
}
