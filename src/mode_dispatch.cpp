#include "mode_dispatch.h"

#include "fj.h"
#include "fpr.h"
#include "io/HighsIO.h"
#include "local_mip.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "portfolio.h"
#include "scylla.h"
#include "solution_pool.h"

#include <chrono>

namespace heuristics {

namespace {

// Emit `[Sequential] heur=<name> effort=<N> wall_ms=<X.X> effort_per_ms=<R>`.
// Parsed by `bench/parse_highs_log.py` and used by
// `bench/check_effort_drift.py` to calibrate kWeight* below.  Zero-effort
// observations are emitted too (local_mip often skips with non-zero
// setup wall_ms when the incumbent is empty; a deadline can fire before
// setup).  `check_effort_drift.py` filters `effort_per_ms <= 0` before
// aggregation, so these lines inform a human reader without poisoning
// the geomean.  The `%.3f` format preserves precision for slow
// heuristics whose rate would otherwise round to 0.
void log_sequential(const HighsLogOptions &log_options, const char *name, size_t effort,
                    double wall_ms) {
    double effort_per_ms =
        (effort > 0 && wall_ms > 0.0) ? static_cast<double>(effort) / wall_ms : 0.0;
    highsLogDev(log_options, HighsLogType::kVerbose,
                "[Sequential] heur=%s effort=%zu wall_ms=%.1f effort_per_ms=%.3f\n", name, effort,
                wall_ms, effort_per_ms);
}

// Weighted effort allocation: each heuristic runs in turn with its
// proportional share of the budget and the full thread pool.
//
// The `opportunistic` flag is forwarded to all heuristics so each
// picks its deterministic vs continuous parallelism strategy.  Scylla
// uses N independent pump chains sharing a mutex-guarded PDLP solver
// (see `ContestedPdlp`), so its det/opp distinction is the same epoch
// vs opportunistic runner split used by FJ/FPR/LocalMIP.
//
// A single `SolutionPool` is constructed here and threaded through all
// heuristics so that solutions found by an earlier heuristic (e.g. FJ)
// become available as pool-restart seeds for later heuristics (FPR,
// LocalMIP).  The pool is seeded once from the incumbent and flushed
// to HiGHS once at the end; each entry carries its originating
// heuristic's source tag (see solution_pool.h / #73).  The portfolio
// modes already manage their own pool inside `portfolio::run_presolve`
// and are not affected.
bool run_sequential(HighsMipSolver &mipsolver, size_t budget, bool opportunistic) {
    const auto *options = mipsolver.options_mip_;

    // Weights tune each heuristic's share of the common effort budget so
    // that equal weights would yield equal wall-clock spend.  The effort
    // counter each heuristic decrements is in a different unit (FJ
    // step-units; FPR/LocalMIP coefficient accesses; Scylla PDLP iters ×
    // nnz), so a naive equal-weight split would cause wildly asymmetric
    // wall-clock spend across heuristics and instances (issue #71).
    //
    // Semantics: the weight is proportional to each heuristic's rate
    // `effort_per_ms`.  With `share_i = budget * w_i / sum(w)` and rate
    // `r_i`, wall-ms is `share_i / r_i`; setting `w_i ∝ r_i` makes the
    // ratio constant across heuristics.  Fast-per-effort heuristics
    // (high effort_per_ms) therefore get a larger share.
    //
    // Calibration procedure (`bench/check_effort_drift.py` automates 3–5):
    //   1. Build with this file's `[Sequential]` logging enabled.
    //   2. Run seq/det (`mip_heuristic_portfolio=false`,
    //      `mip_heuristic_opportunistic=false`) on MIPLIB with all four
    //      heuristics on (see `bench/run_benchmark.py`).
    //   3. `python bench/check_effort_drift.py bench/results/calibration`.
    //   4. Copy each heuristic's suggested weight into the constants
    //      below.  Normalise so the lowest weight rounds to a tidy value
    //      (0.5 or 1.0) — absolute scale does not matter, only ratios.
    //   5. Re-run to confirm the new geomean rates are stable across
    //      seeds.  Note: cross-heuristic drift (max/min effort_per_ms)
    //      is a structural property of the heuristics — recalibrating
    //      kWeight* does not reduce it.  As of round-5 the drift sits
    //      at ~4.7× because LocalMIP's coefficient-access counter and
    //      Scylla's PDLP-iters × nnz counter measure work in genuinely
    //      different units.  The script's default `--max-drift=3.0`
    //      currently fails on this codebase by design; it is consumed
    //      as a one-shot calibration helper, not a CI gate.
    //
    // Recalibrated against `bench/instances_small.txt` (25 MIPLIB
    // instances, 30 s each, mip_heuristic_effort default=0.30,
    // seq/det, mip_root_presolve_only=true, multi-thread default
    // threads=16).  Measured geomean `effort_per_ms` after issue #78
    // (cold-start construction sweep rolled into local_mip's reported
    // effort):
    //   fj=403k  fpr=636k  local_mip=1222k  scylla=261k   drift = 4.68×
    // Weights are proportional to geomean `effort_per_ms` (scylla
    // normalised to 1.0 as the slowest-per-effort heuristic).
    // Re-run `bench/check_effort_drift.py bench/results/calibration_v2`
    // to refresh after any change to effort accounting.  Earlier
    // calibrations live in git history (commits 82c0fbc, 83bc78b).
    constexpr double kWeightFj = 1.54;
    constexpr double kWeightFpr = 2.43;
    constexpr double kWeightLocalMip = 4.68;
    constexpr double kWeightScylla = 1.00;

    const bool fj_on = options->mip_heuristic_run_feasibility_jump;
    const bool fpr_on = options->mip_heuristic_run_fpr;
    const bool lm_on = options->mip_heuristic_run_local_mip;
    const bool sc_on = options->mip_heuristic_run_scylla;

    double total_weight = 0.0;
    if (fj_on) {
        total_weight += kWeightFj;
    }
    if (fpr_on) {
        total_weight += kWeightFpr;
    }
    if (lm_on) {
        total_weight += kWeightLocalMip;
    }
    if (sc_on) {
        total_weight += kWeightScylla;
    }

    if (total_weight == 0.0) {
        return false;
    }

    auto alloc = [&](double w) -> size_t { return static_cast<size_t>(budget * w / total_weight); };

    // Each heuristic's inner loops also poll the deadline, but their setup
    // (build_csc, precompute_var_orders) runs before that first inner poll;
    // checking out here skips the setup entirely once the budget is
    // exhausted.  `terminatorTerminated` is called only from this
    // sequential outer loop — the previous heuristic's parallel region has
    // already joined, so there is no concurrent access.
    const double time_limit = options->time_limit;
    auto *mipdata = mipsolver.mipdata_.get();
    auto deadline_hit = [&]() {
        return mipdata->terminatorTerminated() || mipsolver.timer_.read() >= time_limit;
    };

    // Shared pool across the whole sequential chain.  One seed_pool call
    // tags the incumbent with kSolutionSourceHeuristic; each heuristic
    // worker adds its own entries with a per-heuristic source tag; a
    // single flush at the end lets HiGHS pick up cross-heuristic finds
    // (e.g. FJ solution landing in FPR's restart pool).
    const bool minimize = (mipsolver.model_->sense_ == ObjSense::kMinimize);
    SolutionPool pool(kPoolCapacity, minimize);
    seed_pool(pool, mipsolver);

    bool infeasible = false;

    // All four heuristics return the effort they consumed and this
    // function books it into `mipdata->heuristic_effort_used` (issue #79
    // and its follow-up that extended LocalMIP's contract to FJ, FPR,
    // and Scylla).  mode_dispatch.cpp is therefore the single point of
    // sequential effort accounting — no heuristic self-books.  Note:
    // `fpr_lp` is *not* part of the harmonisation; it runs during B&B
    // dive (not via this `run_sequential` path) and keeps its own
    // self-booking — see `src/fpr_lp.cpp`.  All
    // bookings happen on the main thread after each parallel region has
    // joined, so we read/write the counter below without synchronisation
    // — do not move any of them into a worker without revisiting this.
    // (Historical note: local_mip used to early-return when
    // `mipdata->incumbent.empty()` so its [Sequential] line was absent
    // on a first solve.  Since issue #75 it runs the paper's
    // construction phase on cold start and emits a non-zero effort even
    // when no upstream heuristic produced a feasible solution.)
    //
    // Wall-ms is measured in this outer frame so all four measurements
    // share a clock and include setup (`build_csc`, `precompute_var_orders`,
    // worker construction) — what users actually pay for.
    const HighsLogOptions &log_options = options->log_options;
    auto run_and_log = [&](const char *name, auto &&call) {
        const auto t0 = std::chrono::steady_clock::now();
        const size_t effort = call();
        const auto t1 = std::chrono::steady_clock::now();
        mipdata->heuristic_effort_used += effort;
        const double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        log_sequential(log_options, name, effort, wall_ms);
    };

    if (fj_on && !deadline_hit()) {
        run_and_log("fj", [&]() -> size_t {
            return fj::run_parallel(mipsolver, pool, alloc(kWeightFj), opportunistic);
        });
    }
    if (!infeasible && fpr_on && !deadline_hit()) {
        run_and_log("fpr", [&]() -> size_t {
            return fpr::run_parallel(mipsolver, pool, alloc(kWeightFpr), opportunistic);
        });
    }
    if (!infeasible && lm_on && !deadline_hit()) {
        run_and_log("local_mip", [&]() -> size_t {
            return local_mip::run_parallel(mipsolver, pool, alloc(kWeightLocalMip), opportunistic);
        });
    }
    if (!infeasible && sc_on && !deadline_hit()) {
        run_and_log("scylla", [&]() -> size_t {
            return scylla::run_parallel(mipsolver, pool, alloc(kWeightScylla), opportunistic);
        });
    }

    // Flush pool to HiGHS (best first).  Each entry carries its
    // originating heuristic's source tag (FJ/FPR/LocalMIP/Scylla or
    // the generic kSolutionSourceHeuristic for the seeded incumbent),
    // so HiGHS logs per-heuristic provenance instead of a generic tag.
    for (auto &entry : pool.sorted_entries()) {
        mipdata->trySolution(entry.solution, entry.source);
    }

    return infeasible;
}

}  // namespace

bool run_presolve(HighsMipSolver &mipsolver, size_t budget) {
    const auto *options = mipsolver.options_mip_;
    const bool portfolio = options->mip_heuristic_portfolio;
    const bool opportunistic = options->mip_heuristic_opportunistic;

    if (portfolio) {
        portfolio::run_presolve(mipsolver, budget, opportunistic);
        return false;
    }
    return run_sequential(mipsolver, budget, opportunistic);
}

}  // namespace heuristics
