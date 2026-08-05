#include "mode_dispatch.h"

#include "fj.h"
#include "fpr.h"
#include "io/HighsIO.h"
#include "local_mip.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "parallel/HighsParallel.h"
#include "scylla.h"
#include "solution_pool.h"

#include <chrono>
#include <mutex>

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
// LocalMIP).  The pool is seeded once from the incumbent; an on_accept
// callback then submits each new solution to HiGHS immediately on
// acceptance so incumbent timestamps reflect find time rather than
// flush time.  Each entry carries its originating heuristic's source
// tag (see solution_pool.h / #73).
bool run_sequential(HighsMipSolver &mipsolver, size_t budget, bool opportunistic, bool fj_on,
                    bool fpr_on, bool lm_on, bool sc_on) {
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
    //   2. Run seq/det (`mip_heuristic_opportunistic=false`) on MIPLIB
    //      with all four heuristics on (see `bench/run_benchmark.py`).
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
    // instances, 30 s each, presolve budget at the 0.30 default —
    // then still the overloaded `mip_heuristic_effort`, now
    // `mip_heuristic_presolve_effort` with the same value and formula,
    // so the calibration carries over — seq/det,
    // mip_root_presolve_only=true, multi-thread default threads=16).  Measured geomean
    // `effort_per_ms` after issue #78 (cold-start construction sweep rolled into local_mip's
    // reported effort):
    //   fpr=636k  local_mip=1222k  scylla=261k   drift = 4.68× (FJ excluded:
    //   fixed vanilla budget, not weight-apportioned; see fj_budget below)
    // Weights are proportional to geomean `effort_per_ms` (scylla
    // normalised to 1.0 as the slowest-per-effort heuristic).
    // Re-run `bench/check_effort_drift.py bench/results/calibration_v2`
    // to refresh after any change to effort accounting.  Earlier
    // calibrations live in git history (commits 82c0fbc, 83bc78b).
    //
    // FJ uses a fixed per-worker budget matching vanilla HiGHS's single-thread
    // FJ limit (nnz << 10, hardcoded in HighsFeasibilityJump.cpp; unaffected
    // by either effort option).  N parallel workers each cover nnz*1024 steps
    // so patched FJ is at least as deep as vanilla per thread.  The remaining
    // presolve budget after FJ is split among FPR / LocalMIP / Scylla below.
    constexpr double kWeightFpr = 2.43;
    constexpr double kWeightLocalMip = 4.68;
    constexpr double kWeightScylla = 1.00;

    double rest_weight = 0.0;
    if (fpr_on) {
        rest_weight += kWeightFpr;
    }
    if (lm_on) {
        rest_weight += kWeightLocalMip;
    }
    if (sc_on) {
        rest_weight += kWeightScylla;
    }

    if (!fj_on && rest_weight == 0.0) {
        return false;
    }

    const size_t nnz = static_cast<size_t>(mipsolver.mipdata_->ARindex_.size());
    const size_t N_threads = static_cast<size_t>(std::max(1, highs::parallel::num_threads()));
    const size_t fj_budget = fj_on ? N_threads * (nnz << 10) : 0;
    const size_t used_for_fj = std::min(fj_budget, budget);
    const size_t rest_budget = budget - used_for_fj;

    auto rest_alloc = [&](double w) -> size_t {
        return rest_weight > 0
                   ? static_cast<size_t>(static_cast<double>(rest_budget) * w / rest_weight)
                   : 0;
    };

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
    // worker adds its own entries with a per-heuristic source tag.
    // The on_accept callback submits each solution to HiGHS immediately when
    // accepted, so incumbent timestamps reflect find time rather than flush time.
    const bool minimize = (mipsolver.model_->sense_ == ObjSense::kMinimize);
    SolutionPool pool(kPoolCapacity, minimize);
    seed_pool(pool, mipsolver);

    // Register the on_accept callback after seed_pool to avoid re-submitting
    // the seeded incumbent (already known to HiGHS).  The mutex serializes
    // concurrent trySolution calls since addIncumbent is not thread-safe.
    std::mutex highs_mtx;
    pool.set_on_accept([&](const std::vector<double> &sol, int src) {
        std::lock_guard<std::mutex> guard(highs_mtx);
        mipdata->trySolution(sol, src);
    });

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
            return fj::run_parallel(mipsolver, pool, fj_budget, opportunistic);
        });
    }
    if (fpr_on && !deadline_hit()) {
        run_and_log("fpr", [&]() -> size_t {
            return fpr::run_parallel(mipsolver, pool, rest_alloc(kWeightFpr), opportunistic);
        });
    }
    if (lm_on && !deadline_hit()) {
        run_and_log("local_mip", [&]() -> size_t {
            return local_mip::run_parallel(mipsolver, pool, rest_alloc(kWeightLocalMip),
                                           opportunistic);
        });
    }
    if (sc_on && !deadline_hit()) {
        run_and_log("scylla", [&]() -> size_t {
            return scylla::run_parallel(mipsolver, pool, rest_alloc(kWeightScylla), opportunistic);
        });
    }

    return false;
}

}  // namespace

HeuristicFlags effective_flags(const HighsOptions &options, bool *preset_recognized) {
    // Individual options first; a recognized non-empty preset overrides
    // all five flags.  Unknown presets leave the individual-option values
    // in place (no silent disable-all footgun) — the caller warns.
    HeuristicFlags flags{options.mip_heuristic_run_feasibility_jump, options.mip_heuristic_run_fpr,
                         options.mip_heuristic_run_local_mip, options.mip_heuristic_run_scylla,
                         options.mip_heuristic_opportunistic};

    const auto &preset = options.mip_heuristic_preset;
    bool recognized = false;
    if (!preset.empty()) {
        if (preset == "off") {
            flags = {false, false, false, false, false};
            recognized = true;
        } else if (preset == "fpr") {
            flags = {false, true, false, false, false};
            recognized = true;
        } else if (preset == "all_det") {
            flags = {true, true, true, false, false};
            recognized = true;
        } else if (preset == "all_opp") {
            flags = {true, true, true, false, true};
            recognized = true;
        } else if (preset == "scylla") {
            flags = {false, false, false, true, true};
            recognized = true;
        }
    }
    if (preset_recognized != nullptr) {
        *preset_recognized = recognized;
    }
    return flags;
}

bool run_presolve(HighsMipSolver &mipsolver, size_t budget) {
    const auto *options = mipsolver.options_mip_;

    bool preset_applied = false;
    const HeuristicFlags flags = effective_flags(*options, &preset_applied);
    const bool fj_on = flags.fj;
    const bool fpr_on = flags.fpr;
    const bool lm_on = flags.local_mip;
    const bool sc_on = flags.scylla;
    const bool opportunistic = flags.opportunistic;

    if (!options->mip_heuristic_preset.empty() && !preset_applied) {
        highsLogUser(options->log_options, HighsLogType::kWarning,
                     "Unknown mip_heuristic_preset value \"%s\"; "
                     "ignoring preset and using individual option flags.\n",
                     options->mip_heuristic_preset.c_str());
    }

    // When a preset was applied, write the derived flags back into the options
    // struct so that downstream code that reads options directly sees the
    // preset values.  The underlying object is a non-const HighsOptions member
    // of the enclosing Highs instance, so the cast is safe.  We save the
    // original values first and restore them before returning so that a second
    // highs.run() call (with or without a preset) sees the options the user
    // actually set, not the preset-overwritten ones.
    if (preset_applied) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        auto *w = const_cast<HighsOptions *>(options);

        // Save originals.
        const bool saved_fj = w->mip_heuristic_run_feasibility_jump;
        const bool saved_fpr = w->mip_heuristic_run_fpr;
        const bool saved_lm = w->mip_heuristic_run_local_mip;
        const bool saved_sc = w->mip_heuristic_run_scylla;
        const bool saved_opportunistic = w->mip_heuristic_opportunistic;

        // Apply preset.
        w->mip_heuristic_run_feasibility_jump = fj_on;
        w->mip_heuristic_run_fpr = fpr_on;
        w->mip_heuristic_run_local_mip = lm_on;
        w->mip_heuristic_run_scylla = sc_on;
        w->mip_heuristic_opportunistic = opportunistic;

        // Dispatch.
        const bool result =
            run_sequential(mipsolver, budget, opportunistic, fj_on, fpr_on, lm_on, sc_on);

        // Restore originals so multi-solve on the same Highs instance is safe.
        w->mip_heuristic_run_feasibility_jump = saved_fj;
        w->mip_heuristic_run_fpr = saved_fpr;
        w->mip_heuristic_run_local_mip = saved_lm;
        w->mip_heuristic_run_scylla = saved_sc;
        w->mip_heuristic_opportunistic = saved_opportunistic;

        return result;
    }

    return run_sequential(mipsolver, budget, opportunistic, fj_on, fpr_on, lm_on, sc_on);
}

}  // namespace heuristics
