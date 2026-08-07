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
#include <string>

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
// A single `SolutionPool` is constructed here and threaded through all
// heuristics so that solutions found by an earlier heuristic (e.g. FJ)
// become available as pool-restart seeds for later heuristics (FPR,
// LocalMIP).  The pool is seeded once from the incumbent; an on_accept
// callback then submits each new solution to HiGHS immediately on
// acceptance so incumbent timestamps reflect find time rather than
// flush time.  Each entry carries its originating heuristic's source
// tag (see solution_pool.h / #73).
bool run_sequential(HighsMipSolver &mipsolver, size_t budget, const HeuristicFlags &flags) {
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
    //   2. Run the fixed suite on `bench/instances_small.txt`, all four
    //      heuristics on, at `threads=16`.  Both of those are part of the
    //      measurement, not incidental:
    //        - the instance set, because the constants below were measured
    //          on it and a re-measurement elsewhere is not comparable;
    //        - the worker count, because `effort_per_ms` is a throughput
    //          that scales with N, and *not* identically across
    //          heuristics — FPR and LocalMIP are near-linear in N while
    //          Scylla serialises every PDLP solve behind the
    //          `ContestedPdlp` mutex and scales sublinearly.  The N factor
    //          therefore does not cancel in the ratios: the same binary on
    //          the same set gives local_mip:scylla = 4.68 at 16 workers
    //          and 2.81 at 6.  16 is what round 5 used; deviate and the
    //          numbers are not comparable to the base.
    //      `bench/run_benchmark.py`'s `patched` config pins
    //      `mip_heuristic_suite=all`, which is what this calibration wants;
    //      any config that narrows the suite measures fewer heuristics.
    //   3. `python bench/check_effort_drift.py <results-dir>`.
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
    // Base calibration, round 5: `bench/instances_small.txt` (25 MIPLIB
    // instances, 30 s each, mip_root_presolve_only=true, threads=16,
    // presolve budget at the 0.30 default — then still the overloaded
    // `mip_heuristic_effort`, now `mip_heuristic_presolve_effort` with the
    // same value and formula, so it carries over across the option split).
    // Measured geomean `effort_per_ms` after issue #78 (cold-start
    // construction sweep rolled into local_mip's reported effort):
    //   fpr=636k  local_mip=1222k  scylla=261k
    //
    // Round 6 (#92): the constants below are those rates scaled by the
    // measured effect of this changeset, not a fresh absolute measurement.
    // Why, and how much to trust it:
    //
    //   * #92 removed the epoch-gated runner and, with it, a per-improving-
    //     attempt SolutionPool spin-lock, two per-call allocations, and an
    //     N× redundant LocalMIP cold start.  Those speed up the heuristics
    //     by *different* factors, so the ratios moved.
    //   * Controlled A/B on this exact instance set, both binaries on one
    //     idle machine, back-to-back, two seeds — geomean of the per-seed
    //     rate ratios (pre-#92 -> HEAD):
    //         fpr 1.27x   local_mip 1.36x   scylla 1.03x   (fj 1.23x)
    //     Scylla is the outlier because it is PDLP/mutex-bound: it does not
    //     benefit from per-attempt hot-path work the way the others do.
    //     That separation is the robust part — it reproduced across both
    //     seeds (scylla 1.006/1.061 vs fpr 1.163/1.386).
    //   * Scaling rather than replacing, because the absolute ratios are
    //     strongly machine- and thread-count-dependent and the round-5
    //     numbers are the ones taken on the intended benchmark
    //     configuration.  Same code, same instances, 6 workers instead of
    //     16 gives local_mip:scylla = 2.81 (2.67 and 2.96 on the two
    //     seeds) against round 5's 4.68 — LocalMIP scales near-linearly in
    //     workers and Scylla does not, so a 6-worker box cannot stand in
    //     for a 16-worker one.
    //
    // Trust boundary, and one known bias.  The multipliers are themselves
    // ratios measured at 6 workers, and the two things they mostly
    // capture — a *contended* pool spin-lock and an *N-fold* redundant
    // LocalMIP cold start — both have benefits that grow with worker
    // count by construction.  So the 16-worker multipliers are expected
    // to be >= these, which biases kWeightLocalMip (and to a lesser
    // extent kWeightFpr) low rather than high.  Direction is solid;
    // magnitude is +-10% and n=2 seeds cannot support that as a
    // confidence interval — it is the observed per-seed spread, and
    // switching to a paired per-instance estimator moves the weights
    // ~6% on its own.  These are a budget split
    // with a 3x drift tolerance, so that is within usable range — but a
    // full re-measurement on the 16-worker benchmark machine should
    // confirm before the closeout campaign, and would supersede the
    // scaling entirely.  #96's budget sweep is the natural place.
    //
    // What the round-6 weights do and do not buy, measured rather than
    // assumed — read this before spending time tuning them:
    //
    //   * They move *effort* as intended.  Going from round 5 to round 6
    //     shifts Scylla's share of the post-FJ envelope 12.3% -> 9.9%, and
    //     the measured geomean effort follows: scylla -5%, local_mip +21%,
    //     fpr +14%.
    //   * They do **not** measurably rebalance *wall clock*, which is the
    //     thing the "equal weights -> equal wall" contract above promises.
    //     Geomean wall per heuristic across the calibration set barely
    //     moves (spread 2.76x/2.93x -> 2.76x/2.82x over two seeds), and
    //     Scylla still spends ~2.8x FPR's wall under either set.
    //   * The reason is fixed cost, not a broken split.  At the default
    //     effort on this set each heuristic's dispatch is only ~20-70 ms,
    //     enough that per-dispatch setup (build_csc, FPR's var-order
    //     precompute, Scylla's ContestedPdlp/LP construction) dominates
    //     the budget-driven part.  A 20% budget cut moved Scylla's wall
    //     2.5%.  The contract holds asymptotically, once search dominates
    //     setup — not at the default budget on instances this size.
    //   * The envelope itself does bind for FPR and LocalMIP: raising
    //     `mip_heuristic_presolve_effort` 0.30 -> 1.00 scales their effort
    //     ~4.1x.  Scylla is inconsistent (4.1x on roll3000, 2.6x on
    //     swath3, and on mzzv11 either down or absent from the chain
    //     altogether, depending on the run, once the enlarged envelope
    //     lets FPR/LocalMIP reach the time limit first) because PDLP
    //     stalls and stale rounds bound it before the budget does.  FJ is
    //     flat, as intended.
    //
    // So: tuning these constants is worthwhile for effort accounting and
    // for keeping the w ∝ r invariant honest, but do not expect wall-clock
    // rebalancing from them at default effort.  If equalising wall is the
    // actual goal, the fixed setup cost has to be attacked first.
    //
    // Weights are proportional to geomean `effort_per_ms` (scylla
    // normalised to 1.0 as the slowest-per-effort heuristic).  FJ is
    // excluded — fixed vanilla budget, not weight-apportioned; see
    // fj_budget below.  Re-run `bench/check_effort_drift.py <results-dir>`
    // to refresh after any change to effort accounting.  Earlier
    // calibrations live in git history (commits 82c0fbc, 83bc78b).
    //
    // FJ uses a fixed per-worker budget matching vanilla HiGHS's single-thread
    // FJ limit (nnz << 10, hardcoded in HighsFeasibilityJump.cpp; unaffected
    // by either effort option).  N parallel workers each cover nnz*1024 steps
    // so patched FJ is at least as deep as vanilla per thread.  The remaining
    // presolve budget after FJ is split among FPR / LocalMIP / Scylla below.
    constexpr double kWeightFpr = 2.99;       // round 5: 2.43
    constexpr double kWeightLocalMip = 6.16;  // round 5: 4.68
    constexpr double kWeightScylla = 1.00;

    double rest_weight = 0.0;
    if (flags.fpr) {
        rest_weight += kWeightFpr;
    }
    if (flags.local_mip) {
        rest_weight += kWeightLocalMip;
    }
    if (flags.scylla) {
        rest_weight += kWeightScylla;
    }

    if (!flags.fj && rest_weight == 0.0) {
        return false;
    }

    const size_t nnz = static_cast<size_t>(mipsolver.mipdata_->ARindex_.size());
    const size_t N_threads = static_cast<size_t>(std::max(1, highs::parallel::num_threads()));
    const size_t fj_budget = flags.fj ? N_threads * (nnz << 10) : 0;

    // What FJ *charges* against the shared presolve budget, which is not
    // the same as what it spends: FJ always runs with the full `fj_budget`
    // (a fixed per-worker allowance, see below), and this only decides how
    // much of the envelope is left for FPR / LocalMIP / Scylla.
    //
    // `fj_budget` scales with the worker count while `budget` does not, so
    // an uncapped charge lets FJ eat the entire envelope: at the default
    // `mip_heuristic_presolve_effort` the two meet at 24 workers, and past
    // that `rest_budget` is 0 and the other three heuristics return
    // immediately, reporting `effort=0` with no warning.  HiGHS derives
    // its default worker count from the machine, so that is reachable on
    // any host with ~47+ hardware threads — plausible for a benchmark
    // machine, and silent when it happens.
    //
    // Reserving a quarter of the envelope bounds that.  The floor binds
    // when `N * (nnz<<10) > 0.75 * budget`: from N >= 19 at the default
    // `mip_heuristic_presolve_effort` (0.30), against N >= 24 for total
    // starvation — so the threads=16 / 0.30 configuration the `kWeight*`
    // constants were calibrated at is untouched.  The threshold scales
    // with the effort option though, binding from N >= 4 at 0.05, which
    // matters for a budget sweep (#96).
    //
    // Note this floors what FJ *charges*, which is not a bound on what
    // anyone spends: FJ books its full per-worker allowance regardless,
    // and Scylla routinely books past its share because one attempt is a
    // whole PDLP solve charging `iters x nnz`.  #94'''s unified ledger is
    // where that gets made honest; this just caps a starvation
    // pathology rather than re-balancing the split.
    const size_t rest_floor = budget / 4;  // <= budget by construction
    const size_t used_for_fj = std::min(fj_budget, budget - rest_floor);
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

    if (flags.fj && !deadline_hit()) {
        run_and_log("fj", [&]() -> size_t {
            return fj::run_parallel(mipsolver, pool, fj_budget);
        });
    }
    if (flags.fpr && !deadline_hit()) {
        run_and_log("fpr", [&]() -> size_t {
            return fpr::run_parallel(mipsolver, pool, rest_alloc(kWeightFpr));
        });
    }
    if (flags.local_mip && !deadline_hit()) {
        run_and_log("local_mip", [&]() -> size_t {
            return local_mip::run_parallel(mipsolver, pool, rest_alloc(kWeightLocalMip));
        });
    }
    if (flags.scylla && !deadline_hit()) {
        run_and_log("scylla", [&]() -> size_t {
            return scylla::run_parallel(mipsolver, pool, rest_alloc(kWeightScylla));
        });
    }

    return false;
}

}  // namespace

HeuristicFlags effective_flags(const HighsOptions &options, bool *recognized) {
    const std::string &suite = options.mip_heuristic_suite;

    // Fail open on an unrecognised value: running everything is the same
    // thing the default does, and silently disabling all four heuristics
    // because of a typo is the worse failure.  The caller warns.
    HeuristicFlags flags{true, true, true, true};
    bool known = true;
    if (suite == "off") {
        flags = {false, false, false, false};
    } else if (suite == "fj") {
        flags = {true, false, false, false};
    } else if (suite == "fpr") {
        flags = {false, true, false, false};
    } else if (suite == "local_mip") {
        flags = {false, false, true, false};
    } else if (suite == "scylla") {
        flags = {false, false, false, true};
    } else if (suite != "all") {
        known = false;
    }

    // Upstream's own FJ switch still means what it says.  At suite=off the
    // patch leaves it gating HiGHS's native FJ call site; everywhere else it
    // gates ours, so `mip_heuristic_run_feasibility_jump=false` turns
    // FeasibilityJump off in every configuration rather than only one.
    flags.fj = flags.fj && options.mip_heuristic_run_feasibility_jump;

    if (recognized != nullptr) {
        *recognized = known;
    }
    return flags;
}

bool run_presolve(HighsMipSolver &mipsolver, size_t budget) {
    const HighsOptions &options = *mipsolver.options_mip_;

    bool recognized = false;
    const HeuristicFlags flags = effective_flags(options, &recognized);
    if (!recognized) {
        highsLogUser(options.log_options, HighsLogType::kWarning,
                     "Unknown mip_heuristic_suite value \"%s\"; running all heuristics.\n",
                     options.mip_heuristic_suite.c_str());
    }

    return run_sequential(mipsolver, budget, flags);
}

}  // namespace heuristics
