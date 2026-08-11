#include "mode_dispatch.h"

#include "effort_ledger.h"
#include "fj.h"
#include "fpr.h"
#include "heuristic_context.h"
#include "incumbent_sink.h"
#include "io/HighsIO.h"
#include "local_mip.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "scylla.h"

#include <algorithm>
#include <atomic>
#include <string>

namespace heuristics {

namespace {

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
// Basis change since round 6 (#94): the shared CSC transpose is now built
// once for the whole chain, in `run_sequential` and outside every
// heuristic's timing window.  Rounds 5 and 6 measured it four times, once
// inside each heuristic's own `wall_ms`.  Charged effort did not change, so
// every rate below was measured against a slightly longer wall than a fresh
// run will produce — and not uniformly across the four, since their
// dispatches differ in length while `build_csc` is a fixed cost.  A
// re-measurement is therefore not directly comparable to these numbers:
// recalibrate the whole set together rather than moving one constant
// against them.
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

// One heuristic's entry in the fixed FJ -> FPR -> LocalMIP -> Scylla
// chain.  `run_sequential` is a filtered loop over the table below; the
// four near-identical `if (enabled && !deadline) { ... }` blocks it
// replaced were the last place a fifth heuristic would have had to be
// wired in by hand.
struct HeuristicConfig {
    const char *name;
    // kSolutionSource* tag the sink attributes this heuristic's solutions
    // with, so the HiGHS log credits the right finder.
    int source_tag;
    // Share of the post-FJ envelope.  Meaningless when `fixed_budget`.
    double weight;
    // FJ's budget is a fixed per-worker allowance, not a weighted share —
    // see the fj_budget derivation in run_sequential.  It is the only
    // entry with this set, and the only one whose `weight` is unused.
    bool fixed_budget;
    // Which `mip_heuristic_suite` bit enables this entry.
    bool HeuristicFlags::*flag;
    size_t (*run)(const ProblemView &, const HeuristicBudget &, ExecutionContext &,
                  IncumbentSink &);
};

constexpr HeuristicConfig kChain[] = {
    {"fj", kSolutionSourceFJ, 0.0, true, &HeuristicFlags::fj, &fj::run},
    {"fpr", kSolutionSourceFPR, kWeightFpr, false, &HeuristicFlags::fpr, &fpr::run},
    {"local_mip", kSolutionSourceLocalMIP, kWeightLocalMip, false, &HeuristicFlags::local_mip,
     &local_mip::run},
    {"scylla", kSolutionSourceScylla, kWeightScylla, false, &HeuristicFlags::scylla,
     &scylla::run},
};

// Weighted effort allocation: each heuristic runs in turn with its
// proportional share of the budget and the full thread pool.
//
// A single `IncumbentSink` is constructed here and threaded through all
// heuristics so that solutions found by an earlier heuristic (e.g. FJ)
// become available as pool-restart seeds for later heuristics (FPR,
// LocalMIP).  Each entry carries its originating heuristic's source tag
// (see incumbent_sink.h / #73).
bool run_sequential(HighsMipSolver &mipsolver, size_t budget, const HeuristicFlags &flags) {
    double rest_weight = 0.0;
    bool any_enabled = false;
    for (const HeuristicConfig &h : kChain) {
        if (flags.*h.flag) {
            any_enabled = true;
            // Fixed-budget entries (FJ) are not part of the weighted share,
            // and their `weight` field is documented meaningless.  Enforce
            // that here rather than relying on the table happening to hold
            // 0.0: a nominal weight added for a future non-fixed FJ mode
            // would otherwise shrink everyone else's cut with no test
            // failing, since the effort assertions are all "non-zero".
            if (!h.fixed_budget) {
                rest_weight += h.weight;
            }
        }
    }
    if (!any_enabled) {
        return false;
    }

    ExecutionContext exec = make_exec(mipsolver);

    // Check out before the transpose, not only before each heuristic.  Each
    // heuristic used to build its own CSC behind its own deadline check, so
    // an already-terminated dispatch built none; hoisting the build out of
    // all four would otherwise make it unconditional, and it is the single
    // most expensive piece of setup in this function.
    if (exec.terminated()) {
        return false;
    }

    // Spans the whole chain, shared setup included, and is reported as
    // `[Root] presolve_heur_s` — a different question from the per-
    // heuristic windows below, which stay scoped to what `kWeight*`
    // calibrates.  See `EffortLedger::note_presolve_span`.  Started after
    // the `terminated()` check above, since a dispatch that returns there
    // costs the solver nothing.
    EffortLedger ledger(mipsolver);
    const double chain_t0_s = ledger.now_s();

    // Built once for the whole chain: the CSC transpose and the derived
    // sizes are the same for all four heuristics, and the row-major buffers
    // they come from are frozen by `runSetup()` before dispatch.  Each
    // heuristic used to build its own identical copy.  `csc` owns the
    // storage `problem` views, so it has to outlive the loop below.
    CscMatrix csc;
    const ProblemView problem = make_problem(mipsolver, csc);

    const size_t fj_budget = flags.fj ? exec.num_workers * (problem.nnz << 10) : 0;

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
    // whole PDLP solve charging `iters x nnz`.  #94's ledger unified where
    // that spend is *recorded*, not how much of it any heuristic is allowed
    // to book past its share — this floor just caps a starvation pathology
    // rather than re-balancing the split.  #96's budget sweep is where the
    // overshoot itself gets revisited.
    const size_t rest_floor = budget / 4;  // <= budget by construction
    const size_t used_for_fj = std::min(fj_budget, budget - rest_floor);
    const size_t rest_budget = budget - used_for_fj;

    auto rest_alloc = [&](double w) -> size_t {
        return rest_weight > 0
                   ? static_cast<size_t>(static_cast<double>(rest_budget) * w / rest_weight)
                   : 0;
    };

    // One sink for the whole sequential chain, so a solution found by an
    // earlier heuristic (say FJ) is available as a pool-restart seed for
    // the later ones.  Its constructor seeds the pool from the incumbent
    // with the generic kSolutionSourceHeuristic tag; `set_source` below
    // re-tags it per heuristic so each entry carries its finder's tag.
    IncumbentSink sink(mipsolver, kSolutionSourceHeuristic);

    // All four heuristics return the effort they consumed and hand it to
    // the ledger, which is the single point of effort accounting for the
    // whole patch (issue #79 and its follow-up that extended LocalMIP's
    // contract to FJ, FPR and Scylla; #94 brought the dive-time `fpr_lp`
    // onto the same path).  No heuristic self-books.  All
    // bookings happen on the main thread after each parallel region has
    // joined, so `EffortLedger` reads/writes the counter without
    // synchronisation — do not move any of them into a worker without
    // revisiting this, and the matching note in effort_ledger.h.
    // (Historical note: local_mip used to early-return when
    // `mipdata->incumbent.empty()` so its [Sequential] line was absent
    // on a first solve.  Since issue #75 it runs the paper's
    // construction phase on cold start and emits a non-zero effort even
    // when no upstream heuristic produced a feasible solution.)
    //
    // Wall-ms is measured in this outer frame so all four measurements
    // share a clock and include each heuristic's own setup
    // (`precompute_var_orders`, `ContestedPdlp` construction, worker
    // construction) — what users actually pay for.  The shared CSC build
    // sits outside all four, since it is no longer any one of them.
    auto run_and_charge = [&](const char *name, auto &&call) {
        // `found` is the sink's accepted-offer count moving across this
        // heuristic's dispatch.  Read either side of the call, on this
        // thread, with the parallel region joined at both points.
        const size_t accepted_before = sink.accepted();
        const double t0_s = ledger.now_s();
        const size_t effort = call();
        ledger.charge_presolve(name, effort, sink.accepted() > accepted_before, t0_s,
                               ledger.now_s());
    };

    // Each heuristic's inner loops also poll the deadline, but their own
    // setup (precompute_var_orders, ContestedPdlp construction) runs before
    // that first inner poll; re-checking here skips it once the budget is
    // exhausted.  `exec.terminated()` is safe to call from this sequential
    // outer loop — the previous heuristic's parallel region has already
    // joined, so there is no concurrent access.
    //
    // `slice`, not `budget`: the parameter of that name is the whole-chain
    // envelope these shares are carved out of.
    for (const HeuristicConfig &h : kChain) {
        if (!(flags.*h.flag) || exec.terminated()) {
            continue;
        }
        const HeuristicBudget slice =
            make_budget(h.fixed_budget ? fj_budget : rest_alloc(h.weight), exec.num_workers);
        sink.set_source(h.source_tag);
        run_and_charge(h.name, [&]() -> size_t { return h.run(problem, slice, exec, sink); });
    }

    ledger.note_presolve_span(chain_t0_s, ledger.now_s());
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
    } else if (!flags.fj && !flags.fpr && !flags.local_mip && !flags.scylla &&
               options.mip_heuristic_suite != "off") {
        // Only reachable as `suite=fj` with mip_heuristic_run_feasibility_jump
        // false, which asks for FJ and then takes it away.  That run is
        // heuristic-free without being `off`, so it also loses the native FJ
        // call site — a benchmark row labelled "FJ isolated" would silently
        // measure vanilla-minus-FJ.  Say so rather than leave it silent.
        highsLogUser(options.log_options, HighsLogType::kWarning,
                     "mip_heuristic_suite=\"%s\" selects only FeasibilityJump, which "
                     "mip_heuristic_run_feasibility_jump=false disables; no heuristic will "
                     "run. Use mip_heuristic_suite=off for a vanilla-equivalent run.\n",
                     options.mip_heuristic_suite.c_str());
    }

    return run_sequential(mipsolver, budget, flags);
}

void log_solve_summary(HighsMipSolver &mipsolver) {
    // RENS and RINS each build a sub-MIP with its own HighsMipSolver, and
    // cleanupSolve runs for those too.  Their counters describe a
    // different model, and one `[Native]` line per sub-MIP would make the
    // per-solve records ambiguous, so only the outer solve reports.
    if (mipsolver.submip) {
        return;
    }
    // No null check on `mipdata_`: `cleanupSolve`, the only caller,
    // dereferences it two statements earlier.
    const HighsMipSolverData *mipdata = mipsolver.mipdata_.get();
    const HighsLogOptions &log_options = mipsolver.options_mip_->log_options;

    // `rens` is the whole-solve total and `rens_root` the root-site subset
    // of it.  The root gate is the one a presolve-found incumbent closes —
    // `upper_limit` goes finite and `moreHeuristicsAllowed()` starts
    // mattering — so a suppressed root RENS is the cannibalization signal,
    // and the merged total can hold steady while it disappears.
    //
    // `heur_lp_iters` / `total_lp_iters` are upstream's own counters, but
    // they are *shared*: `EffortLedger::charge_dive` adds to both so
    // fpr_lp competes with RENS/RINS for the same envelope.  Reporting
    // them raw therefore bills our dive work as HiGHS's, which is the
    // confound this whole line exists to remove — so `fpr_lp_lp_iters`
    // reports exactly what we put in, for an analyst to subtract.
    //
    // `%lld` with an explicit cast: the LP-iteration counters are int64_t,
    // whose printf length modifier is platform-dependent.
    highsLogDev(log_options, HighsLogType::kVerbose,
                "[Native] rens=%zu rens_root=%zu rins=%zu rcfix=%zu heur_lp_iters=%lld "
                "total_lp_iters=%lld fpr_lp_lp_iters=%lld\n",
                mipdata->rens_calls.load(std::memory_order_relaxed),
                mipdata->rens_root_calls.load(std::memory_order_relaxed),
                mipdata->rins_calls.load(std::memory_order_relaxed),
                mipdata->rcfix_calls.load(std::memory_order_relaxed),
                static_cast<long long>(mipdata->heuristic_lp_iterations),
                static_cast<long long>(mipdata->total_lp_iterations),
                static_cast<long long>(mipdata->fpr_lp_lp_iterations));

    // `lp_time_s` is negative when the root LP was never reached (presolve
    // solved or proved the model, or a limit fired first); the parser
    // treats that as "no root LP" rather than "at t=0".
    //
    // The two fields are not on the same footing across a restart, and
    // deliberately so: HiGHS's `goto restart` re-enters above both the
    // presolve chain and `evaluateRootNode` without rebuilding `mipdata_`,
    // so `presolve_heur_s` accumulates over every restart while
    // `lp_time_s` pins the *first* root LP.  "How long until the root LP
    // first got to start" and "how much wall time did the presolve chain
    // cost in total" are the two questions being asked; on a restarting
    // instance `presolve_heur_s > lp_time_s` is therefore expected rather
    // than a contradiction.
    highsLogDev(log_options, HighsLogType::kVerbose,
                "[Root] lp_time_s=%.3f presolve_heur_s=%.3f\n", mipdata->root_lp_time,
                mipdata->presolve_heuristic_time);
}

}  // namespace heuristics
