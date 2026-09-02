#include "contested_pdlp.h"

#include "heuristic_context.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "pump_common.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <utility>

namespace {

constexpr HighsInt kMinPdlpIterCap = 100;

// `setOptionValue` reports an unknown or out-of-range option only through
// its return status plus a HiGHS log line.  We set `output_flag=false` on
// this instance, so the log line is suppressed and a typo'd or renamed
// option silently does nothing while looking like it applied — which is
// exactly how `pdlp_scaling` and `pdlp_e_restart_method` survived here
// undetected.  Route every option write through this instead.
template <typename T>
void set_option_or_die(Highs& highs, const char* name, T value) {
    if (highs.setOptionValue(name, value) != HighsStatus::kOk) {
        std::fprintf(stderr,
                     "ContestedPdlp: HiGHS rejected option '%s' (unknown name or invalid value). "
                     "This is a build-time bug, not a solve failure.\n",
                     name);
        // `assert` is a no-op under `NDEBUG` (the default for
        // `-DCMAKE_BUILD_TYPE=Release`, i.e. every build that runs a
        // benchmark), and an option that silently fails to apply is
        // exactly what this helper exists to eliminate — so abort
        // unconditionally, matching `run_locked_with_accounting` below.
        // Every call site passes a compile-time-constant option name, and
        // the two runtime-valued writes are provably in domain (epsilon is
        // floored at `pump::kEpsilonFloor`=1e-8 against `kkt_tolerance`'s
        // 1e-10 minimum; `time_limit` is guarded `> 0` in
        // `run_locked_with_accounting`), so this cannot fire on
        // legitimate solve data.
        assert(false && "ContestedPdlp: unknown or invalid HiGHS option");
        std::abort();
    }
}

}  // namespace

ContestedPdlp::ContestedPdlp(HighsMipSolver& mipsolver, HighsInt pdlp_iter_cap)
    : deadline_(deadline_of(mipsolver)) {
    const auto* model = mipsolver.model_;
    auto* mipdata = mipsolver.mipdata_.get();
    ncol_ = model->num_col_;
    nrow_ = model->num_row_;
    nnz_lp_ = mipdata->ARindex_.size();
    if (ncol_ == 0 || nrow_ == 0 || nnz_lp_ == 0) {
        return;
    }

    auto lp = pump::build_lp_relaxation(*model, *mipdata);
    set_option_or_die(highs_, "solver", "pdlp");
    set_option_or_die(highs_, "output_flag", false);
    // Two options used to be set here, `pdlp_scaling=true` and
    // `pdlp_e_restart_method=2`.  Both existed in HiGHS v1.13.1 and were
    // renamed in v1.14.0, so they have been silently rejected since that
    // bump — every result measured after it, including the round-5
    // effort calibration, was already produced on the HiGHS
    // defaults.  They are deliberately *not* revived under their nearest
    // modern names, because neither would do anything on this code path:
    //   - `pdlp_scaling_mode` is consumed only by HiPDLP
    //     (`pdlp/hipdlp/pdhg.cc`).  We set `solver=pdlp`, which
    //     `HighsSolve.cpp` routes to cuPDLP-C (`solveLpCupdlp`); cuPDLP-C
    //     scaling is governed by `pdlp_features_off & kPdlpScalingOff`,
    //     which defaults to scaling on — the original intent already holds.
    //   - `pdlp_cupdlpc_restart_method` is collapsed to {0,1} inside
    //     `CupdlpWrapper.cpp` (`intParam[E_RESTART_METHOD] = restart_on`),
    //     so the old `2` is indistinguishable from the default `1`; only
    //     `0` (restart off) changes anything.
    // Leaving them unset keeps behaviour bit-identical to what that
    // effort calibration was measured against.
    set_option_or_die(highs_, "pdlp_iteration_limit",
                      pdlp_iter_cap > kMinPdlpIterCap ? pdlp_iter_cap : kMinPdlpIterCap);
    highs_.passModel(std::move(lp));

    initialized_ = true;
}

ContestedPdlp::ContestedPdlp(ForTesting /*unused*/, Deadline deadline) : deadline_(deadline) {
    // Minimal init for unit tests: the subclass overrides `solve_locked`
    // so we never touch `highs_`.  ncol/nrow/nnz stay 0 by default;
    // tests that care can set them via their own friends, but most just
    // drive the lock / snapshot plumbing and don't need real shapes.
    initialized_ = true;
}

ContestedPdlp::SolveResult ContestedPdlp::solve_locked(
    const std::vector<double>& modified_cost, const std::vector<double>& warm_start_col_value,
    const std::vector<double>& warm_start_row_dual, bool warm_start_valid, double epsilon,
    double time_limit) {
    SolveResult result;

    highs_.changeColsCost(0, ncol_ - 1, modified_cost.data());

    // The pump's epsilon schedule drives *all three* of cuPDLP-C's
    // termination tolerances, and it does so through `kkt_tolerance`
    // (#140).  Both halves of that sentence were arrived at the hard way;
    // read the whole comment before changing either.
    //
    // What the paper says (Mexi et al., Sect. 2.2): "The standard stopping
    // criterion for PDLP is a maximum error e on the primal and dual
    // feasibilities. This error can be relaxed ... " — a *single* maximum
    // error, and the two quantities it names are the primal and dual
    // feasibilities, not the gap.  Reading implemented here: epsilon is
    // that one error, and it reaches the primal and dual feasibility
    // thresholds **as well as** the gap, rather than instead of the gap.
    // "As well as" because cuPDLP-C's termination check is a conjunction
    // over all three residuals, so relaxing a subset relaxes nothing.
    //
    // The mapping, verified against the vendored HiGHS v1.15.1
    // (`highs/pdlp/CupdlpWrapper.cpp`, `getCupdlpParams`):
    //     floatParam[D_PRIMAL_TOL] = options.primal_feasibility_tolerance;
    //     floatParam[D_DUAL_TOL]   = options.dual_feasibility_tolerance;
    //     floatParam[D_GAP_TOL]    = options.pdlp_optimality_tolerance;
    //     if (options.kkt_tolerance != kDefaultKktTolerance) {
    //       floatParam[D_PRIMAL_TOL] = floatParam[D_DUAL_TOL] =
    //           floatParam[D_GAP_TOL] = options.kkt_tolerance;
    //     }
    // Until #140 only `pdlp_optimality_tolerance` was written, so the two
    // tolerances the paper actually names sat at `kDefaultKktTolerance`
    // (1e-7) on every solve and the schedule bought only the gap term.
    //
    // WHY `kkt_tolerance` AND NOT THREE EXPLICIT WRITES.  The obvious fix
    // — write the three options directly — is wrong, and the first
    // attempt at #140 shipped it.  `primal_feasibility_tolerance` and
    // `dual_feasibility_tolerance` are not private to the PDLP solve:
    // `Highs::run` always reaches `runPresolve(force_lp_presolve=true)` on
    // this instance — the presolve-skip branch is
    // `(unconstrained_lp || has_basis || without_presolve) &&
    // solver_will_use_basis`, and it is the last conjunct that decides:
    // `solver_will_use_basis` is false for anything but `simplex` and
    // `choose`, so `solver="pdlp"` can never take it (having no basis is
    // true but is not the operative guard) — and `HPresolve` takes
    // `primal_feastol` verbatim from the option and uses it in ~100
    // places — `weaklyDominatedCol` fixes a column to a bound whenever
    // `direction * dualBound >= -dual_feasibility_tolerance`, and the
    // pump's modified costs are O(0.1-1), so at epsilon=0.01 that band is
    // 1-10% of a typical cost.  Writing those two options therefore does
    // not solve the same LP loosely, it solves a *different* LP, with
    // `x_bar` pushed to bounds by presolve rather than by PDLP.
    //
    // Measured on the bundled LPs, `highs --solver=pdlp`, presolve
    // dimensions after reductions:
    //     afiro   default 7/10/28     primal=dual=1e-2 8/11/30
    //     25fv47  default 666/1434/9659   ...        663/1427/9623
    //     afiro   kkt_tolerance=1e-2  7/10/28      (identical)
    //     25fv47  kkt_tolerance=1e-2  666/1434/9659 (identical)
    // `kkt_tolerance` is resolved into the feasibility thresholds only as
    // *function-local* variables (`HighsSolution.cpp`'s `getKktFailures`
    // and friends); nothing writes it back into
    // `options_.primal_feasibility_tolerance`, and `HighsOptions.cpp` has
    // no resolution step for it at all.  So it is the one route that
    // reaches exactly the three cuPDLP parameters and leaves LP presolve
    // bit-identical to what it did before #140 — which is what makes this
    // a pure fidelity fix rather than a fidelity fix plus an unmeasured
    // change of LP.  Same instances, iterations to convergence at 1e-2:
    // afiro 320 -> 120, 25fv47 63240 -> 520, and the objective stays
    // closer to the true LP optimum than the three-write version manages
    // (25fv47: true 5501.8, kkt route 5512.7, three-write route 5688.2).
    //
    // The cost of this route, stated plainly: it depends on an upstream
    // "if changed from its default" branch, which is the silent
    // cross-version fragility this project's HiGHS-bump guidance warns
    // about.  That risk is real but it is *detectable*, and
    // `tests/test_contested_pdlp.cpp` detects it — "a looser epsilon
    // takes strictly fewer PDLP iterations" fails outright if the branch
    // is ever removed or renamed, which no option-name check could catch.
    // What is NOT a reason to avoid this route, contrary to the first
    // attempt's comment: a scheduled epsilon of exactly 1e-7 is harmless
    // — the override does not fire, and all three parameters then sit at
    // their 1e-7 defaults, which is the value that was wanted anyway.
    //
    // Domain (`HighsOptions.h`): `kkt_tolerance` is
    // `[kMinimumKktTolerance = 1e-10, kHighsInf]`, and the schedule runs
    // from `pump::kEpsilonInit` (0.01) down to `pump::kEpsilonFloor`
    // (1e-8), so every value is in domain and no clamping is needed.
    //
    // Second-order effects, considered and accepted.
    //  (1) `getKktFailures` resolves *five* of its own thresholds from
    //      `kkt_tolerance` — both feasibility tolerances plus the primal
    //      residual, dual residual and optimality tolerances — and
    //      `HighsSolution.cpp` demotes `kOptimal` to `kUnknown` when the
    //      relative violation exceeds them.  This route keeps that check
    //      *consistent with the solve*, since both now read the same
    //      constant, and that is a second reason to prefer it: measured on
    //      afiro and 25fv47 at 1e-2, the three-write route demotes to
    //      `Unknown` on both while the `kkt_tolerance` route reports
    //      `Optimal`, exactly as the untouched default does.  It is still
    //      epsilon-dependent in principle, and `SolveResult::model_status`
    //      is a public field, so a future caller testing `== kOptimal`
    //      should read this — but it is benign in any case, because
    //      `ScyllaWorker::absorb_fresh_solve` retires a chain on `kError`
    //      and `kInfeasible` alone.
    //  (2) `analysePdlpSolution` in the same wrapper is
    //      `#if CUPDLP_DEBUG`-gated and dead in our builds.
    //  (3) `Highs.cpp`'s PDLP cleanup pass also reads `kkt_tolerance`,
    //      but sits behind `const bool consider_pdlp_cleanup = false;`.
    //  (4) What the solve hands back can be a looser LP point at large
    //      epsilon, and that is fine: `x_bar` is a *rounding guide*, not
    //      a solution we submit.  It reaches FPR as `FprConfig::lp_ref`
    //      (value selection) and `cont_fallback` (continuous fill), where
    //      every value is clamped into the propagated domain and the
    //      resulting `x_hat` has its rows re-checked by FPR itself, with
    //      anything that survives still going through `IncumbentSink` and
    //      HiGHS's own `trySolution`.
    //
    // AN INDICATIVE LOCAL A/B, not a benchmark.  Six bundled instances,
    // `suite=scylla`, `presolve_only=true`, `threads=1`, seed 0, one seed,
    // effort raised past binding so the wall clock is the only stopping
    // rule.  Pump iterations per second (fresh PDLP solves per second of
    // the scylla dispatch), before -> after:
    //     flugpl 5260 -> 5833 (1.11x)   lseu   909 -> 2482 (2.73x)
    //     egout  2221 -> 3312 (1.49x)   gt2    949 -> 1658 (1.75x)
    //     bell5   145 ->  889 (6.12x)   p0548  184 ->  359 (1.95x)
    // 2.13x geomean.  Charged effort per pump iteration falls to
    // 0.13-0.60x; that counter is `pdlp_iters * nnz` plus the round's FPR
    // rounding effort, so read the drop as a *floor* on the reduction in
    // PDLP iterations per solve, not as that quantity.
    //
    // What this does NOT establish.  It is `threads=1`, so it says nothing
    // about the contended regime `try_solve_or_snapshot` and
    // `kMaxStaleRounds` exist for.  It is one seed on six small bundled
    // instances with no variance statement.  And the issue's
    // time-to-first-incumbent criterion is effectively unmet at this
    // scale: only `lseu` produced an incumbent on both sides (12.1 ms ->
    // 5.8 ms, obj 2269 -> 2030) and `gt2` produced one only after (2.26 s,
    // obj 21166, its known optimum) — two data points are an anecdote.
    // Throughput is the only number here with any weight, and it wants a
    // real campaign before it is quoted as a result.
    //
    // One thing the A/B did establish, worth knowing before timing this
    // code: every dispatch stopped at almost exactly half its wall-clock
    // limit (bell5 measured 2.037 s / 4 s, 4.075 s / 8 s, 8.146 s / 16 s).
    // That is not this code and not a stall — the wrapped `Highs`
    // instance's own `timer_` accumulates across every pump solve while
    // the per-solve limit shrinks with the deadline, so `runPresolve`'s
    // `left = time_limit - timer_.read()` crosses zero at T/2 and returns
    // a cleared solution, which `absorb_fresh_solve` reads as a retired
    // chain.  Pre-existing and out of scope for #140 (filed as #152).  It
    // does not invalidate the comparison above, and the reason is better
    // than "it is symmetric", which was not measured: the trip point is
    // `T/(1+alpha)` for `alpha` the fraction of dispatch wall time spent
    // inside `highs_.run()` — the measured 50.9% implies alpha ~= 0.965,
    // and its constancy across 4/8/16 s is the signature of a constant-
    // alpha model — and this route lowers alpha slightly, so the
    // after-arm's window is slightly *longer*.  The metric is a rate, so a
    // longer window scales numerator and denominator together; and because
    // epsilon decays geometrically, a longer window holds proportionally
    // more of the expensive late iterations.  Any residual asymmetry
    // therefore biases the ratio downward — 2.13x is if anything
    // conservative.
    //
    // Hypothesis worth recording, not observed: a loose tolerance could in
    // principle make a 0-iteration return reachable — cuPDLP checks
    // termination at `nIter == 0`, so a warm start already satisfying
    // 1e-2 would return `pdlp_iteration_count == 0`, and
    // `pump::kMaxPdlpStalls` consecutive such rounds retire the chain.
    // Not seen in any A/B run here: at `threads=1` a retired chain ends
    // the dispatch, and every dispatch ran thousands of fresh solves to
    // its wall-clock stop.  The modified cost changes between rounds,
    // which moves the dual residual and gap even when the primal point
    // does not, so a run of three is harder to reach than it looks.
    set_option_or_die(highs_, "kkt_tolerance", epsilon);
    // The one place the wrapped instance's time limit is written, and the
    // caller's guarantee is that `time_limit > 0` (see
    // `run_locked_with_accounting`) — HiGHS reads `time_limit == 0` as *no
    // limit*, so a zero here would be the opposite of what it looks like.
    //
    // What cuPDLP-C does with it, as of HiGHS v1.15.1: `getCupdlpParams`
    // (`highs/pdlp/CupdlpWrapper.cpp`, lines 700-707) computes a
    // remaining-time adjustment and then assigns the *unadjusted*
    // `options.time_limit` to `floatParam[D_TIME_LIM]` — the adjustment is
    // dead code.  That upstream bug is load-bearing here in our favour:
    // the adjustment subtracts `timer.read()` of the wrapped `Highs`
    // instance, which accumulates across every `run()` we make on it, so
    // the "fixed" version would hand a late solve a limit of 0 — which
    // cuPDLP-C reads as "already over" rather than "no limit", stalling
    // the pump. The solver's own loop then honours the limit properly:
    // `PDHG_Solve` recomputes `dSolvingTime` every iteration and its
    // termination check includes `dSolvingTime > dTimeLim` directly, so
    // one PDLP iteration is the granularity, not one check interval.
    // If a HiGHS bump fixes line 707, re-derive this: the per-solve
    // meaning of the limit changes silently.
    set_option_or_die(highs_, "time_limit", time_limit);

    if (warm_start_valid && std::cmp_equal(warm_start_col_value.size(), ncol_) &&
        std::cmp_equal(warm_start_row_dual.size(), nrow_)) {
        HighsSolution warm;
        warm.col_value = warm_start_col_value;
        warm.row_dual = warm_start_row_dual;
        warm.value_valid = true;
        warm.dual_valid = true;
        highs_.setSolution(warm);
    }

    result.status = highs_.run();
    result.model_status = highs_.getModelStatus();
    highs_.getInfoValue("pdlp_iteration_count", result.pdlp_iters);

    const auto& sol = highs_.getSolution();
    result.col_value = sol.col_value;
    result.row_dual = sol.row_dual;
    result.value_valid = sol.value_valid;
    result.dual_valid = sol.dual_valid;

    return result;
}

ContestedPdlp::SolveTolerances ContestedPdlp::tolerances_for_test() const {
    SolveTolerances t;
    // `getOptionValue` leaves its out-parameter alone on a bad name, and
    // `SolveTolerances` zero-initialises, so a renamed-away option reads
    // back as 0.0 and fails the caller's comparison rather than passing
    // quietly.  The names are nonetheless pinned by "ContestedPdlp: every
    // PDLP option name we write exists in HiGHS", which additionally
    // establishes on a *fresh* `Highs` that the three unwritten options'
    // defaults really are `kDefaultKktTolerance` — the constant the
    // stays-at-default assertions here compare against.
    static_cast<void>(highs_.getOptionValue("kkt_tolerance", t.kkt));
    static_cast<void>(highs_.getOptionValue("primal_feasibility_tolerance", t.primal_feasibility));
    static_cast<void>(highs_.getOptionValue("dual_feasibility_tolerance", t.dual_feasibility));
    static_cast<void>(highs_.getOptionValue("pdlp_optimality_tolerance", t.pdlp_optimality));
    return t;
}

ContestedPdlp::SolveResult ContestedPdlp::run_locked_with_accounting(
    const std::vector<double>& modified_cost, const std::vector<double>& warm_start_col_value,
    const std::vector<double>& warm_start_row_dual, bool warm_start_valid, double epsilon) {
    // The solve's time limit, read here rather than by the caller (#117).
    // `solve()` blocks on `mu_` for as long as a peer's whole solve takes,
    // and a limit computed before that wait is stale by exactly it — the
    // blocking path is the one Scylla forces every `kMaxStaleRounds`, so
    // this was a doubling of the overrun on precisely the workers that had
    // waited longest.  A caller that arrived after the deadline gets no
    // solve: an empty `SolveResult` keeps its `kError` default, which
    // `ScyllaWorker::absorb_fresh_solve` turns into a retired chain.
    const double time_limit = deadline_.remaining();
    if (time_limit <= 0.0) {
        return {};
    }

    // One-solve-in-flight invariant: this counter should see at most
    // one concurrent writer.  `mu_` enforces the invariant; we track
    // the counter as a debug assertion (and a peak the tests read).
    // The RAII wrapper guarantees the decrement runs even if
    // `solve_locked` or `publish_snapshot_locked` throws — without it
    // a thrown exception would wedge `in_flight_count_ >= 1` and the
    // next call's assert fires spuriously.  R2 flagged this.
    struct InFlightGuard {
        std::atomic<int>& counter;
        ~InFlightGuard() { counter.fetch_sub(1, std::memory_order_acq_rel); }
    };
    int observed = in_flight_count_.fetch_add(1, std::memory_order_acq_rel) + 1;
    InFlightGuard guard{in_flight_count_};
    int prev_peak = peak_in_flight_.load(std::memory_order_relaxed);
    while (observed > prev_peak &&
           !peak_in_flight_.compare_exchange_weak(prev_peak, observed, std::memory_order_relaxed)) {
        // retry
    }
    // Release-mode invariant check (NOT just `assert`): a concurrent
    // PDLP solve silently corrupts cuPDLP GPU state and yields
    // garbage primals — a release-mode crash is far better than
    // returning corrupt results that then drive downstream rounding
    // and pollute the solution pool.  `assert` is a no-op under
    // `NDEBUG` (the default for `-DCMAKE_BUILD_TYPE=Release`), so we
    // also fire `std::abort` unconditionally.  The mutex `mu_` is
    // supposed to make this unreachable; if it ever fires it's a
    // structural bug (e.g. a future refactor that grew a second
    // entry path bypassing the lock).
    assert(observed == 1 && "ContestedPdlp: concurrent solve detected (cuPDLP GPU state unsafe)");
    if (observed != 1) {
        std::fprintf(stderr,
                     "ContestedPdlp: concurrent solve detected (in_flight=%d). cuPDLP GPU state "
                     "is unsafe under overlap; aborting to avoid corrupt results.\n",
                     observed);
        std::abort();
    }

    auto result = solve_locked(modified_cost, warm_start_col_value, warm_start_row_dual,
                               warm_start_valid, epsilon, time_limit);
    publish_snapshot_locked(result);
    return result;
}

void ContestedPdlp::publish_snapshot_locked(const SolveResult& result) {
    // Only publish usable snapshots (something a stale worker can round
    // against).  Failed / empty-column solves leave the previous
    // snapshot in place, which is the best we have.
    //
    // Status contract: publish on `kOk` and `kWarning`; suppress on
    // `kError`.  `kWarning` typically means HiGHS hit the iteration or
    // time limit before convergence — the returned primal is still a
    // valid (just sub-optimal) PDLP iterate, and a stale-but-valid
    // snapshot is strictly better than holding nothing for peer workers
    // running through `try_solve_or_snapshot` (issue #76).  A `kError`
    // result, by contrast, may have an undefined / partial primal, so
    // we leave the previous snapshot in place.  The trailing
    // `value_valid` guard is the fail-safe: any solve that flagged its
    // primal unusable is suppressed regardless of status.
    if (result.status == HighsStatus::kError) {
        return;
    }
    if (result.col_value.empty() || !result.value_valid) {
        return;
    }
    auto snap = std::make_shared<Snapshot>();
    snap->col_value = result.col_value;
    snap->row_dual = result.row_dual;
    snap->pdlp_iters = result.pdlp_iters;
    snap->value_valid = result.value_valid;
    snap->dual_valid = result.dual_valid;
    // Stamp the snapshot with its generation under the same mutex that
    // serialises publishes, so generation values are strictly monotonic
    // and pair 1:1 with `Snapshot` instances.  Consumers should compare
    // by generation rather than `shared_ptr` address (heap addresses
    // can be recycled; generations cannot).
    snap->generation = snapshot_generation_.fetch_add(1, std::memory_order_acq_rel) + 1;
    snapshot_.store(std::shared_ptr<const Snapshot>(std::move(snap)), std::memory_order_release);
}

void ContestedPdlp::publish_snapshot_for_test(Snapshot snap) {
    // Contract: the caller MUST hold `mu_` (typically via
    // `acquire_for_test()`).  Test fixtures use the pattern
    //   auto guard = pdlp.acquire_for_test();
    //   pdlp.publish_snapshot_for_test(std::move(seed));
    // to seed an initial snapshot while pretending to be inside a
    // production publish path; locking here ourselves would be a
    // recursive-lock deadlock against that fixture.  The lock-held
    // precondition makes the `fetch_add` → `store` pair effectively
    // atomic for any well-behaved test, matching the production
    // path's serialisation discipline.
    snap.generation = snapshot_generation_.fetch_add(1, std::memory_order_acq_rel) + 1;
    auto sp = std::make_shared<const Snapshot>(std::move(snap));
    snapshot_.store(sp, std::memory_order_release);
}

ContestedPdlp::SolveResult ContestedPdlp::solve(const std::vector<double>& modified_cost,
                                                const std::vector<double>& warm_start_col_value,
                                                const std::vector<double>& warm_start_row_dual,
                                                bool warm_start_valid, double epsilon) {
    SolveResult result;
    if (!initialized_) {
        return result;
    }
    assert(static_cast<HighsInt>(modified_cost.size()) == ncol_ ||
           ncol_ == 0 /* test-double allows empty shapes */);

    std::scoped_lock lock(mu_);
    return run_locked_with_accounting(modified_cost, warm_start_col_value, warm_start_row_dual,
                                      warm_start_valid, epsilon);
}

ContestedPdlp::TrySolveResult ContestedPdlp::try_solve_or_snapshot(
    const std::vector<double>& modified_cost, const std::vector<double>& warm_start_col_value,
    const std::vector<double>& warm_start_row_dual, bool warm_start_valid, double epsilon) {
    TrySolveResult out;
    if (!initialized_) {
        out.stale_snapshot = latest_snapshot();
        return out;
    }
    assert(static_cast<HighsInt>(modified_cost.size()) == ncol_ ||
           ncol_ == 0 /* test-double allows empty shapes */);

    std::unique_lock<std::mutex> lock(mu_, std::try_to_lock);
    if (!lock.owns_lock()) {
        // Contended — fall back to the most recent published snapshot.
        // Note: this path touches NO Highs/PDLP state, so cuPDLP GPU
        // memory is untouched while another worker is inside its solve.
        out.fresh = false;
        out.stale_snapshot = latest_snapshot();
        return out;
    }

    out.solve = run_locked_with_accounting(modified_cost, warm_start_col_value, warm_start_row_dual,
                                           warm_start_valid, epsilon);
    out.fresh = true;
    return out;
}
