#include "contested_pdlp.h"

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
        // floored at `pump::kEpsilonFloor`=1e-8 against a 1e-10 minimum;
        // `time_limit` is guarded `> 0` by the caller), so this cannot
        // fire on legitimate solve data.
        assert(false && "ContestedPdlp: unknown or invalid HiGHS option");
        std::abort();
    }
}

}  // namespace

ContestedPdlp::ContestedPdlp(HighsMipSolver& mipsolver, HighsInt pdlp_iter_cap) {
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
    // `kWeight*` effort calibration, was already produced on the HiGHS
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
    // Leaving them unset keeps behaviour bit-identical to what the
    // `kWeight*` effort calibration was measured against.
    set_option_or_die(highs_, "pdlp_iteration_limit",
                      pdlp_iter_cap > kMinPdlpIterCap ? pdlp_iter_cap : kMinPdlpIterCap);
    highs_.passModel(std::move(lp));

    initialized_ = true;
}

ContestedPdlp::ContestedPdlp(ForTesting /*unused*/) {
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
    set_option_or_die(highs_, "pdlp_optimality_tolerance", epsilon);
    set_option_or_die(highs_, "time_limit", time_limit);

    if (warm_start_valid && static_cast<HighsInt>(warm_start_col_value.size()) == ncol_ &&
        static_cast<HighsInt>(warm_start_row_dual.size()) == nrow_) {
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

ContestedPdlp::SolveResult ContestedPdlp::run_locked_with_accounting(
    const std::vector<double>& modified_cost, const std::vector<double>& warm_start_col_value,
    const std::vector<double>& warm_start_row_dual, bool warm_start_valid, double epsilon,
    double time_limit) {
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
                                                bool warm_start_valid, double epsilon,
                                                double time_limit) {
    SolveResult result;
    if (!initialized_) {
        return result;
    }
    assert(static_cast<HighsInt>(modified_cost.size()) == ncol_ ||
           ncol_ == 0 /* test-double allows empty shapes */);

    std::scoped_lock lock(mu_);
    return run_locked_with_accounting(modified_cost, warm_start_col_value, warm_start_row_dual,
                                      warm_start_valid, epsilon, time_limit);
}

ContestedPdlp::TrySolveResult ContestedPdlp::try_solve_or_snapshot(
    const std::vector<double>& modified_cost, const std::vector<double>& warm_start_col_value,
    const std::vector<double>& warm_start_row_dual, bool warm_start_valid, double epsilon,
    double time_limit) {
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
                                           warm_start_valid, epsilon, time_limit);
    out.fresh = true;
    return out;
}
