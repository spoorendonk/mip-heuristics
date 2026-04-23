#include "contested_pdlp.h"

#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "pump_common.h"

#include <cassert>
#include <memory>
#include <mutex>
#include <utility>

namespace {

constexpr HighsInt kMinPdlpIterCap = 100;

}  // namespace

ContestedPdlp::ContestedPdlp(HighsMipSolver &mipsolver, HighsInt pdlp_iter_cap) {
    const auto *model = mipsolver.model_;
    auto *mipdata = mipsolver.mipdata_.get();
    ncol_ = model->num_col_;
    nrow_ = model->num_row_;
    nnz_lp_ = mipdata->ARindex_.size();
    if (ncol_ == 0 || nrow_ == 0 || nnz_lp_ == 0) {
        return;
    }

    auto lp = pump::build_lp_relaxation(*model, *mipdata);
    highs_.setOptionValue("solver", "pdlp");
    highs_.setOptionValue("output_flag", false);
    highs_.setOptionValue("pdlp_scaling", true);
    highs_.setOptionValue("pdlp_e_restart_method", 2);
    highs_.setOptionValue("pdlp_iteration_limit",
                          pdlp_iter_cap > kMinPdlpIterCap ? pdlp_iter_cap : kMinPdlpIterCap);
    highs_.passModel(std::move(lp));

    initialized_ = true;
}

ContestedPdlp::ContestedPdlp(ForTesting) {
    // Minimal init for unit tests: the subclass overrides `solve_locked`
    // so we never touch `highs_`.  ncol/nrow/nnz stay 0 by default;
    // tests that care can set them via their own friends, but most just
    // drive the lock / snapshot plumbing and don't need real shapes.
    initialized_ = true;
}

ContestedPdlp::SolveResult ContestedPdlp::solve_locked(
    const std::vector<double> &modified_cost, const std::vector<double> &warm_start_col_value,
    const std::vector<double> &warm_start_row_dual, bool warm_start_valid, double epsilon,
    double time_limit) {
    SolveResult result;

    highs_.changeColsCost(0, ncol_ - 1, modified_cost.data());
    highs_.setOptionValue("pdlp_optimality_tolerance", epsilon);
    highs_.setOptionValue("time_limit", time_limit);

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

    const auto &sol = highs_.getSolution();
    result.col_value = sol.col_value;
    result.row_dual = sol.row_dual;
    result.value_valid = sol.value_valid;
    result.dual_valid = sol.dual_valid;

    return result;
}

ContestedPdlp::SolveResult ContestedPdlp::run_locked_with_accounting(
    const std::vector<double> &modified_cost, const std::vector<double> &warm_start_col_value,
    const std::vector<double> &warm_start_row_dual, bool warm_start_valid, double epsilon,
    double time_limit) {
    // One-solve-in-flight invariant: this counter should see at most
    // one concurrent writer.  `mu_` enforces the invariant; we track
    // the counter as a debug assertion (and a peak the tests read).
    // The RAII wrapper guarantees the decrement runs even if
    // `solve_locked` or `publish_snapshot_locked` throws — without it
    // a thrown exception would wedge `in_flight_count_ >= 1` and the
    // next call's assert fires spuriously.  R2 flagged this.
    struct InFlightGuard {
        std::atomic<int> &counter;
        ~InFlightGuard() { counter.fetch_sub(1, std::memory_order_acq_rel); }
    };
    int observed = in_flight_count_.fetch_add(1, std::memory_order_acq_rel) + 1;
    InFlightGuard guard{in_flight_count_};
    int prev_peak = peak_in_flight_.load(std::memory_order_relaxed);
    while (observed > prev_peak &&
           !peak_in_flight_.compare_exchange_weak(prev_peak, observed, std::memory_order_relaxed)) {
        // retry
    }
    assert(observed == 1 && "ContestedPdlp: concurrent solve detected (cuPDLP GPU state unsafe)");

    auto result = solve_locked(modified_cost, warm_start_col_value, warm_start_row_dual,
                               warm_start_valid, epsilon, time_limit);
    publish_snapshot_locked(result);
    return result;
}

void ContestedPdlp::publish_snapshot_locked(const SolveResult &result) {
    // Only publish usable snapshots (something a stale worker can round
    // against).  Failed / empty-column solves leave the previous
    // snapshot in place, which is the best we have.
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
    snapshot_.store(std::shared_ptr<const Snapshot>(std::move(snap)), std::memory_order_release);
    snapshot_generation_.fetch_add(1, std::memory_order_acq_rel);
}

void ContestedPdlp::publish_snapshot_for_test(Snapshot snap) {
    auto sp = std::make_shared<const Snapshot>(std::move(snap));
    snapshot_.store(sp, std::memory_order_release);
    snapshot_generation_.fetch_add(1, std::memory_order_acq_rel);
}

ContestedPdlp::SolveResult ContestedPdlp::solve(const std::vector<double> &modified_cost,
                                                const std::vector<double> &warm_start_col_value,
                                                const std::vector<double> &warm_start_row_dual,
                                                bool warm_start_valid, double epsilon,
                                                double time_limit) {
    SolveResult result;
    if (!initialized_) {
        return result;
    }
    assert(static_cast<HighsInt>(modified_cost.size()) == ncol_ ||
           ncol_ == 0 /* test-double allows empty shapes */);

    std::lock_guard<std::mutex> lock(mu_);
    return run_locked_with_accounting(modified_cost, warm_start_col_value, warm_start_row_dual,
                                      warm_start_valid, epsilon, time_limit);
}

ContestedPdlp::TrySolveResult ContestedPdlp::try_solve_or_snapshot(
    const std::vector<double> &modified_cost, const std::vector<double> &warm_start_col_value,
    const std::vector<double> &warm_start_row_dual, bool warm_start_valid, double epsilon,
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
