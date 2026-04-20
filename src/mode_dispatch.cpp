#include "mode_dispatch.h"

#include "fj.h"
#include "fpr.h"
#include "local_mip.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "portfolio.h"
#include "scylla.h"

namespace heuristics {

namespace {

// Weighted effort allocation: each heuristic runs in turn with its
// proportional share of the budget and the full thread pool.
//
// The `opportunistic` flag is forwarded to all heuristics so each
// picks its deterministic vs continuous parallelism strategy.  Scylla
// uses N independent pump chains sharing a mutex-guarded PDLP solver
// (see `ContestedPdlp`), so its det/opp distinction is the same epoch
// vs opportunistic runner split used by FJ/FPR/LocalMIP.
bool run_sequential(HighsMipSolver &mipsolver, size_t budget, bool opportunistic) {
    const auto *options = mipsolver.options_mip_;

    constexpr double kWeightFj = 1.0;
    constexpr double kWeightFpr = 1.0;
    constexpr double kWeightLocalMip = 1.5;
    constexpr double kWeightScylla = 2.0;

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
    // (build_csc, precompute_var_orders, seed_pool) runs before that first
    // inner poll; checking out here skips the setup entirely once the
    // budget is exhausted.  `terminatorTerminated` is called only from
    // this sequential outer loop — the previous heuristic's parallel
    // region has already joined, so there is no concurrent access.
    const double time_limit = options->time_limit;
    const auto *mipdata = mipsolver.mipdata_.get();
    auto deadline_hit = [&]() {
        return mipdata->terminatorTerminated() || mipsolver.timer_.read() >= time_limit;
    };

    if (fj_on) {
        if (deadline_hit()) {
            return false;
        }
        if (fj::run_parallel(mipsolver, alloc(kWeightFj), opportunistic)) {
            return true;  // proven infeasible
        }
    }
    if (fpr_on) {
        if (deadline_hit()) {
            return false;
        }
        fpr::run_parallel(mipsolver, alloc(kWeightFpr), opportunistic);
    }
    if (lm_on) {
        if (deadline_hit()) {
            return false;
        }
        local_mip::run_parallel(mipsolver, alloc(kWeightLocalMip), opportunistic);
    }
    if (sc_on) {
        if (deadline_hit()) {
            return false;
        }
        scylla::run_parallel(mipsolver, alloc(kWeightScylla), opportunistic);
    }

    return false;
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
