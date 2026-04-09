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
bool run_sequential(HighsMipSolver &mipsolver, size_t budget) {
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

    if (fj_on) {
        if (fj::run_parallel(mipsolver, alloc(kWeightFj))) {
            return true;  // proven infeasible
        }
    }
    if (fpr_on) {
        fpr::run_parallel(mipsolver, alloc(kWeightFpr));
    }
    if (lm_on) {
        local_mip::run_parallel(mipsolver, alloc(kWeightLocalMip));
    }
    if (sc_on) {
        scylla::run_parallel(mipsolver, alloc(kWeightScylla));
    }

    return false;
}

}  // namespace

bool run_presolve(HighsMipSolver &mipsolver, size_t budget) {
    const auto *options = mipsolver.options_mip_;

    if (options->mip_heuristic_portfolio) {
        portfolio::run_presolve(mipsolver, budget);
        // In opportunistic mode, Scylla is not yet a portfolio arm — run
        // standalone.  Deterministic portfolio includes Scylla as an arm.
        if (options->mip_heuristic_portfolio_opportunistic && options->mip_heuristic_run_scylla) {
            if (options->mip_heuristic_scylla_parallel) {
                scylla::run_parallel(mipsolver, budget);
            } else {
                scylla::run(mipsolver, budget);
            }
        }
    } else {
        if (run_sequential(mipsolver, budget)) {
            return true;
        }
    }

    return false;
}

}  // namespace heuristics
