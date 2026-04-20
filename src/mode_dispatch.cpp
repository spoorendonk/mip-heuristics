#include "mode_dispatch.h"

#include "fj.h"
#include "fpr.h"
#include "local_mip.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "portfolio.h"
#include "scylla.h"
#include "solution_pool.h"

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

    if (fj_on && !deadline_hit()) {
        if (fj::run_parallel(mipsolver, pool, alloc(kWeightFj), opportunistic)) {
            infeasible = true;
        }
    }
    if (!infeasible && fpr_on && !deadline_hit()) {
        fpr::run_parallel(mipsolver, pool, alloc(kWeightFpr), opportunistic);
    }
    if (!infeasible && lm_on && !deadline_hit()) {
        local_mip::run_parallel(mipsolver, pool, alloc(kWeightLocalMip), opportunistic);
    }
    if (!infeasible && sc_on && !deadline_hit()) {
        scylla::run_parallel(mipsolver, pool, alloc(kWeightScylla), opportunistic);
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
