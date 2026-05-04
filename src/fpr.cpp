#include "fpr.h"

#include "epoch_runner.h"
#include "fpr_core.h"
#include "fpr_strategies.h"
#include "heuristic_common.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "opportunistic_runner.h"
#include "parallel/HighsParallel.h"
#include "parallel_setup.h"
#include "solution_pool.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace fpr {

// Precomputed variable orders indexed by kFprStrategies position.  Built
// once sequentially before any parallel region (some strategies call
// HighsCliqueTable::cliquePartition which is not thread-safe).
using VarOrderTable = std::vector<std::vector<HighsInt>>;

// EpochWorker driving the lifecycle introduced in issue #77: an attempt
// is the unit of work, and an attempt's DFS may pause at the per-epoch
// budget gate and resume next epoch with state intact.  When an attempt
// verdicts (feasible / failed), the worker advances `attempt_idx_` and
// picks the next (strategy, mode) from a deterministic per-worker
// rotation `(worker_idx_ + attempt_idx_)`.  Attempts never share a
// queue across workers — that determinism rule is what lets two runs
// with the same seed produce bit-identical [Sequential] traces.
//
// Satisfies the EpochWorker concept from epoch_runner.h.  In det mode
// `finished()` always returns false; the outer epoch loop's
// stale_budget is the only termination gate.
class FprWorker {
public:
    FprWorker(HighsMipSolver &mipsolver, const CscMatrix &csc, SolutionPool &pool,
              const VarOrderTable &var_orders, int worker_idx, uint32_t seed,
              size_t attempt_budget);

    EpochResult run_epoch(size_t epoch_budget);

    bool finished() const { return false; }

    // Reset the cross-worker improvement-broadcast bookkeeping.  Called by
    // the runner at the epoch barrier when any peer improved last epoch.
    // The lifecycle's mid-attempt staleness has no per-worker counter to
    // touch (the epoch loop's effort_since_improvement is the source of
    // truth), so this is a no-op in det mode.
    void reset_staleness() {}

private:
    // Pick the (strategy, mode) for `attempt_idx_`.  Cycles the
    // paper-curated `kInitialFprConfigs` list (8 entries) keyed on
    // `(worker_idx + attempt_idx) % kNumInitialFprConfigs`.  See the body
    // comment for why this is the curated list rather than the full 8×5
    // grid (a second `repair_search` activity-undo gap is the residual
    // blocker).
    void select_config_for_current_attempt();

    HighsMipSolver &mipsolver_;
    const CscMatrix &csc_;
    SolutionPool &pool_;
    const VarOrderTable &var_orders_;

    int worker_idx_;
    size_t attempt_budget_;  // hint for cfg.max_effort per attempt

    int strat_idx_ = 0;
    FrameworkMode mode_ = FrameworkMode::kDfs;

    int attempt_idx_ = 0;
    bool attempt_alive_ = false;
    FprAttemptState attempt_state_;

    Rng rng_;
    FprScratch scratch_;
    // Reused across attempts to avoid `std::vector<double>` churn — the
    // multi-attempt loop in `run_epoch` calls `pool_.get_restart` once
    // per attempt, and an unhoisted local would re-allocate every
    // iteration on instances large enough to matter (review R2 CF-1).
    std::vector<double> initial_solution_buf_;
};

static_assert(EpochWorker<FprWorker>, "FprWorker must satisfy EpochWorker concept");

namespace {

// Master strategy pool for all FPR parallel paths.  var_orders are
// precomputed for each entry (see precompute_var_orders) so any strategy
// — including clique-based ones like kStratBadobjcl whose compute_var_order
// calls HighsCliqueTable::cliquePartition — can be used inside a parallel
// region without racing on cliquePartition's internal state.
constexpr FprStrategyConfig kFprStrategies[] = {
    // Strategies used by the paper's curated initial configs.
    kStratBadobjcl,  // 0: type+cliques / badobj
    kStratLocks2,    // 1: locks / loosedyn
    kStratLocks,     // 2: LR / loosedyn
    kStratRandom,    // 3: type+cliques / random
    kStratDomsize,   // 4: domainSize / loosedyn
    // Extra strategies kept for randomization diversity at restart.
    kStratRandom2,  // 5: random / random
    kStratBadobj,   // 6: type / badobj
    kStratGoodobj,  // 7: type / goodobj
};
constexpr int kNumFprStrategies = static_cast<int>(std::size(kFprStrategies));

// Paper Section 6.3, Class 1 — LP-free initial configs.  Each entry gives
// a worker its starting (strategy, mode); strat_idx is an index into
// kFprStrategies.
struct InitialFprConfig {
    int strat_idx;
    FrameworkMode mode;
};
constexpr InitialFprConfig kInitialFprConfigs[] = {
    {0, FrameworkMode::kDfs},           // kStratBadobjcl, dfs
    {1, FrameworkMode::kDfs},           // kStratLocks2, dfs
    {1, FrameworkMode::kDive},          // kStratLocks2, dive
    {2, FrameworkMode::kDfsrep},        // kStratLocks, dfsrep
    {0, FrameworkMode::kDfsrep},        // kStratBadobjcl, dfsrep
    {3, FrameworkMode::kDiveprop},      // kStratRandom, diveprop
    {2, FrameworkMode::kRepairSearch},  // kStratLocks, repairsearch
    {4, FrameworkMode::kDfs},           // kStratDomsize, dfs
};
constexpr int kNumInitialFprConfigs = static_cast<int>(std::size(kInitialFprConfigs));

constexpr FrameworkMode kAllModes[] = {
    FrameworkMode::kDfs,      FrameworkMode::kDfsrep,       FrameworkMode::kDive,
    FrameworkMode::kDiveprop, FrameworkMode::kRepairSearch,
};

constexpr int kNumAllModes = static_cast<int>(std::size(kAllModes));

// Compute variable orders for every strategy in kFprStrategies.  MUST be
// called from a sequential context: clique-based var_strategies invoke
// HighsCliqueTable::cliquePartition which mutates internal state and is
// not thread-safe.
VarOrderTable precompute_var_orders(HighsMipSolver &mipsolver) {
    VarOrderTable orders(kNumFprStrategies);
    const uint32_t base = heuristic_base_seed(mipsolver.options_mip_->random_seed);
    for (int i = 0; i < kNumFprStrategies; ++i) {
        Rng rng(base + static_cast<uint32_t>(i));
        orders[i] = compute_var_order(mipsolver, kFprStrategies[i].var_strategy, rng, nullptr);
    }
    return orders;
}

}  // namespace

// ---------------------------------------------------------------------------
// FprWorker implementation
// ---------------------------------------------------------------------------

FprWorker::FprWorker(HighsMipSolver &mipsolver, const CscMatrix &csc, SolutionPool &pool,
                     const VarOrderTable &var_orders, int worker_idx, uint32_t seed,
                     size_t attempt_budget)
    : mipsolver_(mipsolver),
      csc_(csc),
      pool_(pool),
      var_orders_(var_orders),
      worker_idx_(worker_idx),
      attempt_budget_(attempt_budget),
      rng_(seed) {
    select_config_for_current_attempt();
}

void FprWorker::select_config_for_current_attempt() {
    // Per-worker rotation through the paper-curated `kInitialFprConfigs`
    // list (8 entries), keyed deterministically on
    // `(worker_idx + attempt_idx) % kNumInitialFprConfigs`.  Each worker
    // visits every Class-1 config exactly once before wrapping.  Issue #77's
    // determinism rule is satisfied because the rotation is purely a
    // function of (worker identity, attempt count) — no shared queue, no
    // rng dependency, no per-attempt randomisation.
    //
    // Why not the full 8 × 5 = 40-pair (strategy, mode) grid?  An earlier
    // draft widened to it once `e_pq_mark` threading was in place, but
    // the `(kStratDomsize, kRepairSearch)` pairing exposed a second
    // latent state-restoration gap in `repair_search`'s secondary
    // backtrack: `act_mark` is not threaded through `RepairSearchNode`
    // analogously to `e_pq_mark`, so when `init_activities()` ran in
    // Phase 2 (any `kLoosedyn` value strategy) the activity vectors and
    // `vs_` diverge across the secondary backtrack.  `kStratDomsize` is
    // the only entry that simultaneously uses `init_domain_pq` AND a
    // `kLoosedyn` val strategy AND was widened to a `kRepairSearch`
    // mode the curated list never exercised — so it is the smallest
    // reproducer.  Fix is the same shape as `e_pq_mark` (extend
    // `RepairSearchNode` with `e_act_mark` and pass it to
    // `E.backtrack_to`); kept out of this change to bound scope.  Until
    // then the curated list keeps the rotation safe.  Multi-attempt
    // looping inside `run_epoch` still lets fast workers fill the slice
    // by cycling through the 8-config list, which the issue's #1
    // acceptance bullet (FPR CPU% on tbfp-network) cares about.
    const int idx = ((worker_idx_ + attempt_idx_) % kNumInitialFprConfigs + kNumInitialFprConfigs) %
                    kNumInitialFprConfigs;
    const auto &cfg = kInitialFprConfigs[idx];
    strat_idx_ = cfg.strat_idx;
    mode_ = cfg.mode;
}

EpochResult FprWorker::run_epoch(size_t epoch_budget) {
    EpochResult epoch{};

    // Issue #77 lifecycle.  Two mechanics in play:
    //
    // (1) Pause/resume *across* calls.  When the DFS exhausts the per-call
    //     slice, `fpr_attempt_step` returns `kBudgetGate` and we return so
    //     peers can run their next slice.  The DFS state (var_order_cursor,
    //     nodes_visited, found_complete, dfs_stack, prop_engine) lives in
    //     `attempt_state_` + `scratch_` and is preserved until the next call
    //     resumes the same attempt.  Without this, a long DFS subtree gets
    //     truncated and discarded each epoch — the parallelism bottleneck
    //     this issue exists to fix.
    //
    // (2) Multi-attempt fill *within* a call.  When an attempt verdicts
    //     before exhausting the slice (e.g. a fast-failing strategy on a
    //     hard instance, or a leaf found in 5% of the budget), we start
    //     the next attempt with the next rotation slot rather than idling
    //     at the runner's barrier waiting for slow peers.  This is what
    //     drives FPR's CPU% up on `tbfp-network` (acceptance bullet #1) —
    //     pause/resume alone fixes truncation but leaves fast workers
    //     idle while a slow worker holds the barrier.
    //
    // The safety cap (`kMaxAttemptsPerCall`) and no-progress guard guard
    // against degenerate models where every attempt verdicts with near-zero
    // recorded effort (`infeasible-mip0` initial-propagation short-circuit):
    // without bounds we'd burn the slice on `fpr_attempt_begin`'s
    // O(ncol+nrow) setup churn alone.

    auto *mipdata = mipsolver_.mipdata_.get();
    const double time_limit = mipsolver_.options_mip_->time_limit;

    constexpr int kMaxAttemptsPerCall = 32;
    int attempts_started = 0;
    size_t prev_loop_effort = 0;

    while (epoch.effort < epoch_budget) {
        if (mipdata->terminatorTerminated() || mipsolver_.timer_.read() >= time_limit) {
            break;
        }
        if (attempts_started > 0 && epoch.effort == prev_loop_effort) {
            // Defensive: previous iteration produced zero accounted effort
            // (degenerate model, attempt_idx > 0 random init that no-ops, etc.).
            // Without this guard we would loop forever on a near-zero-cost
            // attempt while never tripping the runner's outer stale gate.
            break;
        }
        prev_loop_effort = epoch.effort;

        // Advance the per-worker rotation BEFORE building cfg so that
        // `cfg.strategy` / `cfg.mode` / `cfg.precomputed_var_order` reflect
        // the current attempt's choice.  Earlier draft built cfg from the
        // previous attempt's strat/mode and re-assigned 4 fields after the
        // rotation advance — a maintenance hazard if a future cfg field
        // is added (review R2 CF-2).
        if (!attempt_alive_) {
            if (attempts_started >= kMaxAttemptsPerCall) {
                break;
            }
            ++attempts_started;
            select_config_for_current_attempt();
        }

        const auto &strat = kFprStrategies[strat_idx_];
        const auto &var_order = var_orders_[strat_idx_];
        FprConfig cfg{};
        // `cfg.max_effort` is the attempt-wide cap consumed by Phase 3 sub-
        // budgets (`cfg.max_effort - total_prop_work` for repair_search /
        // walksat).  Sized at the worker's `attempt_budget_` (=
        // ParallelSetup::stale_budget = max_effort/4), not the per-call
        // `epoch_budget`: when an attempt spans multiple `run_epoch` calls,
        // the cumulative `total_prop_work` arriving at Phase 3 already
        // exceeds any single slice, so a slice-sized cap clamps the repair
        // budget to 0 (review R1 CF-1).  The DFS gate inside
        // `fpr_attempt_step` uses `effort_remaining` (the per-call slice)
        // and is unaffected by this size — Phase 3's iteration counts
        // (`cfg.repair_iterations`, `cfg.walksat_iterations`) self-throttle
        // even when the effort budget is large.
        cfg.max_effort = std::max<size_t>(attempt_budget_, 1);
        cfg.hint = nullptr;
        cfg.scores = nullptr;
        cfg.cont_fallback = nullptr;
        cfg.csc = &csc_;
        cfg.mode = mode_;
        cfg.strategy = &strat;
        cfg.lp_ref = nullptr;
        cfg.precomputed_var_order = var_order.data();
        cfg.precomputed_var_order_size = static_cast<HighsInt>(var_order.size());
        cfg.scratch = &scratch_;

        if (!attempt_alive_) {
            // `initial_solution_buf_` is hoisted to a member to avoid
            // reallocating an `ncol`-sized vector every loop iteration on
            // tbfp-network-scale instances (review R2 CF-1).  pool_.get_restart
            // overwrites the contents.
            initial_solution_buf_.clear();
            const double *init_ptr = nullptr;
            if (pool_.get_restart(rng_, initial_solution_buf_)) {
                init_ptr = initial_solution_buf_.data();
            }
            fpr_attempt_begin(attempt_state_, mipsolver_, cfg, rng_, attempt_idx_, init_ptr);
            attempt_alive_ = true;
            epoch.effort += attempt_state_.effort_consumed;
        }

        if (attempt_state_.phase == FprAttemptState::Phase::kDfs) {
            const size_t before_step = attempt_state_.effort_consumed;
            const size_t budget_remaining =
                epoch_budget > epoch.effort ? epoch_budget - epoch.effort : 0;
            const FprStepResult outcome =
                fpr_attempt_step(attempt_state_, mipsolver_, cfg, rng_, budget_remaining);
            epoch.effort += attempt_state_.effort_consumed - before_step;
            if (outcome == FprStepResult::kBudgetGate) {
                // Attempt paused at the per-call slice boundary — return so
                // peers do their next epoch's work and we resume here next call.
                return epoch;
            }
            // kVerdictReady — DFS ended (leaf found or stack/node-limit
            // exhausted), proceed to finish.
        }

        const size_t before_finish = attempt_state_.effort_consumed;
        HeuristicResult result = fpr_attempt_finish(attempt_state_, mipsolver_, cfg, rng_);
        epoch.effort += attempt_state_.effort_consumed - before_finish;

        if (result.found_feasible) {
            pool_.try_add(result.objective, result.solution, kSolutionSourceFPR);
            epoch.found_improvement = true;
        }

        ++attempt_idx_;
        attempt_alive_ = false;
    }

    return epoch;
}

// ---------------------------------------------------------------------------
// Parallel epoch-gated FPR
// ---------------------------------------------------------------------------

namespace {

size_t run_parallel_deterministic(HighsMipSolver &mipsolver, SolutionPool &pool,
                                  size_t max_effort) {
    ParallelSetup setup(mipsolver, max_effort);

    // Precompute var_orders sequentially before any parallel region.
    VarOrderTable var_orders = precompute_var_orders(mipsolver);

    std::vector<std::unique_ptr<FprWorker>> workers;
    workers.reserve(setup.N);
    for (size_t w = 0; w < setup.N; ++w) {
        uint32_t seed = setup.base_seed + static_cast<uint32_t>(w) * kSeedStride;
        workers.push_back(std::make_unique<FprWorker>(
            mipsolver, setup.csc, pool, var_orders, static_cast<int>(w), seed, setup.stale_budget));
    }

    return run_epoch_loop(
        mipsolver, workers, max_effort, setup.epoch_budget(kEpochsPerWorker),
        [](int) { /* FprWorker::finished() is always false post-#77 — det mode runs
                     until the outer epoch loop's stale_budget fires. */
        },
        setup.stale_budget);
}

size_t run_parallel_opportunistic(HighsMipSolver &mipsolver, SolutionPool &pool,
                                  size_t max_effort) {
    ParallelSetup setup(mipsolver, max_effort);

    // Precompute var_orders sequentially before any parallel region.
    VarOrderTable var_orders = precompute_var_orders(mipsolver);

    std::vector<std::unique_ptr<FprWorker>> workers;
    workers.reserve(setup.N);
    for (size_t w = 0; w < setup.N; ++w) {
        uint32_t seed = setup.base_seed + static_cast<uint32_t>(w) * kSeedStride;
        workers.push_back(std::make_unique<FprWorker>(
            mipsolver, setup.csc, pool, var_orders, static_cast<int>(w), seed, setup.stale_budget));
    }

    struct FprOppState {
        int worker_idx;
    };

    return run_opportunistic_loop(
        mipsolver, static_cast<int>(setup.N), max_effort, setup.stale_budget, setup.default_run_cap,
        setup.base_seed,
        [](int worker_idx, Rng & /*rng*/) -> FprOppState { return FprOppState{worker_idx}; },
        [&](FprOppState &state, Rng & /*rng*/, size_t run_cap) -> HeuristicResult {
            auto &worker = workers[state.worker_idx];
            // FprWorker::finished() returns false unconditionally
            // post-#77; the opportunistic loop's own staleness gate is
            // the termination signal.  No worker-level replacement
            // needed.
            auto epoch = worker->run_epoch(run_cap);
            HeuristicResult result;
            result.effort = epoch.effort;
            if (epoch.found_improvement) {
                result.found_feasible = true;
                result.objective = pool.snapshot().best_objective;
            }
            return result;
        });
}

}  // namespace

size_t run_parallel(HighsMipSolver &mipsolver, SolutionPool &pool, size_t max_effort,
                    bool opportunistic) {
    const auto *model = mipsolver.model_;
    const HighsInt ncol = model->num_col_;
    const HighsInt nrow = model->num_row_;
    if (ncol == 0 || nrow == 0) {
        return 0;
    }

    if (opportunistic) {
        return run_parallel_opportunistic(mipsolver, pool, max_effort);
    }
    return run_parallel_deterministic(mipsolver, pool, max_effort);
}

}  // namespace fpr
